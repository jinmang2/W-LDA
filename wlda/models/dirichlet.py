from collections import OrderedDict
from typing import Optional, List, Dict, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ENet, DNet
from .outputs import WAEModelOutput, WAEReconstructionOutput
from ..args import ModelArguments
from ..utils import NON_LINEARITY

from compute_op import mmd_loss


class Dense(nn.Module):
    """
    A Linear class with non-linearity (mxnet style)
    """

    def __init__(self, *args, non_linearity="sigmoid", **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.activation = NON_LINEARITY.get(non_linearity, nn.Identity)()

    def forward(self, x):
        return self.activation(self.linear(x))


class Encoder(ENet):
    """
    A Neural Network Module Encoder class
    """

    def __init__(self, args):
        super().__init__()

        self.freeze = args.enc_freeze
        self.weights_file = args.enc_weights

        if args.enc_n_layers >= 0:
            if isinstance(args.enc_n_hiddens, list):
                n_hidden = args.enc_n_hiddens[0]
            n_hidden = args.enc_n_layers * [n_hidden]
            n_layers = args.enc_n_layers
        else:
            n_hidden = args.enc_n_hiddens
            n_layers = len(args.enc_n_hiddens)

        n_hidden.insert(0, args.ndim_x)

        main = [
            Dense(n_hidden[i], n_hidden[i+1],
                  non_linearity=args.enc_nonlinearity)
            for i in range(n_layers)
        ]
        main.append(Dense(n_hidden[-1], args.ndim_y, non_linearity=None))
        self.main = nn.Sequential(*main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Decoder(DNet):
    """
    A Neural Network Module Decoder class
    """

    def __init__(self, args):
        super().__init__()

        self.freeze = args.dec_freeze
        self.weights_file = args.dec_weights

        if isinstance(args.dec_n_hiddens, list):
            n_hidden = args.dec_n_hiddens[0]
        else:
            n_hidden = args.dec_n_hiddens
        self.main = Dense(args.ndim_y, n_hidden, non_linearity=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def y_as_topics(self, eps=1e-10):
        # main.in_features == ndim_y
        return torch.eye(self.main.in_features, device=self.device)


class WAEInitModel(AENet, metaclass=ABCMeta):
    def __init__(self, args: ModelArguments):
        self.args = args

    def _init_weights(self, module: Net):
        if self.args.init_type in self.INIT_TYPES:
            getattr(module, "init_weights")(self.args.init_type)


class WAEModel(WAEInitModel):
    """
    Wasserstein Auto Encoder Model
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        if self.encoder.freeze:
            self.encoder.freeze_params()
        if self.decoder.freeze:
            self.decoder.freeze_params()

    @property
    def encoder(self):
        return self.encoder

    @property
    def decoder(self):
        return self.decoder

    @property
    def discriminator(self):
        return None

    @property
    def add_noise(self) -> bool:
        if self.training and self.args.latent_noise > 0:
            return True
        return False

    def forward(
        self, 
        input_embeds: torch.Tensor,
        eps: float = 1e-10,
    ) -> WAEModelOutput:
        batch_size = docs.shape[0] # batch first
        y_onehot_u = self.encoder(docs)
        y_onehot_u_softmax = torch.softmax(y_onehot_u, dim=-1)

        if self.add_noise:
            # Mix-up
            alpha = self.args.latent_noise
            y_noise = self._sampling_y_over_dirich(batch_size)
            y_onehot_u_softmax = (1 - alpha) * y_onehot_u_softmax + alpha * y_noise

        x_reconstruction_u = self.decoder(y_onehot_u_softmax)

        return WAEModelOutput(
            original_documents=docs,
            doc_topic_vec=y_onehot_u,
            doc_topic_vec_prob=y_onehot_u_softmax,
            reconstructed_documents=x_reconstruction_u,
        )

    def _sampling_y_over_dirich(
        self, 
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # @TODO: Consider ``torch.distrubitions.dirichlet.Distribution```
        y = np.random.dirichlet(
            np.ones(self.args.ndim_y) * self.args.dirich_alpha,
            size=batch_size,
        )
        device = device if device is not None else self.device
        return torch.tensor(y, dtype=self.dtype, device=device)


class WAEForTopicModeling(WAEInitModel):
    def __init__(
        self,
        args: Optional[Union[ModelArguments, Dict]] = None,
    ):
        if args is None:
            args = ModelArguments()
        super().__init__(args)
        self.model = WAEModel(args)
        self._init_weights(self.model)

    def forward(
        self, 
        input_embeds: torch.Tensor,
        eps: float = 1e-10,
    ) -> WAELossOutput:
        outputs = self.model(docs, eps)
        logits = torch.log_softmax(outputs.reconstructed_documents, dim=-1)
        # Cross Entropy Loss
        loss_reconstruction = torch.mean(torch.sum(-docs * logits, dim=-1))
        return WAEReconstructionOutput(
            loss=loss_reconstruction,
            logits=logits,
            original_documents=outputs.original_documents,
            doc_topic_vec=outputs.doc_topic_vec,
            doc_topic_vec_prob=outputs.doc_topic_vec_prob,
            reconstructed_documents=outputs.reconstructed_documents,
        )


class WAEForLDASynthetic(WAEInitModel):
    pass