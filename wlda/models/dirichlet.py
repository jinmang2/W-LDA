from collections import OrderedDict
from typing import Optional, List, Dict, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ENet, DNet
from .utils.outputs import WAEOutput
from ..args import ModelArguments
from ..utils import NON_LINEARITY

from compute_op import mmd_loss


def calc_mean_sum(tensor: torch.Tensor, dim: int = -1):
    return torch.mean(torch.sum(tensor, dim=-1))


def calc_entropy(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-10):
    return calc_mean_sum(-tensor * torch.log(tensor + eps), dim=dim)


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


# class Discriminator_y(Net):
#     """
#     A Neural Network Module Disctiminator class
#     Is is similar to Encoder class
#     """

#     def __init__(
#         self,
#         output_dim: int = 2,
#         n_hidden: Union[List[int], int] = 64,
#         ndim_y: int = 16,
#         n_layers: int = 1,
#         weights_file: str = "",
#         non_linearity: Optional[str] = None,
#         apply_softmax: bool = False,
#     ):
#         super(Encoder, self).__init__()

#         self.weights_file = args.dis_freeze
#         self.apply_softmax = apply_softmax

#         if n_layers >= 0:
#             if isinstance(n_hidden, list):
#                 n_hidden = n_hidden[0]
#             n_hidden = n_layers * [n_hidden]
#         else:
#             n_layers = len(n_hidden)

#         n_hidden.insert(0, input_dim)

#         main = [
#             Dense(n_hidden[i], n_hidden[i+1], non_linearity=non_linearity)
#             for i in range(n_layers)
#         ]
#         main.append(Dense(n_hidden[-1], ndim_y, non_linearity=None))
#         self.main = nn.Sequential(main)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         logit = self.main(x)
#         if self.apply_softmax:
#             return torch.softmax(logit)
#         return logit

class WassersteinAutoEncoder(Net):

    def __init__(
        self,
        args: Optional[Union[ModelArguments, Dict]] = None,
    ):
        super().__init__()

        if args is None:
            args = ModelArguments()
        self.args = args
        
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        if self.encoder.freeze:
            self.encoder.freeze_params()
        if self.decoder.freeze:
            self.decoder.freeze_params()

    @property
    def add_noise(self) -> bool:
        if self.training and self.args.latent_noise > 0:
            return True
        return False

    def forward(
        self, 
        docs: torch.Tensor,
        eps: float = 1e-10,
    ) -> WAEOutput:
        batch_size = docs.shape[0] # batch first
        y_onehot_u = self.encoder(docs)
        y_onehot_u_softmax = torch.softmax(y_onehot_u, dim=-1)
        if self.add_noise:
            # Mix-up
            alpha = self.args.latent_noise
            y_noise = self._sampling_y_over_dirich(batch_size)
            y_onehot_u_softmax = (1 - alpha) * y_onehot_u_softmax + alpha * y_noise
        x_reconstruction_u = self.decoder(y_onehot_u_softmax)
        logits = torch.log_softmax(x_reconstruction_u, dim=-1)
        # calc reconstructoin loss (CELoss)
        loss_reconstruction = calc_mean_sum(-docs * logits)
        # calc l2 regularizer
        loss_l2_regularizer = calc_mean_sum(y_onehot_u ** 2)
        # calc MMD loss
        y_true = self._sampling_y_over_dirich(batch_size)
        y_fake = torch.softmax(self.encoder(docs), dim=-1)
        loss_discriminator = mmd_loss(y_true, y_fake, t=self.args.kernel_alpha)

        with torch.no_grad():
            latent_max = torch.zeros(self.args.ndim_y).to(self.device)
            latent_max[y_onehot_u.argmax(-1)] += 1
            latent_max /= batch_size

            latent_entropy = calc_entropy(y_onehot_u_softmax)

            latent_v = torch.mean(y_onehot_u_softmax, axis=0)

            dirich_entropy = calc_entropy(y_true)

        return WAEOutput(
            loss_reconstruction=loss_reconstruction,
            loss_discriminator=loss_discriminator,
            loss_l2_regularizer=loss_l2_regularizer,
            latent_max=latent_max,
            latent_entropy=latenr_entropy,
            latent_v=latent_v,
            dirich_entropy=dirich_entropy,
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
