from collections import OrderedDict
from typing import Optional, List, Dict, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Net, Dense
from ..args import ModelArguments


class Encoder(Net):
    """
    A Neural Network Module Encoder class
    """

    def __init__(self, config):
        super().__init__()

        self.freeze = config.enc_freeze
        self.weights_file = config.enc_weights

        if config.enc_n_layers >= 0:
            if isinstance(config.enc_n_hiddens, list):
                n_hidden = config.enc_n_hiddens[0]
            n_hidden = config.enc_n_layers * [n_hidden]
            n_layers = config.enc_n_layers
        else:
            n_hidden = config.enc_n_hiddens
            n_layers = len(config.enc_n_hiddens)

        n_hidden.insert(0, config.ndim_x)

        main = [
            Dense(n_hidden[i], n_hidden[i+1],
                  non_linearity=config.enc_nonlinearity)
            for i in range(n_layers)
        ]
        main.append(Dense(n_hidden[-1], config.ndim_y, non_linearity=None))
        self.main = nn.Sequential(*main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Decoder(Net):
    """
    A Neural Network Module Decoder class
    """

    def __init__(self, config):
        super().__init__()

        self.freeze = config.dec_freeze
        self.weights_file = config.dec_weights

        if isinstance(config.dec_n_hiddens, list):
            n_hidden = config.dec_n_hiddens[0]
        else:
            n_hidden = config.dec_n_hiddens
        self.main = Dense(config.ndim_y, n_hidden, non_linearity=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


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

#         self.weights_file = config.dis_freeze
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
        config: Optional[Union[ModelArguments, Dict]] = None,
    ):
        super().__init__()

        self.config = config

        if config is None:
            config = ModelArguments()
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        if self.encoder.freeze:
            self.encoder.freeze_params()
        if self.decoder.freeze:
            self.decoder.freeze_params()

    def forward(self, docs: torch.Tensor, enc_out_corrupt: bool = False) -> torch.Tensor:
        y_onehot_u = self.encoder(docs)
        y_onehot_u_softmax = torch.softmax(y_onehot_u, dim=-1)
        if enc_out_corrupt and self.training:
            # @TODO: Consifer ``torch.distrubitions.dirichlet.Distribution```
            alpha = self.config.latent_noise
            y_noise = np.random.dirichlet(
                np.ones(self.config.ndim_y) * self.config.dirich_alpha,
                size=docs.shape[0])
            y_noise = torch.FLoatTensor(y_noise)
            # Mix-up
            y_onehot_u_softmax = (1 - alpha) * y_onehot_u_softmax + alpha * y_noise
        x_reconstruction_u = self.decoder(y_onehot_u_softmax)
        return x_reconstruction_u
