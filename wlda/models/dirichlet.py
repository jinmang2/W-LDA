from collections import OrderedDict
from typing import Optional, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import Net, Dense

from ..args import ModelArguments


class Encoder(Net):
    """
    A Neural Network Module Encoder class
    """

    def __init__(
        self,
        input_dim : int,
        n_hidden : Optional[List[int], int] = 64,
        ndim_y : int = 16,
        n_layers : int = 1,
        weights_file : str = "",
        freeze : bool = False,
        non_linearity : Optional[str] = None,
        **kwargs
    ):
        super(Encoder, self).__init__(self)

        self.freeze = freeze
        self.weights_file = weights_file

        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hidden[0]
            n_hidden = n_laers * [n_hidden]
        else:
            n_layers = len(n_hidden)

        n_hidden.insert(0) = input_dim

        main = [
            Dense(n_hidden[i], n_hidden[i+1], non_linearity=non_linearity)
            for i in range(len(n_layers))
        ]
        main.append(Dense(n_hidden[-1], ndim_y, non_linearity=None))
        self.main = nn.Sequential(main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Decoder(Net):
    """
    A Neural Network Module Decoder class
    """

    def __init__(
        self,
        input_dim: int,
        n_hidden: Optional[List[int], int] = 64,
        ndim_y: int = 16,
        n_layers: int = 1,
        weights_file: str = "",
        freeze: bool = False,
        **kwargs
    ):
        super(Encoder, self).__init__(self)

        self.freeze = freeze
        self.weights_file = weights_file

        self.main = Dense(ndim_y, n_hidden[0], non_linearity=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Discriminator_y(Net):
    """
    A Neural Network Module Disctiminator class
    Is is similar to Encoder class
    """

    def __init__(
        self,
        output_dim: int = 2,
        n_hidden: Optional[List[int], int] = 64,
        ndim_y: int = 16,
        n_layers: int = 1,
        weights_file: str = "",
        non_linearity: Optional[str] = None,
        apply_softmax: bool = False,
        **kwargs
    ):
        super(Encoder, self).__init__(self)

        self.weights_file = weights_file
        self.apply_softmax = apply_softmax

        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hidden[0]
            n_hidden = n_laers * [n_hidden]
        else:
            n_layers = len(n_hidden)

        n_hidden.insert(0) = input_dim

        main = [
            Dense(n_hidden[i], n_hidden[i+1], non_linearity=non_linearity)
            for i in range(len(n_layers))
        ]
        main.append(Dense(n_hidden[-1], ndim_y, non_linearity=None))
        self.main = nn.Sequential(main)

    def forward(self, x: torc.Tensor) -> torch.Tensor:
        logit = self.main(x)
        if self.apply_softmax:
            return torch.softmax(logit)
        return logit

class WassersteinAutoEncoder(Net):

    def __init__(
        self,
        config: Optional[Union[ModelArguments, Dict]] = None,
    ):
        super().__init__()

        self.config = config

        if isinstance(config, ModelArguments):
            config = config.__dict__
        elif config is None:
            config = {}
        
        if config.get("input_dim", None) is None:
            raise AttributeError("`input_dim` is required")
        
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        if config.enc_freeze:
            self.encoder.freeze_params()
        if config.dec_freeze:
            self.decoder.freeze_params()

    def forward(self, docs: torch.Tensor) -> torch.Tensor:
        y_onehot_u = self.Encoder(docs)
        y_onehot_u_softmax = torch.softmax(y_onehot_u)
        if self.training and self.config.latent_noise > 0:
            continue
        x_reconstructoin_u = self.Decoder(y_onehot_u_softmax)
        return x_reconstructoin_u



