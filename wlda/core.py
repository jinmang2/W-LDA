from abc import ABCMeta, abstractmethod
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
import os
import math
import collections
from typing import Optional, List, Dict, Callable, NewType, Tuple, NamedTuple, Union, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from .args import AdvTrainingArguments
from .scheduler import get_scheduler
from transformers import Trainer
from .tokenizers import Tokenizer
from transformers.file_utils import ModelOutput


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ====================================================================
# Data Class
# ====================================================================


class Data(Dataset):
    """
    Data Generator object.
    - awslab에 따르면, main functionality는 ``minibatch``와 
      ``subsampled_labeled_data``라고 함.
    - 그러나 굳이? 현재까진 🤗의 Datasets 라이브러리도 있고 Pytorch의 자체 기능도 있기에
    - `force_reset_data`, `load`, `visualize_series` 등의 메서드만 구현할 예정.
    - ParlAI를 참고해서 객체만들자!
    """
    def __init__(
        self,
        tokenizer: Tokenizer
    ):
        self.data = load_dataset(
            path=path, name=name, cache_dir=cache_dir
        )
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass



# ====================================================================
# Neural Network Base Class
# ====================================================================


class Net(nn.Module, metaclass=ABCMeta):
    """
    A neural network skeleton class.
    This class exists for porting to the ``mx.gluon.HybridBlock``.
    @TODO gluon.HybridBlock, graph version.
    """
    INIT_TYPES = [
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_"
    ]

    def __init__(self):
        super(Net, self).__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def init_weights(self, init_type: str = "xavier_uniform_"):
        if init_types not in self.INIT_TYPES:
            raise AttributeError(f"`init_types` must be in {self.INIT_TYPES}")
        for name, param in self.named_parameters():
            if name in ["bias"]:
                with torch.no_grad():
                    param.zero_()
            else:
                getattr(torch.nn.init, init_types)(param)

    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)


class ENet(Net):
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DNet(Net):
    def forward(self, y: torch.Tensor, z: Optional[torch.Tensor] = None):
        raise NotImplementedError


class AENet(Net, metaclass=ABCMeta):
    @property
    @abstractmethod
    def encoder(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    @property
    @abstractmethod
    def discriminator(self):
        pass


# ====================================================================
# Compute Base Class
# ====================================================================


class Compute(Trainer, metaclass=ABCMeta):
    """ Skeleton class to manage training and testing. """

    def __init__(self, *args, **kwargs):
        """ Constructor for Compute. """
        super(Compute).__init__(*args, **kwargs)

    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        🤗 :func:`Trainer.compute_loss`.
        For custom behavior, override it using `train_op` method.
        """
        loss = self.train_op(self, outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return loss

    @abstractmethod
    def train_op(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """ How the loss is computed by Trainer. """
        pass

    @abstractmethod
    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset="test") -> ModelOutput:
        """ Evaluates the model. """
        pass

    @abstractmethod
    def get_outputs(self, num_samples=None, num_epochs=None, reset=None, dataset="test") -> ModelOutput:
        """ Retrieves raw outputs from model. """
        pass

    def get_optimizer_grouped_parameters(self) -> Union[Dict, List[Union[str, List]]]:
        """ Get the optimizer grouped parameters from models. """        
        # Encoder
        if self.model.encoder is not None:
            enc_named_params = self.model.encoder.named_parameters()
        else:
            enc_named_params = None
        # Decoder
        if self.model.decoder is not None:
            dec_named_params = self.model.decoder.named_parameters()
        else:
            dec_named_params = None
        # Discriminator
        if self.model.discriminator is not None:
            dis_named_params = self.model.discriminator.named_parameters()
        else:
            dis_named_params = None

        def get_params(
            named_params: Optional[
                Generator[Tuple[str, torch.nn.parameter.Parameter], None, None]
            ] = None
        ) -> List[Dict[str, List[torch.nn.parameter.Parameter]]]:
            no_decay = ["bias", "LayerNorm.weight"]
            params = []
            if named_params is not None:
                params += [
                    {"params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
                ]
            return params

        return get_params(enc_named_params) + \
               get_params(dec_named_params) + \
               get_params(dis_named_params)

    def get_optimizer_kwargs(
        self,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_grouped_parameters: Union[Dict, List[Union[str, List]]],
    ) -> Dict:
        """ Get the optimizer keyword arguments """
        optimizer_kwargs = {"lr": self.args.learing_rate}
        if isinstance(optimizer_cls, torch.optim.Adam):
            for i, params in enumerate(optimizer_grouped_parameters):
                if i < 4:
                    # Only Encoder and Decoder, in-place operation
                    params.update({"betas": (0.99, 0.999)})
        elif isisntance(optimizer_cls, torch.optim.RMSprop):
            optimizer_kwargs.update({"eps": 1e-10, "alpha": 0.9})
        
        return optimizer_kwargs
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        🤗 :func:`Trainer.create_optimizer_and_scheduler`.
        Create optimizer and learning rate scheduler.
        Note that: ``Encoder``, ``Decoder``, ``Discriminator`` are optimized by same algorithms
        such as Adam, Adadelta, RMSprop, and SGD.
        For custom behavior, override it using :func:`get_optimizer_grouped_parameters` and
        :func:`get_optimizer_kwargs` method.
        """
        if self.optimizer is None:
            # Get grouped parameters
            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
            if optimizer_grouped_parameters is None:
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
                ]
            # Get optimizer class
            if self.args.optimizer in ["Adam", "Adadelta", "RMSprop", "SGD"]:
                optimizer_cls = getattr(optim, self.args.optimizer)
            else:
                optimizer_cls = optim.Adam
            # Get optimizer keyword arguments
            optimizer_kwargs = self.get_optimizer_kwargs(optimizer_cls, optimizer_grouped_parameters)
            # Get optimizer using grouped parameters and keyword arguements
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_wramup_steps=self.args.num_wramup_steps,
                num_trainig_steps=num_training_steps,
            )