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
from torch.optim import OPtimizer
from torch.optim.lr_scheduler import LambdaLR
from .args import AdvTrainingArguments
from .scheduler import get_scheduler
from transformers import Trainer
import version


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


# ====================================================================
# Data Class
# ====================================================================


class Data(object):
    """
    Data Generator object.
    - awslab에 따르면, main functionality는 ``minibatch``와 
      ``subsampled_labeled_data``라고 함.
    - 그러나 굳이? 현재까진 🤗의 Datasets 라이브러리도 있고 Pytorch의 자체 기능도 있기에
    - `force_reset_data`, `load`, `visualize_series` 등의 메서드만 구현할 예정.
    """
    pass


# ====================================================================
# Neural Network Base Class
# ====================================================================


class Net(nn.Module, metacalss=ABCMeta):
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

    def init_weights(self, init_types: str = "xavier_uniform_"):
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
        """ Combine `train_op` method. """
        loss = self.train_op(self, model, inputs)
        return loss

    @abstractmethod
    def train_op(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> ModelOutputs:
        """ How the loss is computed by Trainer. """
        pass

    @abstractmethod
    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset="test") -> ModelOutputs:
        """ Evaluates the model using num_samples. """
        pass

    @abstractmethod
    def get_outputs(self, num_samples=None, num_epochs=None, reset=None, dataset="test") -> ModelOutputs:
        """ Retrieves raw outputs from model for num_samples. """
        pass

    @abstractmethod
    def get_grouped_parameters(self) -> Union[List[Dict[str, Union[List[torch.nn.Parameter], Any]]]], None]:
        pass
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and learning rate scheduler.
        Note that: ``Encoder``, ``Decoder``, ``Discriminator`` are optimized by same algorithms
        such as Adam, Adadelta, RMSprop, and SGD.
        """
        optimizer_grouped_parameters = self.get_grouped_parameters()
        if optimizer is None:
            if optimizer_grouped_parameters is None:
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
                ]
            # Optimizer Class
            if self.args.optimizer == "Adam":
                optimizer_cls = optim.Adam
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2).
                    "eps": self.args.adam_epsilon,
                }
            elif self.args.optimizer == "Adadelta":
                optimizer_cls = optim.Adadelta
            elif self.args.optimizer == "RMSProp":
                optimizer_cls = optim.RMSProp
                optimizer_kwargs = {"eps": 1e-10, "alpha": 0.9}
            elif self.args.optimizer == "SGD":
                optimizer_cls = optim.SGD
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_wramup_steps=self.args.num_wramup_steps,
                num_trainig_steps=num_training_steps,
            )