import torch
import torch.nn as nn
import torch.nn.functional as F


NON_LINEARITY = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "hardsigmoid": nn.HardSigmoid,
    "silu": nn.SiLU,
    "hardswish": nn.Hardswish,
    "elu": nn.ELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "glu": nn.GLU,
    "gelu": nn.GELU,
    "hardshrink": nn.Hardshrink,
    "leakyrelu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "prelu": nn.PReLU,
    "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink,
    "softmin": nn.Softmin,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
}


class Dense(nn.Linear):
    """
    A Linear class with non-linearity (mxnet style)
    """

    def __init__(self, *args, non_linearity="sigmoid", **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = NON_LINEARITY.get(non_linearity, nn.Identity)

    def forward(self, x):
        out = super().forward(x)
        return self.act(out)


class Net(nn.Module):
    """
    A neural network skeleton class.
    This class exists for porting to the ``mx.gluon.HybridBlock``.
    """
    INIT_TYPES = [
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_"
    ]

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        raise NotImplemented

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self, init_types="xavier_uniform_"):
        if init_types not in self.INIT_TYPES:
            raise AttributeError(f"`init_types` must be in {self.INIT_TYPES}")
        for name, param in self.named_parameters():
            torch.nn.init.xavier_uniform_(param)

    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
