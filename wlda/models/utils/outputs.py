import torch
from dataclasses import dataclass


@dataclass
class WAEOutput:
    loss_reconstruction: float
    loss_l2_regularizer: float
    latent_max: torch.Tensor
    latent_entropy: float
    latent_v: torch.Tensor
    dirich_entropy: float