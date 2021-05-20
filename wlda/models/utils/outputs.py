import torch
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class WAEOutput:
    loss_reconstruction: torch.Tensor
    doc_topic_vec_before_softmax: torch.Tensor
    doc_topic_vec_after_softmax: torch.Tensor
    label_ids: Optional[torch.Tensor] = None