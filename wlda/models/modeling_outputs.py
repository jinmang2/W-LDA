import torch
from transformers.file_utils import ModelOutput
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class WAEModelOutput(ModelOutput):
    original_documents: torch.Tensor = None
    doc_topic_vec: torch.Tensor = None
    doc_topic_vec_prob: torch.Tensor = None
    reconstructed_documents: torch.Tensor = None


@dataclass
class WAEReconstructionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    original_documents: torch.Tensor = None
    doc_topic_vec: torch.Tensor = None
    doc_topic_vec_prob: torch.Tensor = None
    reconstructed_documents: torch.Tensor = None