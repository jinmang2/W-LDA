import os
import sys
import time
import random
import importlib.util
import threading
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import NamedTuple, Union, Tuple, Optional, Dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        import torch

        torch.mamual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


@dataclass
class LabelSmoother:

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        model_loss = model_output["loss"] if isinstance(model_output, dict) else model_output[0]
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[1]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)

        # Look at the ignored index and mask the corresponding log_probs
        padding_mask = labels.unsqueeze(-1).eq(self.ignore_index)
        log_probs.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimentions, then divide the number of active elements (i.e. not-padded)
        smoothed_loss = log_probs.mean(dim=-1).sum() / (padding_mask.numel() - padding_mask.long().sum())
        return (1 - self.epsilon) * model_loss + self.epsilon * smoothed_loss


def speed_metrics(split, start_time, num_samples=None):
    """ Measure and return speed performance metrics """
    runtime = time.time() - start_time
    result =  {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = 1 / (runtime / num_samples)
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    return result


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


class ExplicitEnum(Enum):

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please seelct one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
