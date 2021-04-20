import os
import sys
import time
import random
import importlib.util
import threading
import numpy as np
import torch
from dataclasses import dataclass


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    # Check we're not importing a "datasets" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    _ = importlib_metadata.version("datasets")
    _datasets_metadata = importlib_metadata.metadata("datasets")
    if _datasets_metadata.get("author", "") != "HuggingFace Inc.":
        _datasets_available = False
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


def is_torch_available():
    return _torch_available


def is_datasets_available():
    return _datasets_available


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
