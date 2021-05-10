import os
import math
import collections
from typing import Optional, List, Dict, Callable, NewType, Tuple, NamedTuple

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from .file_utils import is_datasets_available, WEIGHTS_NAME
from .training_args import TrainingArguments
from .trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    set_seed,
    LabelSmoother,
    speed_metrics,
    nested_detach,
)


# from `transformers.data.data_collator.py`
InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType(
    "DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


class Trainer:
    """
    Trainer is a simple but feature-complete training and the eval loop for PyTorch.
    This class is Huggingface style 🤗 and a class that replaces ``Compute`` object
    of awslab/w-lda(https://github.com/awslabs/w-lda)
    See @ https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    @TODO (in the future or other libraries):
        - hyperparameter search
        - deepspeed
        - model parallelizable
        - callback, control, state
        - DDP training
        - apex, amp
        - fp16
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizer: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None):
    ):
        if args is None:
            args = TrainingArguments(output_dir="results")
        self.args = args
        set_seed(self.args.seed)
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        default_label_names = (["labels"])
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

    def _get_train_sampler(self) -> Optional[Sampler]: 
        return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[Sampler]:
        return SequentialSampler(self.eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    @collections.abc.abstractmethod
    def create_optimizer_and_scheduler(self, num_trainig_steps: int):
        return NotImplemented

    def num_examples(self, dataloader: DataLoader) -> int:
        return len(dataloader.dataset)

    def train(self):
        train_dataloader = self.get_train_dataloader()
        max_steps = self.args.max_steps
        num_train_epochs = 1
        num_examples = self.num_examples(train_dataloader)

        # @TODO Checkpoint Learning + optimizer 불러오기도 포함
        start_time = time.time()

        model = self.model
        model.zero_grad()

        for epoch in range(epochs_trained, num_train_epochs):
            for step, inputs in enumerate(train_dataloader):
                # 여기에 checkpoint 이전 step은 건너뛰기 코드 추가!
                # gradient accumulation 추가
                self.training_step(model, inputs)
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.args.max_grad_norm,
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
            # epoch당 eval True면 아래 코드 실행
            # metrics = self.evaluate()
            # 여기서의 evaluate는 출력 찍어주고 hyperparameter search하는 용이네!
            # 나중에 코드 공부하기!
        
        metrics = speed_metrics("train", start_time)

        return TrainOutput # 이 부분 바꿀 수 있음!

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        return loss.detach()

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def save_model(self, output_dir: Optional[str] = None):
        self._save(output_dir)
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state_dit = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        # self.log(output.metrics) # log 코드 구현
        return output.metrics

    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))
        return output

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)

        model.eval()

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
            else:
                losses = None
            
        eval_loss = losses
        preds = logits if not prediction_loss_only else None
        label_ids = labels if not prediction_loss_only else None

        if (self.compute_metrics is not None
            and preds is not None
            and label_ids is not None):
            metrics = self.compute_metrics(EvalPredictoin(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None and "labels" in inputs:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v for outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
        
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)
