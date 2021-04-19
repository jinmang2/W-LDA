import os
import collections
from typing import Optional, List, Dict, Callable, NewType, Tuple, NamedTuple

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from .optimization import Adafactor, AdamW, get_scheduler
from .args import TrainingArguments
from .utils import (
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    set_seed,
    is_datasets_available,
    LabelSmoother,
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
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizer: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None):
    ):
        if args is None:
            args = TrainingArguments(output_dir="tmp_trainer")
        self.args = args
        set_seed(self.args.seed)
        self.data_collator = data_collator # @TODO Default Data Collator 추가
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.args._n_gpu = 1

        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        os.makedirs(self.args.output_dir, exist_ok=True)

        if (not callable(self.data_collator) and 
            callable(getattr(self.data_collator, "collate_batch", None))):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`).")
        
        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(
                    self.train_dataset, description="training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(
                    self.eval_dataset, description="evaluation")

        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        default_label_names = (
            # ["start_positions", "end_positions"] # for QA
            ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

    def _remove_unused_columns(
        self, 
        dataset: "dataset.Dataset", 
        description: Optional[str]
    ):
        if not self.args.remove_unused_columns:
            return
        signature = inspect.isgnature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]
        ignore_columns = list(set(dataset.column_names) - set(signature_columns))
        dset_description = "" if description is None else f"in the {description} set "
        dataset.set_format(type=dataset.format["type"], columns=columns)

    def _get_train_sampler(self) -> Optional[Sampler]: 
        return RandomSampler(self.train_dataset)

    def _get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_dataloader()

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

    def get_test_datalaoder(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
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

    def create_optimizer_and_scheduler(self, num_trainig_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {
                    "scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        pass

    def _tune_save_checkpoint(self):
        pass

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        pass

    def _save_checkpoint(self, model, trial, metrics=None):
        pass

    def _load_optimizer_and_scheduler(self, model_path):
        pass

    def hyperparameter_search(self):
        raise NotImplemented

    def log(self, logs: Dict[str, float]) -> None:
        pass

    def _prepare_inputs(self, input: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        pass

    def compute_loss(self, model, inputs):
        pass

    def is_local_process_zero(self) -> bool:
        pass

    def is_world_process_zero(self) -> bool:
        pass

    def save_model(self, output_dir: Optional[str] = None):
        pass
    
    def _save(self, output_dir: Optional[str] = None):
        pass

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        pass

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        pass

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        pass

    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        pass

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        pass

    def _gather_and_numpify(self, tensors, name):
        pass

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass


class AdversarialTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.optim.lower() not in [
            "adam", "adadelta", "rmsprop", "sgd"]:
            raise AttributeError(
                "``optim`` is not in 'adam', 'adadelta', 'rmsprop', and 'sgd'.")

    def create_optimizer_and_scheduler(self, num_trainig_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {
                    "scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
