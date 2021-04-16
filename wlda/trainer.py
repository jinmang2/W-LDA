from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader


class Trainer:
    """
    Huggingface style trainer class
    """

    def __init__(self):
        pass

    def _remove_unused_columns(
        self, 
        dataset: "dataset.Dataset", 
        description: Optional[str]
    ):
        # signature = inspect.isgnature(self.model.forward)
        pass

    def _get_train_sampler(self) -> Optional[Sampler]:
        pass

    def _get_train_dataloader(self) -> DataLoader:
        pass 

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[Sampler]:
        pass

    def _get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        pass

    def _get_test_datalaoder(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        pass

    def create_optimizer_and_scheduler(self, num_trainig_steps: int):
        pass

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