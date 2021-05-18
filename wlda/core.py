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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    def forward(self, y: torch.Tensor, z: torch.Tensor):
        raise NotImplementedError


class Compute:
    """
    Skeleton class to manage training, testing, and retrieving outputs.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        discriminator: Optional[torch.nn.Module] = None,
        args: AdvTrainingArguments = None,
        tokenizer = None, # type hint 고민 중...
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizer: Tuple[Optimizer, LambdaLR] = (None, None),
    ):
        if args is None:
            args = AdvTrainingArguments(output_dir="output")
        self.args = args
        self.model = model
        self.dis_model = discriminator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizer
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.dis_model, 
            self.tokenizer, self.optimizer, self.lr_scheduler
        )

        self.state = TrainerState()
        self.control = TrainerControl()

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def train(self):
        train_dataloader = self.get_train_dataloader()
        
        # Setting up training control variables:
        # - number of training epochs: num_train_epochs
        # - number of training steps per epoch: num_update_steps_per_epoch
        # - total number of training steps to execute: max_steps
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)
        
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        total_train_batch_size = (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        num_examples = self.num_examples(train_dataloader)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state._total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

        # Reset the past mems state at the beginning of each epoch if necessary.
        if self.args.past_index >= 0:
            self._past = None

        steps_in_epoch = len(epoch_iterator)
        self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

        for step, inputs in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

            # training_step
            tr_loss += self.training_step(model, inputs)
            self._total_flos += self.floating_point_ops(inputs)

            if (step + 1) % self.gradient_accumulation_steps == 0 or (
                steps_in_epoch <= self.args.gradient_accumulation_steps
                and (step + 1) == steps_in_epoch
            ):
                # Gradient clipping
                if self.args.max_grad_norm is not None and self.max_grad_norm > 0:
                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the shared optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm,)

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, model, epoch)
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # Load best model
        # Not Implemented

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        # if self.self._total_flos is not None:
        # self.store_flos() # flos not implemented
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
        

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            if self.dis_model is not None:
                optimizer_grouped_parameters += [
                    {"params": [p for n, p in self.dis_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in self.dis_model.named_parameters() if any(nd in n for nd in no_decay)],
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

    def _maybe_action(self, model, epoch)

        






    def num_examples(self, dataloader: DataLoader) -> int:
        return len(dataloader.dataset)

    def add_callback(self, callback: TrainerCallback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0