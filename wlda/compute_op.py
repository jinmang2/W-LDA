from packaging import version
from typing import Dict, Any, Tuple, List, Optional, Union
import torch
import numpy as np
from transformers.trainer import Trainer

from .callbacks import AdjustReconAlphaCallback


def mmd_loss(
    x: torch.Tensor, 
    y: torch.Tensor, 
    t: float = 0.1, 
    kernel: str = "diffusion",
):
    """
    Computes the mmd loss with information diffusion kernel.
    """
    eps = 1e-06
    n, d = x.size()
    if kernel == "tv":
        # https://math.stackexchange.com/questions/3415641/total-variation-distance-l1-norm
        def calc_var(x, y=None):
            n = x.shape[0]
            start_idx = 1 if y is None else 0
            sum_ = torch.zeros(1, device=x.device)
            for i in range(start_idx, n):
                if y is None:
                    sum_ = sum_ + torch.norm(x[:-i] - x[i:], p=1)
                else:
                    for j in range(y.shape[0]):
                        sum_ = sum_ + torch.norm(x[i] - y[j], p=1)
            m = n - 1 if y is None else y.shape[0]
            return sum_ / (n * (n-1))

        sum_xx = calc_var(x)
        sum_yy = calc_var(y)
        sum_xy = calc_var(x, y)
    else:
        qx = torch.sqrt(torch.clip(x, eps, 1))
        qy = torch.sqrt(torch.clip(y, eps, 1))
        xx = torch.mm(qx, qx.T)
        yy = torch.mm(qy, qy.T)
        xy = torch.mm(qx, qy.T)
        
        def diffusion_kernel(a, tmpt, dim, use_cons=False):
            cons = (4 * np.pi * tmpt)**(-dim / 2) if use_cons else 1
            return cons * torch.exp(-torch.square(torch.arccos(a)) / tmpt)

        off_diag = 1 - torch.eye(n, device=x.device)
        k_xx = diffusion_kernel(torch.clip(xx, 0, 1-eps), t, d-1)
        k_yy = diffusion_kernel(torch.clip(yy, 0, 1-eps), t, d-1)
        k_xy = diffusion_kernel(torch.clip(xy, 0, 1-eps), t, d-1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
        sum_xy = 2 * k_xy.sum() / (n * x)
    return sum_xx + sum_yy - sum_xy


class UnsupervisedTrainer(Trainer):
    """ Unsupervised Trainer class for W-LDA """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.model, "init_weights"):
            self.model.init_weights("xavier_uniform_")
        # Use unsupervised callback
        self.add_callback(AdjustReconAlphaCallback)
        self.control = AdjustReconAlphaCallback.on_init_end(
            self.args, self.state, self.control
        )

    def compute_loss(self, model: torch.nn.Module, inputs: torch.Tensor):
        """ computes the mmd loss with information diffusion kernel """
        outputs = self.model(inputs)

        loss_recon = outputs.loss_reconstruction * self.args.recon_alpha
        loss_mmd = outputs.loss_discriminator
        if self.args.l2_alpha > 0:
            loss_l2 = outputs.loss_l2_regularizer * self.args.l2_alpha
        else:
            loss_l2 = 0.0
            
        if self.args.adverse:
            loss_mmd_return = loss_mmd.cpu().detach().item()
        else:
            loss_mmd_return = 0.0

        self.state.recorder.add({
            "loss_discriminator": loss_mmd_return,
            "loss_reconstruction": loss_recon,
            "latent_max_distr": outputs.latent_max,
            "latent_avg_entropy": outputs.latent_entropy,
            "latent_avg": outputs.latent_v,
            "dirich_avg_entropy": dirich_entropy,
        })

        return loss_recon + loss_mmd + loss_l2

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 1. Topic Uniqueness
        # 2. NPMI --> Not implemented
        # 3. loss and accuracy
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """ Setup the optimizer and the learning rate scheduler. """
        # @TODO Add Adam, Adadelta, RMSProp, SGD
        if self.args.optimizer == "AdamW":
            super().create_optimizer_and_scheduler(num_training_steps)
        else:
            # Write in here! code!
            raise RuntimeError()

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if control.should_log:
            logs = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            # Add state record
            logs.update(self.state.recorder.asdict())
            self.state.recorder.reset()

            self.log(logs)
        
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
