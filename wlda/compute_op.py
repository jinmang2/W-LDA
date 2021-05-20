from packaging import version
from typing import Dict, Any, Tuple, List, Optional, Union
import torch
import numpy as np

from .core import Trainer, Net
from .callbacks import AdjustReconAlphaCallback


def calc_mean_sum(tensor: torch.Tensor, dim: int = -1):
    return torch.mean(torch.sum(tensor, dim=-1))


def calc_entropy(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-10):
    return calc_mean_sum(-tensor * torch.log(tensor + eps), dim=dim)


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


class Unsupervised(Compute):
    """
    Unsupervised trainer class to manage training, testing, retrieving outputs.
    """
    
    def __init__(self, model: Net, discriminator: Net, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.discriminator = discriminator
        if self.args.init_type in Net.INIT_TYPES:
            self.model.init_weights(self.args.init_type)
            if self.discriminator is not None:
                self.discriminator.init_weights(self.args.init_type)
        # Use unsupervised callback
        self.add_callback(AdjustReconAlphaCallback)
        self.control = AdjustReconAlphaCallback.on_init_end(
            self.args, self.state, self.control
        )

    def train_op(
        self, 
        model: Net, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """ Trains the model using one minibatch of data. """
        outputs = model(**inputs)

        if hasattr(model, "_sampling_y_over_dirich"):
            with torch.no_grad():
                y_true = model._sampling_y_over_dirich(self.args.train_batch_size)
        elif inputs["label_ids"] is not None:
            y_true = inputs["label_ids"]
        else:
            raise NotImplementedError
        outputs.label_ids = y_true
        
        # Retrain_enc_only vs MMD vs GAN Training
        if self.args.retrain_enc_only:
            self.model.decoder.freeze_params()
            self.discriminator.freeze_params()
            loss = self.retrain_enc(outputs)
        elif self.args.train_mode == "mmd":
            loss = self.unlabeled_train_op_mmd_combine(outputs)
        elif self.args.train_mode == "adv":
            if self.discriminator is None:
                raise NotImplementedError
            loss = self.unlabeled_train_op_adv_combine(outputs)

        return loss

    def unlabeled_train_op_mmd_combine(self, outputs: ModelOutputs):
        """ Trains the MMD model """
        # Get reconstruction loss
        loss_reconstruction = outputs.loss_reconstruction * self.args.recon_alpha

        # Calc discriminator loss with MMD
        y_fake = outputs.doc_topic_vec_before_softmax
        y_true = outputs.label_ids
        loss_discriminator = mmd_loss(y_true, y_fake, t=self.args.kernel_alpha)

        # Calc L2 regularizer
        loss_l2_regularizer = max(calc_mean_sum(y_fake ** 2) * self.args.l2_alpha, 0.0)

        theta_noise = outputs.doc_topic_vec_after_softmax
        with torch.no_grad():
            latent_max = torch.zeros(self.args.ndim_y).to(self.args.device)
            latent_max[y_onehot_u.argmax(-1)] += 1
            latent_max /= self.args.train_batch_size

            latent_entropy = calc_entropy(theta_noise)

            latent_v = torch.mean(theta_noise, dim=0)

            dirich_entropy = calc_entropy(y_true)
        
        # For adjust recon_alpha callback
        if self.args.adverse:
            loss_mmd_return = loss_mmd.cpu().detach().item()
        else:
            loss_mmd_return = 0.0

        self.state.recorder.update({
            "loss_discriminator": loss_mmd_return,
            "loss_reconstruction": loss_reconstruction,
            "latent_max_distr": latent_max,
            "latent_avg_entropy": latent_entropy,
            "latent_avg": latent_v,
            "dirich_avg_entropy": dirich_entropy,
        })

        return loss_reconstruction + loss_discriminator + loss_l2_regularizer

    def unlabeled_train_op_adv_combine(self, outputs: ModelOutputs):
        """ Trains the GAN model """
        pass
        

    def retrain_enc(self, outputs: ModelOutputs):
        """ Re-train the Encoder only """
        loss_reconstruction = outputs.loss_reconstruction
        theta = outputs.doc_topic_vec_before_softmax

        l1_norm = torch.mean(torch.norm(theta, p=1, dim=-1))

        return loss_reconstruction + self.args.l2_alpha * l1_norm

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        outputs = {**logs, **{"step": self.state.global_step}}
        try:
            outputs.update(self.state.recorder.asdict())
        except NameError:
            pass
        self.state.log_history.append(output)

    def test_op(self):
        pass

    def get_outputs(self):
        pass

    def get_grouped_parameters(self):
        pass