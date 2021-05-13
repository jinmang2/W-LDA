import torch
from transformers.trainer import Trainer

import functools
from typing import overload





class UnsupervisedTrainer(Trainer):
    """
    Unsupervised Trainger class for W-LDA
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.init_weights("xavier_uniform_")        

    def compute_loss(self, model: torch.nn.Module, inputs: torch.Tensor):
        """ computes the mmd loss with information diffusion kernel """
        wae_outputs = self.model(inputs)

        loss_recon = wae_outputs.loss_reconstruction * self.args.recon_alpha
        loss_mmd = wae_outputs.loss_mmd
        loss_l2 = wae_outputs.loss_l2_regularizer * self.args.l2_alpha

        return loss_recon + loss_mmd + loss_l2

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """ Setup the optimizer and the learning rate scheduler. """
        # @TODO Add Adam, Adadelta, RMSProp, SGD
        if self.args.optimizer == "AdamW":
            super().create_optimizer_and_scheduler(num_training_steps)
        else:
            # Write in here! code!
            raise RuntimeError()
