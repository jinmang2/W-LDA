from transformers.trainer_callback import TrainerCallback
from .record import TrainRecorder


class AdjustReconAlphaCallback(TrainerCallback):
    def on_init_begin(self, args, state, control, **kwargs):
        state.recorder = TrainRecorder()

    def on_train_begin(self, args, state, control, **kwargs):
        if not hasattr(state, "recorder"):
            state.recorder = TrainRecorder()

    def on_train_end(self, args, state, control, **kwargs):
        del state.recorder

    def on_evaluate(self, args, state, control, **kwargs):
        if args.recon_alpha_adapt > 0 and state.epoch <= 1.0:
            recon_alpha = args.recon_alpha
            loss_dis= state.recorder.loss_discriminator[-1]
            loss_recons = state.recorder.loss_reconstruction[-1]
            args.recon_alpha = abs(loss_dis / loss_recons) * args.recon_alpha_adapt
            print("recon_alpha {} adjusted to {}".format(recon_alpha, args.recon_alpha))