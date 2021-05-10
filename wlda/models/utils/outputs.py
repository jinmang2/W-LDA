from dataclasses import dataclass, asdict


@dataclass
class WAEOutput:
    loss_reconstruction: float
    loss_discriminator: float
    latent_max_distr: float
    latent_avg_entropy: float
    latent_avg: float
    dirich_avg_entropy: float
    recon_alpha: float

    def __post_init__(self):
        pass
