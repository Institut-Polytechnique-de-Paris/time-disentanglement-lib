"""Basic sampler sampling from a N(0, 1) in the Autoencoder's latent space.

Available models:
------------------

.. autosummary::
    ~disco.models.AE
    ~disco.models.VAE
    ~disco.models.BetaVAE
    ~disco.models.VAE_LinNF
    ~disco.models.VAE_IAF
    ~disco.models.DisentangledBetaVAE
    ~disco.models.FactorVAE
    ~disco.models.BetaTCVAE
    ~disco.models.IWAE
    ~disco.models.MSSSIM_VAE
    ~disco.models.WAE_MMD
    ~disco.models.INFOVAE_MMD
    ~disco.models.VAMP
    ~disco.models.SVAE
    ~disco.models.Adversarial_AE
    ~disco.models.VAEGAN
    ~disco.models.VQVAE
    ~disco.models.HVAE
    ~disco.models.RAE_GP
    ~disco.models.RAE_L2
    ~disco.models.RHVAE
    :nosignatures:
"""

from .normal_config import NormalSamplerConfig
from .normal_sampler import NormalSampler

__all__ = ["NormalSampler", "NormalSamplerConfig"]
