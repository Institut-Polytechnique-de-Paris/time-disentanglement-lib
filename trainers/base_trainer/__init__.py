"""This module implements the base trainer allowing you to train the models implemented in disco.

Available models:
------------------

.. autosummary::
    ~disco.models.AE
    ~disco.models.VAE
    ~disco.models.BetaVAE
    ~disco.models.DisentangledBetaVAE
    ~disco.models.BetaTCVAE
    ~disco.models.IWAE
    ~disco.models.MSSSIM_VAE
    ~disco.models.INFOVAE_MMD
    ~disco.models.WAE_MMD
    ~disco.models.VAMP
    ~disco.models.SVAE
    ~disco.models.VQVAE
    ~disco.models.RAE_GP
    ~disco.models.HVAE
    ~disco.models.RHVAE
    :nosignatures:
"""

from .base_trainer import BaseTrainer
from .base_training_config import BaseTrainerConfig

__all__ = ["BaseTrainer", "BaseTrainerConfig"]
