"""This module implements the base trainer allowing you to train the models implemented in XGen.

Available models:
------------------

.. autosummary::
    ~XGen.models.AE
    ~XGen.models.VAE
    ~XGen.models.BetaVAE
    ~XGen.models.DisentangledBetaVAE
    ~XGen.models.BetaTCVAE
    ~XGen.models.IWAE
    ~XGen.models.MSSSIM_VAE
    ~XGen.models.INFOVAE_MMD
    ~XGen.models.WAE_MMD
    ~XGen.models.VAMP
    ~XGen.models.SVAE
    ~XGen.models.VQVAE
    ~XGen.models.RAE_GP
    ~XGen.models.HVAE
    ~XGen.models.RHVAE
    :nosignatures:
"""

from .base_trainer import BaseTrainer
from .base_training_config import BaseTrainerConfig

__all__ = ["BaseTrainer", "BaseTrainerConfig"]
