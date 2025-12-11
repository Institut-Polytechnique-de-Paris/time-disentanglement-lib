"""This module implements the dual optimizer trainer using two distinct optimizers for the encoder 
and the decoder. It is suitable for all models but must be used in particular to train a 
:class:`~disco.models.RAE_L2`.

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
    ~disco.models.RAE_L2
    ~disco.models.HVAE
    ~disco.models.RHVAE
    :nosignatures:
"""

from .coupled_optimizer_trainer import CoupledOptimizerTrainer
from .coupled_optimizer_trainer_config import CoupledOptimizerTrainerConfig

__all__ = ["CoupledOptimizerTrainer", "CoupledOptimizerTrainerConfig"]
