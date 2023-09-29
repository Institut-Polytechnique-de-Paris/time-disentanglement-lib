"""This module implements the dual optimizer trainer using two distinct optimizers for the encoder 
and the decoder. It is suitable for all models but must be used in particular to train a 
:class:`~XGen.models.RAE_L2`.

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
    ~XGen.models.RAE_L2
    ~XGen.models.HVAE
    ~XGen.models.RHVAE
    :nosignatures:
"""

from .coupled_optimizer_trainer import CoupledOptimizerTrainer
from .coupled_optimizer_trainer_config import CoupledOptimizerTrainerConfig

__all__ = ["CoupledOptimizerTrainer", "CoupledOptimizerTrainerConfig"]
