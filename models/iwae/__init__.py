"""This module is the implementation of the Importance Weighted Autoencoder
proposed in (https://arxiv.org/abs/1509.00519v4).

Available samplers
-------------------

.. autosummary::
    ~disco.samplers.NormalSampler
    ~disco.samplers.GaussianMixtureSampler
    ~disco.samplers.TwoStageVAESampler
    ~disco.samplers.MAFSampler
    ~disco.samplers.IAFSampler
    :nosignatures:

"""

from .iwae_config import IWAEConfig
from .iwae_model import IWAE

__all__ = ["IWAE", "IWAEConfig"]
