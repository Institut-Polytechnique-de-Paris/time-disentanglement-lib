"""This module is the implementation of the Importance Weighted Autoencoder
proposed in (https://arxiv.org/abs/1509.00519v4).

Available samplers
-------------------

.. autosummary::
    ~XGen.samplers.NormalSampler
    ~XGen.samplers.GaussianMixtureSampler
    ~XGen.samplers.TwoStageVAESampler
    ~XGen.samplers.MAFSampler
    ~XGen.samplers.IAFSampler
    :nosignatures:

"""

from .iwae_config import IWAEConfig
from .iwae_model import IWAE

__all__ = ["IWAE", "IWAEConfig"]
