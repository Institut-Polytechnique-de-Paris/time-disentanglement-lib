"""This module is the implementation of the FactorVAE proposed in
(https://arxiv.org/abs/1802.05983).
This model adds a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and the Total Correlation.


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

from .factor_vae_config import FactorVAEConfig
from .factor_vae_model import FactorVAE

__all__ = ["FactorVAE", "FactorVAEConfig"]
