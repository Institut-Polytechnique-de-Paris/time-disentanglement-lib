"""This module is the implementation of the BetaVAE proposed in
(https://openreview.net/pdf?id=Sy2fzU9gl).
This model adds a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and KL term.


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

from .beta_vae_config import BetaVAEConfig
from .beta_vae_model import BetaVAE

__all__ = ["BetaVAE", "BetaVAEConfig"]
