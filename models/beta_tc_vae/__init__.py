"""This module is the implementation of the BetaTCVAE proposed in
(https://arxiv.org/abs/1802.04942).


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

from .beta_tc_vae_config import BetaTCVAEConfig
from .beta_tc_vae_model import BetaTCVAE

__all__ = ["BetaTCVAE", "BetaTCVAEConfig"]
