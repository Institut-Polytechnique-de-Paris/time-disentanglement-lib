"""This module is the implementation of a Vanilla Variational Autoencoder
(https://arxiv.org/abs/1312.6114).

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

from .vae_config import VAEConfig
from .vae_model import VAE

__all__ = ["VAE", "VAEConfig"]
