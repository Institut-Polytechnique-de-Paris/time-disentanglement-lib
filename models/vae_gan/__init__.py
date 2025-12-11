"""This module is the implementation of the VAE-GAN model
proposed in (https://arxiv.org/abs/1512.09300).

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

from .vae_gan_config import VAEGANConfig
from .vae_gan_model import VAEGAN

__all__ = ["VAEGAN", "VAEGANConfig"]
