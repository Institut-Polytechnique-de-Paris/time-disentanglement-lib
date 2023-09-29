"""This module is the implementation of the VAE-GAN model
proposed in (https://arxiv.org/abs/1512.09300).

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

from .vae_gan_config import VAEGANConfig
from .vae_gan_model import VAEGAN

__all__ = ["VAEGAN", "VAEGANConfig"]
