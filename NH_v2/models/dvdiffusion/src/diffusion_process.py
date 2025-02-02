# -*-Encoding: utf-8 -*-
"""
Authors:
    Khalid OUBLAL, PhD IPP/ OneTech
"""
import numpy as np
import torch
from functools import partial
from inspect import isfunction
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Res12_Quadratic
from utils.metric import MSE
from .l_variational import L_VariationalAutoencoder


class Coupled_Diffusion_L_Var(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.target_dim = args.endmembers_dim
        self.input_size = args.embedding_dimension
        self.channels =args.channels
        self.channels = args.channels
        self.scale = args.scale
    
        # Define a fully connected layer to transform from features
        self.proj_feat = nn.Linear(self.input_size, args.hidden_size)

        self.l_variational = L_VariationalAutoencoder(args, prior="Normal") # else prior = "Dirichlet"
        
        self.diffusion = GaussianDiffusion(
            self.l_variational,
            input_size=args.endmembers_dim,
            diff_steps=args.diff_steps,
            loss_type=args.loss_type,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            scale = args.scale,
        )
        self.projection = nn.Linear(args.embedding_dimension+args.hidden_size, args.embedding_dimension)
    
    def forward(self, x, y, t, self_supervised=False):
        """
        Output the generative results and related variables.
        """
        feats = self.proj_feat(x)
        input = torch.cat([feats, x], dim=-1)
        if self_supervised:
            output, y_noisy, total_c, all_z = self.diffusion.log_prob(input, y, t)
        else:
            output, y_noisy, total_c, all_z = self.diffusion.log_prob(input, y, t)
        return output, y_noisy, total_c, all_z
        

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        bvae,
        input_size,
        beta_start=0,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        scale = 0.1,
        beta_schedule="linear",
    ):
        """
        Params:
           bave: The bidirectional vae model.
           beta_start: The start value of the beta schedule.
           beta_end: The end value of the beta schedule.
           beta_schedule: the kind of the beta schedule, here are fixed to linear, you can adjust it as needed.
           diff_steps: The maximum diffusion steps.
           scale: scale parameters for the target time series.
        """
        super().__init__()
        self.generative = bvae
        self.scale = scale
        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = self.get_beta_schedule(beta_schedule, beta_start, beta_end, diff_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
    
        alphas_target = 1.0 - betas*scale
        alphas_target_cumprod = np.cumprod(alphas_target, axis=0)
        self.alphas_target = alphas_target
        self.alphas_target_cumprod = alphas_target_cumprod
        
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
       
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
    
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_alphas_target_cumprod", to_torch(np.sqrt(alphas_target_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_target_cumprod", to_torch(np.sqrt(1.0 - alphas_target_cumprod))
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the initial input.
            :param x_start: [B, T, *]
            :return: [B, T, *]
        """
        noise = self.default(noise, lambda: torch.randn_like(x_start))

        out = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(x_start.device) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(x_start.device) * noise.to(x_start.device)
            )

        out = out.to(x_start.device)
        return out
    
    def q_sample_target(self, y_target, t, noise=None):
        """
        Diffuse the target.
            :param y_target: [B1, T1, *]
            :return: (tensor) [B1, T1, *]
        """
        noise = self.default(noise, lambda: torch.randn_like(y_target)).to(y_target.device)

        return (
            self.extract(self.sqrt_alphas_target_cumprod, t, y_target.shape).to(y_target.device) * y_target
            + self.extract(self.sqrt_one_minus_alphas_target_cumprod, t, y_target.shape).to(y_target.device) * noise
        )
           
    def p_losses(self, x_start, y_target, t,  noise=None, noise1=None, self_supervised=False):
        B, T, _ = x_start.shape
        B1, T1, _ = y_target.shape
        x_start = x_start.reshape(B, 1, T, -1)
        y_target = y_target.reshape(B1, 1, T1, -1)
        noise = self.default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise.to(x_start.device))

        x_noisy = x_noisy.reshape(B,1, T,-1)

        logits, total_c, all_z = self.generative(x_noisy)
       
        noise1 = self.default(noise1, lambda: torch.randn_like(y_target))
        y_noisy = self.q_sample_target(y_target=y_target, t=t, noise=noise1.to(y_target.device))

        y_noisy = y_noisy.reshape(B1,1, T1,-1)
    
        output = self.generative.decoder_output(logits)

        return output, y_noisy, total_c, all_z

    def log_prob(self, x_input, y_target, time, self_supervised=False):
        output, y_noisy, total_c, all_z = self.p_losses(
            x_input, y_target, time, self_supervised=self_supervised
        )
        return output, y_noisy, total_c, all_z
    



    def get_beta_schedule(self, beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
        if beta_schedule == 'quad':
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'const':
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas


    def default(self, val, d):
        if val is not None:
            return val
        return d() if isfunction(d) else d


    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        a = a.to(t.device)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


    def noise_like(self, shape, device, repeat=False):
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()

