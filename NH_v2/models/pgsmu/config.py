import math
from os import stat
from pyexpat import model
from typing import *
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from lightning import LightningModule
import torch.nn.functional as F
import pandas as pd
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
from sklearn.metrics import mean_squared_error
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
pl.seed_everything(42)
import matplotlib.pyplot as plt
from torch.distributions import Normal
from models.dvdiffusion.src.utils import kl_balancer


class TrainerPGMSU(LightningModule):
    def __init__(self, model = None, args = None):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=['model'])

         # gains on the networks nonlinear parts ---------------------
        self.gain_nlin_a = 0.1
        self.gain_nlin_y = 0.1
        
        # (importance) sampling sizes -------------------------------
        self.K1 = 1
        self.K2 = 5
        self.criterion = nn.MSELoss()
        self.args = args
        # Create model
        self.PGMSU = model
        self.df = pd.DataFrame()
        self.ground_ = []
        self.z_ = []
        self.agg_ = []
        self.y_hat_ = []
        self.init()

    def init(self, verbose : bool = True):
        self.final_preds = []
        self.index = []
        self.aggregate = []
        self.ground = []

        self.abundance = []
        self.endmemebers = []
        self.y_hat_gen = []
        self.y_hat_dirichlet = []
        
        self.abundance_trues = []
        self.endmemeber_trues = []
        self.y_trues = []


    def configure_optimizers(self):
        lr =  1e-4 # self.args.learning_rate
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr)
        #return denoise_optim
        # optimizer = getattr(torch.optim, self.optimizer_config['type'])
        # del self.optimizer_config['type']
        # optimizer = optimizer(self.parameters(), **self.optimizer_config)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-2, verbose=True)

        return [optimizer], [scheduler]

        #return optimizer

    def forward(self, batch_y_unsup, batch_y_sup, batch_m, batch_a, t):
        # Forward function that is run when visualizing the graph
        return self.PGMSU(batch_y_sup, batch_y_sup,batch_m, batch_a, t, self.current_epoch)
    

    def training_step_older(self, batch, batch_idx):
        # Old viresion x must be in shape [batch_size, 1, channels]
        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch

        # print(f"  batch_y_unsup shape: {batch_y_unsup.shape}")
        # print(f"  batch_y_sup shape: {batch_y_sup.shape}")
        # print(f"  batch_m shape: {batch_m.shape}")
        # print(f"  batch_a shape: {batch_a.shape}")
        
        t = torch.randint(0, self.args.diff_steps, (self.args.batch_size,)).long()
        batch_y_unsup = batch_y_unsup.float()
        batch_y_unsup = batch_y_unsup[...,-self.args.endmembers_dim:].float()
        # end_memebers_sample, end_memebers_diffused, total_correlation, l_variational_latents, loss_diffusion
        
        (y_hat_sup, y_hat_unsup), (outputs_dirichlet_sup,
                                   outputs_genearitv_diffusion_sup,
                                   loss_diffusion_sup,
                                   outputs_dirichlet_unsup,
                                   outputs_genearitv_diffusion_unsup,
                                   loss_diffusion_unsup) = self(batch_y_sup, batch_y_sup,batch_m, batch_a, t)

        
        endmemebers_sample, endmemebers_diffused, kl_loss_sup, l_var_latents = outputs_genearitv_diffusion_sup
        _, _, kl_loss_unsup, l_var_latents = outputs_genearitv_diffusion_unsup

        kl_loss_sup_balanced, kl_coeffs, kl_vals = kl_balancer(kl_loss_sup)
        kl_loss_unsup_balanced, kl_coeffs_unsup, kl_vals_unsup = kl_balancer(kl_loss_unsup)

        ## KL
        kl_loss = (kl_loss_sup_balanced + kl_loss_unsup_balanced)*self.args.gamma

        ## DIFFUSION
        loss_diffusion = self.args.weight_diffusion*(loss_diffusion_sup+loss_diffusion_unsup)

        ## DIRICHLET LOSS
        (y_rec_sup, sampled_abundance_sup, implicit_kldir_sup, max_kld_sampled_sup) = outputs_dirichlet_sup
        (y_rec_unsup, sampled_abundance_unsup, implicit_kldir_unsup, max_kld_sampled_unsup) = outputs_dirichlet_unsup 

        # Implicit Gradient for dirichlet
        implicit_kl_dirichlet = (implicit_kldir_sup + implicit_kldir_unsup)*self.args.kl_dirichlet

        ## MSE 1st model 
        mes_loss_dirichlet = self.criterion(y_rec_unsup, batch_y_unsup) + self.criterion(y_rec_unsup, batch_y_sup)
        #recon = endmemebers_sample.log_prob(endmemebers_sample.sample())

        ## MSE 2nd model 
        mse_loss = self.criterion(endmemebers_sample.sample(), batch_m) + self.criterion(y_hat_unsup, batch_y_unsup) + self.criterion(y_hat_sup, batch_y_sup) #(endmemebers_diffused)
        #log_prob = - torch.mean(torch.sum(recon, dim=[1, 2, 3]))

        #loss = torch.tensor(mse_loss, requires_grad=True)
        self.loss = mse_loss + mes_loss_dirichlet + kl_loss + loss_diffusion

        #self.log("Performance_val", {"total_loss":total_loss, "MSELoss": loss, "KLdivergenceLoss":info_loss})
        self.log('loss_diffusion', loss_diffusion, prog_bar=True)
        self.log('loss_mse_generative', mse_loss, prog_bar=True)
        self.log('loss_mse_dirichlet', mes_loss_dirichlet, prog_bar=True)
        self.log('implicit_kl_dirichlet', implicit_kl_dirichlet, prog_bar=True)
        self.log('kl_loss', kl_loss, prog_bar=True)

        #self.log('TC_loss', tc, prog_bar=True)
        #self.log('MSELoss', loss, prog_bar=True)
        #self.log('KLdivergenceLoss', info_loss, prog_bar=True)
        self.log('loss', self.loss, prog_bar=True)
        tensorboard_logs = {'train_loss': self.loss}
        return {'loss': self.loss, 'log': tensorboard_logs}

    def training_step(self, batch, batch_idx)-> Dict:
        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch

        batch_y_unsup = batch_y_unsup.float()
        batch_y_sup = batch_y_sup.float()
        batch_m = batch_m.float()
        batch_a = batch_a.float()
        batch_m = batch_m.squeeze(-1).permute(0,2,1)
        
        batch_y_unsup = batch_y_unsup.squeeze(-1)
        batch_y_sup = batch_y_sup.squeeze(-1) 
        # Supervised
        y_hat_sup, mu_sup, log_var_sup, abundance_pred_sup, endmemebers_pred_sup = self.PGMSU(batch_y_sup)
        # print(f"endmemebers_pred_sup {endmemebers_pred_sup.shape}")
        # print(f"batch_m {batch_m.shape}")
        # print(f"batch_y_sup {batch_y_sup.shape}")
        # print(f"y_hat_sup {y_hat_sup.shape}")
        loss_supervised, _ = self.loss_func(
                            y_hat_sup = y_hat_sup,
                            batch_y_sup = batch_y_sup,
                            mu_sup = mu_sup,
                            log_var_sup = log_var_sup,
                            abundance_pred_sup = abundance_pred_sup,
                            endmemebers_pred_sup = endmemebers_pred_sup,
                            endmemebers_target = batch_m,
                            supervised=True)
        ## Unsupervised.
        y_hat_unsup, mu_unsup, log_var_unsup, abundance_pred_unsup, endmemebers_pred_unsup = self.PGMSU(batch_y_sup)
        y_hat_unsup_product = torch.bmm(endmemebers_pred_unsup.float().permute(0, 2, 1), abundance_pred_unsup.unsqueeze(2)).squeeze(-1)

        _, loss_unsupervised = self.loss_func(
                    y_hat_sup = y_hat_unsup_product,
                    batch_y_sup = batch_y_unsup,
                    mu_sup = mu_unsup,
                    log_var_sup = log_var_unsup,
                    abundance_pred_sup = abundance_pred_unsup,
                    endmemebers_pred_sup = endmemebers_pred_unsup,
                    endmemebers_target = None,
                    supervised=False)
        loss = loss_unsupervised + 0.5*loss_supervised
        
        self.log('loss', loss, prog_bar=True)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    def compute_nll_gaussian(self, target, mu, sigma_value=1.0):
        """
        Computes the negative log likelihood of the target under a Gaussian distribution
        with mean (mu) and fixed standard deviation (sigma).
        
        Args:
        - target (torch.Tensor): The target tensor.
        - mu (torch.Tensor): The mean of the Gaussian distribution.
        - sigma_value (float): The assumed standard deviation of the Gaussian distribution (default is 1.0).
        
        Returns:
        - nll (torch.Tensor): The negative log likelihood.
        """
        # Create a normal distribution with fixed sigma_value
        sigma = torch.full_like(mu, sigma_value)
        dist = Normal(mu, sigma)
        
        # Compute log probabilities
        log_prob = dist.log_prob(target)
        nll = -log_prob.mean()  # Mean negative log likelihood
        return nll
    
    def loss_func(self,
                y_hat_sup = None,
                batch_y_sup = None,
                mu_sup = None,
                log_var_sup = None,
                abundance_pred_sup = None,
                endmemebers_pred_sup = None,
                endmemebers_target = None,
                supervised=True):
        
        lambda_kl = 0.1
        lambda_sad = 3
        lambda_vol = 7
        
        loss_rec = self.compute_nll_gaussian(batch_y_sup, y_hat_sup)

        #loss_rec = self.criterion(y_hat_sup, batch_y_sup) # ** 2).sum() / batch_y_sup.shape[0]

        kl_div = -0.5 * (log_var_sup + 1 - mu_sup ** 2 - log_var_sup.exp())

        kl_div = kl_div.sum() / y_hat_sup.shape[0]
        # KL balance of VAE
        kl_div = torch.max(kl_div, torch.tensor(0.2).cuda())

        loss_supervised = 0
        if supervised:
            # pre-train process
            loss_vca = (endmemebers_pred_sup - endmemebers_target).square().sum() / batch_y_sup.shape[0]
            loss_supervised = loss_rec + lambda_kl * kl_div + 0.1 * loss_vca
        #else:
            #training process
            #constrain 1 min_vol of EMs
        em_bar = endmemebers_pred_sup.mean(dim=1, keepdim=True)
        loss_minvol = ((endmemebers_pred_sup - em_bar) ** 2).sum() / y_hat_sup.shape[0] / 5 / 162

        # constrain 2 SAD for same materials
        em_bar = endmemebers_pred_sup.mean(dim=0, keepdim=True)  # [1,5,198] [1,z_dim,Channel]
        aa = (endmemebers_pred_sup * em_bar).sum(dim=2)
        em_bar_norm = em_bar.square().sum(dim=2).sqrt()
        em_tensor_norm = endmemebers_pred_sup.square().sum(dim=2).sqrt()

        sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
        loss_sad = sad.sum() / y_hat_sup.shape[0] / 5
        loss_unsupervised = loss_rec + lambda_kl * kl_div + lambda_vol * loss_minvol + lambda_sad * loss_sad

        return loss_supervised, loss_unsupervised
    

    def A_training_step_A(self, batch, batch_idx) -> Dict:

        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch
        batch_y_unsup = batch_y_unsup.float()
        batch_y_sup = batch_y_sup.float()
        batch_m = batch_m.float()
        batch_a = batch_a.float()


        batch_y_sup = batch_y_sup.squeeze(-1) 
        # Supervised
        y_hat_sup, mu_sup, log_var_sup, abundance_pred_sup, endmemebers_pred_sup = self.PGMSU(batch_y_sup)
        loss_enmembers_sup = self.criterion(endmemebers_pred_sup, batch_m.permute(0,2,1))
        loss_abundance_sup = self.criterion(abundance_pred_sup.unsqueeze(2), batch_a)
        loss_y_rec_sup = self.criterion(y_hat_sup, batch_y_sup)

        kl_div_sup = torch.mean(-0.5 * (log_var_sup + 1 - mu_sup ** 2 - log_var_sup.exp()))
        self.loss_sup = loss_enmembers_sup +  loss_abundance_sup + loss_y_rec_sup + kl_div_sup

        self.log("loss_enmembers", loss_enmembers_sup, prog_bar=True)
        self.log("loss_abundance", loss_abundance_sup, prog_bar=True)
        self.log("loss_y_rec_sup", loss_y_rec_sup, prog_bar=True)
        self.log("kl_div_sup", kl_div_sup, prog_bar=True)

        # SSL
        batch_y_unsup = batch_y_unsup.squeeze(-1) 
        y_hat_unsup, mu_unsup, log_var_unsup, abundance_pred_unsup, endmemebers_pred_unsup = self.PGMSU(batch_y_unsup)
        y_hat_unsup_product = torch.bmm(endmemebers_pred_unsup.float().permute(0, 2, 1), abundance_pred_unsup.unsqueeze(2)).squeeze(-1)
        loss_y_rec_unsup = self.criterion(y_hat_unsup, batch_y_sup)
        loss_y_hat_unsup_product = self.criterion(y_hat_unsup_product, batch_y_sup)
        kl_div_unsup = torch.mean(-0.5 * (log_var_unsup + 1 - mu_unsup ** 2 - log_var_unsup.exp()))
        self.loss_unsup = loss_y_rec_unsup + loss_y_hat_unsup_product + kl_div_unsup

        self.log("loss_y_rec_unsup", loss_y_rec_unsup, prog_bar=True)
        self.log("loss_y_hat_unsup_product", loss_y_hat_unsup_product, prog_bar=True)
        self.log("kl_div_unsup", kl_div_unsup, prog_bar=True)
        self.loss = self.loss_sup  + self.loss_unsup

        self.log('loss', self.loss, prog_bar=True)
        tensorboard_logs = {'train_loss': self.loss}
        return {'loss': self.loss, 'log': tensorboard_logs}
    
    def train_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `training_step`
        train_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        self.log("loss", train_loss)


    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Dict:
        #copy_parameters(self.dv_diffusion, self.pred_net)
        # batch_y_unsup, batch_y_sup, batch_m, batch_a
        batch_y_unsup, batch_y_sup, batch_m, batch_a = val_batch

        # batch_y_sup = batch_y_sup.float()
        # batch_m = batch_m.float()
        # batch_a = batch_a.float()
        # batch_y_sup = batch_y_sup[...,-self.args.endmembers_dim:].float()
        # batch_y_unsup = batch_y_unsup[...,-self.args.endmembers_dim:].float()

        batch_y_unsup = batch_y_unsup.float()
        batch_y_sup = batch_y_sup.float()
        batch_m = batch_m.float()
        batch_a = batch_a.float()

        batch_y_sup = batch_y_sup.squeeze(-1)
        batch_y_unsup = batch_y_unsup.squeeze(-1) 

        # _, out, _, _ = self.PGMSU.inference(batch_y_sup) #self.pred_net(batch_y_sup)        
        # self.loss = self.criterion(out, batch_m)
        # # self.log("RMSE_A", RMSE_A, prog_bar=True)
        # # self.log("RMSE_M", RMSE_M, prog_bar=True)
        # # self.log("RMSE_Y", RMSE_Y, prog_bar=True)
        # self.log("val_loss", self.loss)
        # return {"vloss":  self.loss, "val_loss": self.loss}
        y_hat, mu_unsup, log_var_unsup, abundance_pred_unsup, endmemebers_pred_unsup = self.PGMSU(batch_y_unsup)
        y_hat_unsup_product = torch.bmm(endmemebers_pred_unsup.float().permute(0, 2, 1), abundance_pred_unsup.unsqueeze(2)).squeeze(-1)

        _, loss_unsupervised = self.loss_func(
                    y_hat_sup = y_hat_unsup_product,
                    batch_y_sup = batch_y_unsup,
                    mu_sup = mu_unsup,
                    log_var_sup = log_var_unsup,
                    abundance_pred_sup = abundance_pred_unsup,
                    endmemebers_pred_sup = endmemebers_pred_unsup,
                    endmemebers_target = None,
                    supervised=False)
        
        self.val_loss = loss_unsupervised

        self.log("val_loss", self.val_loss, prog_bar=True)
        self.log("vloss", self.val_loss, prog_bar=True)

        return {"vloss":  self.val_loss, "val_loss": self.val_loss}


    def test_step(self, batch, batch_idx):
        #copy_parameters(self.dv_diffusion, self.pred_net)
        # batch_y_unsup, batch_y_sup, batch_m, batch_a
        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch

        batch_y_unsup = batch_y_unsup.float()
        batch_y_sup = batch_y_sup.float()
        batch_m = batch_m.float()
        batch_a = batch_a.float()
        #batch_y_sup = batch_y_sup[...,-self.args.endmembers_dim:].float()

        # _, out, _, _ = self.PGMSU.inference(batch_y_sup) #self.pred_net(batch_y_sup)        
        # self.loss = self.criterion(out, batch_m)
        # # self.log("RMSE_A", RMSE_A, prog_bar=True)
        # # self.log("RMSE_M", RMSE_M, prog_bar=True)
        # # self.log("RMSE_Y", RMSE_Y, prog_bar=True)
        # self.log("val_loss", self.loss)
        # return {"vloss":  self.loss, "val_loss": self.loss}
        y_hat, mu, log_var, abundance_pred, endmemebers_pred = self.PGMSU(batch_y_unsup.squeeze(-1))

        # print("-"*10)
        # print("Shape de endmembers_pred:", endmemebers_pred.float().shape)
        # print("Shape de batch_m:", batch_m.shape)
        # print("Shape de batch_a:", batch_a.shape)
        # print("Shape de abundance_pred:", abundance_pred.shape)
        # print("Shape de y_hat_gen:", y_hat.shape)
        # print("Shape de batch_y_sup:", batch_y_sup.shape)
        # print("-"*10)

        y_hat_gen = torch.bmm(endmemebers_pred.float().permute(0, 2, 1), abundance_pred.unsqueeze(2)).squeeze(-1)
        
        loss_enmembers = self.criterion(endmemebers_pred, batch_m.permute(0,2,1))
        loss_abundance = self.criterion(abundance_pred.unsqueeze(2), batch_a)

        loss_y_rec_dirichlet = self.criterion(y_hat, batch_y_sup)
        loss_y_rec_generative = self.criterion(y_hat_gen, batch_y_sup)


        self.abundance.append(abundance_pred.detach().cpu())
        self.endmemebers.append(endmemebers_pred.float().detach().cpu())
        self.y_hat_gen.append(y_hat_gen.detach().cpu())
        self.y_trues.append(y_hat_gen.detach().cpu())
        self.y_hat_dirichlet.append(y_hat_gen.detach().cpu())


        # print("Shape de self.abundance:", self.abundance[0].shape)
        # print("Shape de self.endmemebers:", self.endmemebers[0].shape)
        # print("Shape de batch_a:", batch_a.shape)
        # print("Shape de self.y_hat_gen:", self.y_hat_gen[0].shape)
        # print("Shape de self.y_hat_dirichlet:", self.y_hat_dirichlet[0].shape)

        return {"vloss":  loss_enmembers, "val_loss": loss_enmembers}

    def on_test_epoch_end(self):
    
        self.A_pred = torch.concatenate(self.abundance, axis=0)
        self.Mn_pred = torch.concatenate(self.endmemebers, axis=0)
        self.y_pred = torch.concatenate(self.y_hat_gen, axis=0)

        self.pred = (self.A_pred,
                    self.Mn_pred,
                    self.y_pred,
                    self.y_pred,
                    self.y_pred,
                    )
        
        return {"pred": self.pred}
