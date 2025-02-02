
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

class TrainerIDNet(LightningModule):
    def __init__(self, model = None, H = None, P = None, L=None, optimizer_config=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=['model'])
        self.H = H
        self.P = P
        self.L = L
        self.optimizer_config = optimizer_config

         # gains on the networks nonlinear parts ---------------------
        self.gain_nlin_a = 0.1
        self.gain_nlin_y = 0.1
        
        # (importance) sampling sizes -------------------------------
        self.K1 = 1
        self.K2 = 5
        # Create model
        self.model = model
        self.df = pd.DataFrame()
        self.ground_ = []
        self.z_ = []
        self.agg_ = []
        self.y_hat_ = []
        self.validation_step_outputs = []
        self.init()

    def init(self, verbose : bool = True):
        self.final_preds = []
        self.index = []
        self.aggregate = []
        self.ground = []
        self.A_pred = []
        self.Mn_pred = []
        self.y_pred = []
        self.Mn_avg_pred = []

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_config['type'])
        del self.optimizer_config['type']
        optimizer = optimizer(self.parameters(), **self.optimizer_config)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Dict:
        
        batch_y_unsup, batch_y_sup, batch_m, batch_a = val_batch

        batch_y_unsup = batch_y_unsup.float()
        batch_y_sup = batch_y_sup.float()
        batch_y_sup = batch_y_sup.float()
        batch_m = batch_m.float()
        batch_a = batch_a.float()
        
        batch_y_unsup = batch_y_unsup.squeeze(-1)
        batch_y_sup = batch_y_sup.squeeze(-1)
        A_u_true = batch_a.squeeze(-1)
        M_u_true = batch_m.squeeze(-1)
        Y_true = batch_y_sup.squeeze(-1)

        # data_usup, data_sup , (A_u_true, M_u_true, Y_true) = val_batch
        A_est, Mn_est, M_avg, Y_rec, a_nlin_deg = self.unmix_step(batch_y_unsup)

        A_est = A_est.permute(1, 0)
        Mn_est = Mn_est.permute(2, 0, 1)
        Y_rec = Y_rec.permute(1, 0)

        # print(f"A_u_true shape: {A_u_true.shape}, {A_est.shape}")
        # print(f"M_avg shape: {M_u_true.shape}, {Mn_est.shape}")
        # print(f"A_u_true shape: {Y_true.shape}, {Y_rec.shape}")

        RMSE_A, NRSME_A = self.compute_metrics(A_u_true, A_est)
        RMSE_M, NRMSE_M = self.compute_metrics(M_u_true, Mn_est)
        RMSE_Y, NRMSE_Y = self.compute_metrics(Y_true, Y_rec)

        self.log("RMSE_A", RMSE_A, prog_bar=True)
        self.log("RMSE_M", RMSE_M, prog_bar=True)
        self.log("RMSE_Y", RMSE_Y, prog_bar=True)
        self.log("val_loss", self.loss)

        self.validation_step_outputs.append(self.loss)

        return {"vloss": self.loss, "val_loss": self.loss}


    def on_validation_epoch_end(self):
    
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('avg_val_loss', avg_val_loss)
        self.validation_step_outputs.clear()  # free memory
        
        # Optionally compute additional metrics, e.g., accuracy
        # y_hats = torch.cat([x['y_hat'] for x in outputs], dim=0)
        # ys = torch.cat([x['y'] for x in outputs], dim=0)
        # acc = (y_hats.argmax(dim=1) == ys).float().mean()
        # self.log('val_acc', acc, prog_bar=True)


    def forward(self, batch_y_unsup, batch_y_sup, batch_m, batch_a):
        # Forward function that is run when visualizing the graph
        return self.model(batch_y_unsup, batch_y_sup, batch_m, batch_a, self.current_epoch)
    
    def unmix_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        Y = batch
        N = Y.shape[0]
        M_avg, _, _ = self.model.samp_q_M_Z(Z=torch.zeros((1,self.H,self.P)).to(self.device), batch_size=1, K1=1)
        Z_y_mean, Z_y_sigma, Z_y_samp = self.model.samp_q_Z_y(Y, batch_size=N, K1=1)
        Mn_est, M_Z_sigma, M_Z_samp = self.model.samp_q_M_Z(Z=Z_y_samp, batch_size=N, K1=1)
        q_a_My, alphas, a_Zy_samp, a_nlin_deg = self.model.samp_q_a_My(M=Mn_est, y=Y, batch_size=N, K1=1, compute_nlin_deg=True)
        A_est = (alphas/torch.kron(torch.ones(1,self.P).to(self.device), alphas.sum(dim=1).unsqueeze(1))).T

        # compute the reconstructed image
        Y_rec = nn.functional.relu(torch.bmm(Mn_est, A_est.T.unsqueeze(2)).squeeze() \
            + self.gain_nlin_y*self.model.fcy_Ma_nlin(torch.cat((Mn_est.reshape((N,self.L*self.P)),A_est.T), dim=1))).T # nonlinear part
        return A_est, Mn_est.permute(1,2,0), M_avg.squeeze(), Y_rec, a_nlin_deg


    def training_step(self, batch, batch_idx):
        # Old viresion x must be in shape [batch_size, 1, sequence_length]
        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch
        #data_usup, data_sup, _ = batch
        log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all = self(batch_y_unsup, batch_y_sup, batch_m, batch_a)
        self.loss = self.my_loss_function(log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all)
        #self.log("Performance_val", {"total_loss":total_loss, "MSELoss": loss, "KLdivergenceLoss":info_loss})
        #self.log('total_loss', total_loss, prog_bar=True)
        #self.log('TC_loss', tc, prog_bar=True)
        #self.log('MSELoss', loss, prog_bar=True)
        #self.log('KLdivergenceLoss', info_loss, prog_bar=True)
        self.log('loss', self.loss, prog_bar=True)
        tensorboard_logs = {'train_loss': self.loss}
        return {'loss': self.loss, 'log': tensorboard_logs}

    def train_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `training_step`
        train_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        self.log("loss", train_loss)

    def test_step(self, batch, batch_idx):
        batch_y_unsup, batch_y_sup, batch_m, batch_a = batch
        
        batch_y_unsup = batch_y_unsup.squeeze(-1)
        batch_y_sup = batch_y_sup.squeeze(-1)
        A_u_true = batch_a.squeeze(-1)
        M_u_true = batch_m.squeeze(-1)
        Y_true = batch_y_sup.squeeze(-1)

        A_est, Mn_est, M_avg, Y_rec, a_nlin_deg = self.unmix_step(batch_y_unsup)

        A_est = A_est.permute(1, 0)
        Mn_est = Mn_est.permute(2, 0, 1)
        Y_rec = Y_rec.permute(1, 0)

        RMSE_A, NRSME_A = self.compute_metrics(A_u_true, A_est)
        RMSE_M, NRMSE_M = self.compute_metrics(M_u_true, Mn_est)
        RMSE_Y, NRMSE_Y = self.compute_metrics(Y_true, Y_rec)

        self.log("RMSE_A", RMSE_A, prog_bar=True)
        self.log("RMSE_M", RMSE_M, prog_bar=True)
        self.log("RMSE_Y", RMSE_Y, prog_bar=True)
        
        self.A_pred.append(A_est.detach().cpu())
        self.Mn_pred.append(Mn_est.detach().cpu())
        self.y_pred.append(Y_rec.detach().cpu())
        self.Mn_avg_pred.append(M_avg)
        return {'test_loss': None}

    def on_test_epoch_end(self):
        # self.A_pred_out = np.concatenate(np.array(self.A_pred,  dtype=object),axis=0)
        # self.Mn_pred_out = np.concatenate(np.array(self.Mn_pred,  dtype=object),axis=0)
        # self.y_pred_out = np.concatenate(np.array(self.y_pred,  dtype=object),axis=0)

        self.A_pred = torch.concatenate(self.A_pred, axis=0)
        self.Mn_pred = torch.concatenate(self.Mn_pred, axis=0)
        self.y_pred = torch.concatenate(self.y_pred, axis=0)

        self.pred = (self.A_pred,
                    self.Mn_pred,
                    self.y_pred,
                    self.y_pred,
                    self.y_pred,
                    )
        
        return {"pred": self.pred}
    
        # return {"A_pred_out": self.A_pred_out, "Mn_pred_out": self.Mn_pred_out, "y_pred_out": self.y_pred_out}


    def my_loss_function(self, log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all):        
        my_llambda = 10
        my_tau     = 0
        my_lamb_We = 1e4
        my_lamb_Wd = 0.001
        
        llambda = my_llambda; # regularization between supervised and unsupervised part
        bbeta   = 10; # extra regularization (high likelihood of endmembers and abundances training data in the posterior)
        tau     = my_tau; # extra extra regularization (sparsity)
        lamb_We = my_lamb_We; # penalizes network weights of nonlinear encoder
        lamb_Wd = my_lamb_Wd; # penalizes network weights of nonlinear decoder

        K1 = log_probs_unsup['log_py_Ma'].shape[1]
        K2 = omegas_nrmlzd.shape[1]
        batch_size = omegas_nrmlzd.shape[0]
        
        # unsupervised part of the cost function --------------
        cost_unsup = 0
        for i in range(0,batch_size):
            for j in range(0,K1):
                # terms in the numerator
                cost_unsup = cost_unsup + log_probs_unsup['log_py_Ma'][i,j] 
                cost_unsup = cost_unsup + log_probs_unsup['log_pa'][i,j]
                cost_unsup = cost_unsup + log_probs_unsup['log_pM_Z'][i,j] 
                cost_unsup = cost_unsup + log_probs_unsup['log_pZ'][i,j]
                # terms in the denominator
                cost_unsup = cost_unsup - log_probs_unsup['log_qa_My'][i,j]
                cost_unsup = cost_unsup - log_probs_unsup['log_qM_Z'][i,j]
                cost_unsup = cost_unsup - log_probs_unsup['log_qZ_y'][i,j]
                
        # supervised part of the cost function  
        cost_sup = 0
        for i in range(0,batch_size):
            for j in range(0,K2):
                temp = 0
                # terms in the numerator
                temp = temp + log_probs_sup['log_py_Ma'][i,j] 
                temp = temp + log_probs_sup['log_pa'][i,j] 
                temp = temp + log_probs_sup['log_pM_Z'][i,j] 
                temp = temp + log_probs_sup['log_pZ'][i,j] 
                # terms in the denominator
                temp = temp - log_probs_sup['log_qa_My'][i,j] 
                temp = temp - log_probs_sup['log_qM_Z'][i,j] 
                temp = temp - log_probs_sup['log_qZ_y'][i,j] 
                # importance weight normalization (the omegas are already normalized now)
                temp = omegas_nrmlzd[i,j] * temp
                # accumulate in the cost function
                cost_sup = cost_sup + temp
                
        # regularization term
        cost_reg = 0
        for i in range(0,batch_size):
            for j in range(0,K2):
                cost_reg = cost_reg + log_probs_sup['log_qa_My'][i,j] 
                cost_reg = cost_reg + log_omegas[i,j]
        
        # yet another regularization term (sparsity on alphas)
        cost_reg_sprs = 0
        for i in range(0,batch_size):
            # this one computes it over the unsupervised data
            for j in range(0,K1):
                cost_reg_sprs = cost_reg_sprs + torch.linalg.norm(alphas_all['alphas_unsup'][:,i,j], ord=0.5) / K1
            # this one computes it over the supervised data
            for j in range(0,K2):
                cost_reg_sprs = cost_reg_sprs + torch.linalg.norm(alphas_all['alphas_sup'][:,i,j], ord=0.5) / K2
        
        
        # regularizes nonlinear mixing weights
        reg_nlin_d_weights = 0
        # for param in model.fcy_Ma_nlin.parameters():
        #     reg_nlin_d_weights = reg_nlin_d_weights + torch.norm(param, p="fro")
        
        reg_nlin_e_weights = 0
        # for param in model.fca_My_alphas.parameters():
        #     reg_nlin_e_weights = reg_nlin_e_weights + torch.norm(param, p="fro")    
            
        
        # now the total cost functions
        cost = cost_unsup/K1 + llambda * cost_sup/K2 + (1+bbeta) * cost_reg/K2 \
            - tau * cost_reg_sprs - lamb_We * reg_nlin_e_weights - lamb_Wd * reg_nlin_d_weights
        return -cost # maximize cost
    

    def compute_metrics(self, t_true, t_est):
        RMSE = torch.sqrt(torch.sum((t_true-t_est)**2)/t_true.shape.numel())
        NRSME = torch.sqrt(torch.sum((t_true-t_est)**2)/torch.sum(t_true**2))
        return RMSE, NRSME

