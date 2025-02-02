
from data_load.data_loader import Dataset_Custom, HyperspectralDataset
from trainer.exp_basic import Exp_Basic
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from gluonts.torch.util import copy_parameters
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
from utils.metric import RMSE
import numpy as np
import math
import collections
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer
import os
import time
import warnings
import matplotlib.pyplot as plt
from datetime import datetime


from models.dvdiffusion.src.utils import kl_balancer
from models.dvdiffusion.src.resnet import Res12_Quadratic
from models.dvdiffusion.src.diffusion_process import GaussianDiffusion
from models.dvdiffusion.src.embedding import DataEmbedding

### Models
from models.dvdiffusion import DVDiffusion, TrainerDVDiffusion
from models.idnet import IDNet, TrainerIDNet
from models.pgsmu import PGMSU, TrainerPGMSU

warnings.filterwarnings('ignore')
import yaml
from easydict import EasyDict
import torch
import matplotlib.pyplot as plt



import torch
import matplotlib.pyplot as plt

def plot_abundance(A, nr, nc,
                   colorscale = "rainbow", 
                   thetitle: list = None,
                   savepath : str =None):
    
    ''' plots abundance maps, should be P by N '''
    P = A.shape[0] # number of endmembers
    N = A.shape[1] # number of pixels
    A_cube = np.reshape(np.transpose(A), (nc, nr, P))
    fig, axs = plt.subplots(1, P, figsize=(20,5))
    for i in range(P):
        axs[i].imshow(A_cube[:,:,i].T, cmap=colorscale, vmin=0, vmax=1) #cmap='gray'
        axs[i].axis('off')
        if thetitle:
            axs[i].set_title(thetitle[i], fontsize=12)
    if savepath is not None: # save a figure is specified
        plt.savefig(savepath, dpi=300, format='png')
    plt.show()
    
def plot_endmembers(endmembers, thetitle:list = None, savepath=None):
    P = endmembers.shape[1]  # number of endmembers
    L = endmembers.shape[0]  # number of bands

    fig, axes = plt.subplots(1, P, figsize=(20, 5), sharex=True)

    for i in range(P):
        axes[i].plot(np.linspace(1, L, L), endmembers[:, i])
        axes[i].set_ylabel(f'Endmember {i}')                                                                                                                                                                           
        if thetitle:
            axes[i].set_title(thetitle[i], fontsize=12)

    plt.xlabel('Band')
    
    plt.tight_layout()

    if savepath is not None:  # save a figure if specified
        plt.savefig(savepath, dpi=300, format='png')

    return fig

class NeuralHyperspectral(Exp_Basic):
    def __init__(self, args):
        super(NeuralHyperspectral, self).__init__(args)
        self.args = args
        print("self.device", self.device)
        self.model_train = None
        self.diff_step = args.diff_steps
        self.save_dir = None
        self.checkpoints = None

    def _get_data(self, flag):
        args = self.args
        if flag in ['test']: # 'val', 'validation'
            shuffle_flag = False
            drop_last = False
            validation_flag = True
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            validation_flag = False
            batch_size = args.batch_size
        
        data_set = HyperspectralDataset(self.args,
                    validation_flag = validation_flag)
        
        print(f"Flag: {flag} - data size: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        return data_set, data_loader

    def _select_optimizer(self):
        denoise_optim = optim.Adam(
            self.dv_diffusion.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95), weight_decay=self.args.weight_decay
        )
        return denoise_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
        
    def vali(self, vali_data, vali_loader, criterion):
        #copy_parameters(self.dv_diffusion, self.pred_net)
        total_mse = []
        total_mae = []

        # batch_y_unsup, batch_y_sup, batch_m, batch_a
        for i, (batch_y_unsup, batch_y_sup, batch_m, batch_a) in enumerate(vali_loader):
            batch_y_sup = batch_y_sup.float().to(self.device)
            batch_y_sup = batch_y_sup.float().to(self.device)
            batch_m = batch_m.float().to(self.device)
            batch_a = batch_a.float().to(self.device)

            batch_y_sup = batch_y_sup[...,-self.args.endmembers_dim:].float().to(self.device)
            _, out, _, _ = self.dv_diffusion.inference(batch_y_sup) #self.pred_net(batch_y_sup)

            mse = criterion(out, batch_m)
            total_mse.append(mse.item())
        total_mse = np.average(total_mse)
        return total_mse
    
    def train(self,
            save_dir=None,
            load_checkpoint=None,
            save_checkpoint : bool = True) -> None:
        
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        # test_data, test_loader = self._get_data(flag = 'test')

        os.makedirs(save_dir, exist_ok=True)

        d = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.save_dir = os.path.join(save_dir, f"outputs-{d}/")
        os.makedirs(self.save_dir, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor=  "val_loss",
                min_delta= 0.000001,
                patience=self.args.patience,
                verbose=True, mode="min"
            )
        ]
        use_model = "IDNet"
        
        if use_model == "IDNet":
            ## --------------
            params_file = "/tsi/data_education/Ladjal/koublal/open-source/NH_v2/experiment_sh_run/config.yaml"
            with open(params_file, "r") as f:
                config = yaml.safe_load(f)
            # N:2500, nr:50, nc:50, H:2, L:162, P:5
            P = 5
            L = 162
            H = 2
            training_optim = EasyDict(config["training_optim"])
            model = IDNet(P, L, H=H)
            self.model_train = TrainerIDNet(model, P=P, H=H, L=L, **training_optim)
            #ckpt = torch.load("/tsi/data_education/Ladjal/koublal/open-source/IDNet/results/model_bugs.pth",map_location=torch.device('cpu'))
            #self.model_train.model.load_state_dict(ckpt)
            ## --------------

        elif use_model == "PGMSU":
            model = PGMSU(P=self.args.abundance_dim, Channel=self.args.channels, z_dim=self.args.abundance_dim)
            self.model_train = TrainerPGMSU(model, self.args)

        elif use_model == "DVDiffusion":
            model = DVDiffusion(self.args)
            self.model_train = TrainerDVDiffusion(model, self.args)
            #ckpt = torch.load("/tsi/data_education/Ladjal/koublal/open-source/PGMSU/PGMSU_weight/Diffusion_alphav45_best.pt",map_location=torch.device('cpu'))
            #self.model_train.model.model.load_state_dict(ckpt)
            print("Strat training from 0")

        if load_checkpoint is not None:
            checkpoint = torch.load(load_checkpoint)
            self.model_train.load_state_dict(checkpoint)


        trainer = Trainer(
            callbacks=callbacks,
            num_sanity_val_steps=0,
            #gradient_clip_val = 0.8,
            strategy = 'ddp_find_unused_parameters_true',
            max_epochs=self.args.train_epochs)
        
        trainer.fit(self.model_train, train_loader, vali_loader)

        if save_checkpoint:
            self.checkpoints_dir = self.save_dir+f'best_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}.pth'
            self.checkpoints = self.model_train.state_dict()
            torch.save(self.checkpoints, self.checkpoints_dir)
            print(f"Weight are saved in {self.checkpoints_dir}")



    def test(self, save_dir=None, load_checkpoint = None, flag : str = "test"):
        self.args.batch_size = 1
        test_data, test_loader = self._get_data(flag=flag)
        

        if self.checkpoints is None and load_checkpoint is not None:
            self.checkpoints  = torch.load(load_checkpoint)

        if self.save_dir is None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

        ## --------------
        use_model = "IDNet"

        if use_model == "IDNet":
            if self.model_train is None:
                params_file = "/tsi/data_education/Ladjal/koublal/open-source/NH_v2/experiment_sh_run/config.yaml"
                with open(params_file, "r") as f:
                    config = yaml.safe_load(f)
                # N:2500, nr:50, nc:50, H:2, L:162, P:5
                P = 5
                L = 162
                H = 2
                training_optim = EasyDict(config["training_optim"])
                #if self.model_train is None:
                self.model_train = IDNet(P, L, H=H)
                self.model_train = TrainerIDNet(self.model_train, P=P, H=H, L=L, **training_optim)
                ckpt = torch.load("/tsi/data_education/Ladjal/koublal/open-source/IDNet/results/model_bugs.pth",map_location=torch.device('cuda'))
                self.model_train.model.load_state_dict(ckpt)
            else:
                print("use IDENET from Training Function")
        ## -------------- 
        elif use_model == "PGMSU":
            if self.model_train is None:
                model = PGMSU(P=self.args.abundance_dim, Channel=self.args.channels, z_dim=self.args.abundance_dim)
                self.model_train = TrainerPGMSU(model, self.args)
                ckpt = torch.load("/tsi/data_education/Ladjal/koublal/open-source/NH_v2/resutls/Synthetic_Variability/outputs-2024.05.26_18.25.13/best_2024.05.26_18.25.35.pth",map_location=torch.device('cpu'))
                self.model_train.load_state_dict(ckpt)
                print("self.model_train  is build")
            else:
                print("self.model_train  is available")
        
        elif use_model == "DVDiffusion":
            if self.model_train is None:
                model = DVDiffusion(self.args)
                self.model_train = TrainerDVDiffusion(model, self.args)
                ckpt = torch.load("/tsi/data_education/Ladjal/koublal/open-source/NH_v2/resutls/Synthetic_Variability/outputs-2024.05.26_18.25.13/best_2024.05.26_18.25.35.pth",map_location=torch.device('cpu'))
                self.model_train.load_state_dict(ckpt)
                print("self.model_train  is build")
            else:
                print("self.model_train  is available")

            # Load model weights
            #if load_checkpoint is not None:
            # try:
            #     self.model_train.load_state_dict(self.checkpoints)
            #     print("loaded load_state_dict from")
            # except:
            #     self.model_train.model.load_state_dict(self.checkpoints['state_dict'])
            #     print("loaded load_state_dict with ckpt['state_dict']")
            
            #else:
            #    print("self.model_train use trainable prams from train function")

        trainer = Trainer()
        
        trainer.test(self.model_train, test_loader)
        abundance, endmemebers,  y_hat_gen, y_hat_dirichlet, y_trues = self.model_train.pred

        if use_model == "IDNet":
            # They are torch tensors
            abundance = abundance.permute(1,0).numpy()
            endmemebers = endmemebers.permute(1,2,0).numpy()
            y_hat_gen = y_hat_gen.permute(1,0).numpy()
            y_hat_dirichlet = y_hat_dirichlet.permute(1,0).numpy()
            y_trues = y_trues.permute(1,0).numpy()

            print("use_idenet", use_model)
            print(f"abundance shape: {abundance.shape}")
            print(f"endmemebers shape: {endmemebers.shape}")
            print(f"y_hat_dirichlet shape: {y_hat_dirichlet.shape}")
            print(f"y_hat_gen shape: {y_hat_gen.shape}")
            print(f"y_trues shape: {y_trues.shape}")

        else:
            # print(f"abundance shape: {abundance[0].shape}")
            # print(f"endmemebers shape: {endmemebers[0].shape}")
            # print(f"y_hat_dirichlet shape: {y_hat_dirichlet[0].shape}")
            # print(f"y_hat_gen shape: {y_hat_gen[0].shape}")
            # print(f"y_trues shape: {y_trues[0].shape}")
            abundance = abundance.T.numpy()
            endmemebers = endmemebers.numpy()
            y_hat_gen = y_hat_gen.numpy()
            y_hat_dirichlet = y_hat_dirichlet.numpy()
            y_trues = y_trues.T.numpy()

        # Plot results 
        # SynthVaria
        # N:2500, nr:50, nc:50, H:2, L:162, P:5
        plot_abundance(abundance, nr=50, nc=50,
                    thetitle=None, # Should be a list like ["water", "metal" ....]
                    savepath=self.save_dir + "plot_abundance.png") #(A, nr, nc, thetitle='', savepath=None)
        
        plot_endmembers(endmemebers,
                    thetitle= None, # Should be a list like ["water", "metal" ....]
                    savepath = self.save_dir + "plot_endmembers.png")


        np.save(self.save_dir + 'abundance.npy', abundance)
        np.save(self.save_dir + 'endmemebers.npy', endmemebers)
        np.save(self.save_dir + 'y_hat_gen.npy', y_hat_gen)
        np.save(self.save_dir + 'y_hat_dirichlet.npy', y_hat_dirichlet)
        np.save(self.save_dir + 'y_trues.npy', y_trues)
        mae = 0
        mse = 0
        return mae, mse
    
    def train_over(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        train_steps = len(train_loader)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        denoise_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        train = []

        for epoch in range(self.args.train_epochs):
            all_loss_log_prob = []
            all_diffusion_loss = []
            all_loss = []
            mse = []
            self.dv_diffusion.train()
            epoch_time = time.time()

            for i, (batch_y_unsup, batch_y_sup, batch_m, batch_a) in enumerate(train_loader):


                batch_y_unsup, batch_y_sup, batch_m, batch_a = batch_y_unsup.to(self.device), batch_y_sup.to(self.device), batch_m.to(self.device), batch_a.to(self.device)
                t = torch.randint(0, self.diff_step, (self.args.batch_size,)).long().to(self.device)
                batch_y_unsup = batch_y_unsup.float().to(self.device)
                batch_y_unsup = batch_y_unsup[...,-self.args.endmembers_dim:].float().to(self.device)
                denoise_optim.zero_grad()

                # end_memebers_sample, end_memebers_diffused, total_correlation, l_variational_latents, loss_diffusion
                endmemebers_sample, endmemebers_diffused, kl_loss, l_var_latents, _ = self.dv_diffusion(batch_y_sup, batch_y_sup,batch_m, batch_a, t)
                
                kl_loss, kl_coeffs, kl_vals = kl_balancer(kl_loss)

                #recon = endmemebers_sample.log_prob(endmemebers_sample.sample())

                mse_loss = criterion(endmemebers_sample.sample(), batch_m) #(endmemebers_diffused)
                #log_prob = - torch.mean(torch.sum(recon, dim=[1, 2, 3]))

                #loss = torch.tensor(mse_loss, requires_grad=True)
                loss = mse_loss + self.args.gamma*kl_loss

                all_loss.append(loss.item())
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.dv_diffusion.parameters(), max_norm=.54)  # Adjust max_norm as needed
                denoise_optim.step()
                #all_loss_log_prob.append(log_prob.item()*self.args.psi)
                #all_diffusion_loss.append(diff_loss.item()*self.args.lambda1)
                mse.append(mse_loss.item())
                if i%40==0:
                    print(f"loss iter {i}: {loss}")
                    
            all_loss = np.average(all_loss)
            train.append(all_loss)
            all_loss_log_prob = 0 #np.average(all_loss_log_prob)
            all_diffusion_loss = 0 #np.average(all_diffusion_loss)
            mse = np.average(mse)

            print(f"Training Epoch: {epoch + 1}, Steps: {train_steps} | Train L1 : {all_loss_log_prob:.7f} Train diffusion_loss: {all_diffusion_loss:.7f} Train MSE: {mse:.7f} Train loss:{all_loss:.7f}")
            print(f"Validation Epoch: {epoch + 1} \n")
            vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_mse = self.vali(test_data, test_loader, criterion)
            print("vali_mse:{0:.7f}, test_mse:{1:.7f}".format(vali_mse, test_mse), "--"*10, "\n")
            # print("Epoch: {0}, Steps: {1} | Train L1 : {2:.7f} Train diffusion_loss: {3:.7f} Train loss3: {4:.7f} Train loss:{5:.7f}".format(
            #    epoch + 1, train_steps, all_loss_log_prob, all_diffusion_loss, mse, all_loss))
            
            early_stopping(vali_mse, self.dv_diffusion, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(denoise_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.dv_diffusion.load_state_dict(torch.load(best_model_path))

    def test_over(self, setting):
        #copy_parameters(self.dv_diffusion, self.pred_net)
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        noisy = []
        input = []
        for i, (batch_x, batch_y_unsup, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y_unsup = batch_y_unsup[...,-self.args.endmembers_dim:].float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            noisy_out, out, _, _ = self.pred_net(batch_x, batch_x_mark)

            noisy.append(noisy_out.squeeze(1).detach().cpu().numpy())
            preds.append(out.squeeze(1).detach().cpu().numpy())
            trues.append(batch_y_unsup.detach().cpu().numpy())
            input.append(batch_x[...,-1:].detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        noisy = np.array(noisy)
        input = np.array(input)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'noisy.npy', noisy)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', input)
        return mae, mse
