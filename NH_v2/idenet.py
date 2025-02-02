import argparse
import os
import torch
import yaml
from data_load.data_loader import HyperspectralDataset
from model.idnet import IDNet, TrainerIDNet
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer
from easydict import EasyDict as edict
from utils.utils import viz

#### --------
from data_load.data_loader import Dataset_Custom, HyperspectralDataset
from trainer.exp_basic import Exp_Basic
from model.DVDiffusion import DVDiffusion
from model.utils import kl_balancer
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from gluonts.torch.util import copy_parameters
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
from utils.metric import RMSE
from model.resnet import Res12_Quadratic
from model.diffusion_process import GaussianDiffusion
from model.embedding import DataEmbedding
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
from model.dvdiffusion.config import TrainerDVDiffusion
warnings.filterwarnings('ignore')

def main(model_name, params_file):
    with open(params_file, "r") as f:
        config = yaml.safe_load(f)

    training_optim = edict(config["training_optim"])

    
    # Your code for setting up data loaders and other configurations
    train_loader = torch.utils.data.DataLoader(HyperspectralDataset(config, validation_flag=False), 
                                        batch_size=training_optim.batch_size,
                                        shuffle=training_optim.batch_shuffle,
                                        num_workers = training_optim.num_workers if training_optim.checkpoint \
                                                    is not None else os.cpu_count()
                                        )
    
    L = train_loader.dataset.data_sup[0][1].shape[0]
    P = train_loader.dataset.data_sup[0][1].shape[1]
    N = len(train_loader.dataset.data_unsup)
    nr, nc = train_loader.dataset.A_u_cube.shape[0], train_loader.dataset.A_u_cube.shape[1]
    H = 2
    
    # Initialize your model and trainer
    model = IDNet(P, L, H=H)
    model_train = TrainerIDNet(model, P=P, H=H, L=L, **training_optim)

    if training_optim.checkpoint is not None:
        # Load the checkpoint
        checkpoint = torch.load(training_optim.checkpoint)
        # Load model weights
        model_train.load_state_dict(checkpoint['model_state_dict'])

    elif training_optim.checkpoint_ckpt is not None:
        model_train = model_train.load_from_checkpoint(training_optim.checkpoint_ckpt)
        
    callbacks = [
        EarlyStopping(
            monitor= training_optim.monitor_loss,
            min_delta= training_optim.min_delta,
            patience=training_optim.patience,
            verbose=True, mode="min"
        )
    ]

    trainer = Trainer(
        callbacks=callbacks,
        num_sanity_val_steps=0,
        strategy = 'ddp_find_unused_parameters_true',
        max_epochs=training_optim.max_epochs)

    trainer.fit(model_train, train_loader, train_loader)

    model_train.init()
    trainer.test(model_train, train_loader)


    fig = viz().Abunds(pred = torch.Tensor(model_train.A_pred_out.T),
                ground = train_loader.dataset.A_u,
                nr = nr,
                nc = nc,
                colorscale = "rainbow",
                thetitle = f'Residual Abundances {model_name}',
                savepath=f'{training_optim.save_path}/abondances_{model_name}.pdf')

    with open(f"{training_optim.save_path}/abondances_{model_name}.html", 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for the model")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--config", type=str, help="File containing model parameters")
    args = parser.parse_args()
    main(args.model_name, args.config)