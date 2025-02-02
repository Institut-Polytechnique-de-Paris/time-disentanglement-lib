# -*-Encoding: utf-8 -*-
"""
Authors: Khalid Oublal, PhD IPP / OneTech (khalid.oublal@poytechnique.edu)
"""


import argparse
import torch
import numpy as np
from datetime import datetime
import random
from trainer.exp_model import NeuralHyperspectral
from utils.debuglogger import DebugLogger
parser = argparse.ArgumentParser(description='generating')
parser.add_argument('--fix_seed', type=int, default=123456, help='Fix seed') 
# This line of code `parser.add_argument('--data_option', type=str, default='Synthetic_Variability',
# help='data option from implemented ones[Synthetic_Nonlinear_Mixture,Synthetic_Variability,Samson,
# Jasper, Cuprite]')` is adding a command-line argument to the script.

# LOADING: load data
parser.add_argument('--data_option', type=str, default='Synthetic_Variability', help='data option from implemented ones[Synthetic_Nonlinear_Mixture,Synthetic_Variability,Samson, Jasper, Cuprite]') 
parser.add_argument('--data_path', type=str, default='//tsi/data_education/Ladjal/koublal/open-source/IDNet/DATA/synth_DC1/data_ex_nl1.mat', help='data file')  # # load image !! change if accordingly
parser.add_argument('--data_path_extract', type=str, default='//tsi/data_education/Ladjal/koublal/open-source/IDNet/DATA/synth_DC1/extracted_bundles_nl_ex1.mat', help='data file')  # # load spectral !! change if accordingly
parser.add_argument('--root_path', type=str, default='./data/', help='where should I put resutls')
parser.add_argument('--endmembers', type=str, default='endmembers_dim', help='target variable endmembers_dim')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')

# HARD SETTING
parser.add_argument('--channels', type=int, default=224, help='number of channels or L')
parser.add_argument('--number_columns', type=int, default=100, help='number_columns for the image length')
parser.add_argument('--number_rows', type=int, default=100, help='number_rows for the image')
parser.add_argument('--percentage', type=float, default=0.02, help='the percentage of the whole dataset')
parser.add_argument('--endmembers_dim', type=int, default=3, help='dimension of Endmember')
parser.add_argument('--abundance_dim', type=int, default=5, help='dimension of Abundance')
parser.add_argument('--input_dim', type=int, default=1, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=32, help='dimension of input')
parser.add_argument('--embedding_dimension', type=int, default=128, help='feature embedding dimension')

# DIRICHLET PARAMS:
parser.add_argument('--alpha_min', type=float, default=0.5, help='alpha')
parser.add_argument('--dir_prior', type=float, default=0.3, help='dir_prior')
parser.add_argument('--use_dirichlet_mix', type=bool, default=True, help='use_dirichlet_mix')
parser.add_argument('--n_componenet', type=int, default=3, help='n_componenet of dirichlet')
parser.add_argument('--kl_dirichlet', type=float, default=1e-2, help='trade off Implicit Gradient dirichlet')


# DIFFUSION PRAMS MODEL:
parser.add_argument('--score_hw', type=int, default=32, help='score_hw for model score net')
parser.add_argument('--score_hdim', type=int, default=64, help='score_hdim  dim of score net')
parser.add_argument('--diff_steps', type=int, default=2, help='number of the diff step')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
parser.add_argument('--beta_end', type=float, default=0.01, help='end of the beta')
parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')
parser.add_argument('--psi', type=float, default=0.5, help='trade off parameter psi')
parser.add_argument('--lambda1', type=float, default=1.0, help='trade off parameter lambda')
parser.add_argument('--gamma', type=float, default=0.01, help='trade off parameter gamma')
parser.add_argument('--weight_diffusion', type=float, default=1e-3, help='trade off diffusion parameter weight_diffusion')




# BACKBONE: l-variational Inference (now prams are arround 10M --> need to go for 6M)

parser.add_argument('--mult', type=float, default=1, help='mult of channels')
parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')
parser.add_argument('--res_dist', type=bool, default=False, help='use res_dist')
parser.add_argument('--without_diffusion', type=bool, default=False, help='without_diffusion')

# parser.add_argument('--mult', type=float, default=1, help='mult of channels')
# parser.add_argument('--num_layers', type=int, default=2, help='num of layers')
# parser.add_argument('--num_channels_enc', type=int, default=8, help='number of channels in encoder')
# parser.add_argument('--channel_mult', type=int, default=5, help='number of channels in encoder')
# parser.add_argument('--num_preprocess_blocks', type=int, default=2, help='number of preprocessing blocks')
# parser.add_argument('--num_preprocess_cells', type=int, default=2, help='number of cells per block')
# parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
# parser.add_argument('--num_postprocess_blocks', type=int, default=2, help='number of postprocessing blocks')
# parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
# parser.add_argument('--num_channels_dec', type=int, default=8, help='number of channels in decoder')
# parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')
# parser.add_argument('--res_dist', type=bool, default=False, help='use res_dist')
# parser.add_argument('--without_diffusion', type=bool, default=False, help='without_diffusion')


# TRAIING SETTING ########################################################
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiment times')
parser.add_argument('--dim', type=int, default=-1, help='forecasting dims')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0200, help='weight decay')
parser.add_argument('--loss_type', type=str, default='kl',help='loss function')
parser.add_argument('--debuglogger', type=bool, default=False,help='debug script')


# USE GPU / JeanZay config on slurm Multi process
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multiple gpus')
parser.add_argument('--training_testiong', type=bool, default='0',help='both training and testiong')

args = parser.parse_args()
args.use_gpu = False #True if torch.cuda.is_available() and args.use_gpu else False

random.seed(args.fix_seed)
torch.manual_seed(args.fix_seed)
np.random.seed(args.fix_seed)

if args.debuglogger is True:
    print("🤖 Debugging Mode:", args.debuglogger)
    debug_logger = DebugLogger(log_dir='.logs')
    debug_logger.enable_logging()

args.use_multi_gpu = True
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [0] #[int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

all_mse = []
all_mae = []

args.training_testiong = True

# setting record of experiments
setting = f'{args.data_option}'
exp = NeuralHyperspectral(args)  # set experiments
if args.training_testiong:
    print('🚀 start training : {}'.format(setting))
    model = exp.train(save_dir=f"{args.root_path}/resutls/{args.data_option}/")

    print('🚀 start testing : {}'.format(setting))
    mae, mse = exp.test(save_dir="/tsi/data_education/Ladjal/koublal/open-source/NH_v2/resutls/Synthetic_Variability/outputs/")
else:
    print('🚀 start testing : {}'.format(setting))
    mae, mse = exp.test(save_dir="/tsi/data_education/Ladjal/koublal/open-source/NH_v2/resutls/Synthetic_Variability/outputs-2024.05.24_01.59.59/",
                    load_checkpoint="/tsi/data_education/Ladjal/koublal/open-source/NH_v2/resutls/Synthetic_Variability/outputs-2024.05.24_01.51.28/best_2024.05.24_01.52.36.pth")


torch.cuda.empty_cache()

print(f'END EXP AT: {datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}')

    
