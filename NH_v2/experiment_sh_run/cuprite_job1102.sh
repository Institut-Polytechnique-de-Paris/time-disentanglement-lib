#!/bin/bash -l
#####################
# job-array example #
#####################
#SBATCH --output=//tsi/data_education/Ladjal/koublal/experiences/Multimodels/Multimodal_nlayer_4_all%j.out
#SBATCH --error=//tsi/data_education/Ladjal/koublal/experiences/Multimodels/Multimodal_nlayer_4_all%j.err
#SBATCH --job-name=HVAE
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1 # nombre de GPU réservés par nœud (ici 8 soit tous les GPU)
#SBATCH --nodes=1 # nombre total de nœuds (N à définir)
#SBATCH --mem=256G
#SBATCH --partition=A100
#SBATCH --cpus-per-task=21 # nombre de cœurs par tache (donc 8x3 = 24 cœurs soit tous les cœurs)

export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_LAUNCH_BLOCKING=1
#set -x
cd //tsi/data_education/Ladjal/koublal/open-source/NH_v2/
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
# # Please @rasim use srun in ids cluster to avoid swap issues (here python just to test)
# Config
# Cuprite --> N:47750, nr:250, nc:191, H:2, L(channels):188, P:14

python main.py \
--root_path //tsi/data_education/Ladjal/koublal/open-source/NH_v2 \
--data_option 'Cuprite' \
--data_path /tsi/data_education/Ladjal/koublal/open-source/IDNet/DATA/real_Cuprite/alldata_real_Cuprite.mat \
--data_path_extract //tsi/data_education/Ladjal/koublal/open-source/IDNet/DATA/real_Cuprite/extracted_bundles.mat \
--percentage 0.05 \
--input_dim 1 \
--endmembers_dim 14 \
--abundance_dim 14 \
--channels 188 \
--hidden_size 8 \
--embedding_dimension 8 \
--score_hw 4 \
--score_hdim 4 \
--num_channels_enc 4 \
--num_channels_dec 4 \
--diff_steps 100 \
--dropout_rate 0.1 \
--beta_schedule 'linear' \
--beta_start 0.0 \
--beta_end 0.01 \
--beta_end 0.1 \
--alpha_min 0.4 \
--dir_prior 0.3 \
--use_dirichlet_mix False \
--n_componenet 3 \
--use_gpu True \
--without_diffusion True \
--num_workers 0 \
--patience 10 \
--train_epochs 10 \
--batch_size 64 \
--learning_rate 0.001 \
--weight_decay 0.0200 \


#### HERE OTHER PARAMS TO CHECK

# python main.py \
# --root_path //tsi/data_education/Ladjal/koublal/open-source/NH_v2 \
# --data_path //tsi/data_education/Ladjal/koublal/open-source/NH_v2/data_load/ETTh1.csv \
# --data_option 'Synthetic_Variability' \
# --checkpoints './checkpoints/' \
# --arch_instance 'res_mbconv' \
# --channels 162 \
# --channels 162 \
# --percentage 0.02 \
# --endmembers_dim 5 \
# --input_dim 1 \
# --hidden_size 32 \
# --embedding_dimension 128 \
# --diff_steps 2 \
# --dropout_rate 0.1 \
# --beta_schedule 'linear' \
# --beta_start 0.0 \
# --beta_end 0.01 \
# --scale 0.1 \
# --psi 0.5 \
# --lambda1 1.0 \
# --gamma 0.01 \
# --mult 1 \
# --num_layers 2 \
# --num_channels_enc 32 \
# --num_channels_dec 32 \
# --channel_mult 2 \
# --num_preprocess_blocks 1 \
# --num_preprocess_cells 3 \
# --groups_per_scale 2 \
# --num_postprocess_blocks 1 \
# --num_postprocess_cells 2 \

# --num_latent_per_group 8 \
# --res_dist False \
# --without_diffusion True \
# --loss_type 'kl' \
# --use_gpu True \
# --gpu 0 \
# --devices '0'
# #--use_multi_gpu \
