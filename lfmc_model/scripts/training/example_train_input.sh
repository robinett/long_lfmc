#!/bin/bash

# note that this model is way smaller than we would ever want
# but keep it at this size to not overwrite existing outputs

python3 -u train_multitarget_longweather_vvvh.py \
    --input_data_dir '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs_base' \
    --save_dir '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/testing' \
    --batch_size 128 \
    --lr 1e-4 \
    --val_split 0.25 \
    --adam_wd 1e-4 \
    --d_model 64 \
    --nhead 2 \
    --num_layers 2 \
    --dim_feedforward 128 \
    --dropout 0.15 \
    --long_d_model 256 \
    --long_nhead 4 \
    --long_num_layers 3 \
    --long_dim_feedforward 512 \
    --long_out_dim 32 \
    --num_gradnorm_tasks 1