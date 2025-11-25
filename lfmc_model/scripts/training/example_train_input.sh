#!/bin/bash

# note that this model is way smaller than we would ever want
# but keep it at this size to not overwrite existing outputs

python3 -u train_multitarget_longweather_vvvh.py \
    --input_data_dir '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs_base' \
    --save_dir '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/testing' \
    --batch_size 128 \
    --lr 1e-4 \
    --val_split 0.2 \
    --adam_wd 1e-4 \
    --d_model 16 \
    --nhead 1 \
    --num_layers 1 \
    --dim_feedforward 32 \
    --dropout 0.2 \
    --long_d_model 16 \
    --long_nhead 1 \
    --long_num_layers 1 \
    --long_dim_feedforward 32 \
    --long_out_dim 16