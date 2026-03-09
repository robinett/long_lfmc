#!/bin/bash

# Interactive smoke test for a single ensemble member.
# Re-run with a different seed, batch_seed, and run_tag for additional members.

python3 -u train_multitarget_longweather_vvvh.py \
    --input_data_dir '/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vh_vv' \
    --save_dir '/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/testing_ens' \
    --batch_size 128 \
    --lr 5e-4 \
    --val_split 0.15 \
    --adam_wd 1e-4 \
    --d_model 32 \
    --nhead 2 \
    --num_layers 2 \
    --dim_feedforward 64 \
    --dropout 0.15 \
    --long_d_model 64 \
    --long_nhead 2 \
    --long_num_layers 3 \
    --long_dim_feedforward 128 \
    --long_out_dim 16 \
    --num_tasks 3 \
    --task_weight_type 'manual' \
    --manual_task_weights 3.0 1.0 1.0 \
    --seed 1000 \
    --batch_seed 1000 \
    --split_seed 42 \
    --run_tag 'seed000'
