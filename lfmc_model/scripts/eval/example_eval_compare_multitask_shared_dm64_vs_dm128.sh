#!/bin/bash

model_a_name="multitask_dm64"
model_a_ensemble_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_shared_ensemble"
model_a_ensemble_member_name_prefix="transformer_dm64_"
model_b_name="multitask_dm128"
model_b_ensemble_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_shared_ensemble"
model_b_ensemble_member_name_prefix="transformer_dm128_"
plot_dir=""
fontsize=16

cmd=(
    python3 eval_compare_models.py
    --model_a_name "$model_a_name"
    --model_b_name "$model_b_name"
    --model_a_ensemble_outputs_root "$model_a_ensemble_root"
    --model_b_ensemble_outputs_root "$model_b_ensemble_root"
    --model_a_ensemble_member_name_prefix "$model_a_ensemble_member_name_prefix"
    --model_b_ensemble_member_name_prefix "$model_b_ensemble_member_name_prefix"
    --fontsize "$fontsize"
)

if [ -n "$plot_dir" ]; then
    cmd+=(--plot_dir "$plot_dir")
fi

"${cmd[@]}"
