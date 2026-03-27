#!/bin/bash

# Example run for comparing two ensemble/model outputs with eval_compare_models.py.

model_a_name="lfmc_365_ens"
model_a_ensemble_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_365_ens"
model_a_ensemble_member_name_prefix=""
model_b_name="lfmc_vh_vv_365_ens"
model_b_ensemble_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_ens"
model_b_ensemble_member_name_prefix=""
plot_dir=""
fontsize=16

cmd=(
    python3 eval_compare_models.py
    --model_a_name "$model_a_name"
    --model_b_name "$model_b_name"
    --model_a_ensemble_outputs_root "$model_a_ensemble_root"
    --model_b_ensemble_outputs_root "$model_b_ensemble_root"
    --fontsize "$fontsize"
)

if [ -n "$model_a_ensemble_member_name_prefix" ]; then
    cmd+=(--model_a_ensemble_member_name_prefix "$model_a_ensemble_member_name_prefix")
fi

if [ -n "$model_b_ensemble_member_name_prefix" ]; then
    cmd+=(--model_b_ensemble_member_name_prefix "$model_b_ensemble_member_name_prefix")
fi

if [ -n "$plot_dir" ]; then
    cmd+=(--plot_dir "$plot_dir")
fi

"${cmd[@]}"
