#!/bin/bash

ensemble_outputs_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_shared_ensemble"
ensemble_member_name_prefix="transformer_dm128_"
ensemble_subset_size=16
ensemble_subset_seed=0
plot_dir=""
hexbin_gridsize=70
fontsize=16

cmd=(
    python3 eval_dm128_test_uncertainty.py
    --ensemble_outputs_root "$ensemble_outputs_root"
    --ensemble_member_name_prefix "$ensemble_member_name_prefix"
    --ensemble_subset_size "$ensemble_subset_size"
    --ensemble_subset_seed "$ensemble_subset_seed"
    --hexbin_gridsize "$hexbin_gridsize"
    --fontsize "$fontsize"
)

if [ -n "$plot_dir" ]; then
    cmd+=(--plot_dir "$plot_dir")
fi

"${cmd[@]}"
