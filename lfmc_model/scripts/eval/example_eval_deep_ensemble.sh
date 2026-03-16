#!/bin/bash

# Example run for eval_deep.py using an ensemble root.
# Edit the settings here for quick local runs.

ensemble_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_ens"
plot_dir=""
hexbin_gridsize=60
fontsize=16

cmd=(
    python3 eval_deep.py
    --ensemble_outputs_root "$ensemble_root"
    --hexbin_gridsize "$hexbin_gridsize"
    --fontsize "$fontsize"
)

if [ -n "$plot_dir" ]; then
    cmd+=(--plot_dir "$plot_dir")
fi

"${cmd[@]}"
