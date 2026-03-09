#!/bin/bash

# Example run for eval_deep.py.
# Edit the settings here for quick local runs.
#
# Use model_dir when you already know the exact model to evaluate.
# Otherwise use model_root plus model_df_index to match the selection style
# used in example_compare_timeseries.sh.

model_dir=""
model_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv"
model_df_index="24"
sort_metric="test_insitu_rmse"
ascending=true

plot_dir=""
hexbin_gridsize=60
fontsize=16

cmd=(
    python3 eval_deep.py
    --outputs_root "$model_root"
    --sort_metric "$sort_metric"
    --hexbin_gridsize "$hexbin_gridsize"
    --fontsize "$fontsize"
)

if [ -n "$model_dir" ]; then
    cmd+=(--model_dir "$model_dir")
fi

if [ -n "$plot_dir" ]; then
    cmd+=(--plot_dir "$plot_dir")
fi

if [ -n "$model_df_index" ]; then
    cmd+=(--model_df_index "$model_df_index")
fi

if [ "$ascending" = true ]; then
    cmd+=(--ascending)
fi

"${cmd[@]}"
