#!/bin/bash

# Example run for compare_timeseries.py.
# Edit the settings here for quick local runs.
#
# To compare multiple model families at once, replace the single-value
# model_root/model_name settings below with arrays and pass them through, e.g.
# model_names=("lfmc" "lfmc_vh_vv")
# model_roots=(
#     "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc"
#     "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv"
# )
# Then update the cmd array to use:
# --model_roots "${model_roots[@]}"
# --model_names "${model_names[@]}"
# Note: model_df_index is currently one value applied to every selected root.

model_name="lfmc_vh_vv"
model_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc"
model_df_index="27"
num_sites_per_criterion=3
min_measurements=10
padding_days=60
max_years=3
plot_vv=false
plot_vh=false

plot_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/model_comparisons"
inputs_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs"

cmd=(
    python3 compare_timeseries.py
    --num_sites_per_criterion "$num_sites_per_criterion"
    --min_measurements "$min_measurements"
    --plot_dir "$plot_dir"
    --inputs_root "$inputs_root"
    --padding_days "$padding_days"
    --max_years "$max_years"
    --model_roots "$model_root"
    --model_names "$model_name"
)

if [ -n "$model_df_index" ]; then
    cmd+=(--model_df_index "$model_df_index")
fi

if [ "$plot_vv" = true ]; then
    cmd+=(--plot_vv)
else
    cmd+=(--no-plot_vv)
fi

if [ "$plot_vh" = true ]; then
    cmd+=(--plot_vh)
else
    cmd+=(--no-plot_vh)
fi

"${cmd[@]}"
