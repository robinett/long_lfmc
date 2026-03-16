#!/bin/bash

# Example run for compare_timeseries.py using an ensemble root.
# Edit the settings here for quick local runs.
#
# For LFMC-only ensemble:
# ensemble_model_name="lfmc_ens"
# ensemble_model_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_ens"
# ensemble_input_data_name="lfmc"
# plot_vv=false
# plot_vh=false
#
# For multitask LFMC/VH/VV ensemble:
# ensemble_model_name="lfmc_vh_vv_ens"
# ensemble_model_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_ens"
# ensemble_input_data_name="lfmc_vh_vv"
# plot_vv=true
# plot_vh=true
#
# For full-random multitask LFMC/VH/VV ensemble:
# ensemble_model_name="lfmc_vh_vv_ens_fullrandom"
# ensemble_model_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_ens_fullrandom"
# ensemble_input_data_name="ensemble/lfmc_vh_vv_ens_fullrandom"
# plot_vv=true
# plot_vh=true

ensemble_model_names=(
    #"lfmc_ens"
    "lfmc_vh_vv_365_ens"
)
ensemble_model_roots=(
    #"/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_ens"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365_ens"
)
ensemble_input_data_names=(
    #"lfmc"
    "ensemble/lfmc_vh_vv_365_ens"
)

num_sites_per_criterion=10
min_measurements=10
padding_days=60
max_years=50
plot_vv=false
plot_vh=false

plot_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/plots/timeseries"
inputs_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs"

cmd=(
    python3 compare_timeseries.py
    --num_sites_per_criterion "$num_sites_per_criterion"
    --min_measurements "$min_measurements"
    --plot_dir "$plot_dir"
    --inputs_root "$inputs_root"
    --padding_days "$padding_days"
    --max_years "$max_years"
    --ensemble_model_roots "${ensemble_model_roots[@]}"
    --ensemble_model_names "${ensemble_model_names[@]}"
    --ensemble_input_data_names "${ensemble_input_data_names[@]}"
)

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
