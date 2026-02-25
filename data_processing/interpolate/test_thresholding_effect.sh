#!/bin/bash
set -euo pipefail

# Edit these defaults as needed.
BASE_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid"
CHECK_YEAR="2018"
CHECK_SAMPLE_SIZE="1000"
CHECK_PLOT_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/interpolation_check_2023.png"
CHECK_MAP_PLOT_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/interpolation_check_2023_may_missing_map.png"

# Space-separated thresholds.
CHECK_THRESHOLDS=(0 7 15 30 60 90)

# Leave empty to auto-select a reflectance band.
CHECK_BAND=""

cmd=(
    python3 interpolate_new.py
    --check_interpolation
    --base_path "${BASE_PATH}"
    --check_year "${CHECK_YEAR}"
    --check_sample_size "${CHECK_SAMPLE_SIZE}"
    --check_plot_path "${CHECK_PLOT_PATH}"
    --check_map_plot_path "${CHECK_MAP_PLOT_PATH}"
    --check_thresholds "${CHECK_THRESHOLDS[@]}"
)

if [[ -n "${CHECK_BAND}" ]]; then
    cmd+=(--check_band "${CHECK_BAND}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Saved plot to ${CHECK_PLOT_PATH}"
echo "Saved May missingness map to ${CHECK_MAP_PLOT_PATH}"
