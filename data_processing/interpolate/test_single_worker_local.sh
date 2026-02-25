#!/bin/bash
set -euo pipefail

# Edit these defaults for interactive debugging.
START_DATE="2000-01-01"
END_DATE="2024-12-31"
BASE_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid"
OUTPUT_ZARR="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/testing/debug_single_worker_local.zarr"
MAX_INTERPOLATION_DAYS="30"
BUFFER_DAYS=""
XY_CHUNK_SIZE="512"
NUM_WORKERS="1"
WORKER_ID="0"
OVERWRITE_ZARR="1"
MAX_CHUNKS_PER_WORKER="3"
ZERO_FILL_SKIPPED_CHUNKS="1"
DIAGNOSTIC_PLOT_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/debug_single_worker_local_timeseries.png"
DIAGNOSTIC_MAP_PLOT_PATH="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/debug_single_worker_local_map.png"
DIAGNOSTIC_BAND=""
DIAGNOSTIC_N_POINTS="3"
DIAGNOSTIC_SEED="0"

#source ~/.bashrc
source ~/uv_activations/activate_lfmc_process_py312.sh

common_args=(
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --base_path "${BASE_PATH}"
    --output_zarr "${OUTPUT_ZARR}"
    --max_interpolation_days "${MAX_INTERPOLATION_DAYS}"
    --xy_chunk_size "${XY_CHUNK_SIZE}"
    --num_workers "${NUM_WORKERS}"
)

if [[ -n "${BUFFER_DAYS}" ]]; then
    common_args+=(--buffer_days "${BUFFER_DAYS}")
fi

echo "Step 1/3: init"
cmd_init=(python3 interpolate_new.py --mode init "${common_args[@]}")
if [[ "${OVERWRITE_ZARR}" == "1" ]]; then
    cmd_init+=(--overwrite_zarr)
fi
echo "Running: ${cmd_init[*]}"
"${cmd_init[@]}"

echo "Step 2/3: worker"
cmd_worker=(
    python3 interpolate_new.py
    --mode worker
    "${common_args[@]}"
    --worker_id "${WORKER_ID}"
)
if [[ -n "${MAX_CHUNKS_PER_WORKER}" ]]; then
    cmd_worker+=(--max_chunks_per_worker "${MAX_CHUNKS_PER_WORKER}")
    if [[ "${ZERO_FILL_SKIPPED_CHUNKS}" == "1" ]]; then
        cmd_worker+=(--zero_fill_skipped_chunks)
    fi
fi
echo "Running: ${cmd_worker[*]}"
"${cmd_worker[@]}"

echo "Step 3/3: finalize (auto-generates diagnostics)"
cmd_finalize=(python3 interpolate_new.py --mode finalize "${common_args[@]}")
cmd_finalize+=(
    --diagnostic_plot_path "${DIAGNOSTIC_PLOT_PATH}"
    --diagnostic_map_plot_path "${DIAGNOSTIC_MAP_PLOT_PATH}"
    --diagnostic_n_points "${DIAGNOSTIC_N_POINTS}"
    --diagnostic_seed "${DIAGNOSTIC_SEED}"
)
if [[ -n "${DIAGNOSTIC_BAND}" ]]; then
    cmd_finalize+=(--diagnostic_band "${DIAGNOSTIC_BAND}")
fi
echo "Running: ${cmd_finalize[*]}"
"${cmd_finalize[@]}"

echo "Done. Output Zarr: ${OUTPUT_ZARR}"
echo "Diagnostic plot: ${DIAGNOSTIC_PLOT_PATH}"
echo "Diagnostic map: ${DIAGNOSTIC_MAP_PLOT_PATH}"
