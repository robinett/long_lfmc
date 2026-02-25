#!/usr/bin/env bash
set -euo pipefail

source /home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh

cd ~/long_lfmc/data_processing/convert_to_zarr

ROOT="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid"
OUT="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars_1month_smoketest.zarr"
COORD_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_queue_coord_1month_smoketest"

python3 -u daymet_to_zarr_worker.py \
  --coord-dir "${COORD_DIR}" \
  --root "${ROOT}" \
  --out "${OUT}" \
  --max-months 1 \
  --rebuild-index
