#!/usr/bin/env bash
#SBATCH -J daymet_to_zarr_finalize
#SBATCH -p serc
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
source /home/users/trobinet/uv_activations/activate_lfmc_process.sh

cd ~/long_lfmc/data_processing/convert_to_zarr

python3 -u daymet_to_zarr_worker.py \
  --coord-dir "/scratch/users/trobinet/long_lfmc/daymet_queue_coord" \
  --finalize
