#!/usr/bin/env bash
#SBATCH -J daymet_to_zarr_workers
#SBATCH -p serc
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-4%5  # 5 workers

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 4096

# Activate env
source /home/users/trobinet/uv_activations/activate_lfmc_process.sh

cd ~/long_lfmc/data_processing/convert_to_zarr
mkdir -p logs

COORD_DIR="/scratch/users/trobinet/long_lfmc/daymet_queue_coord"

python3 -u daymet_to_zarr_worker.py \
  --coord-dir "${COORD_DIR}"
