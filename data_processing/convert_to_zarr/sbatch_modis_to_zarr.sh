#!/usr/bin/env bash
#SBATCH -J modis_to_zarr
#SBATCH -p serc,konings
#SBATCH -t 36:00:00
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 4096

# Activate your environment
source /home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh

# Move to the script directory
cd ~/long_lfmc/data_processing/convert_to_zarr

# Ensure log dir exists
mkdir -p logs

# Run
python3 -u modis_to_zarr.py
