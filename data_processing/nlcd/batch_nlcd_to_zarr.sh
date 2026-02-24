#!/usr/bin/env bash
#SBATCH -J nlcd_zarr
#SBATCH -p serc
#SBATCH -t 12:00:00
#SBATCH --mem=200G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH -o logs/nlcd_zarr_%j.out
#SBATCH -e logs/nlcd_zarr_%j.err

set -euo pipefail

echo "[$(date)] Job starting on $(hostname)"
mkdir -p "$HOME/long_lfmc/data_processing/nlcd/logs"

# 1) activate uv env
source "$HOME/uv_activations/activate_lfmc_process_py312.sh"

# 2) cd to working dir
cd "$HOME/long_lfmc/data_processing/nlcd"

# 3) run the script
python3 nlcd_to_zarr.py

echo "[$(date)] Job finished"

