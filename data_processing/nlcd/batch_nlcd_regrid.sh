#!/usr/bin/env bash
#SBATCH -J nlcd_regrid
#SBATCH -p serc
#SBATCH -t 12:00:00
#SBATCH --mem=256G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH -o logs/nlcd_regrid_%j.out
#SBATCH -e logs/nlcd_regrid_%j.err

set -euo pipefail

echo "[$(date)] Job starting on $(hostname)"
mkdir -p "$HOME/long_lfmc/data_processing/nlcd/logs"

# 1) activate poetry env
source "$HOME/uv_activations/activate_lfmc_process.sh"

# 2) cd to working dir
cd "$HOME/long_lfmc/data_processing/nlcd"

# 3) run the script
python3 -u nlcd_zarr_to_target.py

echo "[$(date)] Job finished"

