#!/usr/bin/env bash
#SBATCH -J nlcd_raw
#SBATCH -p serc
#SBATCH -t 05:00:00
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH -o logs/nlcd_raw_%j.out
#SBATCH -e logs/nlcd_raw_%j.err

set -euo pipefail

echo "[$(date)] Job starting on $(hostname)"

cd "$HOME/long_lfmc/data_processing/nlcd"
mkdir -p logs

# Activate processing env (swap if your cluster only has the non-py312 script)
source "$HOME/uv_activations/activate_lfmc_process_py312.sh"

bash ./get_raw_nlcd.sh

echo "[$(date)] Job finished"
