#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/generate_longweather_fold_info_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/generate_longweather_fold_info_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

cd "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training"
mkdir -p "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u /home/users/trobinet/long_lfmc/lfmc_model/scripts/training/generate_longweather_fold_info.py "$@"
