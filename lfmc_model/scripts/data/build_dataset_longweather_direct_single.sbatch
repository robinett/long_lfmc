#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs/build_dataset_longweather_single_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs/build_dataset_longweather_single_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/data"
cd "${script_dir}"
mkdir -p /home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u /home/users/trobinet/long_lfmc/lfmc_model/scripts/data/build_dataset_longweather_direct.py "$@"
