#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs/build_sample_index_longweather_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs/build_sample_index_longweather_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/data"
cd "${script_dir}"
mkdir -p /home/users/trobinet/long_lfmc/lfmc_model/scripts/data/logs

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u /home/users/trobinet/long_lfmc/lfmc_model/scripts/data/build_sample_index_longweather.py "$@"
