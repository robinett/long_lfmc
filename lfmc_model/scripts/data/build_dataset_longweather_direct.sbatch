#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=logs/build_dataset_longweather_%j.out
#SBATCH --error=logs/build_dataset_longweather_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

sample_index_path="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc.parquet"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p logs

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u build_dataset_longweather_direct.py "${sample_index_path}"
