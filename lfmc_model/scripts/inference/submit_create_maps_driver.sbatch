#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/create_maps_driver_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/create_maps_driver_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ -z "${CONFIG_PATH:-}" ]]; then
    CONFIG_PATH="${script_dir}/map_configs_full_2024.yaml"
fi
export CONFIG_PATH

echo "[create_maps_driver] Running manifest/submission workflow"
echo "[create_maps_driver] CONFIG_PATH=${CONFIG_PATH}"

bash "${script_dir}/submit_create_maps_ensemble.sh"
