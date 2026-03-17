#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/build_map_manifest_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/build_map_manifest_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

export MPLCONFIGDIR="${script_dir}/logs/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
    CONFIG_PATH="${script_dir}/map_configs_full_2024.yaml"
fi
export CONFIG_PATH

echo "Running manifest build job ${SLURM_JOB_ID}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "manifest args: $*"

python3 -u "${script_dir}/create_map_manifest.py" "$@"
