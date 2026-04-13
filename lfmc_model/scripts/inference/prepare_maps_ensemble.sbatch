#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/prepare_maps_%A_%a.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/prepare_maps_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "MANIFEST_PATH must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

prepare_task_id="${PREPARE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${prepare_task_id}" ]]; then
    echo "PREPARE_TASK_ID or SLURM_ARRAY_TASK_ID must be set before running ${BASH_SOURCE[0]}" >&2
    exit 1
fi

echo "Running CPU prepare task ${prepare_task_id}"
echo "manifest_path=${MANIFEST_PATH}"

python3 -u "${script_dir}/prepare_map_task.py" \
    --manifest_path "${MANIFEST_PATH}" \
    --task_id "${prepare_task_id}"
