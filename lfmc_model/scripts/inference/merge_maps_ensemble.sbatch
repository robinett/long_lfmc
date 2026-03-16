#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/merge_maps_%A_%a.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/merge_maps_%A_%a.err
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

overwrite_flag=""
if [[ "${OVERWRITE_MERGE:-0}" == "1" ]]; then
    overwrite_flag="--overwrite"
fi

if [[ "${MERGE_INITIALIZE_ONLY:-0}" == "1" ]]; then
    echo "Initializing merged store for manifest ${MANIFEST_PATH}"
    python3 -u "${script_dir}/merge_map_shards.py" \
        --manifest_path "${MANIFEST_PATH}" \
        --initialize_only \
        ${overwrite_flag}
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID must be set for merge array tasks" >&2
    exit 1
fi

echo "Merging manifest ${MANIFEST_PATH} for merge_task_id=${SLURM_ARRAY_TASK_ID}"
python3 -u "${script_dir}/merge_map_shards.py" \
    --manifest_path "${MANIFEST_PATH}" \
    --merge_task_id "${SLURM_ARRAY_TASK_ID}"
