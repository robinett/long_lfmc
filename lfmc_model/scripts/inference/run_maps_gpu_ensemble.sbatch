#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/run_maps_gpu_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/run_maps_gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

lock_file="${LOCK_FILE:-}"
if [[ -n "${lock_file}" ]]; then
  cleanup() { rm -f "${lock_file}"; }
  trap cleanup EXIT INT TERM
fi

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "MANIFEST_PATH must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

if [[ -z "${GPU_TASK_ID:-}" ]]; then
    echo "GPU_TASK_ID must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

model_type="${MODEL_TYPE:-standard}"

echo "Running GPU forward task ${GPU_TASK_ID}"
echo "manifest_path=${MANIFEST_PATH}"
echo "model_type=${model_type}"

python3 -u "${script_dir}/run_map_task_gpu.py" \
    --manifest_path "${MANIFEST_PATH}" \
    --task_id "${GPU_TASK_ID}" \
    --model_type "${model_type}"
