#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

GPU_TASK_ID="${GPU_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${GPU_TASK_ID:-}" ]]; then
    echo "GPU_TASK_ID or SLURM_ARRAY_TASK_ID must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

model_type="${MODEL_TYPE:-standard}"
dynamic_work_queue="${DYNAMIC_GPU_WORK_QUEUE:-false}"
gpu_job_time_limit_seconds="${GPU_JOB_TIME_LIMIT_SECONDS:-}"

echo "Running GPU forward task ${GPU_TASK_ID}"
echo "manifest_path=${MANIFEST_PATH}"
echo "model_type=${model_type}"
echo "dynamic_work_queue=${dynamic_work_queue}"
echo "gpu_job_time_limit_seconds=${gpu_job_time_limit_seconds}"

run_args=(
    --manifest_path "${MANIFEST_PATH}"
    --model_type "${model_type}"
)

if [[ "${dynamic_work_queue}" == "true" ]]; then
    run_args+=(--worker_id "${GPU_TASK_ID}" --dynamic_work_queue)
else
    run_args+=(--task_id "${GPU_TASK_ID}")
fi

python3 -u "${script_dir}/run_map_task_gpu.py" "${run_args[@]}"
