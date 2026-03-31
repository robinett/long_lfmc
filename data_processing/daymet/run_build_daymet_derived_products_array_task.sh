#!/bin/bash

#SBATCH --job-name=daymet_vars_and_anoms_array
#SBATCH --output=./logs/daymet_vars_and_anoms_array_%A_%a.out
#SBATCH --error=./logs/daymet_vars_and_anoms_array_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
cd "$SCRIPT_DIR"

mkdir -p ./logs

config_path="${1}"
num_shards="${2}"
smoke_test="${3}"
shift 3

if [[ $# -lt 1 ]]; then
    echo "Expected at least one anomaly variable name"
    exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
var_count="$#"
total_tasks=$((var_count * num_shards))

if (( task_id < 0 || task_id >= total_tasks )); then
    echo "Task id ${task_id} out of range for ${var_count} vars x ${num_shards} shards"
    exit 1
fi

var_index=$((task_id / num_shards))
shard_index=$((task_id % num_shards))
var_name="${@:$((var_index + 1)):1}"

echo "Array task ${task_id}/${total_tasks} -> var=${var_name} shard=$((shard_index + 1))/${num_shards}"

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u build_daymet_derived_products.py \
    --config "${config_path}" \
    --mode build-anomaly-var \
    --var "${var_name}" \
    --shard-index "${shard_index}" \
    --num-shards "${num_shards}" \
    $([[ "${smoke_test}" -eq 1 ]] && printf '%s' "--smoke-test")

echo "Daymet array task complete for ${var_name} shard ${shard_index}/${num_shards}"
