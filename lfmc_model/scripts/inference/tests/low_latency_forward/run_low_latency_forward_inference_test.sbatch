#!/bin/bash

#SBATCH --job-name=ll_fwd_coord
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/low_latency_forward_inference_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/low_latency_forward_inference_%j.err
#SBATCH --time=168:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
log_dir="/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup"
source_registry="${log_dir}/source_registry_test.yaml"
map_config="${log_dir}/map_configs_low_latency_test.yaml"
test_lock_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/tests/low_latency_forward_setup/production/metadata/locks/low_latency_daily.lock"

unset RUN_ROOT
unset EXISTING_RUN_DIR
unset START_FROM_GPU
unset AUTO_RESUME_COMPLETE_RUN
unset REQUESTED_START_DATE
unset REQUESTED_END_DATE
unset CONFIG_PATH
unset REGISTRY_PATH

if [[ -d "${test_lock_dir}" ]]; then
    echo "Removing stale low-latency test lock: ${test_lock_dir}"
    rmdir "${test_lock_dir}" || true
fi

GPU_MAX_JOBS="${GPU_MAX_JOBS:-0}" \
OWNERS_GPU_MAX_JOBS="${OWNERS_GPU_MAX_JOBS:-100}" \
OWNERS_PARTITION="${OWNERS_PARTITION:-owners}" \
OWNERS_GPU_CONSTRAINT="${OWNERS_GPU_CONSTRAINT:-}" \
TODAY_OVERRIDE="${TODAY_OVERRIDE:-2025-01-07}" \
STRICT_CONFIG_RUN_ROOT=true \
AUTO_RESUME_COMPLETE_RUN=true \
CONFIG_PATH="${map_config}" \
REGISTRY_PATH="${source_registry}" \
SKIP_OAK_SYNC=1 \
bash "${repo_root}/lfmc_model/scripts/inference/orchestrate_low_latency_daily_update.sh"
