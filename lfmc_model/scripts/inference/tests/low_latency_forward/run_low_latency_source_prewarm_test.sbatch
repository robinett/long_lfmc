#!/bin/bash

#SBATCH --job-name=ll_src_prewarm
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/low_latency_source_prewarm_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/low_latency_source_prewarm_%j.err
#SBATCH --time=168:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
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
unset CONFIG_PATH
unset REGISTRY_PATH

if [[ -d "${test_lock_dir}" ]]; then
    echo "Removing stale low-latency test lock: ${test_lock_dir}"
    rmdir "${test_lock_dir}" || true
fi

GPU_MAX_JOBS=0 \
OWNERS_GPU_MAX_JOBS=0 \
TODAY_OVERRIDE="${TODAY_OVERRIDE:-2025-01-07}" \
REQUESTED_START_DATE="${REQUESTED_START_DATE:-2023-01-01}" \
REQUESTED_END_DATE="${REQUESTED_END_DATE:-2023-12-31}" \
STRICT_CONFIG_RUN_ROOT=true \
CONFIG_PATH="${map_config}" \
REGISTRY_PATH="${source_registry}" \
SKIP_INFERENCE=1 \
SKIP_OAK_SYNC=1 \
bash "${repo_root}/lfmc_model/scripts/inference/orchestrate_low_latency_daily_update.sh"
