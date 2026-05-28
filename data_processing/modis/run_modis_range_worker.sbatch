#!/bin/bash
#SBATCH --job-name=modis_range
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o /home/users/trobinet/long_lfmc/logs/low_latency_parallel/modis_range_%j.out
#SBATCH -e /home/users/trobinet/long_lfmc/logs/low_latency_parallel/modis_range_%j.err

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
script_path="${repo_root}/data_processing/modis/update_modis_range.py"

mkdir -p "${repo_root}/logs/low_latency_parallel"
source "${process_env}"

cmd=(
    python3 "${script_path}"
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --tail_context_days "${TAIL_CONTEXT_DAYS}"
    --registry_path "${REGISTRY_PATH}"
)

echo "Running MODIS range worker"
echo "  start_date=${START_DATE}"
echo "  end_date=${END_DATE}"
echo "  tail_context_days=${TAIL_CONTEXT_DAYS}"
"${cmd[@]}"
