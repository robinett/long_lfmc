#!/bin/bash
#SBATCH --job-name=ll_climate
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /home/users/trobinet/long_lfmc/logs/low_latency_parallel/low_latency_climate_%j.out
#SBATCH -e /home/users/trobinet/long_lfmc/logs/low_latency_parallel/low_latency_climate_%j.err

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
script_path="${repo_root}/data_processing/climate_low_latency/update_low_latency_climate_range.py"

mkdir -p "${repo_root}/logs/low_latency_parallel"
source "${process_env}"

cmd=(
    python3 "${script_path}"
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --registry_path "${REGISTRY_PATH}"
)

echo "Running low-latency climate worker"
echo "  start_date=${START_DATE}"
echo "  end_date=${END_DATE}"
"${cmd[@]}"
