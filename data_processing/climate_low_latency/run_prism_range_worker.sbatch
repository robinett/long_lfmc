#!/bin/bash
#SBATCH --job-name=prism_range
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/users/trobinet/long_lfmc/logs/low_latency_parallel/prism_range_%j.out
#SBATCH -e /home/users/trobinet/long_lfmc/logs/low_latency_parallel/prism_range_%j.err

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
script_path="${repo_root}/data_processing/prism/update_prism_range.py"

mkdir -p "${repo_root}/logs/low_latency_parallel"
source "${process_env}"

cmd=(
    python3 "${script_path}"
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
    --registry_path "${REGISTRY_PATH}"
    --grid_path "${GRID_PATH}"
    --raw_root "${RAW_ROOT}"
    --extracted_root "${EXTRACTED_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --plots_dir "${PLOTS_DIR}"
    --release_latency_days "${RELEASE_LATENCY_DAYS}"
)

if [[ "${OVERWRITE_EXISTING:-0}" == "1" ]]; then
    cmd+=(--overwrite_existing)
fi

echo "Running PRISM range worker"
echo "  start_date=${START_DATE}"
echo "  end_date=${END_DATE}"
echo "  output_root=${OUTPUT_ROOT}"
"${cmd[@]}"
