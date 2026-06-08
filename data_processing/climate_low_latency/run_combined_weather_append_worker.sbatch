#!/bin/bash
#SBATCH --job-name=combined_weather
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o /home/users/trobinet/long_lfmc/logs/low_latency_parallel/combined_weather_%j.out
#SBATCH -e /home/users/trobinet/long_lfmc/logs/low_latency_parallel/combined_weather_%j.err

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
script_path="${repo_root}/data_processing/climate_low_latency/append_low_latency_range_from_saved_climatology.py"

mkdir -p "${repo_root}/logs/low_latency_parallel"
source "${process_env}"

cmd=(
    python3 "${script_path}"
    --standard_zarr "${STANDARD_ZARR}"
    --combined_zarr "${COMBINED_ZARR}"
    --climatology_zarr "${CLIMATOLOGY_ZARR}"
    --start_date "${START_DATE}"
    --end_date "${END_DATE}"
)

echo "Running combined weather append worker"
echo "  start_date=${START_DATE}"
echo "  end_date=${END_DATE}"
echo "  standard_zarr=${STANDARD_ZARR}"
echo "  combined_zarr=${COMBINED_ZARR}"
"${cmd[@]}"
