#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/run_site_pixel_grids_mitch_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/run_site_pixel_grids_mitch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

site_csv="${SITE_CSV:-/scratch/users/trobinet/long_lfmc/final_lfmc/inference/sites_for_mitch/valid_psinet_sites.csv}"
output_dir="${OUTPUT_DIR:-/scratch/users/trobinet/long_lfmc/final_lfmc/inference/sites_for_mitch/pixel_grids_multisource_fusion_clim20}"
forward_batch_size="${FORWARD_BATCH_SIZE:-8192}"
max_plot_years="${MAX_PLOT_YEARS:-2}"
use_full_period="${USE_FULL_PERIOD:-true}"
extra_args="${EXTRA_ARGS:-}"

cmd=(
    python3 -u
    "${script_dir}/run_site_pixel_grids.py"
    --site_csv "${site_csv}"
    --output_dir "${output_dir}"
    --forward_batch_size "${forward_batch_size}"
    --max_plot_years "${max_plot_years}"
)

if [[ "${use_full_period}" == "true" ]]; then
    cmd+=(--max_period_2001_2024)
fi

if [[ -n "${extra_args}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${extra_args})
    cmd+=("${extra_args_array[@]}")
fi

echo "Running site pixel grids for Mitch"
echo "site_csv=${site_csv}"
echo "output_dir=${output_dir}"
echo "forward_batch_size=${forward_batch_size}"
echo "max_plot_years=${max_plot_years}"
echo "use_full_period=${use_full_period}"
echo "extra_args=${extra_args}"
echo "command=${cmd[*]}"

"${cmd[@]}"
