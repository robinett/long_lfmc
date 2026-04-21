#!/bin/bash

#SBATCH --job-name=daymet_split
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/daymet_parallel/daymet_to_daily_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/daymet_parallel/daymet_to_daily_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

source /home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh

repo_root="/home/users/trobinet/long_lfmc"
input_root="${INPUT_ROOT:?INPUT_ROOT must be set}"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
year="${YEAR:?YEAR must be set}"
var_name="${VAR:?VAR must be set}"
skip_existing="${SKIP_EXISTING:-1}"
progress_every="${PROGRESS_EVERY:-10}"

echo "Running Daymet to_daily worker"
echo "  year=${year}"
echo "  var=${var_name}"
echo "  input_root=${input_root}"
echo "  output_root=${output_root}"

cmd=(
    python3
    "${repo_root}/data_processing/daymet/to_daily.py"
    --input_root "${input_root}"
    --output_root "${output_root}"
    --year "${year}"
    --var "${var_name}"
    --progress_every "${progress_every}"
    --no-plot_first_day
)

if [[ "${skip_existing}" == "1" ]]; then
    cmd+=(--skip_existing)
fi

printf '  cmd='
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "Completed Daymet to_daily worker for var=${var_name} year=${year}"
