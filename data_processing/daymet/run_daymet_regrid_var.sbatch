#!/bin/bash

#SBATCH --job-name=daymet_regrid
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/daymet_parallel/daymet_regrid_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/daymet_parallel/daymet_regrid_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

source /home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh

repo_root="/home/users/trobinet/long_lfmc"
year="${YEAR:?YEAR must be set}"
var_name="${VAR:?VAR must be set}"
grid_path="${GRID_PATH:?GRID_PATH must be set}"
src_dir="${SRC_DIR:?SRC_DIR must be set}"
target_dir="${TARGET_DIR:?TARGET_DIR must be set}"
chunk_size="${CHUNK_SIZE:?CHUNK_SIZE must be set}"
chunk_buffer="${CHUNK_BUFFER:?CHUNK_BUFFER must be set}"
tmp_root="${TMP_FILTER_ROOT:-/tmp/daymet_regrid_filters}"
src_crs="${SRC_CRS:-+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs}"
target_crs="${TARGET_CRS:-EPSG:5070}"

tmp_dir="${tmp_root}/year_${year}_${var_name}_${SLURM_JOB_ID}"
mkdir -p "${tmp_dir}"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT INT TERM

echo "Running Daymet regrid worker"
echo "  year=${year}"
echo "  var=${var_name}"
echo "  src_dir=${src_dir}"
echo "  target_dir=${target_dir}"
echo "  tmp_dir=${tmp_dir}"

shopt -s nullglob
matches=("${src_dir}"/daymet_v4_daily_na_"${var_name}"_*.nc)
shopt -u nullglob
if [[ ${#matches[@]} -eq 0 ]]; then
    echo "No source files found for var=${var_name} year=${year} in ${src_dir}" >&2
    exit 1
fi

for src_path in "${matches[@]}"; do
    ln -sf "${src_path}" "${tmp_dir}/$(basename "${src_path}")"
done

cmd=(
    python3
    "${repo_root}/data_processing/regrid/main.py"
    --target_grid "${grid_path}"
    --src_dir "${tmp_dir}"
    --target_dir "${target_dir}"
    --src_crs "${src_crs}"
    --target_crs "${target_crs}"
    --chunk_size "${chunk_size}"
    --chunk_buffer "${chunk_buffer}"
)

printf '  cmd='
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "Completed Daymet regrid worker for var=${var_name} year=${year}"
