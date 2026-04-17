#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/promote_maps_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/promote_maps_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

for required_var in MANIFEST_PATH STAGING_ZARR PRODUCTION_ZARR METADATA_DIR START_DATE END_DATE PROMOTION_MODE PRODUCT_TIER; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "${required_var} must be set before submitting ${BASH_SOURCE[0]}" >&2
        exit 1
    fi
done

initialize_flag=()
if [[ "${INITIALIZE_IF_MISSING:-0}" == "1" ]]; then
    initialize_flag+=(--initialize_if_missing)
fi

echo "[promote_maps] staging_zarr=${STAGING_ZARR}"
echo "[promote_maps] production_zarr=${PRODUCTION_ZARR}"
echo "[promote_maps] start_date=${START_DATE} end_date=${END_DATE}"
echo "[promote_maps] mode=${PROMOTION_MODE} tier=${PRODUCT_TIER}"

python3 -u "${script_dir}/promote_map_output.py" \
    --staging_zarr "${STAGING_ZARR}" \
    --production_zarr "${PRODUCTION_ZARR}" \
    --metadata_dir "${METADATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --mode "${PROMOTION_MODE}" \
    --tier "${PRODUCT_TIER}" \
    "${initialize_flag[@]}"

echo "[promote_maps] verifying promoted production store"
python3 -u "${script_dir}/verify_map_store.py" \
    --manifest_path "${MANIFEST_PATH}" \
    --kind production \
    --store_path "${PRODUCTION_ZARR}" \
    --metadata_dir "${METADATA_DIR}" \
    --staging_zarr "${STAGING_ZARR}" \
    --production_zarr "${PRODUCTION_ZARR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}"
