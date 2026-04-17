#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/verify_maps_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/verify_maps_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

for required_var in MANIFEST_PATH VERIFY_KIND; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "${required_var} must be set before submitting ${BASH_SOURCE[0]}" >&2
        exit 1
    fi
done

verify_args=(
    --manifest_path "${MANIFEST_PATH}"
    --kind "${VERIFY_KIND}"
)

if [[ -n "${STORE_PATH:-}" ]]; then
    verify_args+=(--store_path "${STORE_PATH}")
fi
if [[ -n "${OUTPUT_PATH:-}" ]]; then
    verify_args+=(--output_path "${OUTPUT_PATH}")
fi
if [[ -n "${SAMPLE_N:-}" ]]; then
    verify_args+=(--sample_n "${SAMPLE_N}")
fi
if [[ -n "${METADATA_DIR:-}" ]]; then
    verify_args+=(--metadata_dir "${METADATA_DIR}")
fi
if [[ -n "${STAGING_ZARR:-}" ]]; then
    verify_args+=(--staging_zarr "${STAGING_ZARR}")
fi
if [[ -n "${PRODUCTION_ZARR:-}" ]]; then
    verify_args+=(--production_zarr "${PRODUCTION_ZARR}")
fi
if [[ -n "${START_DATE:-}" ]]; then
    verify_args+=(--start_date "${START_DATE}")
fi
if [[ -n "${END_DATE:-}" ]]; then
    verify_args+=(--end_date "${END_DATE}")
fi

echo "[verify_maps] manifest_path=${MANIFEST_PATH}"
echo "[verify_maps] kind=${VERIFY_KIND}"

python3 -u "${script_dir}/verify_map_store.py" "${verify_args[@]}"
