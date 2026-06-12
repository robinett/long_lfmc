#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/source_coop_transfer_configs.yaml"
product_prefix="rseg/sentinel1-lfmc/"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cd "${script_dir}"
source "${env_path}"

upload_dataset() {
    local dataset_key="$1"
    if [[ "${dry_run}" -eq 1 ]]; then
        python3 "${script_dir}/upload_source_coop.py" \
            --config_path "${config_path}" \
            --dataset_key "${dataset_key}" \
            --product_prefix "${product_prefix}" \
            --delete_extra_remote_files \
            --dry_run \
            --skip_verify
    else
        python3 "${script_dir}/upload_source_coop.py" \
            --config_path "${config_path}" \
            --dataset_key "${dataset_key}" \
            --product_prefix "${product_prefix}" \
            --delete_extra_remote_files
    fi
}

log "Uploading Rao S1 scientific LFMC zarr to Source"
upload_dataset rao_s1_scientific_lfmc_maps

log "Uploading Rao S1 viewer 3857 LFMC zarr to Source"
upload_dataset rao_s1_viewer_3857_lfmc_maps

log "Uploading Rao S1 viewer assets to Source"
upload_dataset rao_s1_viewer_3857_assets

if [[ "${dry_run}" -eq 1 ]]; then
    log "Dry run complete; skipping remote verification"
    exit 0
fi

log "Verifying Rao S1 Source products"
python3 "${script_dir}/verify_remote_rao_s1_source_products.py" \
    --config_path "${config_path}" \
    --product_prefix "${product_prefix}"

log "Finished uploading Rao S1 Source products"
