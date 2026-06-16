#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/source_coop_transfer_configs.yaml"
product_prefix="rseg/sentinel1-lfmc/"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
dry_run=0
target_date=""
manifest_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --target-date)
            target_date="$2"
            shift 2
            ;;
        --manifest-dir)
            manifest_dir="$2"
            shift 2
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

if [[ -n "${manifest_dir}" ]]; then
    manifest_dir="$(cd "${manifest_dir}" && pwd)"
fi

cd "${script_dir}"
source "${env_path}"

upload_dataset() {
    local dataset_key="$1"
    local manifest_path=""
    local transfer_args=()
    if [[ -n "${manifest_dir}" ]]; then
        manifest_path="${manifest_dir}/${dataset_key}.txt"
        if [[ ! -f "${manifest_path}" ]]; then
            echo "[ERROR] Missing manifest for ${dataset_key}: ${manifest_path}" >&2
            exit 1
        fi
        transfer_args=(--transfer_mode manifest --manifest_path "${manifest_path}" --verify_mode sample)
    else
        transfer_args=(--delete_extra_remote_files)
    fi
    if [[ "${dry_run}" -eq 1 ]]; then
        python3 "${script_dir}/upload_source_coop.py" \
            --config_path "${config_path}" \
            --dataset_key "${dataset_key}" \
            --product_prefix "${product_prefix}" \
            "${transfer_args[@]}" \
            --dry_run \
            --skip_verify
    else
        python3 "${script_dir}/upload_source_coop.py" \
            --config_path "${config_path}" \
            --dataset_key "${dataset_key}" \
            --product_prefix "${product_prefix}" \
            "${transfer_args[@]}"
    fi
}

if [[ -n "${manifest_dir}" ]]; then
    log "Using targeted manifest uploads from ${manifest_dir}"
    if [[ -n "${target_date}" ]]; then
        log "Target date: ${target_date}"
    fi
fi

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
