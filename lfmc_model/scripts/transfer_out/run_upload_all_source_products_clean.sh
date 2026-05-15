#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/source_coop_transfer_configs.yaml"
api_refresh_url="${LONG_LFMC_API_REFRESH_URL:-https://long-lfmc-api.onrender.com/api/refresh}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cd "${script_dir}"

log "Preparing scientific LFMC dataset for Source upload"
source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 "${script_dir}/prepare_scientific_lfmc_dataset_for_source.py"

log "Uploading scientific LFMC dataset to Source with remote cleanup enabled"
python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key scientific_lfmc_maps \
    --delete_extra_remote_files

log "Verifying remote scientific LFMC dataset"
source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh
python3 "${script_dir}/verify_remote_scientific_lfmc_dataset.py"

log "Preparing viewer 3857 LFMC dataset for Source upload"
source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

python3 "${script_dir}/prepare_viewer_3857_dataset_for_source.py"

log "Uploading viewer 3857 LFMC dataset to Source with remote cleanup enabled"
python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key viewer_3857_lfmc_maps \
    --delete_extra_remote_files

log "Verifying remote viewer 3857 LFMC dataset"
python3 "${script_dir}/verify_remote_viewer_dataset_3857.py"

log "Uploading viewer 3857 assets to Source with remote cleanup enabled"
python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key viewer_3857_assets \
    --delete_extra_remote_files

if [[ -n "${LONG_LFMC_API_REFRESH_TOKEN:-}" ]]; then
    log "Refreshing deployed viewer API after Source upload"
    curl -fsS \
        -X POST \
        -H "Authorization: Bearer ${LONG_LFMC_API_REFRESH_TOKEN}" \
        "${api_refresh_url}"
    printf '\n'
else
    log "Skipping deployed viewer API refresh because LONG_LFMC_API_REFRESH_TOKEN is not set"
fi

log "Finished uploading all Source products"
