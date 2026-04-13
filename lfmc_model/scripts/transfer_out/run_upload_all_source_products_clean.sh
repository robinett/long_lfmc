#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/source_coop_transfer_configs.yaml"
scientific_source_path="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/final_products/lfmc_vh_vv_365_multisource_fusion_clim20_2023_07_08.zarr"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cd "${script_dir}"

log "Preparing scientific LFMC dataset for Source upload"
source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -c "import zarr; zarr.consolidate_metadata('${scientific_source_path}')"

log "Uploading scientific LFMC dataset to Source with remote cleanup enabled"
python3 "${script_dir}/upload_source_coop.py" \
    --config_path "${config_path}" \
    --dataset_key scientific_lfmc_maps \
    --source_path "${scientific_source_path}" \
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

log "Finished uploading all Source products"
