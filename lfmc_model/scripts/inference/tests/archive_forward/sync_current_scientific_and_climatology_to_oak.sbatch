#!/bin/bash

#SBATCH --job-name=sync_science_oak
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/archive_forward_setup_20260417_111322/sync_current_scientific_and_climatology_to_oak_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/archive_forward_setup_20260417_111322/sync_current_scientific_and_climatology_to_oak_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

scratch_root="/scratch/users/trobinet/long_lfmc/final_lfmc"
oak_root="/oak/stanford/groups/konings/trobinet/long_lfmc/final_lfmc"

scientific_src="${scratch_root}/lfmc_model/inference/final_products/lfmc_vh_vv_365_multisource_fusion_clim20_2001_2024.zarr"
scientific_dst="${oak_root}/lfmc_model/inference/final_products/lfmc_vh_vv_365_multisource_fusion_clim20_2001_2024.zarr"
clim_src="${scratch_root}/daymet/daymet_vars_and_anoms_clim20_climatology.zarr"
clim_dst="${oak_root}/daymet/daymet_vars_and_anoms_clim20_climatology.zarr"

sync_path() {
    local src_path="$1"
    local dst_path="$2"
    if [[ -d "${src_path}" ]]; then
        mkdir -p "${dst_path}"
        rsync -a "${src_path}/" "${dst_path}/"
    else
        mkdir -p "$(dirname "${dst_path}")"
        rsync -a "${src_path}" "${dst_path}"
    fi
}

sync_path "${scientific_src}" "${scientific_dst}"
sync_path "${clim_src}" "${clim_dst}"
