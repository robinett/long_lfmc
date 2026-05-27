#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/viewer_3857_rechunk_t32/rechunk_viewer_dataset_t32_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/viewer_3857_rechunk_t32/rechunk_viewer_dataset_t32_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

repo_root="/home/users/trobinet/long_lfmc"
script_dir="${repo_root}/lfmc_model/scripts/viewer_3857"
logs_dir="${repo_root}/logs/viewer_3857_rechunk_t32"
config_path="${script_dir}/viewer_pipeline_config.yaml"
source_zarr="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_3857/lfmc_maps_3857.zarr"
destination_zarr="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_3857/lfmc_maps_3857_rechunk_t32.zarr"
time_chunk_size=32
spatial_chunk_size=256
workers=32

mkdir -p "${logs_dir}"
cd "${repo_root}"

source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

echo "Starting viewer 3857 rechunk"
echo "config_path=${config_path}"
echo "source_zarr=${source_zarr}"
echo "destination_zarr=${destination_zarr}"
echo "time_chunk_size=${time_chunk_size}"
echo "spatial_chunk_size=${spatial_chunk_size}"
echo "workers=${workers}"

python3 -u "${script_dir}/rechunk_viewer_dataset.py" \
    --config "${config_path}" \
    --source "${source_zarr}" \
    --destination "${destination_zarr}" \
    --time-chunk-size "${time_chunk_size}" \
    --spatial-chunk-size "${spatial_chunk_size}" \
    --workers "${workers}" \
    --overwrite
