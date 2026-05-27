#!/usr/bin/env bash
#SBATCH --job-name=viewer_t32_source_transfer
#SBATCH --partition=serc
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/source_transfer/viewer_t32_source_transfer_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/source_transfer/viewer_t32_source_transfer_%j.err

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
log_dir="/home/users/trobinet/long_lfmc/logs/source_transfer"

mkdir -p "${log_dir}"
cd "${script_dir}"

bash "${script_dir}/run_upload_viewer_3857_dataset_t32_fresh.sh"
