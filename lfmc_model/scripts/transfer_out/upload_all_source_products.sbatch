#!/usr/bin/env bash
#SBATCH --job-name=source_transfer
#SBATCH --partition=serc
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out/logs/source_transfer_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out/logs/source_transfer_%j.err

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"

mkdir -p "${script_dir}/logs"
cd "${script_dir}"

bash "${script_dir}/run_upload_all_source_products_clean.sh"
