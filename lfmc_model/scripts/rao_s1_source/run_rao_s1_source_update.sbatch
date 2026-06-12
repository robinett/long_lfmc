#!/usr/bin/env bash
#SBATCH --job-name=rao_s1_source_update
#SBATCH --partition=serc
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source/logs/rao_s1_source_update_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source/logs/rao_s1_source_update_%j.err

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source"

mkdir -p "${script_dir}/logs"
cd "${script_dir}"

bash "${script_dir}/run_rao_s1_source_update.sh" "$@"
