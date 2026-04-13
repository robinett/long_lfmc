#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/logs/build_viewer_assets_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/logs/build_viewer_assets_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
logs_dir="${script_dir}/logs"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

echo "Building viewer 3857 assets"
echo "build_config=${script_dir}/viewer_build_config.yaml"

python3 -u "${script_dir}/build_viewer_assets.py"
