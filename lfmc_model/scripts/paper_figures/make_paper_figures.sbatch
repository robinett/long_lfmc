#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures/logs/make_paper_figures_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures/logs/make_paper_figures_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures"
config_path="${script_dir}/paper_figure_configs.yaml"
logs_dir="${script_dir}/logs"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source ~/uv_activations/activate_lfmc_model_py312.sh

python3 -u "${script_dir}/make_paper_figures.py" \
    --config "${config_path}"
