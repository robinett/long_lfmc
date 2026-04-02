#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures"
config_path="${script_dir}/paper_figure_configs_new.yaml"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u "${script_dir}/make_paper_figures_new.py" \
    --config "${config_path}"
