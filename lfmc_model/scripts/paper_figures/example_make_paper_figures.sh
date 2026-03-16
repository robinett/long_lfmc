#!/bin/bash

config_path="/home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures/paper_figure_configs.yaml"

source ~/uv_activations/activate_lfmc_model_py312.sh

python3 /home/users/trobinet/long_lfmc/lfmc_model/scripts/paper_figures/make_paper_figures.py \
    --config "$config_path"
