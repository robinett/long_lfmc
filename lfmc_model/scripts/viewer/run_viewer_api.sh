#!/usr/bin/env bash

set -eo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer"
cd "${script_dir}"

source ~/.bashrc || true
set -u
source /home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh

python3 api/serve_viewer_api.py
