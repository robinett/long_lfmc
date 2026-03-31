#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p logs

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

python3 -u "${script_dir}/submit_multitask_multisource_fusion_sweep.py" "$@"
