#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /abs/path/to/run_root" >&2
    exit 1
fi

run_root="$1"
script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"

if [[ ! -d "${run_root}" ]]; then
    echo "Run root not found: ${run_root}" >&2
    exit 1
fi

cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

echo "Re-running validation for ${run_root}"
python3 -u "${script_dir}/validate_map_outputs.py" \
    --run_root "${run_root}"
