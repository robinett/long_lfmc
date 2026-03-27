#!/usr/bin/env bash

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_shared_dm128_multiyear.yaml"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"

source ~/.bashrc
source "${model_env}"

set -euo pipefail

manifest_only="${MANIFEST_ONLY:-0}"
resume_runs="${RESUME_RUNS:-1}"
start_year="${START_YEAR:-}"
end_year="${END_YEAR:-}"

cmd=(
    python3 "${script_dir}/create_maps_multiyear.py"
    --config_path "${config_path}"
)

if [[ -n "${start_year}" ]]; then
    cmd+=(--start_year "${start_year}")
fi

if [[ -n "${end_year}" ]]; then
    cmd+=(--end_year "${end_year}")
fi

if [[ "${manifest_only}" == "1" ]]; then
    cmd+=(--manifest_only)
fi

if [[ "${resume_runs}" == "0" ]]; then
    cmd+=(--no-resume)
fi

echo "Running shared dm128 multiyear map workflow"
echo "config_path=${config_path}"
if [[ -n "${start_year}" ]]; then
    echo "start_year=${start_year}"
fi
if [[ -n "${end_year}" ]]; then
    echo "end_year=${end_year}"
fi
if [[ "${manifest_only}" == "1" ]]; then
    echo "manifest_only=1"
fi
if [[ "${resume_runs}" == "0" ]]; then
    echo "resume_runs=0"
fi

"${cmd[@]}"
