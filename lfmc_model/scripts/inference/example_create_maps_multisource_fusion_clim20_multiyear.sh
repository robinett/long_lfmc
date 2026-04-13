#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_multisource_fusion_clim20_multiyear.yaml"

cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

start_year="${START_YEAR:-}"
end_year="${END_YEAR:-}"
manifest_only="${MANIFEST_ONLY:-0}"
resume_mode="${RESUME_MODE:-1}"

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

if [[ "${resume_mode}" == "0" ]]; then
    cmd+=(--no-resume)
fi

echo "Running multisource-fusion clim20 multiyear map workflow"
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
if [[ "${resume_mode}" == "0" ]]; then
    echo "resume=0"
else
    echo "resume=1"
fi

"${cmd[@]}"
