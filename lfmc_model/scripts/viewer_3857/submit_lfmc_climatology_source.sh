#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
config_path="${script_dir}/viewer_pipeline_config.yaml"
logs_dir="${script_dir}/logs"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source "${env_path}"

echo "Initializing source-grid LFMC climatology plan"
python3 -u "${script_dir}/build_lfmc_climatology_source.py" \
    --config "${config_path}" \
    --mode init

block_count="$(python3 -c "import json, yaml; cfg=yaml.safe_load(open('${config_path}')); plan=cfg['climatology']['source_state_dir'] + '/source_climatology_plan.json'; print(json.load(open(plan))['block_count'])")"
max_concurrent="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_path}')); print(int(cfg['climatology'].get('max_concurrent_tasks', 64)))")"

if [[ "${block_count}" -lt 1 ]]; then
    echo "No climatology source blocks selected; nothing to submit."
    exit 0
fi

last_index=$((block_count - 1))
array_spec="0-${last_index}%${max_concurrent}"

climatology_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=06:00:00 \
    --cpus-per-task=2 \
    --mem=32G \
    --array="${array_spec}" \
    --output="${logs_dir}/lfmc_climatology_source_%A_%a.out" \
    --error="${logs_dir}/lfmc_climatology_source_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_lfmc_climatology_source.py --config ${config_path} --mode worker --use-slurm-array")"

echo "Submitted source-grid climatology array ${climatology_job_id} (${array_spec})"
echo "Logs: ${logs_dir}/lfmc_climatology_source_${climatology_job_id}_*.out"
echo "Status command:"
echo "  cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_lfmc_climatology_source.py --config ${config_path} --mode status"
