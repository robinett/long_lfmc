#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
config_path="${script_dir}/viewer_pipeline_config.yaml"
logs_dir="${script_dir}/logs"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source "${env_path}"

echo "Detecting viewer append/update dates from source dates and quality flags"
python3 -u "${script_dir}/detect_viewer_updates.py" \
    --config "${config_path}" \
    --mode append

block_count="$(python3 -c "import json, yaml; from pathlib import Path; cfg=yaml.safe_load(open('${config_path}')); plan=Path(cfg['output']['state_dir'])/'viewer_update_plan.json'; print(json.load(open(plan))['block_count'])")"
date_count="$(python3 -c "import json, yaml; from pathlib import Path; cfg=yaml.safe_load(open('${config_path}')); plan=Path(cfg['output']['state_dir'])/'viewer_update_plan.json'; print(json.load(open(plan))['date_count'])")"

if [[ "${block_count}" -lt 1 ]]; then
    echo "No new dates or changed quality flags detected; no viewer append work submitted."
    exit 0
fi

last_index=$((block_count - 1))
array_spec="0-${last_index}"

init_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --output="${logs_dir}/viewer_append_init_%j.out" \
    --error="${logs_dir}/viewer_append_init_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/init_viewer_dataset_3857_pipeline.py --config ${config_path} --mode append")"

dataset_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --array="${array_spec}" \
    --dependency="afterok:${init_job_id}" \
    --output="${logs_dir}/viewer_append_dataset_%A_%a.out" \
    --error="${logs_dir}/viewer_append_dataset_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/run_viewer_dataset_dates.py --config ${config_path} --use-slurm-array")"

tiles_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=04:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --array="${array_spec}" \
    --dependency="afterok:${dataset_job_id}" \
    --output="${logs_dir}/viewer_append_tiles_%A_%a.out" \
    --error="${logs_dir}/viewer_append_tiles_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/run_viewer_tile_dates.py --config ${config_path} --use-slurm-array")"

finalize_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --dependency="afterok:${tiles_job_id}" \
    --output="${logs_dir}/viewer_append_manifest_%j.out" \
    --error="${logs_dir}/viewer_append_manifest_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/finalize_viewer_manifest.py --config ${config_path} --require-all-dates")"

echo "Detected ${date_count} dates in ${block_count} date blocks"
echo "Submitted viewer append init job ${init_job_id}"
echo "Submitted viewer append dataset array ${dataset_job_id} (${array_spec})"
echo "Submitted viewer append tile array ${tiles_job_id} (${array_spec})"
echo "Submitted viewer append manifest finalize job ${finalize_job_id}"
