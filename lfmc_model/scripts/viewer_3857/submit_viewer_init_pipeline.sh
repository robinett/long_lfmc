#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
config_path="${script_dir}/viewer_pipeline_config.yaml"
logs_dir="${script_dir}/logs"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source "${env_path}"

echo "Detecting full viewer initialization date plan"
python3 -u "${script_dir}/detect_viewer_updates.py" \
    --config "${config_path}" \
    --mode init

block_count="$(python3 -c "import json, yaml; from pathlib import Path; cfg=yaml.safe_load(open('${config_path}')); plan=Path(cfg['output']['state_dir'])/'viewer_update_plan.json'; print(json.load(open(plan))['block_count'])")"

if [[ "${block_count}" -lt 1 ]]; then
    echo "No viewer dates selected; nothing to submit."
    exit 0
fi

last_index=$((block_count - 1))
array_spec="0-${last_index}"

init_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --output="${logs_dir}/viewer_init_%j.out" \
    --error="${logs_dir}/viewer_init_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/init_viewer_dataset_3857_pipeline.py --config ${config_path} --mode rebuild")"

dataset_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --array="${array_spec}" \
    --dependency="afterok:${init_job_id}" \
    --output="${logs_dir}/viewer_dataset_%A_%a.out" \
    --error="${logs_dir}/viewer_dataset_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/run_viewer_dataset_dates.py --config ${config_path} --use-slurm-array")"

tiles_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=04:00:00 \
    --cpus-per-task=8 \
    --mem=96G \
    --array="${array_spec}" \
    --dependency="afterok:${dataset_job_id}" \
    --output="${logs_dir}/viewer_tiles_%A_%a.out" \
    --error="${logs_dir}/viewer_tiles_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/run_viewer_tile_dates.py --config ${config_path} --use-slurm-array")"

finalize_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --dependency="afterok:${tiles_job_id}" \
    --output="${logs_dir}/viewer_manifest_%j.out" \
    --error="${logs_dir}/viewer_manifest_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/finalize_viewer_manifest.py --config ${config_path} --require-all-dates")"

echo "Submitted viewer init job ${init_job_id}"
echo "Submitted viewer dataset array ${dataset_job_id} (${array_spec})"
echo "Submitted viewer tile array ${tiles_job_id} (${array_spec})"
echo "Submitted viewer manifest finalize job ${finalize_job_id}"
