#!/usr/bin/env bash

set -euo pipefail

repo_dir="/home/users/trobinet/long_lfmc"
script_dir="${repo_dir}/lfmc_model/scripts/viewer_3857"
transfer_script="${repo_dir}/lfmc_model/scripts/transfer_out/upload_source_coop.py"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh"

workflow_name="viewer_alpha96_full_tiles"
logs_dir="${repo_dir}/logs/${workflow_name}"
base_config="${script_dir}/viewer_pipeline_config.yaml"
render_config="${logs_dir}/viewer_alpha96_full_config.yaml"
plan_path="${logs_dir}/viewer_alpha96_full_plan.json"

viewer_dataset_path="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_3857/lfmc_maps_3857_rechunk_t32.zarr"
asset_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_assets/web_mercator_3857"
state_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_3857/state_alpha96_full_tiles"

evergreen_alpha=96
block_days=16
tile_array_concurrency=24

mkdir -p "${logs_dir}"
cd "${script_dir}"

source "${env_path}"

echo "Preparing full MODIS tile refresh"
echo "  viewer_dataset_path=${viewer_dataset_path}"
echo "  asset_root=${asset_root}"
echo "  state_dir=${state_dir}"
echo "  evergreen_alpha=${evergreen_alpha}"
echo "  block_days=${block_days}"

python3 -u "${script_dir}/prepare_viewer_tile_refresh.py" \
    --base-config "${base_config}" \
    --output-config "${render_config}" \
    --plan-path "${plan_path}" \
    --viewer-dataset-path "${viewer_dataset_path}" \
    --asset-root "${asset_root}" \
    --state-dir "${state_dir}" \
    --evergreen-alpha "${evergreen_alpha}" \
    --block-days "${block_days}"

block_count="$(python3 -c "import json; print(json.load(open('${plan_path}'))['block_count'])")"
date_count="$(python3 -c "import json; print(json.load(open('${plan_path}'))['date_count'])")"

if [[ "${block_count}" -lt 1 ]]; then
    echo "No dates selected; nothing to submit."
    exit 0
fi

last_index=$((block_count - 1))
array_spec="0-${last_index}%${tile_array_concurrency}"

tiles_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=06:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --array="${array_spec}" \
    --output="${logs_dir}/viewer_alpha96_tiles_%A_%a.out" \
    --error="${logs_dir}/viewer_alpha96_tiles_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/run_viewer_tile_dates.py --config ${render_config} --plan-path ${plan_path} --use-slurm-array")"

finalize_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --dependency="afterok:${tiles_job_id}" \
    --output="${logs_dir}/viewer_alpha96_manifest_%j.out" \
    --error="${logs_dir}/viewer_alpha96_manifest_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/finalize_viewer_manifest.py --config ${render_config} --require-all-dates")"

upload_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=48:00:00 \
    --cpus-per-task=8 \
    --mem=128G \
    --dependency="afterok:${finalize_job_id}" \
    --output="${logs_dir}/viewer_alpha96_source_upload_%j.out" \
    --error="${logs_dir}/viewer_alpha96_source_upload_%j.err" \
    --wrap="cd ${repo_dir}; source ${env_path}; python3 -u ${transfer_script} --dataset_key viewer_3857_assets --source_path ${asset_root} --destination_relpath viewer_3857/assets/web_mercator_3857 --transfer_mode fresh-prefix --verify_mode sample --sample_verify_count 128")"

job_summary="${logs_dir}/viewer_alpha96_full_jobs.json"
python3 -c "import json; from pathlib import Path; Path('${job_summary}').write_text(json.dumps({'date_count': int('${date_count}'), 'block_count': int('${block_count}'), 'array_spec': '${array_spec}', 'tiles_job_id': '${tiles_job_id}', 'finalize_job_id': '${finalize_job_id}', 'upload_job_id': '${upload_job_id}', 'render_config': '${render_config}', 'plan_path': '${plan_path}', 'asset_root': '${asset_root}', 'state_dir': '${state_dir}'}, indent=2, sort_keys=True) + '\n')"

echo "Prepared ${date_count} dates in ${block_count} tile blocks"
echo "Submitted tile array ${tiles_job_id} (${array_spec})"
echo "Submitted manifest finalize job ${finalize_job_id}"
echo "Submitted Source upload job ${upload_job_id}"
echo "Job summary: ${job_summary}"
echo "Logs: ${logs_dir}"
