#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source"
config_path="${script_dir}/rao_s1_source_config.yaml"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
logs_dir="${script_dir}/logs"

mkdir -p "${logs_dir}"
cd "${script_dir}"
source "${env_path}"

python3 -u "${script_dir}/build_rao_s1_viewer_climatology.py" \
    --config "${config_path}" \
    --mode init

block_count="$(python3 - "${config_path}" <<'PY'
import json
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
    cfg = yaml.safe_load(file_obj)
plan_path = cfg["climatology"]["state_dir"].rstrip("/") + "/rao_s1_viewer_climatology_plan.json"
with open(plan_path, "r", encoding="utf-8") as file_obj:
    print(json.load(file_obj)["block_count"])
PY
)"
max_concurrent="$(python3 - "${config_path}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
    cfg = yaml.safe_load(file_obj)
print(int(cfg["climatology"].get("max_concurrent_tasks", 24)))
PY
)"

if [[ "${block_count}" -lt 1 ]]; then
    echo "No climatology blocks selected; nothing to submit."
    exit 0
fi

last_index=$((block_count - 1))
array_spec="0-${last_index}%${max_concurrent}"
worker_job_id="$(sbatch --parsable \
    --job-name=rao_s1_climatology \
    --partition=serc \
    --time=12:00:00 \
    --cpus-per-task=1 \
    --mem=24G \
    --array="${array_spec}" \
    --output="${logs_dir}/rao_s1_climatology_%A_%a.out" \
    --error="${logs_dir}/rao_s1_climatology_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_rao_s1_viewer_climatology.py --config ${config_path} --mode worker --use-slurm-array")"

finalize_job_id="$(sbatch --parsable \
    --job-name=rao_s1_clim_final \
    --partition=serc \
    --time=2:00:00 \
    --cpus-per-task=1 \
    --mem=16G \
    --dependency="afterok:${worker_job_id}" \
    --output="${logs_dir}/rao_s1_climatology_finalize_%j.out" \
    --error="${logs_dir}/rao_s1_climatology_finalize_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_rao_s1_viewer_climatology.py --config ${config_path} --mode finalize")"

echo "Submitted Rao S1 climatology array ${worker_job_id} (${array_spec})"
echo "Submitted Rao S1 climatology finalize job ${finalize_job_id}"
echo "Logs: ${logs_dir}/rao_s1_climatology_${worker_job_id}_*.out"
