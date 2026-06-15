#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source"
config_path="${script_dir}/rao_s1_source_config.yaml"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
logs_dir="${script_dir}/logs"

mkdir -p "${logs_dir}"
cd "${script_dir}"
source "${env_path}"

plan_path="$(python3 - "${config_path}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
    cfg = yaml.safe_load(file_obj)
print(cfg["climatology"]["state_dir"].rstrip("/") + "/rao_s1_anomaly_tile_backfill_plan.json")
PY
)"

python3 - "${config_path}" "${plan_path}" <<'PY'
import json
import math
import sys
import yaml
import zarr
import numpy as np
from pathlib import Path

config_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
root = zarr.open_group(str(cfg["paths"]["viewer_zarr_path"]), mode="r")
dates = [np.datetime_as_string(value, unit="D") for value in np.asarray(root["time"][:]).astype("datetime64[D]")]
block_days = 8
blocks = []
for start in range(0, len(dates), block_days):
    block_dates = dates[start:start + block_days]
    blocks.append({"block_index": len(blocks), "dates": block_dates})
plan_path.parent.mkdir(parents=True, exist_ok=True)
plan_path.write_text(json.dumps({
    "status": "initialized",
    "mode": "rao_s1_anomaly_tile_backfill",
    "date_count": len(dates),
    "block_days": block_days,
    "block_count": len(blocks),
    "blocks": blocks,
}, indent=2, sort_keys=True), encoding="utf-8")
print(plan_path)
print(len(blocks))
PY

block_count="$(python3 - "${plan_path}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], "r", encoding="utf-8"))["block_count"])
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

last_index=$((block_count - 1))
array_spec="0-${last_index}%${max_concurrent}"
tiles_job_id="$(sbatch --parsable \
    --job-name=rao_s1_anom_tiles \
    --partition=serc \
    --time=8:00:00 \
    --cpus-per-task=1 \
    --mem=32G \
    --array="${array_spec}" \
    --output="${logs_dir}/rao_s1_anomaly_tiles_%A_%a.out" \
    --error="${logs_dir}/rao_s1_anomaly_tiles_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_rao_s1_viewer_tiles.py --config ${config_path} --plan-path ${plan_path} --use-slurm-array --layers anomaly --skip-manifest")"

manifest_job_id="$(sbatch --parsable \
    --job-name=rao_s1_anom_manifest \
    --partition=serc \
    --time=1:00:00 \
    --cpus-per-task=1 \
    --mem=16G \
    --dependency="afterok:${tiles_job_id}" \
    --output="${logs_dir}/rao_s1_anomaly_manifest_%j.out" \
    --error="${logs_dir}/rao_s1_anomaly_manifest_%j.err" \
    --wrap="cd ${script_dir}; source ${env_path}; python3 -u ${script_dir}/build_rao_s1_viewer_tiles.py --config ${config_path} --layers lfmc anomaly --manifest-only")"

echo "Submitted Rao S1 anomaly tile array ${tiles_job_id} (${array_spec})"
echo "Submitted Rao S1 anomaly manifest job ${manifest_job_id}"
echo "Plan: ${plan_path}"
