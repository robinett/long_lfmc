#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
config_path="${script_dir}/viewer_pipeline_config.yaml"
logs_dir="${script_dir}/logs"
viewer_env_path="/home/users/trobinet/uv_activations/activate_lfmc_viewer_py312.sh"
model_env_path="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"

mkdir -p "${logs_dir}"
cd "${script_dir}"

source "${viewer_env_path}"

echo "Initializing viewer-grid LFMC climatology plan"
python3 -u "${script_dir}/sample_lfmc_climatology_viewer.py" \
    --config "${config_path}" \
    --mode init

viewer_block_count="$(python3 -c "import json, yaml; cfg=yaml.safe_load(open('${config_path}')); plan=cfg['climatology']['viewer_state_dir'] + '/viewer_climatology_plan.json'; print(json.load(open(plan))['block_count'])")"
viewer_max_concurrent="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_path}')); print(int(cfg['climatology'].get('viewer_max_concurrent_tasks', 32)))")"

if [[ "${viewer_block_count}" -lt 1 ]]; then
    echo "No viewer climatology blocks selected; nothing to submit."
    exit 0
fi

viewer_last_index=$((viewer_block_count - 1))
viewer_array_spec="0-${viewer_last_index}%${viewer_max_concurrent}"

viewer_clim_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=04:00:00 \
    --cpus-per-task=2 \
    --mem=48G \
    --array="${viewer_array_spec}" \
    --output="${logs_dir}/viewer_climatology_%A_%a.out" \
    --error="${logs_dir}/viewer_climatology_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${viewer_env_path}; python3 -u ${script_dir}/sample_lfmc_climatology_viewer.py --config ${config_path} --mode worker --use-slurm-array")"

viewer_finalize_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=01:00:00 \
    --cpus-per-task=2 \
    --mem=32G \
    --dependency="afterok:${viewer_clim_job_id}" \
    --output="${logs_dir}/viewer_climatology_finalize_%j.out" \
    --error="${logs_dir}/viewer_climatology_finalize_%j.err" \
    --wrap="cd ${script_dir}; source ${viewer_env_path}; python3 -u ${script_dir}/sample_lfmc_climatology_viewer.py --config ${config_path} --mode finalize")"

qc_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=96G \
    --dependency="afterok:${viewer_finalize_job_id}" \
    --output="${logs_dir}/viewer_climatology_qc_%j.out" \
    --error="${logs_dir}/viewer_climatology_qc_%j.err" \
    --wrap="cd ${script_dir}; source ${model_env_path}; python3 -u ${script_dir}/qc_viewer_climatology.py --config ${config_path} --date 2024-12-31")"

anomaly_plan_path="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_path}')); print(cfg['climatology']['anomaly_tile_plan_path'])")"

python3 - <<PY
import json
import time
from pathlib import Path
import yaml
import xarray as xr

cfg = yaml.safe_load(open("${config_path}"))
viewer_path = Path(cfg["output"]["viewer_dataset_path"])
status_dir = Path(cfg["output"]["state_dir"]) / "tile_dates"
plan_path = Path(cfg["climatology"]["anomaly_tile_plan_path"])
block_days = int(cfg["climatology"].get("anomaly_tile_block_days", 90))
layer_key = "anomaly"

ds = xr.open_zarr(viewer_path, consolidated=False)
try:
    dates = [str(value)[:10] for value in ds["time"].values]
finally:
    ds.close()

selected_dates = []
for date_str in dates:
    status_path = status_dir / f"{date_str}.json"
    if not status_path.exists():
        selected_dates.append(date_str)
        continue
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if layer_key not in status.get("layers", {}):
        selected_dates.append(date_str)

blocks = []
for block_idx, start in enumerate(range(0, len(selected_dates), block_days)):
    block_dates = selected_dates[start:start + block_days]
    blocks.append({
        "block_index": block_idx,
        "start_date": block_dates[0],
        "end_date": block_dates[-1],
        "dates": block_dates,
    })

payload = {
    "status": "completed",
    "mode": "anomaly_tiles",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "viewer_dataset_path": str(viewer_path),
    "layer": layer_key,
    "block_days": block_days,
    "date_count": len(selected_dates),
    "block_count": len(blocks),
    "dates": selected_dates,
    "blocks": blocks,
}
plan_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = plan_path.with_suffix(plan_path.suffix + ".tmp")
tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
tmp_path.replace(plan_path)
print(f"Wrote anomaly tile plan {plan_path} with {len(selected_dates)} dates in {len(blocks)} blocks")
PY

anomaly_block_count="$(python3 -c "import json; print(json.load(open('${anomaly_plan_path}'))['block_count'])")"
anomaly_date_count="$(python3 -c "import json; print(json.load(open('${anomaly_plan_path}'))['date_count'])")"

if [[ "${anomaly_block_count}" -lt 1 ]]; then
    echo "No missing anomaly tiles detected; skipping anomaly tile submission."
    echo "Submitted viewer climatology array ${viewer_clim_job_id} (${viewer_array_spec})"
    echo "Submitted viewer climatology finalize job ${viewer_finalize_job_id}"
    echo "Submitted viewer climatology QC job ${qc_job_id}"
    exit 0
fi

anomaly_last_index=$((anomaly_block_count - 1))
anomaly_max_concurrent="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_path}')); print(int(cfg['climatology'].get('anomaly_tile_max_concurrent_tasks', 16)))")"
anomaly_array_spec="0-${anomaly_last_index}%${anomaly_max_concurrent}"

anomaly_tiles_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=08:00:00 \
    --cpus-per-task=8 \
    --mem=128G \
    --array="${anomaly_array_spec}" \
    --dependency="afterok:${qc_job_id}" \
    --output="${logs_dir}/viewer_anomaly_tiles_%A_%a.out" \
    --error="${logs_dir}/viewer_anomaly_tiles_%A_%a.err" \
    --wrap="cd ${script_dir}; source ${viewer_env_path}; python3 -u ${script_dir}/run_viewer_tile_dates.py --config ${config_path} --plan-path ${anomaly_plan_path} --use-slurm-array --layers anomaly")"

manifest_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=64G \
    --dependency="afterok:${anomaly_tiles_job_id}" \
    --output="${logs_dir}/viewer_anomaly_manifest_%j.out" \
    --error="${logs_dir}/viewer_anomaly_manifest_%j.err" \
    --wrap="cd ${script_dir}; source ${viewer_env_path}; python3 -u ${script_dir}/finalize_viewer_manifest.py --config ${config_path} --require-all-dates")"

echo "Submitted viewer climatology array ${viewer_clim_job_id} (${viewer_array_spec})"
echo "Submitted viewer climatology finalize job ${viewer_finalize_job_id}"
echo "Submitted viewer climatology QC job ${qc_job_id}"
echo "Detected ${anomaly_date_count} dates needing anomaly tiles in ${anomaly_block_count} blocks"
echo "Submitted anomaly tile array ${anomaly_tiles_job_id} (${anomaly_array_spec})"
echo "Submitted anomaly manifest finalize job ${manifest_job_id}"
