#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${CONFIG_PATH:-${script_dir}/map_configs_low_latency_update.yaml}"
registry_path="${REGISTRY_PATH:-${script_dir}/source_registry.yaml}"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
map_submit_script="${script_dir}/submit_create_maps_ensemble.sh"
promote_script="${script_dir}/promote_map_output.py"
current_model_script="${script_dir}/current_model_family_utils.py"

requested_start_date="${REQUESTED_START_DATE:-}"
requested_end_date="${REQUESTED_END_DATE:-}"

if [[ -z "${requested_start_date}" || -z "${requested_end_date}" ]]; then
    readarray -t range_values < <(CONFIG_PATH="${config_path}" python3 - <<'PY1'
import os
import yaml
with open(os.environ['CONFIG_PATH'], 'r') as f:
    cfg = yaml.safe_load(f)
print(str(cfg['data']['requested_start_date']))
print(str(cfg['data']['requested_end_date']))
PY1
    )
    requested_start_date="${requested_start_date:-${range_values[0]}}"
    requested_end_date="${requested_end_date:-${range_values[1]}}"
fi

echo "Running low-latency range promotion for ${requested_start_date} -> ${requested_end_date}"
source "${model_env}"

readarray -t model_values < <(python3 "${current_model_script}" --variant multitask --format lines)
active_model_label="${model_values[0]}"
ensemble_root="${ENSEMBLE_ROOT:-${model_values[1]}}"
input_data_name="${INPUT_DATA_NAME:-${model_values[2]}}"
echo "Using active model family ${active_model_label}"
echo "  ensemble_root=${ensemble_root}"
echo "  input_data_name=${input_data_name}"

CONFIG_PATH="${config_path}" \
REQUESTED_START_DATE="${requested_start_date}" \
REQUESTED_END_DATE="${requested_end_date}" \
ENSEMBLE_ROOT="${ensemble_root}" \
INPUT_DATA_NAME="${input_data_name}" \
bash "${map_submit_script}"

readarray -t promotion_values < <(CONFIG_PATH="${config_path}" REGISTRY_PATH="${registry_path}" python3 - <<'PY2'
import os
import sys
from pathlib import Path

script_dir = Path("/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference")
sys.path.insert(0, str(script_dir))

from map_config import get_cfg, load_map_config
from input_source_resolver import load_source_registry
from map_runtime_utils import latest_run_dir

cfg = load_map_config(os.environ['CONFIG_PATH'])
registry = load_source_registry(os.environ['REGISTRY_PATH'])
run_root = get_cfg(cfg, 'paths', 'run_root')
latest = latest_run_dir(run_root)
merged_subdir = get_cfg(cfg, 'paths', 'merged_subdir', default='merged')
merged_store_name = get_cfg(cfg, 'paths', 'merged_store_name', default='lfmc_maps.zarr')
production_path = registry['sources']['production']['zarr_path']
metadata_dir = registry['sources']['production']['metadata_dir']
staging = str(Path(latest) / merged_subdir / merged_store_name)
print(staging)
print(production_path)
print(metadata_dir)
PY2
)
if [[ "${#promotion_values[@]}" -ne 3 ]]; then
    echo "Failed to resolve low-latency promotion paths from config=${config_path} registry=${registry_path}" >&2
    exit 1
fi

staging_zarr="${promotion_values[0]}"
production_zarr="${promotion_values[1]}"
metadata_dir="${promotion_values[2]}"
promotion_mode="$(
    PRODUCTION_ZARR="${production_zarr}" START_DATE="${requested_start_date}" END_DATE="${requested_end_date}" python3 - <<'PY3'
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

production_zarr = Path(os.environ["PRODUCTION_ZARR"])
start_date = pd.Timestamp(os.environ["START_DATE"]).normalize()
end_date = pd.Timestamp(os.environ["END_DATE"]).normalize()
if not production_zarr.exists():
    print("append_time_range")
    raise SystemExit(0)
root = zarr.open_group(str(production_zarr), mode="r")
if "time" not in root:
    print("append_time_range")
    raise SystemExit(0)
time_array = root["time"]
time_size = int(time_array.shape[0]) if len(time_array.shape) > 0 else 0
if time_size == 0:
    print("append_time_range")
    raise SystemExit(0)
times = pd.to_datetime(np.asarray(time_array[:], dtype=np.int64)).normalize()
max_time = pd.Timestamp(times.max()).normalize()
if start_date > max_time:
    print("append_time_range")
elif max_time <= end_date:
    print("replace_tail_range")
else:
    print("overwrite_time_range")
PY3
)"
echo "  promotion_mode=${promotion_mode}"

python3 "${promote_script}" \
    --staging_zarr "${staging_zarr}" \
    --production_zarr "${production_zarr}" \
    --metadata_dir "${metadata_dir}" \
    --start_date "${requested_start_date}" \
    --end_date "${requested_end_date}" \
    --mode "${promotion_mode}" \
    --tier low_latency \
    --initialize_if_missing

echo "Low-latency range promotion complete"
echo "  staging_zarr=${staging_zarr}"
echo "  production_zarr=${production_zarr}"
