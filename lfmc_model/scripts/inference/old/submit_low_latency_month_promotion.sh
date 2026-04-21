#!/usr/bin/env bash

# Archived monthly low-latency promotion driver. Retained for historical reference only.

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
config_path="${script_dir}/map_configs_low_latency_update.yaml"
registry_path="${script_dir}/source_registry.yaml"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
map_submit_script="${script_dir}/submit_create_maps_ensemble.sh"
promote_script="${script_dir}/promote_map_output.py"

target_month="${TARGET_MONTH:-}"
if [[ -z "${target_month}" ]]; then
    target_month="$(CONFIG_PATH="${config_path}" python3 - <<'PY2'
import os
import yaml
with open(os.environ['CONFIG_PATH'], 'r') as f:
    cfg = yaml.safe_load(f)
print(str(cfg['data']['requested_start_date'])[:7])
PY2
)"
fi

requested_start_date="$(TARGET_MONTH="${target_month}" python3 - <<'PY3'
import os
import pandas as pd
month = os.environ['TARGET_MONTH']
start = pd.Timestamp(f'{month}-01').normalize()
print(start.strftime('%Y-%m-%d'))
PY3
)"
requested_end_date="$(TARGET_MONTH="${target_month}" python3 - <<'PY4'
import os
import pandas as pd
month = os.environ['TARGET_MONTH']
start = pd.Timestamp(f'{month}-01').normalize()
end = (start + pd.offsets.MonthEnd(1)).normalize()
print(end.strftime('%Y-%m-%d'))
PY4
)"

echo "Running low-latency monthly promotion for ${target_month}"
source ~/.bashrc
source "${model_env}"

CONFIG_PATH="${config_path}" REQUESTED_START_DATE="${requested_start_date}" REQUESTED_END_DATE="${requested_end_date}" bash "${map_submit_script}"

readarray -t promotion_values < <(CONFIG_PATH="${config_path}" REGISTRY_PATH="${registry_path}" python3 - <<'PY5'
import os
from pathlib import Path
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
PY5
)

staging_zarr="${promotion_values[0]}"
production_zarr="${promotion_values[1]}"
metadata_dir="${promotion_values[2]}"

python3 "${promote_script}" \
    --staging_zarr "${staging_zarr}" \
    --production_zarr "${production_zarr}" \
    --metadata_dir "${metadata_dir}" \
    --start_date "${requested_start_date}" \
    --end_date "${requested_end_date}" \
    --mode append_time_range \
    --tier low_latency \
    --initialize_if_missing

echo "Low-latency monthly promotion complete for ${target_month}"
echo "  staging_zarr=${staging_zarr}"
echo "  production_zarr=${production_zarr}"
