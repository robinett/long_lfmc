#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
repo_root="/home/users/trobinet/long_lfmc"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"

update_year="${UPDATE_YEAR:-2025}"
config_path="${CONFIG_PATH:-${script_dir}/map_configs_final_update.yaml}"
registry_path="${REGISTRY_PATH:-${script_dir}/source_registry.yaml}"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
current_model_script="${script_dir}/current_model_family_utils.py"

daymet_update_script="${repo_root}/data_processing/daymet/update_daymet_archive_year.py"
nlcd_update_script="${repo_root}/data_processing/nlcd/update_nlcd_year.py"
map_submit_script="${script_dir}/submit_create_maps_ensemble.sh"
promote_script="${script_dir}/promote_map_output.py"

requested_start_date="${update_year}-01-01"
requested_end_date="${update_year}-12-31"
skip_source_update="${SKIP_SOURCE_UPDATE:-0}"

echo "Running final annual update for ${update_year}"
source "${process_env}"

if [[ "${skip_source_update}" != "1" ]]; then
    echo "Updating archive Daymet for ${update_year}"
    python3 "${daymet_update_script}" --year "${update_year}"

    echo "Updating NLCD for ${update_year}"
    python3 "${nlcd_update_script}" --year "${update_year}"
else
    echo "Skipping source updates because SKIP_SOURCE_UPDATE=1"
fi

echo "Submitting final-year map production run ${requested_start_date} -> ${requested_end_date}"
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
    echo "Failed to resolve final-year promotion paths from config=${config_path} registry=${registry_path}" >&2
    exit 1
fi

staging_zarr="${promotion_values[0]}"
production_zarr="${promotion_values[1]}"
metadata_dir="${promotion_values[2]}"

echo "Promoting staged final-year output into production store"
python3 "${promote_script}"     --staging_zarr "${staging_zarr}"     --production_zarr "${production_zarr}"     --metadata_dir "${metadata_dir}"     --start_date "${requested_start_date}"     --end_date "${requested_end_date}"     --mode overwrite_time_range     --tier final     --initialize_if_missing

echo "Final annual update complete for ${update_year}"
echo "  staging_zarr=${staging_zarr}"
echo "  production_zarr=${production_zarr}"
