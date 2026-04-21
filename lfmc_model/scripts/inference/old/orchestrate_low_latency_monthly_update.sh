#!/usr/bin/env bash

# Archived monthly low-latency orchestration. Retained for historical reference only.

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
repo_root="/home/users/trobinet/long_lfmc"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"

config_path="${script_dir}/map_configs_low_latency_update.yaml"
registry_path="${script_dir}/source_registry.yaml"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"

modis_update_script="${repo_root}/data_processing/modis/update_modis_month.py"
daymet_update_script="${repo_root}/data_processing/daymet/update_daymet_monthly_latency_month.py"
promotion_script="${script_dir}/submit_low_latency_month_promotion.sh"

source ~/.bashrc
source "${process_env}"

readarray -t registry_values < <(REGISTRY_PATH="${registry_path}" CONFIG_PATH="${config_path}" python3 - <<'PY1'
import os
import yaml

with open(os.environ['REGISTRY_PATH'], 'r') as f:
    registry = yaml.safe_load(f)
with open(os.environ['CONFIG_PATH'], 'r') as f:
    cfg = yaml.safe_load(f)
values = [
    registry['storage']['scratch_root'],
    registry['storage']['oak_root'],
    registry['sources']['modis']['path'],
    registry['processing']['modis']['regrid_root'],
    registry['sources']['daymet']['archive_path'],
    registry['sources']['daymet']['monthly_latency_path'],
    registry['sources']['nlcd']['annual_path'],
    registry['sources']['static']['path'],
    registry['sources']['climate_zone']['path'],
    registry['sources']['production']['zarr_path'],
    registry['sources']['production']['metadata_dir'],
    cfg['data']['grid_path'],
    cfg['ensemble']['outputs_root'],
    str(cfg['data']['requested_start_date'])[:7],
]
for value in values:
    print(value)
PY1
)

scratch_root="${registry_values[0]}"
oak_root="${registry_values[1]}"
modis_path="${registry_values[2]}"
modis_regrid_root="${registry_values[3]}"
daymet_archive_path="${registry_values[4]}"
daymet_monthly_latency_path="${registry_values[5]}"
nlcd_annual_path="${registry_values[6]}"
static_path="${registry_values[7]}"
climate_path="${registry_values[8]}"
production_zarr="${registry_values[9]}"
metadata_dir="${registry_values[10]}"
grid_path="${registry_values[11]}"
ensemble_outputs_root="${registry_values[12]}"
default_target_month="${registry_values[13]}"

batch_stamp="$(date +%Y%m%d_%H%M%S)"
status_dir="${metadata_dir}/status_reports"
lock_dir="${metadata_dir}/locks/low_latency_monthly.lock"
mkdir -p "${status_dir}" "${metadata_dir}/locks"
status_path="${status_dir}/low_latency_monthly_update_${batch_stamp}.json"

bootstrap_modis="not_started"
bootstrap_modis_regrid="not_started"
bootstrap_daymet_archive="not_started"
bootstrap_daymet_monthly="not_started"
bootstrap_nlcd="not_started"
bootstrap_static="not_started"
bootstrap_climate="not_started"
bootstrap_grid="not_started"
bootstrap_models="not_started"
bootstrap_production="not_started"
modis_check_status="not_started"
daymet_check_status="not_started"
modis_update_status="not_started"
daymet_update_status="not_started"
inference_status="not_started"
sync_back_status="not_started"
final_status="started"
message="started"
target_month="${TARGET_MONTH:-}"

cleanup() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

scratch_to_oak() {
    local scratch_path="$1"
    SCRATCH_ROOT="${scratch_root}" OAK_ROOT="${oak_root}" SCRATCH_PATH="${scratch_path}" python3 - <<'PY2'
import os
scratch_root = os.environ['SCRATCH_ROOT']
oak_root = os.environ['OAK_ROOT']
path = os.environ['SCRATCH_PATH']
if not path.startswith(scratch_root + '/') and path != scratch_root:
    print('')
    raise SystemExit(0)
rel = os.path.relpath(path, scratch_root)
print(os.path.join(oak_root, rel))
PY2
}

ensure_on_scratch() {
    local scratch_path="$1"
    local label="$2"
    local allow_missing="$3"
    local oak_path
    if [[ -e "${scratch_path}" ]]; then
        echo "Using scratch ${label}: ${scratch_path}"
        return 0
    fi
    oak_path="$(scratch_to_oak "${scratch_path}")"
    if [[ -n "${oak_path}" && -e "${oak_path}" ]]; then
        mkdir -p "$(dirname "${scratch_path}")"
        echo "Copying ${label} from OAK to scratch: ${oak_path} -> ${scratch_path}"
        rsync -a "${oak_path}" "${scratch_path}"
        return 0
    fi
    if [[ "${allow_missing}" == "1" ]]; then
        echo "${label} is missing on scratch and OAK; continuing because allow_missing=1"
        return 0
    fi
    echo "Required ${label} is missing on scratch and OAK: ${scratch_path}"
    return 1
}

sync_to_oak() {
    local scratch_path="$1"
    local label="$2"
    local oak_path
    if [[ ! -e "${scratch_path}" ]]; then
        echo "Skipping OAK sync for missing ${label}: ${scratch_path}"
        return 0
    fi
    oak_path="$(scratch_to_oak "${scratch_path}")"
    if [[ -z "${oak_path}" ]]; then
        echo "Skipping OAK sync for ${label}; could not map scratch path: ${scratch_path}"
        return 0
    fi
    mkdir -p "$(dirname "${oak_path}")"
    echo "Syncing ${label} back to OAK: ${scratch_path} -> ${oak_path}"
    rsync -a "${scratch_path}" "${oak_path}"
}

write_status() {
    STATUS_PATH="${status_path}" python3 - <<'PY3'
import json
import os
from pathlib import Path

record = {
    'target_month': os.environ['TARGET_MONTH_VALUE'],
    'final_status': os.environ['FINAL_STATUS'],
    'message': os.environ['MESSAGE'],
    'bootstrap': {
        'modis_zarr': os.environ['BOOTSTRAP_MODIS'],
        'modis_regrid_root': os.environ['BOOTSTRAP_MODIS_REGRID'],
        'daymet_archive': os.environ['BOOTSTRAP_DAYMET_ARCHIVE'],
        'daymet_monthly_latency': os.environ['BOOTSTRAP_DAYMET_MONTHLY'],
        'nlcd_annual': os.environ['BOOTSTRAP_NLCD'],
        'static': os.environ['BOOTSTRAP_STATIC'],
        'climate_zone': os.environ['BOOTSTRAP_CLIMATE'],
        'grid': os.environ['BOOTSTRAP_GRID'],
        'ensemble_outputs': os.environ['BOOTSTRAP_MODELS'],
        'production_zarr': os.environ['BOOTSTRAP_PRODUCTION'],
    },
    'checks': {
        'modis': os.environ['MODIS_CHECK_STATUS'],
        'daymet_monthly_latency': os.environ['DAYMET_CHECK_STATUS'],
    },
    'updates': {
        'modis': os.environ['MODIS_UPDATE_STATUS'],
        'daymet_monthly_latency': os.environ['DAYMET_UPDATE_STATUS'],
    },
    'inference_status': os.environ['INFERENCE_STATUS'],
    'sync_back_status': os.environ['SYNC_BACK_STATUS'],
    'paths': {
        'modis_path': os.environ['MODIS_PATH'],
        'modis_regrid_root': os.environ['MODIS_REGRID_ROOT'],
        'daymet_archive_path': os.environ['DAYMET_ARCHIVE_PATH'],
        'daymet_monthly_latency_path': os.environ['DAYMET_MONTHLY_LATENCY_PATH'],
        'nlcd_annual_path': os.environ['NLCD_ANNUAL_PATH'],
        'production_zarr': os.environ['PRODUCTION_ZARR'],
        'metadata_dir': os.environ['METADATA_DIR'],
        'grid_path': os.environ['GRID_PATH'],
        'ensemble_outputs_root': os.environ['ENSEMBLE_OUTPUTS_ROOT'],
    },
}
path = Path(os.environ['STATUS_PATH'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True))
print(path)
PY3
}

update_status_env() {
    export TARGET_MONTH_VALUE="${target_month}"
    export BOOTSTRAP_MODIS="${bootstrap_modis}"
    export BOOTSTRAP_MODIS_REGRID="${bootstrap_modis_regrid}"
    export BOOTSTRAP_DAYMET_ARCHIVE="${bootstrap_daymet_archive}"
    export BOOTSTRAP_DAYMET_MONTHLY="${bootstrap_daymet_monthly}"
    export BOOTSTRAP_NLCD="${bootstrap_nlcd}"
    export BOOTSTRAP_STATIC="${bootstrap_static}"
    export BOOTSTRAP_CLIMATE="${bootstrap_climate}"
    export BOOTSTRAP_GRID="${bootstrap_grid}"
    export BOOTSTRAP_MODELS="${bootstrap_models}"
    export BOOTSTRAP_PRODUCTION="${bootstrap_production}"
    export MODIS_CHECK_STATUS="${modis_check_status}"
    export DAYMET_CHECK_STATUS="${daymet_check_status}"
    export MODIS_UPDATE_STATUS="${modis_update_status}"
    export DAYMET_UPDATE_STATUS="${daymet_update_status}"
    export INFERENCE_STATUS="${inference_status}"
    export SYNC_BACK_STATUS="${sync_back_status}"
    export FINAL_STATUS="${final_status}"
    export MESSAGE="${message}"
    export MODIS_PATH="${modis_path}"
    export MODIS_REGRID_ROOT="${modis_regrid_root}"
    export DAYMET_ARCHIVE_PATH="${daymet_archive_path}"
    export DAYMET_MONTHLY_LATENCY_PATH="${daymet_monthly_latency_path}"
    export NLCD_ANNUAL_PATH="${nlcd_annual_path}"
    export PRODUCTION_ZARR="${production_zarr}"
    export METADATA_DIR="${metadata_dir}"
    export GRID_PATH="${grid_path}"
    export ENSEMBLE_OUTPUTS_ROOT="${ensemble_outputs_root}"
}

resolve_target_month() {
    if [[ -n "${target_month}" ]]; then
        echo "${target_month}"
        return 0
    fi
    TARGET_MONTH_DEFAULT="${default_target_month}" PRODUCTION_ZARR="${production_zarr}" python3 - <<'PY4'
import calendar
import os
import pandas as pd
import zarr
import numpy as np

production_zarr = os.environ['PRODUCTION_ZARR']
default_month = os.environ['TARGET_MONTH_DEFAULT']
if os.path.exists(production_zarr):
    root = zarr.open_group(production_zarr, mode='r')
    if 'time' in root and len(root['time']) > 0:
        times = pd.to_datetime(np.asarray(root['time'][:], dtype=np.int64)).normalize()
        max_time = pd.Timestamp(times.max()).normalize()
        month_end = (max_time + pd.offsets.MonthEnd(0)).normalize()
        if max_time < month_end:
            print(max_time.strftime('%Y-%m'))
        else:
            print((max_time + pd.offsets.MonthBegin(1)).strftime('%Y-%m'))
        raise SystemExit(0)
print(default_month)
PY4
}

echo "Running low-latency monthly orchestration"

if ! mkdir "${lock_dir}" 2>/dev/null; then
    final_status="failed_lock_exists"
    message="Another low-latency monthly update appears to be running: ${lock_dir}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

target_month="$(resolve_target_month)"
echo "  target_month=${target_month}"

target_month_state="$(TARGET_MONTH="${target_month}" python3 - <<'PY5'
import os
import pandas as pd
month = os.environ['TARGET_MONTH']
start = pd.Timestamp(f'{month}-01').normalize()
current_month_start = pd.Timestamp.today().normalize().replace(day=1)
if start >= current_month_start:
    print('future_or_incomplete')
else:
    print('ok')
PY5
)"
if [[ "${target_month_state}" != "ok" ]]; then
    final_status="not_ready"
    message="Target month ${target_month} is not a fully completed past month"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

if ensure_on_scratch "${modis_path}" "canonical MODIS zarr" "0"; then
    bootstrap_modis="ready"
else
    bootstrap_modis="missing"
    final_status="failed_missing_modis"
    message="MODIS canonical zarr is required on scratch or OAK before low-latency update can run"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

if ensure_on_scratch "${modis_regrid_root}" "MODIS regrid root" "0"; then
    bootstrap_modis_regrid="ready"
else
    bootstrap_modis_regrid="missing"
    final_status="failed_missing_modis_regrid"
    message="MODIS regrid root is required on scratch or OAK before low-latency update can run"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

if ensure_on_scratch "${daymet_archive_path}" "archive Daymet zarr" "0"; then
    bootstrap_daymet_archive="ready"
else
    bootstrap_daymet_archive="missing"
    final_status="failed_missing_daymet_archive"
    message="Archive Daymet zarr is required on scratch or OAK before low-latency update can run"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

if ensure_on_scratch "${daymet_monthly_latency_path}" "monthly-latency Daymet zarr" "1"; then
    bootstrap_daymet_monthly="ready_or_missing_allowed"
fi
if ensure_on_scratch "${nlcd_annual_path}" "annual NLCD zarr" "0"; then
    bootstrap_nlcd="ready"
else
    bootstrap_nlcd="missing"
    final_status="failed_missing_nlcd"
    message="Annual NLCD zarr is required on scratch or OAK before low-latency update can run"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi
if ensure_on_scratch "${static_path}" "static dataset" "0"; then bootstrap_static="ready"; fi
if ensure_on_scratch "${climate_path}" "climate dataset" "0"; then bootstrap_climate="ready"; fi
if ensure_on_scratch "${grid_path}" "grid" "0"; then bootstrap_grid="ready"; fi
if ensure_on_scratch "${ensemble_outputs_root}" "ensemble outputs" "0"; then bootstrap_models="ready"; fi
if ensure_on_scratch "${production_zarr}" "production LFMC zarr" "1"; then bootstrap_production="ready_or_missing_allowed"; fi

if python3 "${modis_update_script}" --month "${target_month}" --check_only; then
    modis_check_status="ready"
else
    modis_check_status="not_ready"
    final_status="not_ready"
    message="MODIS is not yet fully available for ${target_month}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

if python3 "${daymet_update_script}" --month "${target_month}" --check_only; then
    daymet_check_status="ready"
else
    daymet_check_status="not_ready"
    final_status="not_ready"
    message="Monthly-latency Daymet is not yet available for ${target_month}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

python3 "${modis_update_script}" --month "${target_month}"
modis_update_status="completed"

python3 "${daymet_update_script}" --month "${target_month}"
daymet_update_status="completed"

source "${model_env}"
TARGET_MONTH="${target_month}" bash "${promotion_script}"
inference_status="completed"

sync_to_oak "${modis_path}" "canonical MODIS zarr"
sync_to_oak "${modis_regrid_root}" "MODIS regrid root"
sync_to_oak "${daymet_monthly_latency_path}" "monthly-latency Daymet zarr"
sync_to_oak "${production_zarr}" "production LFMC zarr"
sync_to_oak "${metadata_dir}" "production metadata"
sync_back_status="completed"

final_status="completed"
message="Low-latency monthly update completed for ${target_month}"
update_status_env
write_status >/dev/null

echo "${message}"
echo "  status_path=${status_path}"
