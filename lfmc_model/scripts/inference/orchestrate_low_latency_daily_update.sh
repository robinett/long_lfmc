#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
repo_root="/home/users/trobinet/long_lfmc"
config_path="${script_dir}/map_configs_low_latency_update.yaml"
registry_path="${script_dir}/source_registry.yaml"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"

modis_update_script="${repo_root}/data_processing/modis/update_modis_month.py"
climate_update_script="${repo_root}/data_processing/climate_low_latency/update_low_latency_climate_range.py"
promotion_script="${script_dir}/submit_low_latency_range_promotion.sh"

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
    registry['sources']['climate_low_latency']['path'],
    registry['sources']['nlcd']['annual_path'],
    registry['sources']['static']['path'],
    registry['sources']['climate_zone']['path'],
    registry['sources']['production']['zarr_path'],
    registry['sources']['production']['metadata_dir'],
    cfg['data']['grid_path'],
    cfg['ensemble']['outputs_root'],
    str(cfg['data']['requested_start_date']),
    str(registry['processing']['prism']['release_latency_days']),
]
for value in values:
    print(value)
PY1
)

scratch_root="${registry_values[0]}"
oak_root="${registry_values[1]}"
modis_path="${registry_values[2]}"
modis_regrid_root="${registry_values[3]}"
low_latency_climate_path="${registry_values[4]}"
nlcd_annual_path="${registry_values[5]}"
static_path="${registry_values[6]}"
climate_path="${registry_values[7]}"
production_zarr="${registry_values[8]}"
metadata_dir="${registry_values[9]}"
grid_path="${registry_values[10]}"
ensemble_outputs_root="${registry_values[11]}"
default_start_date="${registry_values[12]}"
prism_latency_days="${registry_values[13]}"

batch_stamp="$(date +%Y%m%d_%H%M%S)"
status_dir="${metadata_dir}/status_reports"
lock_dir="${metadata_dir}/locks/low_latency_daily.lock"
mkdir -p "${status_dir}" "${metadata_dir}/locks"
status_path="${status_dir}/low_latency_daily_update_${batch_stamp}.json"

bootstrap_modis="not_started"
bootstrap_modis_regrid="not_started"
bootstrap_low_latency_climate="not_started"
bootstrap_nlcd="not_started"
bootstrap_static="not_started"
bootstrap_climate="not_started"
bootstrap_grid="not_started"
bootstrap_models="not_started"
bootstrap_production="not_started"
modis_check_status="not_started"
climate_check_status="not_started"
modis_update_status="not_started"
climate_update_status="not_started"
inference_status="not_started"
sync_back_status="not_started"
final_status="started"
message="started"
requested_start_date="${REQUESTED_START_DATE:-}"
requested_end_date="${REQUESTED_END_DATE:-}"

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
    'requested_start_date': os.environ['REQUESTED_START_DATE_VALUE'],
    'requested_end_date': os.environ['REQUESTED_END_DATE_VALUE'],
    'final_status': os.environ['FINAL_STATUS'],
    'message': os.environ['MESSAGE'],
    'bootstrap': {
        'modis_zarr': os.environ['BOOTSTRAP_MODIS'],
        'modis_regrid_root': os.environ['BOOTSTRAP_MODIS_REGRID'],
        'low_latency_climate': os.environ['BOOTSTRAP_LOW_LATENCY_CLIMATE'],
        'nlcd_annual': os.environ['BOOTSTRAP_NLCD'],
        'static': os.environ['BOOTSTRAP_STATIC'],
        'climate_zone': os.environ['BOOTSTRAP_CLIMATE'],
        'grid': os.environ['BOOTSTRAP_GRID'],
        'ensemble_outputs': os.environ['BOOTSTRAP_MODELS'],
        'production_zarr': os.environ['BOOTSTRAP_PRODUCTION'],
    },
    'checks': {
        'modis': os.environ['MODIS_CHECK_STATUS'],
        'low_latency_climate': os.environ['CLIMATE_CHECK_STATUS'],
    },
    'updates': {
        'modis': os.environ['MODIS_UPDATE_STATUS'],
        'low_latency_climate': os.environ['CLIMATE_UPDATE_STATUS'],
    },
    'inference_status': os.environ['INFERENCE_STATUS'],
    'sync_back_status': os.environ['SYNC_BACK_STATUS'],
}
path = Path(os.environ['STATUS_PATH'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True))
print(path)
PY3
}

update_status_env() {
    export REQUESTED_START_DATE_VALUE="${requested_start_date}"
    export REQUESTED_END_DATE_VALUE="${requested_end_date}"
    export BOOTSTRAP_MODIS="${bootstrap_modis}"
    export BOOTSTRAP_MODIS_REGRID="${bootstrap_modis_regrid}"
    export BOOTSTRAP_LOW_LATENCY_CLIMATE="${bootstrap_low_latency_climate}"
    export BOOTSTRAP_NLCD="${bootstrap_nlcd}"
    export BOOTSTRAP_STATIC="${bootstrap_static}"
    export BOOTSTRAP_CLIMATE="${bootstrap_climate}"
    export BOOTSTRAP_GRID="${bootstrap_grid}"
    export BOOTSTRAP_MODELS="${bootstrap_models}"
    export BOOTSTRAP_PRODUCTION="${bootstrap_production}"
    export MODIS_CHECK_STATUS="${modis_check_status}"
    export CLIMATE_CHECK_STATUS="${climate_check_status}"
    export MODIS_UPDATE_STATUS="${modis_update_status}"
    export CLIMATE_UPDATE_STATUS="${climate_update_status}"
    export INFERENCE_STATUS="${inference_status}"
    export SYNC_BACK_STATUS="${sync_back_status}"
    export FINAL_STATUS="${final_status}"
    export MESSAGE="${message}"
}

resolve_range() {
    TARGET_END_DATE="${requested_end_date}" TARGET_START_DATE="${requested_start_date}" DEFAULT_START_DATE="${default_start_date}" PRODUCTION_ZARR="${production_zarr}" PRISM_LATENCY_DAYS="${prism_latency_days}" python3 - <<'PY4'
import os
import pandas as pd
import zarr
import numpy as np

requested_start = os.environ['TARGET_START_DATE']
requested_end = os.environ['TARGET_END_DATE']
default_start = pd.Timestamp(os.environ['DEFAULT_START_DATE']).normalize()
safe_end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=int(os.environ['PRISM_LATENCY_DAYS'])))
if requested_end:
    end_date = pd.Timestamp(requested_end).normalize()
else:
    end_date = safe_end
if end_date > safe_end:
    end_date = safe_end
if requested_start:
    start_date = pd.Timestamp(requested_start).normalize()
else:
    start_date = default_start
    if os.path.exists(os.environ['PRODUCTION_ZARR']):
        root = zarr.open_group(os.environ['PRODUCTION_ZARR'], mode='r')
        if 'time' in root and len(root['time']) > 0:
            times = pd.to_datetime(np.asarray(root['time'][:], dtype=np.int64)).normalize()
            start_date = pd.Timestamp(times.max()).normalize() + pd.Timedelta(days=1)
print(start_date.strftime('%Y-%m-%d'))
print(end_date.strftime('%Y-%m-%d'))
PY4
}

months_in_range() {
    RANGE_START="$1" RANGE_END="$2" python3 - <<'PY5'
import os
import pandas as pd
start = pd.Timestamp(os.environ['RANGE_START']).normalize()
end = pd.Timestamp(os.environ['RANGE_END']).normalize()
months = pd.period_range(start=start, end=end, freq='M')
for month in months:
    print(str(month))
PY5
}

echo "Running low-latency daily orchestration"

if ! mkdir "${lock_dir}" 2>/dev/null; then
    final_status="failed_lock_exists"
    message="Another low-latency daily update appears to be running: ${lock_dir}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

readarray -t range_values < <(resolve_range)
requested_start_date="${range_values[0]}"
requested_end_date="${range_values[1]}"

if [[ "${requested_start_date}" > "${requested_end_date}" ]]; then
    final_status="nothing_to_do"
    message="No low-latency dates need updating"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

if ensure_on_scratch "${modis_path}" "canonical MODIS zarr" "0"; then bootstrap_modis="ready"; else bootstrap_modis="missing"; fi
if ensure_on_scratch "${modis_regrid_root}" "MODIS regrid root" "0"; then bootstrap_modis_regrid="ready"; else bootstrap_modis_regrid="missing"; fi
if ensure_on_scratch "${low_latency_climate_path}" "low-latency climate zarr" "1"; then bootstrap_low_latency_climate="ready_or_missing_allowed"; fi
if ensure_on_scratch "${nlcd_annual_path}" "annual NLCD zarr" "0"; then bootstrap_nlcd="ready"; else bootstrap_nlcd="missing"; fi
if ensure_on_scratch "${static_path}" "static dataset" "0"; then bootstrap_static="ready"; fi
if ensure_on_scratch "${climate_path}" "climate dataset" "0"; then bootstrap_climate="ready"; fi
if ensure_on_scratch "${grid_path}" "grid" "0"; then bootstrap_grid="ready"; fi
if ensure_on_scratch "${ensemble_outputs_root}" "ensemble outputs" "0"; then bootstrap_models="ready"; fi
if ensure_on_scratch "${production_zarr}" "production LFMC zarr" "1"; then bootstrap_production="ready_or_missing_allowed"; fi

readarray -t months < <(months_in_range "${requested_start_date}" "${requested_end_date}")
for month in "${months[@]}"; do
    if python3 "${modis_update_script}" --month "${month}" --check_only; then
        modis_check_status="ready"
    else
        modis_check_status="not_ready"
        final_status="not_ready"
        message="MODIS is not yet fully available for ${month}"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 0
    fi
done

if python3 "${climate_update_script}" --start_date "${requested_start_date}" --end_date "${requested_end_date}" --check_only; then
    climate_check_status="ready"
else
    climate_check_status="not_ready"
    final_status="not_ready"
    message="Low-latency climate source is not yet fully available for ${requested_start_date} -> ${requested_end_date}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

for month in "${months[@]}"; do
    python3 "${modis_update_script}" --month "${month}"
done
modis_update_status="completed"

python3 "${climate_update_script}" --start_date "${requested_start_date}" --end_date "${requested_end_date}"
climate_update_status="completed"

source "${model_env}"
REQUESTED_START_DATE="${requested_start_date}" REQUESTED_END_DATE="${requested_end_date}" bash "${promotion_script}"
inference_status="completed"

sync_to_oak "${modis_path}" "canonical MODIS zarr"
sync_to_oak "${modis_regrid_root}" "MODIS regrid root"
sync_to_oak "${low_latency_climate_path}" "low-latency climate zarr"
sync_to_oak "${production_zarr}" "production LFMC zarr"
sync_to_oak "${metadata_dir}" "production metadata"
sync_back_status="completed"

final_status="completed"
message="Low-latency daily update completed for ${requested_start_date} -> ${requested_end_date}"
update_status_env
write_status >/dev/null

echo "${message}"
echo "  status_path=${status_path}"
