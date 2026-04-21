#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs/operational_daily"
mkdir -p "${logs_dir}"

low_latency_script="${script_dir}/orchestrate_low_latency_daily_update.sh"
final_year_script="${script_dir}/orchestrate_final_year_update.sh"

run_low_latency="${RUN_LOW_LATENCY:-1}"
run_final_upgrade="${RUN_FINAL_UPGRADE:-1}"
run_downstream="${RUN_DOWNSTREAM:-0}"

batch_stamp="$(date +%Y%m%d_%H%M%S)"
lock_dir="${logs_dir}/operational_daily.lock"
status_path="${logs_dir}/operational_daily_${batch_stamp}.json"

low_latency_status="skipped"
final_upgrade_status="skipped"
downstream_status="skipped"
final_status="started"
message="started"

cleanup() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

write_status() {
    STATUS_PATH="${status_path}" python3 - <<'PY1'
import json
import os
from pathlib import Path

record = {
    "final_status": os.environ["FINAL_STATUS"],
    "message": os.environ["MESSAGE"],
    "low_latency_status": os.environ["LOW_LATENCY_STATUS"],
    "final_upgrade_status": os.environ["FINAL_UPGRADE_STATUS"],
    "downstream_status": os.environ["DOWNSTREAM_STATUS"],
}
path = Path(os.environ["STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True))
print(path)
PY1
}

update_status_env() {
    export LOW_LATENCY_STATUS="${low_latency_status}"
    export FINAL_UPGRADE_STATUS="${final_upgrade_status}"
    export DOWNSTREAM_STATUS="${downstream_status}"
    export FINAL_STATUS="${final_status}"
    export MESSAGE="${message}"
}

echo "Running operational daily pipeline"
echo "  run_low_latency=${run_low_latency}"
echo "  run_final_upgrade=${run_final_upgrade}"
echo "  run_downstream=${run_downstream}"

if ! mkdir "${lock_dir}" 2>/dev/null; then
    final_status="failed_lock_exists"
    message="Another operational daily update appears to be running: ${lock_dir}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

if [[ "${run_low_latency}" == "1" ]]; then
    if bash "${low_latency_script}"; then
        low_latency_status="completed"
    else
        low_latency_status="failed"
        final_status="failed_low_latency"
        message="Low-latency daily orchestration failed"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
fi

if [[ "${run_final_upgrade}" == "1" ]]; then
    if bash "${final_year_script}"; then
        final_upgrade_status="completed"
    else
        final_upgrade_status="failed"
        final_status="failed_final_upgrade"
        message="Final-year upgrade orchestration failed"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
fi

if [[ "${run_downstream}" == "1" ]]; then
    downstream_status="not_implemented"
    echo "Downstream rebuild hooks are not implemented yet; skipping"
else
    downstream_status="skipped"
fi

final_status="completed"
message="Operational daily pipeline completed"
update_status_env
write_status >/dev/null

echo "${message}"
echo "  status_path=${status_path}"
