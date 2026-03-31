#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/lfmc_longweather_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/lfmc_longweather_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

cd "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training"
mkdir -p "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

lock_file="${LOCK_FILE:-}"
coord_state_dir="${COORD_STATE_DIR:-}"
coord_run_name="${COORD_RUN_NAME:-}"
coord_run_tag="${COORD_RUN_TAG:-}"
coord_pool="${COORD_POOL:-}"
heartbeat_pid=""

write_state_marker() {
  local marker_name="$1"
  local exit_code="${2:-}"
  if [[ -z "${coord_state_dir}" ]]; then
    return 0
  fi

  mkdir -p "${coord_state_dir}"
  python3 - "${coord_state_dir}" "${marker_name}" "${coord_run_name}" "${coord_run_tag}" "${coord_pool}" "${SLURM_JOB_ID:-}" "${SLURM_JOB_NODELIST:-}" "${exit_code}" <<'PY'
import datetime as dt
import json
import os
import sys

state_dir, marker_name, run_name, run_tag, pool, job_id, node_name, exit_code = sys.argv[1:9]
now_utc = dt.datetime.now(dt.UTC)
payload = {
    "timestamp_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "timestamp_unix": now_utc.timestamp(),
    "marker": marker_name,
    "run_name": run_name,
    "run_tag": run_tag,
    "pool": pool,
    "job_id": job_id,
    "node_name": node_name,
}
if exit_code not in ("", "None"):
    payload["exit_code"] = int(exit_code)
path = os.path.join(state_dir, marker_name + ".json")
tmp_path = path + ".tmp"
with open(tmp_path, "w") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp_path, path)
PY
}

start_heartbeat_loop() {
  if [[ -z "${coord_state_dir}" ]]; then
    return 0
  fi
  (
    while true; do
      write_state_marker heartbeat
      sleep 300
    done
  ) &
  heartbeat_pid="$!"
}

cleanup() {
  local exit_code="$?"
  trap - EXIT INT TERM

  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi

  if [[ -n "${coord_state_dir}" ]]; then
    if [[ "${exit_code}" -eq 0 ]]; then
      write_state_marker completed 0
    else
      write_state_marker failed "${exit_code}"
    fi
  fi

  if [[ -n "${lock_file}" ]]; then
    rm -f "${lock_file}"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ -n "${coord_state_dir}" ]]; then
  rm -f "${coord_state_dir}/started.json" "${coord_state_dir}/heartbeat.json" "${coord_state_dir}/completed.json" "${coord_state_dir}/failed.json"
  write_state_marker started
  start_heartbeat_loop
fi

python3 -u "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/train_multitarget_longweather_vvvh.py" "$@"
