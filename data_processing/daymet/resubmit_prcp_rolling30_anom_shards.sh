#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

mkdir -p ./logs

SHARD_COUNT=24
RUN_SCRIPT="${SCRIPT_DIR}/run_build_daymet_derived_products.sh"
PRCP_JOB_ID=19810399
FINALIZE_JOB_ID=19810405
COORD_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_vars_and_anoms_coord"

echo "Cancelling existing precipitation anomaly and finalize jobs"
scancel "${PRCP_JOB_ID}" "${FINALIZE_JOB_ID}" || true

echo "Waiting briefly for Slurm state to settle"
sleep 5

echo "Clearing existing prcp rolling30 anomaly markers"
python3 - <<'PY'
from pathlib import Path

coord_dir = Path("/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_vars_and_anoms_coord")

anomaly_dir = coord_dir / "anomaly"
direct_marker = anomaly_dir / "prcp_rolling30_anom.json"
if direct_marker.exists():
    direct_marker.unlink()
for shard_marker in anomaly_dir.glob("prcp_rolling30_anom__shard_*_of_*.json"):
    shard_marker.unlink()
workflow_finalize = coord_dir / "workflow" / "finalize_anomaly.json"
if workflow_finalize.exists():
    workflow_finalize.unlink()
PY

sbatch_submit() {
    sbatch --parsable --chdir "$SCRIPT_DIR" "$@"
}

shard_job_ids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    shard_job_id=$(
        sbatch_submit \
            --job-name="daymet_anom_prcp_r30_s${shard_index}" \
            --output="./logs/daymet_anom_prcp_r30_s${shard_index}_%j.out" \
            --error="./logs/daymet_anom_prcp_r30_s${shard_index}_%j.err" \
            "$RUN_SCRIPT" \
            --mode build-anomaly-var \
            --var prcp_rolling30_anom \
            --shard-index "${shard_index}" \
            --num-shards "${SHARD_COUNT}"
    )
    shard_job_ids+=("${shard_job_id}")
    echo "Submitted shard ${shard_index}/${SHARD_COUNT}: ${shard_job_id}"
done

shard_dep=$(IFS=:; echo "${shard_job_ids[*]}")
finalize_job_id=$(
    sbatch_submit \
        --dependency=afterok:"${shard_dep}" \
        --job-name=daymet_finalize_anomaly \
        --output=./logs/daymet_finalize_anomaly_%j.out \
        --error=./logs/daymet_finalize_anomaly_%j.err \
        "$RUN_SCRIPT" \
        --mode finalize-anomaly
)

echo "Submitted replacement finalize-anomaly job: ${finalize_job_id}"
echo "Sharded precipitation anomaly resubmission complete."
