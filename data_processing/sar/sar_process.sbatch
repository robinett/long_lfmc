#!/usr/bin/env bash
#SBATCH --job-name=sar_process
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/sar_daily_%A_%a.out
#SBATCH --error=logs/sar_daily_%A_%a.err
#SBATCH -p serc
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

# Expect these from sbatch --export in submit script
: "${PY_SCRIPT:?Missing PY_SCRIPT}"
: "${START_DATE:?Missing START_DATE}"
: "${END_DATE:?Missing END_DATE}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_NUM="${TASK_ID}"   # pass through to python as job_num

mkdir -p logs

# environment activation

source /home/users/trobinet/uv_activations/activate_lfmc_process.sh

# Compute this task's 6-month range
chunk_start="$(date -d "${START_DATE} + $((6 * TASK_ID)) months" +%F)"
chunk_end="$(date -d "${chunk_start} + 6 months - 1 day" +%F)"

# Clamp to END_DATE (today)
if [[ "$(date -d "$chunk_end" +%s)" -gt "$(date -d "$END_DATE" +%s)" ]]; then
  chunk_end="$END_DATE"
fi

# If chunk_start is after END_DATE, exit cleanly (shouldn't happen if submit script is right)
if [[ "$(date -d "$chunk_start" +%s)" -gt "$(date -d "$END_DATE" +%s)" ]]; then
  echo "chunk_start $chunk_start is after END_DATE $END_DATE; nothing to do."
  exit 0
fi

echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID TASK_ID=$TASK_ID"
echo "[$(date)] Running range: $chunk_start -> $chunk_end"
echo "[$(date)] PY_SCRIPT=$PY_SCRIPT"
echo "[$(date)] job_num=$JOB_NUM"

python3 "$PY_SCRIPT" \
  --start_date "$chunk_start" \
  --end_date "$chunk_end" \
  --job_num "$JOB_NUM"

echo "[$(date)] Done."
