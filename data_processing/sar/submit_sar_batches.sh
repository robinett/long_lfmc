#!/bin/bash

set -euo pipefail

# Usage:
#   bash submit_sar_batches.sh /path/to/your_script.py
#
# Example:
#   bash submit_sar_batches.sh /oak/stanford/groups/konings/trobinet/long_lfmc/data_processing/sar/get_sar_raw.py

PY_SCRIPT="${1:?Pass the path to the python script as arg 1}"

START="2021-01-01"
END="2021-08-31"
#END="$(date +%F)"   # "present" = today in local cluster time
CHUNK_SIZE=1 # how to chunk up the date range (in months)

# Count how many 6-month chunks we need.
# We step in 6-month increments starting at START until START_i > END.
n=0
cur="$START"
while true; do
  if [[ "$(date -d "$cur" +%s)" -gt "$(date -d "$END" +%s)" ]]; then
    break
  fi
  n=$((n+1))
  cur="$(date -d "$cur + $CHUNK_SIZE months" +%F)"
done

if [[ "$n" -le 0 ]]; then
  echo "No chunks to submit (START=$START, END=$END)."
  exit 1
fi

echo "Submitting $n jobs from $START to $END"
echo "Python script: $PY_SCRIPT"

# Submit as a job array: task IDs 0..n-1
sbatch \
  --export=ALL,PY_SCRIPT="$PY_SCRIPT",START_DATE="$START",END_DATE="$END",CHUNK_SIZE="$CHUNK_SIZE" \
  --array=0-$((n-1))%10 \
  sar_process.sbatch

