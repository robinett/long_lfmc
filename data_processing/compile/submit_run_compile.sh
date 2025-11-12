#!/usr/bin/env bash
# Submit many jobs to process MODIS data in parallel over time windows.
# MODE 1: Manual ranges (edit PERIODS or set PERIODS_FILE)
# MODE 2: Auto ranges (set start/end + deltas like before)

set -euo pipefail

# ---------- MODE 1: MANUAL RANGES ----------
# Either hard-code ranges here...
# Format: "START,END" per entry (inclusive). ISO dates: YYYY-MM-DD
# Example:
PERIODS=(
  "2003-08-01,2003-12-31"
  "2004-01-01,2004-12-31"
  "2005-01-01,2005-12-31"
  "2006-01-01,2006-12-31"
  "2007-01-01,2007-12-31"
  "2008-01-01,2008-12-31"
  "2009-01-01,2009-12-31"
  "2010-01-01,2010-12-31"
  "2011-01-01,2011-12-31"
  "2012-01-01,2012-12-31"
  "2013-01-01,2013-12-31"
  "2014-01-01,2014-12-31"
  "2015-01-01,2015-12-31"
  "2016-01-01,2016-12-31"
  "2017-01-01,2017-12-31"
  "2018-01-01,2018-12-31"
  "2019-01-01,2019-12-31"
  "2020-01-01,2020-12-31"
  "2021-01-01,2021-12-31"
  "2022-01-01,2022-12-31"
  "2023-01-01,2023-12-31"
)
#PERIODS=()

# ...or provide a file with one "START,END" per line; lines starting with # are ignored.
# Example content:
# 2003-04-01,2004-03-31
# 2004-04-01,2006-12-31
PERIODS_FILE=""  # e.g., "./periods.csv"

# The script/job you want to submit:
SUBMIT_SCRIPT="run_compile.sh"

# ---------- MODE 2: AUTO RANGES (used only if no manual ranges found) ----------
start_month="2003-04-01"
end_month="2023-12-31"   # inclusive
month_delta=0
year_delta=1

# ---------- helpers ----------
parse_periods_file() {
  local file="$1"
  mapfile -t PERIODS < <(grep -v '^[[:space:]]*#' "$file" | awk 'NF' | sed 's/[[:space:]]//g')
}

validate_date() {
  local d="$1"
  if ! date -d "$d" > /dev/null 2>&1; then
    echo "Error: invalid date '$d'" >&2
    exit 1
  fi
}

submit_range() {
  local start="$1"
  local end="$2"
  validate_date "$start"
  validate_date "$end"
  # Ensure start <= end
  if [[ $(date -d "$start" +%s) -gt $(date -d "$end" +%s) ]]; then
    echo "Error: start > end for range $start,$end" >&2
    exit 1
  fi
  echo "Submitting job for $start to $end"
  sbatch "$SUBMIT_SCRIPT" "$start" "$end"
}

# ---------- main ----------
# Prefer manual: file > inline array
if [[ -n "${PERIODS_FILE}" ]]; then
  if [[ ! -f "$PERIODS_FILE" ]]; then
    echo "Error: PERIODS_FILE not found: $PERIODS_FILE" >&2
    exit 1
  fi
  parse_periods_file "$PERIODS_FILE"
fi

if (( ${#PERIODS[@]} > 0 )); then
  # Manual mode
  for rng in "${PERIODS[@]}"; do
    # Split "START,END"
    start="${rng%%,*}"
    end="${rng##*,}"
    if [[ -z "$start" || -z "$end" ]]; then
      echo "Error: bad range '$rng' (expected START,END)" >&2
      exit 1
    fi
    submit_range "$start" "$end"
  done
else
  # Auto mode (original behavior)
  current_date=$(date -d "$start_month" +%Y-%m-%d)
  final_date=$(date -d "$end_month" +%Y-%m-%d)
  next_date=$(date -d "$start_month + ${year_delta} year + ${month_delta} month - 1 day" +%Y-%m-%d)

  while [[ $(date -d "$current_date" +%s) -le $(date -d "$final_date" +%s) ]]; do
    if [[ $(date -d "$next_date" +%s) -gt $(date -d "$final_date" +%s) ]]; then
      next_date="$final_date"
    fi
    submit_range "$current_date" "$next_date"
    current_date=$(date -d "$current_date + ${year_delta} year + ${month_delta} month" +%Y-%m-%d)
    next_date=$(date -d "$current_date + ${year_delta} year + ${month_delta} month - 1 day" +%Y-%m-%d)
  done
fi
