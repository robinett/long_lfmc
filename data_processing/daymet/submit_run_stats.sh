#!/usr/bin/env bash
# submit many jobs to process MODIS data in time chunks
# MODE: "manual" = use CHUNKS below; "auto" = old loop

set -euo pipefail

MODE="manual"   # "manual" or "auto"

# ---------- manual chunks (inclusive) ----------
# Each entry: "START END" (YYYY-MM-DD YYYY-MM-DD)
# Edit/add lines to control exactly what gets submitted.
CHUNKS=(
  "2003-07-05 2003-12-31"
  "2004-01-01 2004-12-31"
  "2005-01-01 2005-12-31"
  "2006-01-01 2006-12-31"
  "2007-01-01 2007-12-31"
  "2008-01-01 2008-12-31"
  "2009-01-01 2009-12-31"
  "2010-01-01 2010-12-31"
  "2011-01-01 2011-12-31"
  "2012-01-01 2012-12-31"
  "2013-01-01 2013-12-31"
  "2014-01-01 2014-12-31"
  "2015-01-01 2015-12-31"
  "2016-01-01 2016-12-31"
  "2017-01-01 2017-12-31"
  "2018-01-01 2018-12-31"
  "2019-01-01 2019-12-31"
  "2020-01-01 2020-12-31"
  "2021-01-01 2021-12-31"
  "2022-01-01 2022-12-31"
  "2023-01-01 2023-12-31"
)

# ---------- auto mode settings (fallback) ----------
start_month="2003-07-05"
end_month="2023-12-31"   # inclusive
month_delta=0
year_delta=1

# ---------- helper: safe date to epoch ----------
to_epoch() {
  date -d "$1" +%s
}

# ---------- helper: validate YYYY-MM-DD ----------
valid_date() {
  date -d "$1" +%Y-%m-%d >/dev/null 2>&1
}

# ---------- helper: submit one chunk ----------
submit_range() {
  local s="$1"; local e="$2"
  if ! valid_date "$s" || ! valid_date "$e"; then
    echo "bad date(s): '$s' '$e'" >&2
    return 1
  fi
  if [[ $(to_epoch "$s") -gt $(to_epoch "$e") ]]; then
    echo "start after end: '$s' > '$e'" >&2
    return 1
  fi
  echo "Submitting: $s to $e"
  sbatch run_stats.sh "$s" "$e"
}

# ---------- manual mode ----------
if [[ "$MODE" == "manual" ]]; then
  for pair in "${CHUNKS[@]}"; do
    # split on space into start/end
    s="${pair%% *}"
    e="${pair##* }"
    submit_range "$s" "$e"
  done
  exit 0
fi

# ---------- auto mode (original behavior) ----------
current_date=$(date -d "$start_month" +%Y-%m-%d)
final_date=$(date -d "$end_month" +%Y-%m-%d)
next_date=$(date -d \
  "$start_month + ${year_delta} year + ${month_delta} month - 1 day" \
  +%Y-%m-%d)

while [[ $(to_epoch "$current_date") -le $(to_epoch "$final_date") ]]
do
  # clamp next_date to final_date
  if [[ $(to_epoch "$next_date") -gt $(to_epoch "$final_date") ]]; then
    next_date="$final_date"
  fi

  submit_range "$current_date" "$next_date"

  # advance window
  current_date=$(date -d \
    "$current_date + ${year_delta} year + ${month_delta} month" \
    +%Y-%m-%d)
  next_date=$(date -d \
    "$current_date + ${year_delta} year + ${month_delta} month - 1 day" \
    +%Y-%m-%d)
done
