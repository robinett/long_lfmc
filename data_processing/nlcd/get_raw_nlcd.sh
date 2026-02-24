#!/usr/bin/env bash
set -euo pipefail

# Config
URL_FILE="/home/users/trobinet/long_lfmc/data_processing/nlcd/raw_download_urls.txt"
OUT_DIR="/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_raw"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# 👇 key change: add `|| [[ -n $raw ]]` to process last line even w/o newline
while IFS= read -r raw || [[ -n $raw ]]; do
  # strip stray carriage return if present
  url=${raw%$'\r'}

  # skip blank lines and comments
  [[ -z "${url// }" || "${url}" =~ ^# ]] && continue

  fname=$(basename "${url}")
  tif="${fname%.zip}.tif"

  # check if .tif already exists
  if [[ -f "${tif}" ]]; then
    echo "[SKIP] ${tif} already exists"
    continue
  fi

  echo "[DL] ${fname}"
  wget -c -O "${fname}" "${url}"

  dest="${fname%.zip}"
  echo "[UNZIP] ${fname} -> ${dest}/"
  mkdir -p "${dest}"
  unzip -oq "${fname}" -d "${dest}"

  found_tif=$(find "${dest}" -type f -name "*.tif" | head -n 1)
  if [[ -n "${found_tif}" ]]; then
    mv "${found_tif}" "${tif}"
    echo "[OK] Extracted ${tif}"
  else
    echo "[WARN] No .tif found in ${fname}"
  fi

  rm -rf "${dest}" "${fname}"
done < "${URL_FILE}"

echo "[DONE]"
