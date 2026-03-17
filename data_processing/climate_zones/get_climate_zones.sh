#!/usr/bin/env bash

set -euo pipefail

scratch_root="/scratch/users/trobinet/long_lfmc/final_lfmc/climate_zones"
zip_path="${scratch_root}/koppen_geiger_tif.zip"
raw_dir="${scratch_root}/raw"

mkdir -p "${raw_dir}"

wget --header="User-Agent: Mozilla/5.0" \
  "https://ndownloader.figshare.com/files/61012822" \
  -O "${zip_path}"

unzip -oq "${zip_path}" -d "${raw_dir}"

echo "Downloaded climate zones archive to ${zip_path}"
echo "Extracted climate zone files under ${raw_dir}"
find "${raw_dir}" -type f | sort
