#!/bin/bash

# ====== CONFIGURE THIS SECTION ======
main_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/daymet/daymet_download"
vars_to_check=("dayl" "prcp" "srad" "swe" "tmax" "tmin" "vp")
start_date="2003-01-01"
end_date="2023-12-31"
# ====================================

# Function to loop over dates
date_loop() {
    local start=$1
    local end=$2
    local current=$start

    while [ "$current" != "$(date -I -d "$end + 1 day")" ]; do
        echo "$current"
        current=$(date -I -d "$current + 1 day")
    done
}

# Check files
for var in "${vars_to_check[@]}"; do
    echo "Missing files for variable '$var':"
    missing_any=false

    for date in $(date_loop "$start_date" "$end_date"); do
        y=$(date -d "$date" +%Y)
        m=$(date -d "$date" +%m)
        d=$(date -d "$date" +%d)
        file_path="$main_dir/$var/${y}/${m}/${var}_${y}_${m}_${d}.nc"

        if [ ! -f "$file_path" ]; then
            echo "  $file_path"
            missing_any=true
        fi
    done

    if [ "$missing_any" = false ]; then
        echo "  None! All files present."
    fi

    echo
done

