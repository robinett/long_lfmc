#!/bin/bash

# This script removes all files in the current directory that are not fully downloaded.

main_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/daymet_download"
directories=(
    "${main_dir}/dayl"
    "${main_dir}/prcp"
    "${main_dir}/srad"
    "${main_dir}/swe"
    "${main_dir}/tmax"
    "${main_dir}/tmin"
    "${main_dir}/vp"
)
# pick threshold size
threshold=$((124*1024*1024)) # 124MB
# loop through each directory
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"
    find "$dir" -type f -size -"${threshold}c" -print -delete
done
