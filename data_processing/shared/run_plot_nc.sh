#!/bin/bash

# Script to plot any file (.nc, .nc4, .tif) by changing parameters
# If file_name is a directory, the script will plot all matching files
# in that directory and subdirectories.

file_name="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_regridded/quality_1/2003/02/modis_reflectance_20030201_regridded.nc4"
var_name="Nadir_Reflectance_Band1"
proj_in="EPSG:5070"
proj_out="EPSG:5070"
save_root="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_plots"

# Function to build output filename
get_save_name() {
  input_path="$1"
  base_name=$(basename "$input_path")
  name_no_ext="${base_name%.*}"
  echo "${save_root}/${name_no_ext}_plot.png"
}

# Check input type
if [ -d "$file_name" ]; then
  # Directory: find all matching files
  find "$file_name" -type f \( \
    -name "*.nc" -o -name "*.nc4" -o -name "*.tif" \) | while read f; do
      save_name=$(get_save_name "$f")
      echo "Plotting $f -> $save_name"
      python3 plot_nc.py --file_name "$f" --var_name "$var_name" \
        --proj_in "$proj_in" --proj_out "$proj_out" \
        --save_name "$save_name"
  done
elif [ -f "$file_name" ]; then
  # Single file
  save_name=$(get_save_name "$file_name")
  echo "Plotting $file_name -> $save_name"
  python3 plot_nc.py --file_name "$file_name" --var_name "$var_name" \
    --proj_in "$proj_in" --proj_out "$proj_out" \
    --save_name "$save_name"
else
  echo "Error: $file_name is not a valid file or directory"
  exit 1
fi

