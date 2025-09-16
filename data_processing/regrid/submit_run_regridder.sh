#!/bin/bash
# script to submit many jobs to process modis data so that it can be
# parallelized in time

# for all the files that we want to download
start_month="2016-01"
end_month="2022-12" # inclusive
src_top_level_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_raw_from_gee_api"
target_top_level_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_regrid"
target_grid_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/grid/epsg5070_500m_westUS_grid.nc4"
src_crs="EPSG:4326"
target_crs="EPSG:5070"
chunk_buffer=750
fill_value=-9999
# delta between each request submitted
month_delta=0
year_delta=1
#initialize the current date
current_date=$(date -d "$start_month-01" +%Y-%m-%d)
final_date=$(date -d "$end_month-01 +1 month" +%Y-%m-%d) # exclusive
while [[ "$current_date" < "$final_date" ]]; do
    # get the year and month from the current date
    year=$(date -d "$current_date" +%Y)
    month=$(date -d "$current_date" +%m)
    echo "Submitting job for $current_date"
    if [[ $month_delta > 0 ]]; then
        src_dir="${src_top_level_dir}/${year}/${month}"
        target_dir="${target_top_level_dir}/${year}/${month}"
    else
        src_dir="${src_top_level_dir}/${year}"
        target_dir="${target_top_level_dir}/${year}"
    fi
    # submit the job
    sbatch run_regridder.sh "$target_grid_dir" "$src_dir" "$target_dir" "$src_crs" "$target_crs" "$chunk_buffer" "$fill_value"
    # increment the current date by one month
    current_date=$(date -d "$current_date + ${year_delta} year + ${month_delta} month" +%Y-%m-%d)
done
