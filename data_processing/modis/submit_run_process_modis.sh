#!/bin/bash
# script to submit many jobs to process modis data so that it can be
# parallelized in time

# for all the files that we want to download
start_date="2022-10-01"
end_date="2022-10-31" # inclusive
quality_flag="3"
out_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_processed_daily_w_quality/quality_${quality_flag}"
# delta between each request submitted
day_delta=0
month_delta=1
year_delta=0
#initialize the current date
current_date=$(date -I -d "$start_date")
final_date=$(date -I -d "$end_date")
# loop through dates and start the processing
while [[ "$current_date" < "$final_date" ]]; do
    next_date=$(date -I -d "$current_date + ${year_delta} year + ${month_delta} month + ${day_delta} day - 1 day")
    # clamp next date to final date if goes beyond
    if [[ "$next_date" > "$final_date" ]]; then
        next_date="$final_date"
    fi
    echo "Submitting job for $current_date to $next_date"
    sbatch run_process_modis.sh "$current_date" "$next_date" "$out_dir" "$quality_flag"
    # increment the current date
    current_date=$(date -I -d "$next_date + 1 day")
done
