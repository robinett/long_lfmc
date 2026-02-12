#!/bin/bash

# for all the files taht we want to download
start_date="2000-01-01"
end_date="2025-12-31" # inclusive
# delta between each request submitted
day_delta=0
month_delta=0
year_delta=1
#initialize the current date
current_date=$(date -I -d "$start_date")
final_date=$(date -I -d "$end_date")
# loop through dates and start the download script
while [[ "$current_date" < "$final_date" ]]; do
    next_date=$(date -I -d "$current_date + ${year_delta} year + ${month_delta} month + ${day_delta} day - 1 day")
    # clamp next date to final date if goes beyond
    if [[ "$next_date" > "$final_date" ]]; then
        next_date="$final_date"
    fi
    echo "Submitting job for $current_date to $next_date"
    sbatch run_get_modis.sh "$current_date" "$next_date"
    # increment the current date
    current_date=$(date -I -d "$next_date + 1 day")
done
