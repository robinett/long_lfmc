#!/bin/bash

# for all the files taht we want to download
start_date="2021-01-01"
end_date="2024-01-01" # not inclusive
# delta between each request submitted
day_delta=0
month_delta=2
year_delta=0
#initialize the current date
current_date=$(date -I -d "$start_date")
final_date=$(date -I -d "$end_date")
# loop through dates and start the download script
while [[ "$current_date" < "$final_date" ]]; do
    next_date=$(date -I -d "$current_date + ${year_delta} year + ${month_delta} month + ${day_delta} day")
    # clamp next date to final date if goes beyond
    if [[ "$next_date" > "$final_date" ]]; then
        next_date="$final_date"
    fi
    echo "Submitting job for $current_date to $next_date"
    sbatch get_daymet.sh "$current_date" "$next_date"
    # increment the current date
    current_date=$next_date
done
