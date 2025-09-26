#!/bin/bash
# script to submit many jobs to process modis data so that it can be
# parallelized in time

# for all the files that we want to download
start_month="2003-04-01"
end_month="2023-12-31" # inclusive
# delta between each request submitted
month_delta=0
year_delta=1
#initialize the current date
current_date=$(date -d "$start_month" +%Y-%m-%d)
next_date=$(date -d "$start_month + ${year_delta} year + ${month_delta} month - 1 day" +%Y-%m-%d)
final_date=$(date -d "$end_month" +%Y-%m-%d)
while [[ $(date -d "$current_date" +%s) -le $(date -d "$final_date" +%s) ]]; do
    # if next date is greater than final date, set it to final date
    if [[ $(date -d "$next_date" +%s) -gt $(date -d "$final_date" +%s) ]]; then
        next_date="$final_date"
    fi
    # get the year and month from the current date
    echo "Submitting job for $current_date to $next_date"
    # submit the job
    sbatch run_compile.sh "$current_date" "$next_date"
    # now set the current date to be the final date
    current_date=$(date -d "$current_date + ${year_delta} year + ${month_delta} month" +%Y-%m-%d)
    # and the final date to be the current date + delta
    next_date=$(date -d "$current_date + ${year_delta} year + ${month_delta} month - 1 day" +%Y-%m-%d)
done


