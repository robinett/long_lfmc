#!/bin/bash

# This script sorts files in the current directory into subdirectories
# based on the inclusive of various timestamps in their filenames.
# There are lots of different ways to make small changes to ammend this to
# different directory structures, time periods, etc.
# for all the files that we want to download
start_month="2016-04" # inclusive
end_month="2022-01" # inclusive
file_date_format="*{YYYY}-{MM}*"
# do we want yearly
# top level dir where we are organizing files from
top_level_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_raw"
# set the current and final dates
current_date=$(date -d "$start_month-01" +%Y-%m-%d)
final_date=$(date -d "$end_month-01 +1 month" +%Y-%m-%d) # exclusive
while [[ "$current_date" < "$final_date" ]]; do
    # get the year and month from the current date
    year=$(date -d "$current_date" +%Y)
    month=$(date -d "$current_date" +%m)
    echo "processing $year-$month"
    # create the directory for the current year and month
    dest_dir="$top_level_dir/$year/$month"
    mkdir -p "$dest_dir"
    pattern="*${year}-${month}*"
    find "$top_level_dir" -maxdepth 1 -type f -name "$pattern" -exec mv {} "$dest_dir/" \;
    # increment the current date by one month
    current_date=$(date -d "$current_date +1 month" +%Y-%m-%d)
done
