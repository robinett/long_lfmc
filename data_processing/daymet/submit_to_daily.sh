#!/bin/bash

start_year=2000
end_year=2025

for year in $(seq $start_year $end_year); do
    echo "Submitting job for year $year"
    sbatch run_to_daily.sh "$year"
done
