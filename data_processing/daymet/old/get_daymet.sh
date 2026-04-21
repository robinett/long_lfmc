#!/bin/bash

#SBATCH --job-name=daymet_download  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc            # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH -C CLASS:SH3_CBASE          # Only Sherlock 3 nodes
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

# This is an example script to subset and download Daymet gridded daily data utilizing the netCDF Subset Service RESTful API 
# available through the ORNL DAAC THREDDS Data Server
#
# Daymet data and an interactive netCDF Subset Service GUI are available from the THREDDS web interface:
# https://thredds.daac.ornl.gov/thredds/catalogs/ornldaac/Regional_and_Global_Data/DAYMET_COLLECTIONS/DAYMET_COLLECTIONS.html
#
# Usage:  This is a sample script and not intended to run without user updates.
# Update the inputs under each section of "VARIABLES" for temporal, spatial, and Daymet weather variables.
# More information on Daymet NCSS gridded subset web service is found at:  https://daymet.ornl.gov/web_services
#
# The current Daymet NCSS has a size limit of 6GB for each single subset request. 
#
# Daymet dataset information including citation is available at:
# https://daymet.ornl.gov/
#
# Michele Thornton
# ORNL DAAC
# November 5, 2018
#
#################################################################################
# VARIABLES - Temporal subset - This example is set to subset the 31 days of January for each years 1980, 1981, and 1982
# Note:  The Daymet calendar is based on a standard calendar year. All Daymet years have 1 - 365 days, including leap years. For leap years, 
# the Daymet database includes leap day. Values for December 31 are discarded from leap years to maintain a 365-day year.

source ~/.bashrc

# Time should be in the form yyyy-mm-dd
start_date=$1
end_date=$2 # not inclusive

echo "Start date: $start_date"
echo "End date: $end_date"

# VARIABLES - Region - na is used a example. The complete list of regions is: na (North America), hi(Hawaii), pr(Puerto Rico)
region="na"

# VARIABLES - Daymet variables - tmin and tmax are used as examples, variables should be space separated. 
# The complete list of Daymet variables is: tmin, tmax, prcp, srad, vp, swe, dayl
#vars=("tmax" "prcp" "srad" "vp" "swe")
vars=("tmax")

# VARIABLES - Spatial subset - bounding box in decimal degrees.  
north=52.10
west=-128.00
east=-92.00 #-104
south=22.50 # 31
################################################################################
	
current_date="$start_date"
while [ "$current_date" != "$end_date" ]; do
    yr=$(date -d "$current_date" +%Y)
    mn=$(date -d "$current_date" +%m)
    day=$(date -d "$current_date" +%d)
    for var in "${vars[@]}"; do
        # make sure directory exists
        dir="/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_download/${var}/${yr}/${mn}"
        mkdir -p "$dir"
        file_path="${dir}/${var}_${yr}_${mn}_${day}.nc"
        if [ ! -f "$file_path" ]; then
	        wget -O --netrc "${file_path}" "https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_${region}_${var}_${yr}.nc?var=lat&var=lon&var=${var}&north=${north}&west=${west}&east=${east}&south=${south}&horizStride=1&time_start=${yr}-${mn}-${day}T12:00:00Z&time_end=${yr}-${mn}-${day}T12:00:00Z&timeStride=1&accept=netcdf"
            if [ -f "$file_path" ] && [ $(stat -c%s "$file_path") -gt $((120 * 1024 * 1024)) ]; then
                echo "${file_path} downloaded successfully"
            else
                echo "Error downloading ${file_path}"
            fi
        else
            echo "File already exists: ${var}_${yr}_${mn}_${day}.nc"
        fi
    done
    current_date=$(date -I -d "$current_date + 1 day")
    sleep 5
done
echo Downloads Complete
