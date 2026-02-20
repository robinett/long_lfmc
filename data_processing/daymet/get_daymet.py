import earthaccess
import sys
import argparse
import xarray as xr
import os
import pandas as pd

def main():
    # pass the start and end dates in from the submitting script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date in YYYY-MM-DD format",
        required=True
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format",
        required=True
    )
    start_date = parser.parse_args().start_date
    end_date = parser.parse_args().end_date
    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='Y')
    print('start date:', start_date)
    print('end date:', end_date)
    # do we need to login?
    login = True
    if login:
        print('logging in')
        auth = earthaccess.login()
    # do we want to see collections?
    see_collections = False
    if see_collections:
        print('getting collections')
        collections = earthaccess.search_datasets(
            short_name='Daymet_Daily_V4R1_2129'
        )
        # Step 2: See how many were returned
        print('Found {} collection(s)'.format(len(collections)))
        # Step 3: Inspect metadata for the first collection
        collection = collections[0]
        print('first collection:')
        print(collection)
        sys.exit()
        # Print summary information
        print("Title:", collection["umm"]["EntryTitle"])
        print("Version:", collection["umm"]["Version"])
        print("Start Date:", collection["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
        print("End Date:",collection["umm"]["TemporalExtent"]["RangeDateTime"]["EndingDateTime"])
    # download the data
    print('searching for data')
    #bounding_box = (-124.73,25.85,-93.51,49.00)
    grid = xr.open_dataset(
        '/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4'
    )
    min_lat = grid['lat'].min().values - 1.0
    max_lat = grid['lat'].max().values + 1.0
    min_lon = grid['lon'].min().values - 1.0
    max_lon = grid['lon'].max().values + 1.0
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    print('bounding box:', bounding_box)
    daymet_vars = ['tmax','tmin','prcp','vp','swe','srad']
    # get the lat/lon bounds of this
    #start_date = '2003-01-01'
    #end_date = '2003-12-31'
    #results = earthaccess.search_data(
    #    short_name='Daymet_Daily_V4R1_2129',
    #    #temporal=(start_date,end_date),
    #    #bounding_box=bounding_box
    #)
    result_template = 'https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/daymet_v4_daily_na_{var}_{year}.nc'
    for d,date in enumerate(date_range):
        print(f'Downloading data for year {date.strftime("%Y")}')
        links = []
        for var in daymet_vars:
            links.append(result_template.format(var=var, year=date.strftime('%Y')))
        out_dir = os.path.join(
            '/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess',
            date.strftime("%Y")
        )
        # create directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        files = earthaccess.download(
            links,
            out_dir,
            threads=8,
            show_progress=True
        )

if __name__ == "__main__":
    main()
