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
            short_name='MCD43A4'
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
    min_lat = grid['lat'].min().values - 0.5
    max_lat = grid['lat'].max().values + 0.5
    min_lon = grid['lon'].min().values - 0.5
    max_lon = grid['lon'].max().values + 0.5
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    print('bounding box:', bounding_box)
    # get the lat/lon bounds of this
    #start_date = '2003-01-01'
    #end_date = '2003-12-31'
    out_dir = os.path.join(
        '/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_earthaccess',
        start_date_pd.strftime("%Y")
    )
    # create directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    results = earthaccess.search_data(
        short_name='MCD43A4',
        version='061',
        temporal=(start_date,end_date),
        bounding_box=bounding_box
    )
    links = []
    for r in results:
        this_urls = r.data_links()
        # keep only the .hdf
        this_urls = [url for url in this_urls if url.endswith('.hdf')]
        links.extend(this_urls)
    if not links:
        print('no data found')
        sys.exit()
    print('found {} data results'.format(len(links)))
    files = earthaccess.download(
        links,
        out_dir
    )
    results_quality = earthaccess.search_data(
        short_name='MCD43A2',
        version='061',
        temporal=(start_date,end_date),
        bounding_box=bounding_box
    )
    links_quality = []
    for r in results_quality:
        this_urls = r.data_links()
        # keep only the .hdf
        this_urls = [url for url in this_urls if url.endswith('.hdf')]
        links_quality.extend(this_urls)
    print('found {} quality results'.format(len(links_quality)))
    files = earthaccess.download(
        links_quality,
        out_dir
    )

if __name__ == "__main__":
    main()
