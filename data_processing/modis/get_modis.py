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
    date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='Y')
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
    min_lat = grid['lat'].min().values - 1.0
    max_lat = grid['lat'].max().values + 1.0
    min_lon = grid['lon'].min().values - 1.0
    max_lon = grid['lon'].max().values + 1.0
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    #bounding_box = (-180.0,0.0,0.0,90.0)
    print('bounding box:', bounding_box)
    # get the lat/lon bounds of this
    #start_date = '2003-01-01'
    #end_date = '2003-12-31'
    for d,date in enumerate(date_range):
        out_dir = os.path.join(
            '/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_earthaccess',
            date.strftime("%Y")
        )
        this_start = pd.to_datetime(f'{date.strftime("%Y")}-01-01')
        this_end = pd.to_datetime(f'{date.strftime("%Y")}-12-31')
        this_year_days = pd.date_range(start=this_start, end=this_end, freq='D')
        tiles_v = [4,4,4,4,5,5,5,5,6,6]
        tiles_h = [8,9,10,11,7,8,9,10,8,9]
        data_links = []
        # create directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        results_data = earthaccess.search_data(
            short_name='MCD43A4',
            version='061',
            temporal=(this_start,this_end),
            bounding_box=bounding_box,
        )
        data_links = []
        num_desired_links = len(this_year_days) * len(tiles_v)
        for date in this_year_days:
            this_date_strf = f'A{date.strftime("%Y%j")}'
            for v,h in zip(tiles_v, tiles_h):
                found = 0
                this_tile_strf = f'h{h:02d}v{v:02d}'
                for res in results_data:
                    this_urls = res.data_links()
                    for this_url in this_urls:
                        if this_date_strf in this_url and this_tile_strf in this_url and this_url.endswith('.hdf'):
                            data_links.append(this_url)
                            found = 1
                        if found:
                            break
                if not found:
                    print(f"Warning: No data link found for {this_date_strf} {this_tile_strf}")
        if len(data_links) != num_desired_links:
            print(f"Issue:: {len(data_links)} of {num_desired_links} desired data links found.")
            sys.exit()
        else:
            print(f'Found {len(data_links)} data links, as desired.')
        # now for quality results
        results_quality = earthaccess.search_data(
            short_name='MCD43A2',
            version='061',
            temporal=(this_start,this_end),
            bounding_box=bounding_box
        )
        quality_links = []
        for date in this_year_days:
            this_date_strf = f'A{date.strftime("%Y%j")}'
            for v,h in zip(tiles_v, tiles_h):
                found = 0
                this_tile_strf = f'h{h:02d}v{v:02d}'
                for res in results_quality:
                    this_urls = res.data_links()
                    #sys.exit()
                    for this_url in this_urls:
                        if this_date_strf in this_url and this_tile_strf in this_url and this_url.endswith('.hdf'):
                            quality_links.append(this_url)
                            found = 1
                        if found:
                            break
                if not found:
                    print(f"Warning: No data link found for {this_date_strf} {this_tile_strf}")
        if len(quality_links) != num_desired_links:
            print(f"Issue:: {len(quality_links)} of {num_desired_links} desired quality links found.")
            sys.exit()
        else:
            print(f'Found {len(quality_links)} quality links, as desired.')
        files_data = earthaccess.download(
            data_links,
            out_dir,
            threads=16,
        )
        files_quality = earthaccess.download(
            links_quality,
            out_dir,
            threads=16
        )

if __name__ == "__main__":
    main()
