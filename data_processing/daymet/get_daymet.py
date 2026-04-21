import argparse
import os
from pathlib import Path

import earthaccess
import pandas as pd
import xarray as xr

DAYMET_VARS = ['tmax', 'prcp', 'vp', 'swe', 'srad']
DEFAULT_GRID_PATH = '/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4'
DEFAULT_OUTPUT_ROOT = '/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess'
URL_TEMPLATE = (
    'https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/'
    'data/daymet_v4_daily_na_{var}_{year}.nc'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--grid_path', type=str, default=DEFAULT_GRID_PATH)
    parser.add_argument('--skip_existing', action='store_true')
    return parser.parse_args()


def requested_years(start_date: str, end_date: str):
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    if end_year < start_year:
        raise ValueError('end_date must be on or after start_date')
    return list(range(start_year, end_year + 1))


def download_year(year: int, output_root: str, skip_existing: bool = False):
    out_dir = Path(output_root) / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    links = []
    missing_targets = []
    for var in DAYMET_VARS:
        target_path = out_dir / f'daymet_v4_daily_na_{var}_{year}.nc'
        if skip_existing and target_path.exists():
            print(f'[SKIP] {target_path.name} already exists')
            continue
        links.append(URL_TEMPLATE.format(var=var, year=year))
        missing_targets.append(str(target_path))

    if len(links) == 0:
        print(f'All Daymet annual files already exist for {year}; nothing to download')
        return []

    print(f'Downloading {len(links)} Daymet annual files for {year} into {out_dir}')
    files = earthaccess.download(links, str(out_dir), threads=8, show_progress=True)
    print(f'Downloaded {len(files)} files for {year}')
    return files


def main():
    args = parse_args()
    print('start date:', args.start_date)
    print('end date:', args.end_date)
    print('output_root:', args.output_root)
    print('skip_existing:', args.skip_existing)

    print('logging in')
    earthaccess.login()

    grid = xr.open_dataset(args.grid_path)
    min_lat = grid['lat'].min().values - 1.0
    max_lat = grid['lat'].max().values + 1.0
    min_lon = grid['lon'].min().values - 1.0
    max_lon = grid['lon'].max().values + 1.0
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    print('bounding box:', bounding_box)

    years = requested_years(args.start_date, args.end_date)
    print('requested years:', years)
    for year in years:
        download_year(year, output_root=args.output_root, skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
