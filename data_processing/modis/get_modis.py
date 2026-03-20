import argparse
import os
from pathlib import Path

import earthaccess
import pandas as pd
import xarray as xr


DEFAULT_GRID_PATH = '/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4'
DEFAULT_OUTPUT_ROOT = '/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_earthaccess'
TILES_V = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6]
TILES_H = [8, 9, 10, 11, 7, 8, 9, 10, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--grid_path', type=str, default=DEFAULT_GRID_PATH)
    parser.add_argument('--skip_existing', action='store_true')
    return parser.parse_args()


def expected_filenames_for_month(month_days, short_name):
    expected = {}
    for date in month_days:
        date_tag = f"A{date.strftime('%Y%j')}"
        for v, h in zip(TILES_V, TILES_H):
            tile_tag = f'h{h:02d}v{v:02d}'
            fname = f'{short_name}.{date_tag}.{tile_tag}.061.hdf'
            expected[fname] = (date_tag, tile_tag)
    return expected


def filter_missing_links(out_dir: Path, links, short_name: str, month_days, skip_existing: bool):
    if not skip_existing:
        return links
    expected = expected_filenames_for_month(month_days, short_name)
    existing = {path.name for path in out_dir.glob(f'{short_name}*.hdf')}
    missing_expected = set(expected) - existing
    if not missing_expected:
        print(f'All expected {short_name} files already exist under {out_dir}; skipping download')
        return []
    filtered = [url for url in links if Path(url).name in missing_expected]
    print(f'{short_name}: existing={len(existing)} missing_expected={len(missing_expected)} download_candidates={len(filtered)}')
    return filtered


def collect_links(results, month_days, short_name: str):
    links = []
    desired = len(month_days) * len(TILES_V)
    for date in month_days:
        date_tag = f"A{date.strftime('%Y%j')}"
        for v, h in zip(TILES_V, TILES_H):
            found = False
            tile_tag = f'h{h:02d}v{v:02d}'
            for res in results:
                for url in res.data_links():
                    if date_tag in url and tile_tag in url and url.endswith('.hdf'):
                        links.append(url)
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f'Warning: No {short_name} link found for {date_tag} {tile_tag}')
    print(f'Found {len(links)} {short_name} links, out of {desired} desired.')
    return links


def main():
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)
    print('start date:', start_date)
    print('end date:', end_date)
    print('output_root:', args.output_root)
    print('skip_existing:', args.skip_existing)
    date_range = pd.date_range(start=start_date_pd, end=end_date_pd, freq='ME')

    print('logging in')
    earthaccess.login()

    print('searching for data')
    grid = xr.open_dataset(args.grid_path)
    min_lat = grid['lat'].min().values - 1.0
    max_lat = grid['lat'].max().values + 1.0
    min_lon = grid['lon'].min().values - 1.0
    max_lon = grid['lon'].max().values + 1.0
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    print('bounding box:', bounding_box)

    for month_end in date_range:
        out_dir = Path(args.output_root) / month_end.strftime('%Y')
        this_start = pd.Timestamp(f'{month_end.year}-{month_end.month:02d}-01')
        this_end = month_end
        print(f'Downloading MODIS data from {this_start.date()} to {this_end.date()}')
        month_days = pd.date_range(start=this_start, end=this_end, freq='D')
        out_dir.mkdir(parents=True, exist_ok=True)

        results_data = earthaccess.search_data(
            short_name='MCD43A4',
            version='061',
            temporal=(this_start, this_end),
            bounding_box=bounding_box,
            downloadable=True,
        )
        data_links = collect_links(results_data, month_days, 'MCD43A4')
        data_links = filter_missing_links(out_dir, data_links, 'MCD43A4', month_days, args.skip_existing)

        results_quality = earthaccess.search_data(
            short_name='MCD43A2',
            version='061',
            temporal=(this_start, this_end),
            bounding_box=bounding_box,
            downloadable=True,
        )
        quality_links = collect_links(results_quality, month_days, 'MCD43A2')
        quality_links = filter_missing_links(out_dir, quality_links, 'MCD43A2', month_days, args.skip_existing)

        if len(data_links) == 0 and len(quality_links) == 0:
            print('No files to download for this month, skipping download step.')
            continue
        if len(data_links) > 0:
            earthaccess.download(data_links, str(out_dir), threads=8, show_progress=True)
        if len(quality_links) > 0:
            earthaccess.download(quality_links, str(out_dir), threads=8, show_progress=True)


if __name__ == "__main__":
    main()
