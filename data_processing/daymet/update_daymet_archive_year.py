#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import earthaccess
import pandas as pd
import xarray as xr

REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
GRID_PATH = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
DAYMET_EARTHACCESS_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess')
DAYMET_DAILY_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess_daily')
DAYMET_REGRID_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid')
DAYMET_ARCHIVE_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars.zarr')
DAYMET_VARS = ['tmax', 'tmin', 'prcp', 'vp', 'swe', 'srad']
DAYMET_PROBE_VARS = ['tmax', 'vp']
DAYMET_URL_TEMPLATE = (
    'https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/'
    'data/daymet_v4_daily_na_{var}_{year}.nc'
)
DAYMET_CRS = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
TARGET_CRS = 'EPSG:5070'


def parse_args():
    parser = argparse.ArgumentParser(description='Download, split, regrid, and append one archive-quality Daymet year.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--grid_path', type=Path, default=GRID_PATH)
    parser.add_argument('--earthaccess_root', type=Path, default=DAYMET_EARTHACCESS_ROOT)
    parser.add_argument('--daily_root', type=Path, default=DAYMET_DAILY_ROOT)
    parser.add_argument('--regrid_root', type=Path, default=DAYMET_REGRID_ROOT)
    parser.add_argument('--archive_zarr', type=Path, default=DAYMET_ARCHIVE_ZARR)
    parser.add_argument('--chunk_buffer', type=int, default=200)
    parser.add_argument('--chunk_size', type=int, default=2000)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def archive_has_year(archive_zarr: Path, year: int) -> bool:
    if not archive_zarr.exists():
        return False
    ds = xr.open_zarr(archive_zarr)
    times = pd.DatetimeIndex(pd.to_datetime(ds['time'].values)).normalize()
    year_times = times[times.year == int(year)]
    if len(year_times) == 0:
        return False
    expected_days = 366 if pd.Timestamp(f'{year}-01-01').is_leap_year else 365
    return len(year_times.unique()) >= expected_days


def _probe_remote_daymet_var(session, year: int, var_name: str) -> bool:
    url = DAYMET_URL_TEMPLATE.format(var=var_name, year=year)
    try:
        response = session.head(url, allow_redirects=True, timeout=30)
        if response.ok:
            return True
        response = session.get(url, stream=True, timeout=30)
        try:
            return response.ok
        finally:
            response.close()
    except Exception:
        return False


def remote_daymet_year_available(year: int) -> bool:
    earthaccess.login()
    session = earthaccess.get_requests_https_session()
    return all(_probe_remote_daymet_var(session, year, var_name) for var_name in DAYMET_PROBE_VARS)


def check_year_available(year: int, archive_zarr: Path, earthaccess_root: Path):
    del earthaccess_root
    if archive_has_year(archive_zarr, year):
        return True, 'archive_zarr_already_contains_year'
    if remote_daymet_year_available(year):
        return True, f'remote_probe_ok:{"-".join(DAYMET_PROBE_VARS)}'
    return False, f'remote_probe_failed:{"-".join(DAYMET_PROBE_VARS)}'


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def ensure_annual_downloads(year: int, earthaccess_root: Path):
    year_dir = earthaccess_root / str(year)
    missing = [
        var for var in DAYMET_VARS
        if not (year_dir / f'daymet_v4_daily_na_{var}_{year}.nc').exists()
    ]
    if len(missing) == 0:
        print(f'All Daymet annual files already present for {year} in {year_dir}')
        return
    print(f'Missing annual Daymet files for {year}: {missing}')
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/daymet/get_daymet.py',
        '--start_date', f'{year}-01-01',
        '--end_date', f'{year}-12-31',
        '--output_root', earthaccess_root,
        '--skip_existing',
    ])


def ensure_daily_split(year: int, earthaccess_root: Path, daily_root: Path):
    year_dir = earthaccess_root / str(year)
    annual_files = sorted(year_dir.glob('daymet_v4_daily_na_*_*.nc'))
    if len(annual_files) == 0:
        raise FileNotFoundError(f'No annual Daymet files found for {year} in {year_dir}')

    daily_year_dir = daily_root / str(year)
    expected_min = len(annual_files) * (366 if pd.Timestamp(f'{year}-01-01').is_leap_year else 365)
    existing_daily = sorted(daily_year_dir.glob('daymet_v4_daily_na_*_*.nc')) if daily_year_dir.exists() else []
    if len(existing_daily) >= expected_min:
        print(f'Daymet daily split already looks populated for {year}: {len(existing_daily)} files')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/daymet/to_daily.py',
        '--year', str(year),
        '--input_root', earthaccess_root,
        '--output_root', daily_root,
        '--skip_existing',
        '--no-plot_first_day',
    ])


def ensure_regridded_year(year: int, grid_path: Path, daily_root: Path, regrid_root: Path, chunk_size: int, chunk_buffer: int):
    src_dir = daily_root / str(year)
    target_dir = regrid_root / str(year)
    if not src_dir.exists():
        raise FileNotFoundError(f'Daymet daily directory missing for {year}: {src_dir}')
    src_count = len(list(src_dir.glob('daymet_v4_daily_na_*_*.nc')))
    regrid_count = len(list(target_dir.glob('daymet_v4_daily_na_*_regridded.nc*'))) if target_dir.exists() else 0
    if regrid_count >= src_count and src_count > 0:
        print(f'Daymet regridded output already looks populated for {year}: {regrid_count} files')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/regrid/main.py',
        '--target_grid', grid_path,
        '--src_dir', src_dir,
        '--target_dir', target_dir,
        '--src_crs', DAYMET_CRS,
        '--target_crs', TARGET_CRS,
        '--chunk_size', str(chunk_size),
        '--chunk_buffer', str(chunk_buffer),
    ])


def append_archive_year(year: int, archive_zarr: Path, regrid_root: Path):
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/convert_to_zarr/append_daymet_archive_year.py',
        '--root', regrid_root,
        '--out_zarr', archive_zarr,
        '--year', str(year),
    ])


def main():
    args = parse_args()
    year = int(args.year)
    print(f'Updating archive Daymet for year {year}')
    available, reason = check_year_available(year, args.archive_zarr, args.earthaccess_root)
    if args.check_only:
        print(f'Daymet availability check for {year}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)
    if archive_has_year(args.archive_zarr, year):
        print(f'Archive Daymet zarr already contains year {year}; nothing to do')
        return
    ensure_annual_downloads(year, args.earthaccess_root)
    ensure_daily_split(year, args.earthaccess_root, args.daily_root)
    ensure_regridded_year(year, args.grid_path, args.daily_root, args.regrid_root, args.chunk_size, args.chunk_buffer)
    append_archive_year(year, args.archive_zarr, args.regrid_root)
    print(f'Finished archive Daymet update for year {year}')


if __name__ == '__main__':
    main()
