#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import earthaccess
import pandas as pd
import xarray as xr
import yaml


REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'
GRID_PATH = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
DAYMET_DATASET_ID = 1904
DAYMET_METADATA_URL = 'https://daac.ornl.gov/cgi-bin/metadata/granule.py'
DAYMET_DATAPOOL = 'https://data.ornldaac.earthdata.nasa.gov/protected'
DAYMET_VARS = ['tmax', 'tmin', 'prcp', 'vp', 'swe', 'srad']
DAYMET_PROBE_VARS = ['tmax', 'vp']
DAYMET_CRS = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
TARGET_CRS = 'EPSG:5070'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download, split, regrid, and append one monthly-latency Daymet month into the canonical zarr.'
    )
    parser.add_argument('--month', type=str, required=True, help='Target month in YYYY-MM format')
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=GRID_PATH)
    parser.add_argument('--earthaccess_root', type=Path, default=None)
    parser.add_argument('--daily_root', type=Path, default=None)
    parser.add_argument('--regrid_root', type=Path, default=None)
    parser.add_argument('--monthly_latency_zarr', type=Path, default=None)
    parser.add_argument('--plots_dir', type=Path, default=None)
    parser.add_argument('--chunk_buffer', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    proc = registry.get('processing', {}).get('daymet', {})
    sources = registry.get('sources', {}).get('daymet', {})
    args.earthaccess_root = args.earthaccess_root or Path(proc['monthly_latency_earthaccess_root'])
    args.daily_root = args.daily_root or Path(proc['monthly_latency_daily_root'])
    args.regrid_root = args.regrid_root or Path(proc['monthly_latency_regrid_root'])
    args.monthly_latency_zarr = args.monthly_latency_zarr or Path(sources['monthly_latency_path'])
    args.plots_dir = args.plots_dir or Path(proc['monthly_latency_plots_dir'])
    args.chunk_buffer = args.chunk_buffer or int(proc['regrid_chunk_buffer'])
    args.chunk_size = args.chunk_size or int(proc['regrid_chunk_size'])
    return args


def month_bounds(month_str: str):
    try:
        month_start = pd.Timestamp(f'{month_str}-01').normalize()
    except Exception as exc:
        raise ValueError(f'month must be YYYY-MM; got {month_str!r}') from exc
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    return month_start, month_end


def month_dates(month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(month_start, month_end, freq='D')


def monthly_file_name(var_name: str, month_start: pd.Timestamp) -> str:
    return f'daymet_v4ll_daily_na_{var_name}_{month_start.strftime("%Y%m")}.nc'


def month_fully_present(zarr_path: Path, month_start: pd.Timestamp, month_end: pd.Timestamp) -> bool:
    if not zarr_path.exists():
        return False
    ds = xr.open_zarr(zarr_path)
    times = pd.to_datetime(ds['time'].values).normalize()
    wanted = month_dates(month_start, month_end)
    present = set(times[(times >= month_start) & (times <= month_end)])
    return all(ts in present for ts in wanted)


def _metadata_rows():
    import requests

    response = requests.get(DAYMET_METADATA_URL, params={'ds_id': DAYMET_DATASET_ID, 'length': -1}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload.get('data', [])


def _granule_url(pathname: str, granule_name: str) -> str:
    return '/'.join([DAYMET_DATAPOOL.rstrip('/'), pathname.strip('/'), granule_name])


def _probe_url(url: str) -> bool:
    earthaccess.login()
    session = earthaccess.get_requests_https_session()
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


def resolve_month_links(month_start: pd.Timestamp, var_names):
    wanted = {monthly_file_name(var_name, month_start): var_name for var_name in var_names}
    links = {}
    for row in _metadata_rows():
        granule_name = row.get('granule_name')
        if granule_name not in wanted:
            continue
        pathname = row.get('pathname', '')
        links[wanted[granule_name]] = _granule_url(pathname, granule_name)
    return links


def check_month_available(month_start: pd.Timestamp, month_end: pd.Timestamp, monthly_latency_zarr: Path):
    if month_fully_present(monthly_latency_zarr, month_start, month_end):
        return True, 'monthly_latency_zarr_already_contains_month'
    links = resolve_month_links(month_start, DAYMET_PROBE_VARS)
    missing = [var_name for var_name in DAYMET_PROBE_VARS if var_name not in links]
    if missing:
        return False, f'metadata_missing:{"-".join(missing)}'
    probe_failed = [var_name for var_name, url in links.items() if not _probe_url(url)]
    if probe_failed:
        return False, f'remote_probe_failed:{"-".join(probe_failed)}'
    return True, f'remote_probe_ok:{"-".join(DAYMET_PROBE_VARS)}'


def _download_url(url: str, target_path: Path):
    earthaccess.login()
    session = earthaccess.get_requests_https_session()
    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_month_downloads(month_start: pd.Timestamp, earthaccess_root: Path):
    month_dir = earthaccess_root / month_start.strftime('%Y')
    month_dir.mkdir(parents=True, exist_ok=True)
    links = resolve_month_links(month_start, DAYMET_VARS)
    missing = [var_name for var_name in DAYMET_VARS if var_name not in links]
    if missing:
        raise RuntimeError(f'Monthly-latency Daymet is missing files for {month_start:%Y-%m}: {missing}')
    for var_name, url in links.items():
        target_path = month_dir / monthly_file_name(var_name, month_start)
        if target_path.exists():
            print(f'[SKIP] {target_path.name} already exists')
            continue
        print(f'Downloading {target_path.name} from {url}')
        _download_url(url, target_path)


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def ensure_daily_split(month_start: pd.Timestamp, earthaccess_root: Path, daily_root: Path, plots_dir: Path):
    daily_year_dir = daily_root / month_start.strftime('%Y')
    expected = [
        daily_year_dir / f'daymet_v4_daily_na_{var_name}_{ts.strftime("%Y%m%d")}.nc'
        for var_name in DAYMET_VARS
        for ts in month_dates(month_start, month_start + pd.offsets.MonthEnd(1))
    ]
    if all(path.exists() for path in expected):
        print(f'Daymet monthly-latency daily files already exist for {month_start:%Y-%m}')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/daymet/to_daily.py',
        '--year', month_start.strftime('%Y'),
        '--month', month_start.strftime('%m'),
        '--input_root', earthaccess_root,
        '--output_root', daily_root,
        '--plots_dir', plots_dir,
        '--skip_existing',
        '--no-plot_first_day',
    ])


def ensure_regridded_month(month_start: pd.Timestamp, month_end: pd.Timestamp, grid_path: Path, daily_root: Path, regrid_root: Path, chunk_size: int, chunk_buffer: int):
    src_dir = daily_root / month_start.strftime('%Y')
    target_dir = regrid_root / month_start.strftime('%Y')
    expected = [
        target_dir / f'daymet_v4_daily_na_{var_name}_{ts.strftime("%Y%m%d")}_regridded.nc'
        for var_name in DAYMET_VARS
        for ts in month_dates(month_start, month_end)
    ]
    expected_nc4 = [path.with_suffix('.nc4') for path in expected]
    if all(path.exists() or nc4.exists() for path, nc4 in zip(expected, expected_nc4)):
        print(f'Daymet monthly-latency regridded files already exist for {month_start:%Y-%m}')
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


def append_month(month_start: pd.Timestamp, monthly_latency_zarr: Path, regrid_root: Path):
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/convert_to_zarr/append_daymet_archive_year.py',
        '--root', regrid_root,
        '--out_zarr', monthly_latency_zarr,
        '--year', month_start.strftime('%Y'),
        '--start_month', month_start.strftime('%m'),
        '--end_month', month_start.strftime('%m'),
    ])


def main():
    args = apply_registry_defaults(parse_args())
    month_start, month_end = month_bounds(args.month)
    print(f'Updating monthly-latency Daymet for {args.month}: {month_start.date()} -> {month_end.date()}')
    available, reason = check_month_available(month_start, month_end, args.monthly_latency_zarr)
    if args.check_only:
        print(f'Monthly-latency Daymet availability check for {args.month}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)
    if not available:
        raise SystemExit(f'Monthly-latency Daymet is not yet available for {args.month}: {reason}')
    if month_fully_present(args.monthly_latency_zarr, month_start, month_end):
        print(f'Monthly-latency Daymet zarr already contains full month {args.month}; nothing to do')
        return
    ensure_month_downloads(month_start, args.earthaccess_root)
    ensure_daily_split(month_start, args.earthaccess_root, args.daily_root, args.plots_dir)
    ensure_regridded_month(month_start, month_end, args.grid_path, args.daily_root, args.regrid_root, args.chunk_size, args.chunk_buffer)
    append_month(month_start, args.monthly_latency_zarr, args.regrid_root)
    print(f'Finished monthly-latency Daymet update for {args.month}')


if __name__ == '__main__':
    main()
