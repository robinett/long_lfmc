#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

from get_modis import DEFAULT_GRID_PATH, TILES_V, collect_links


REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download, process, interpolate, and append one monthly MODIS update into the canonical zarr.'
    )
    parser.add_argument('--month', type=str, required=True, help='Target month in YYYY-MM format')
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=Path(DEFAULT_GRID_PATH))
    parser.add_argument('--raw_root', type=Path, default=None)
    parser.add_argument('--regrid_root', type=Path, default=None)
    parser.add_argument('--canonical_zarr', type=Path, default=None)
    parser.add_argument('--staging_root', type=Path, default=None)
    parser.add_argument('--plots_dir', type=Path, default=None)
    parser.add_argument('--max_interpolation_days', type=int, default=None)
    parser.add_argument('--buffer_days', type=int, default=None)
    parser.add_argument('--xy_chunk_size', type=int, default=None)
    parser.add_argument('--time_chunk_size', type=int, default=None)
    parser.add_argument('--quality_flag', type=int, default=None)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    proc = registry.get('processing', {}).get('modis', {})
    sources = registry.get('sources', {})
    args.raw_root = args.raw_root or Path(proc['raw_root'])
    args.regrid_root = args.regrid_root or Path(proc['regrid_root'])
    args.canonical_zarr = args.canonical_zarr or Path(sources['modis']['path'])
    args.staging_root = args.staging_root or Path(proc['staging_root'])
    args.plots_dir = args.plots_dir or Path(proc['plots_dir'])
    args.max_interpolation_days = args.max_interpolation_days or int(proc['interpolation_max_days'])
    args.buffer_days = args.buffer_days or int(proc['interpolation_buffer_days'])
    args.xy_chunk_size = args.xy_chunk_size or int(proc['interpolation_xy_chunk_size'])
    args.time_chunk_size = args.time_chunk_size or int(proc['interpolation_time_chunk_size'])
    args.quality_flag = args.quality_flag if args.quality_flag is not None else int(proc['quality_flag'])
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


def expected_regrid_path(regrid_root: Path, dt_value: pd.Timestamp) -> Path:
    return (
        regrid_root
        / dt_value.strftime('%Y')
        / dt_value.strftime('%m')
        / f'modis_reflectance_{dt_value.strftime("%Y%m%d")}_regridded.nc4'
    )


def _load_time_index_from_zarr(zarr_path: Path) -> pd.DatetimeIndex:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    return pd.to_datetime(ds['time'].values).normalize()


def canonical_has_full_month(canonical_zarr: Path, month_start: pd.Timestamp, month_end: pd.Timestamp) -> bool:
    if not canonical_zarr.exists():
        return False
    times = _load_time_index_from_zarr(canonical_zarr)
    wanted = month_dates(month_start, month_end)
    present = set(times[(times >= month_start) & (times <= month_end)])
    return all(ts in present for ts in wanted)


def regridded_month_complete(regrid_root: Path, month_start: pd.Timestamp, month_end: pd.Timestamp) -> bool:
    expected = [expected_regrid_path(regrid_root, ts) for ts in month_dates(month_start, month_end)]
    return all(path.exists() for path in expected)


def modis_bounding_box(grid_path: Path):
    grid = xr.open_dataset(grid_path)
    min_lat = float(grid['lat'].min().values) - 1.0
    max_lat = float(grid['lat'].max().values) + 1.0
    min_lon = float(grid['lon'].min().values) - 1.0
    max_lon = float(grid['lon'].max().values) + 1.0
    return (min_lon, min_lat, max_lon, max_lat)


def remote_month_available(month_start: pd.Timestamp, month_end: pd.Timestamp, grid_path: Path):
    earthaccess.login()
    bbox = modis_bounding_box(grid_path)
    dates = month_dates(month_start, month_end)
    desired = len(dates) * len(TILES_V)
    results_data = earthaccess.search_data(
        short_name='MCD43A4',
        version='061',
        temporal=(month_start, month_end),
        bounding_box=bbox,
        downloadable=True,
    )
    data_links = collect_links(results_data, dates, 'MCD43A4')
    results_quality = earthaccess.search_data(
        short_name='MCD43A2',
        version='061',
        temporal=(month_start, month_end),
        bounding_box=bbox,
        downloadable=True,
    )
    quality_links = collect_links(results_quality, dates, 'MCD43A2')
    available = len(data_links) >= desired and len(quality_links) >= desired
    reason = (
        f'data_links={len(data_links)}/{desired};quality_links={len(quality_links)}/{desired}'
    )
    return available, reason


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def ensure_raw_downloads(month_start: pd.Timestamp, month_end: pd.Timestamp, raw_root: Path, grid_path: Path):
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/modis/get_modis.py',
        '--start_date', str(month_start.date()),
        '--end_date', str(month_end.date()),
        '--output_root', raw_root,
        '--grid_path', grid_path,
        '--skip_existing',
    ])


def ensure_regridded_month(
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    regrid_root: Path,
    quality_flag: int,
):
    if regridded_month_complete(regrid_root, month_start, month_end):
        print(f'Regridded MODIS files already cover {month_start:%Y-%m}')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/modis/main.py',
        '--start_date', str(month_start.date()),
        '--end_date', str(month_end.date()),
        '--out_dir', regrid_root,
        '--quality_flag', str(quality_flag),
    ])


def canonical_time_bounds(canonical_zarr: Path):
    if not canonical_zarr.exists():
        return None, None
    times = _load_time_index_from_zarr(canonical_zarr)
    if len(times) == 0:
        return None, None
    return pd.Timestamp(times.min()).normalize(), pd.Timestamp(times.max()).normalize()


def compute_refresh_start(canonical_zarr: Path, month_start: pd.Timestamp, max_interpolation_days: int):
    current_min, current_max = canonical_time_bounds(canonical_zarr)
    if current_max is None:
        return month_start
    refresh_candidate = current_max - pd.Timedelta(days=max_interpolation_days)
    refresh_start = min(month_start, refresh_candidate)
    if current_min is not None:
        refresh_start = max(refresh_start, current_min)
    return pd.Timestamp(refresh_start).normalize()


def build_staging_zarr(args, refresh_start: pd.Timestamp, month_end: pd.Timestamp) -> Path:
    args.staging_root.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    month_stamp = month_end.strftime('%Y%m')
    staging_zarr = args.staging_root / f'modis_interp_update_{month_stamp}.zarr'
    if staging_zarr.exists():
        shutil.rmtree(staging_zarr)
    diagnostic_plot = args.plots_dir / f'modis_interp_update_{month_stamp}_diagnostic.png'
    diagnostic_map = args.plots_dir / f'modis_interp_update_{month_stamp}_diagnostic_map.png'
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/interpolate/interpolate_new.py',
        '--base_path', args.regrid_root,
        '--start_date', str(refresh_start.date()),
        '--end_date', str(month_end.date()),
        '--output_zarr', staging_zarr,
        '--max_interpolation_days', str(args.max_interpolation_days),
        '--buffer_days', str(args.buffer_days),
        '--xy_chunk_size', str(args.xy_chunk_size),
        '--time_chunk_size', str(args.time_chunk_size),
        '--overwrite_zarr',
        '--diagnostic_plot_path', diagnostic_plot,
        '--diagnostic_map_plot_path', diagnostic_map,
    ])
    return staging_zarr


def _array_dimensions(array):
    dims = array.attrs.get('_ARRAY_DIMENSIONS')
    if dims:
        return list(dims)
    metadata = getattr(array, 'metadata', None)
    if metadata is not None:
        dims = getattr(metadata, 'dimension_names', None)
        if dims:
            return list(dims)
    return None


def _time_array_names(root):
    names = []
    for name in root.array_keys():
        dims = _array_dimensions(root[name])
        if dims and dims[0] == 'time':
            names.append(name)
    return sorted(set(names))


def _ensure_static_coords_match(staging_root, canonical_root):
    for coord_name in ['x', 'y', 'lat', 'lon']:
        if coord_name in staging_root and coord_name in canonical_root:
            if not np.array_equal(staging_root[coord_name][:], canonical_root[coord_name][:]):
                raise ValueError(f'Coordinate mismatch for {coord_name}')


def _time_slice(index: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp) -> slice:
    positions = np.where((index >= start_date) & (index <= end_date))[0]
    if len(positions) == 0:
        raise ValueError(f'No time coordinates found between {start_date.date()} and {end_date.date()}')
    expected = np.arange(positions[0], positions[0] + len(positions))
    if not np.array_equal(positions, expected):
        raise ValueError('Selected time range is not contiguous')
    return slice(int(positions[0]), int(positions[-1]) + 1)


def promote_staging_window(staging_zarr: Path, canonical_zarr: Path, refresh_start: pd.Timestamp, refresh_end: pd.Timestamp):
    if not canonical_zarr.exists():
        print(f'Canonical MODIS zarr missing; initializing from staging {staging_zarr}')
        shutil.copytree(staging_zarr, canonical_zarr)
        return 'initialized_from_staging'

    staging_root = zarr.open_group(str(staging_zarr), mode='r')
    canonical_root = zarr.open_group(str(canonical_zarr), mode='a')
    _ensure_static_coords_match(staging_root, canonical_root)

    staging_time = pd.to_datetime(np.asarray(staging_root['time'][:], dtype=np.int64)).normalize()
    canonical_time = pd.to_datetime(np.asarray(canonical_root['time'][:], dtype=np.int64)).normalize()
    staging_slice = _time_slice(staging_time, refresh_start, refresh_end)
    staging_window = staging_time[staging_slice]
    canonical_end = canonical_time.max() if len(canonical_time) > 0 else None
    time_names = _time_array_names(staging_root)

    overlap_mask = np.ones(len(staging_window), dtype=bool) if canonical_end is None else (staging_window <= canonical_end)
    append_mask = np.zeros(len(staging_window), dtype=bool) if canonical_end is None else (staging_window > canonical_end)
    if canonical_end is None:
        append_mask = np.ones(len(staging_window), dtype=bool)
        overlap_mask = np.zeros(len(staging_window), dtype=bool)

    if overlap_mask.any():
        overlap_times = staging_window[overlap_mask]
        production_slice = _time_slice(canonical_time, overlap_times[0], overlap_times[-1])
        production_times = canonical_time[production_slice]
        if not np.array_equal(overlap_times.values.astype('datetime64[ns]'), production_times.values.astype('datetime64[ns]')):
            raise ValueError('Staging and canonical MODIS times differ for overlap refresh range')
        overlap_start = int(np.where(overlap_mask)[0][0])
        overlap_stop = int(np.where(overlap_mask)[0][-1]) + 1
        staging_overlap_slice = slice(staging_slice.start + overlap_start, staging_slice.start + overlap_stop)
        print(f'Overwriting canonical MODIS overlap window {overlap_times[0].date()} -> {overlap_times[-1].date()}')
        for name in time_names:
            canonical_root[name][production_slice] = staging_root[name][staging_overlap_slice]

    if append_mask.any():
        append_times = staging_window[append_mask]
        append_start = int(np.where(append_mask)[0][0])
        append_stop = int(np.where(append_mask)[0][-1]) + 1
        staging_append_slice = slice(staging_slice.start + append_start, staging_slice.start + append_stop)
        old_n = len(canonical_time)
        new_n = old_n + len(append_times)
        print(f'Appending canonical MODIS window {append_times[0].date()} -> {append_times[-1].date()}')
        for name in time_names:
            arr = canonical_root[name]
            if name == 'time':
                arr.resize(new_n)
                arr[old_n:new_n] = staging_root[name][staging_append_slice]
            else:
                arr.resize(new_n, *arr.shape[1:])
                arr[old_n:new_n] = staging_root[name][staging_append_slice]

    canonical_root.attrs['last_monthly_update'] = refresh_end.strftime('%Y-%m')
    canonical_root.attrs['last_refresh_start'] = str(refresh_start.date())
    return 'updated'


def main():
    args = apply_registry_defaults(parse_args())
    month_start, month_end = month_bounds(args.month)
    print(f'Updating MODIS for month {args.month}: {month_start.date()} -> {month_end.date()}')

    if args.check_only:
        if canonical_has_full_month(args.canonical_zarr, month_start, month_end):
            print(f'MODIS availability check for {args.month}: available=True reason=canonical_zarr_already_contains_month')
            raise SystemExit(0)
        available, reason = remote_month_available(month_start, month_end, args.grid_path)
        print(f'MODIS availability check for {args.month}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)

    if canonical_has_full_month(args.canonical_zarr, month_start, month_end):
        print(f'Canonical MODIS zarr already contains full month {args.month}; nothing to do')
        return

    ensure_raw_downloads(month_start, month_end, args.raw_root, args.grid_path)
    ensure_regridded_month(month_start, month_end, args.regrid_root, args.quality_flag)

    refresh_start = compute_refresh_start(args.canonical_zarr, month_start, args.max_interpolation_days)
    print(f'Refreshing canonical MODIS from {refresh_start.date()} through {month_end.date()} to preserve interpolation consistency')
    staging_zarr = build_staging_zarr(args, refresh_start, month_end)
    try:
        result = promote_staging_window(staging_zarr, args.canonical_zarr, refresh_start, month_end)
        print(f'MODIS monthly update complete for {args.month}: {result}')
        print(f'  staging_zarr={staging_zarr}')
        print(f'  canonical_zarr={args.canonical_zarr}')
    finally:
        if staging_zarr.exists():
            shutil.rmtree(staging_zarr)
            print(f'Removed staging zarr {staging_zarr}')


if __name__ == '__main__':
    main()
