#!/usr/bin/env python3

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

from get_modis import DEFAULT_GRID_PATH, TILES_V, collect_links

REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
INFERENCE_SCRIPT_DIR = REPO_ROOT / 'lfmc_model/scripts/inference'
if str(INFERENCE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_SCRIPT_DIR))

from low_latency_rollback import capture_zarr_rollback_from_env


DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'
INTERPOLATION_WORKER_SCRIPT = REPO_ROOT / 'data_processing/interpolate/run_interpolation_worker.sbatch'
MODIS_SINUSOIDAL_CRS = '+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs'
DAILY_DATE_PATTERN = re.compile(r'(\d{8})')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download, process, interpolate, and append one monthly MODIS update into the canonical zarr.'
    )
    parser.add_argument('--month', type=str, required=True, help='Target month in YYYY-MM format')
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=Path(DEFAULT_GRID_PATH))
    parser.add_argument('--raw_root', type=Path, default=None)
    parser.add_argument('--mosaic_root', type=Path, default=None)
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
    args.mosaic_root = args.mosaic_root or Path(proc['mosaic_root'])
    args.regrid_root = args.regrid_root or Path(proc['regrid_root'])
    args.canonical_zarr = args.canonical_zarr or Path(sources['modis']['path'])
    args.staging_root = args.staging_root or Path(proc['staging_root'])
    args.plots_dir = args.plots_dir or Path(proc['plots_dir'])
    args.max_interpolation_days = args.max_interpolation_days or int(proc['interpolation_max_days'])
    args.buffer_days = args.buffer_days or int(proc['interpolation_buffer_days'])
    args.xy_chunk_size = args.xy_chunk_size or int(proc['interpolation_xy_chunk_size'])
    args.time_chunk_size = args.time_chunk_size or int(proc['interpolation_time_chunk_size'])
    args.interpolation_num_workers = int(proc.get('interpolation_num_workers', 32))
    args.interpolation_worker_cpus = int(proc.get('interpolation_worker_cpus', 4))
    args.interpolation_worker_mem = str(proc.get('interpolation_worker_mem', '16G'))
    args.interpolation_worker_time = str(proc.get('interpolation_worker_time', '08:00:00'))
    args.interpolation_array_max_retries = int(proc.get('interpolation_array_max_retries', 3))
    args.regrid_chunk_buffer = int(proc.get('regrid_chunk_buffer', 100))
    args.regrid_chunk_size = int(proc.get('regrid_chunk_size', 2000))
    args.retain_staging_after_success = bool(proc.get('retain_staging_after_success', True))
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


def daily_mosaic_path(mosaic_root: Path, dt_value: pd.Timestamp) -> Path:
    base_dir = mosaic_root / dt_value.strftime('%Y') / dt_value.strftime('%m')
    date_tag = dt_value.strftime('%Y%m%d')
    return base_dir / f'modis_reflectance_{date_tag}.nc4'


def daily_regrid_path(regrid_root: Path, dt_value: pd.Timestamp) -> Path:
    base_dir = regrid_root / dt_value.strftime('%Y') / dt_value.strftime('%m')
    date_tag = dt_value.strftime('%Y%m%d')
    return base_dir / f'modis_reflectance_{date_tag}_regridded.nc4'


def candidate_regrid_paths(regrid_root: Path, dt_value: pd.Timestamp) -> list[Path]:
    return [daily_regrid_path(regrid_root, dt_value)]


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
    return all(
        any(path.exists() for path in candidate_regrid_paths(regrid_root, ts))
        for ts in month_dates(month_start, month_end)
    )


def mosaics_complete(mosaic_root: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    return all(
        daily_mosaic_path(mosaic_root, ts).exists()
        for ts in pd.date_range(start_date, end_date, freq='D')
    )


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
    available = True
    data_missing = max(desired - len(data_links), 0)
    quality_missing = max(desired - len(quality_links), 0)
    reason = (
        f'data_links={len(data_links)}/{desired};quality_links={len(quality_links)}/{desired}'
    )
    if data_missing:
        reason = (
            f'{reason};data_missing={data_missing};'
            'data_partial_allowed=nan_fallback'
        )
    if quality_missing:
        reason = (
            f'{reason};quality_missing={quality_missing};'
            'quality_partial_allowed=nan_fallback'
        )
    return available, reason


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def _run_capture(cmd):
    result = subprocess.run(
        [str(part) for part in cmd],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _wait_for_slurm_job(job_id: str, label: str) -> bool:
    inactive_nonterminal_polls = 0
    while True:
        states = subprocess.run(
            ['sacct', '-j', str(job_id), '--format=State', '-n'],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        clean_states = [line.strip().split()[0] for line in states if line.strip()]
        bad_states = {
            'BOOT_FAIL',
            'CANCELLED',
            'DEADLINE',
            'FAILED',
            'NODE_FAIL',
            'OUT_OF_MEMORY',
            'PREEMPTED',
            'TIMEOUT',
        }
        if any(state in bad_states for state in clean_states):
            print(f'  {label} job={job_id} failed with states={clean_states}')
            return False
        active = subprocess.run(
            ['squeue', '-h', '-j', str(job_id)],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if active:
            inactive_nonterminal_polls = 0
            print(f'  {label} job={job_id} state=ACTIVE; sleeping 60s')
            time.sleep(60)
            continue
        if clean_states and all(state == 'COMPLETED' for state in clean_states):
            print(f'  {label} job={job_id} completed with states={clean_states}')
            return True
        nonterminal_states = {
            'CONFIGURING',
            'COMPLETING',
            'PENDING',
            'REQUEUED',
            'REQUEUE_FED',
            'REQUEUE_HOLD',
            'RESIZING',
            'RUNNING',
            'SPECIAL_EXIT',
            'STAGE_OUT',
            'SUSPENDED',
        }
        if clean_states and any(state in nonterminal_states for state in clean_states):
            inactive_nonterminal_polls += 1
            if inactive_nonterminal_polls <= 10:
                print(
                    f'  {label} job={job_id} no longer in squeue but accounting '
                    f'has nonterminal states={clean_states}; sleeping 30s'
                )
                time.sleep(30)
                continue
            print(
                f'  {label} job={job_id} accounting did not settle after '
                f'{inactive_nonterminal_polls} inactive polls; states={clean_states}'
            )
            return False
        print(f'  {label} job={job_id} ended with unexpected states={clean_states}')
        return False


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


def ensure_daily_mosaics(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    raw_root: Path,
    mosaic_root: Path,
    quality_flag: int,
) -> None:
    if mosaics_complete(mosaic_root, start_date, end_date):
        print(f'Native MODIS mosaics already cover {start_date.date()} -> {end_date.date()}')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/modis/main.py',
        '--start_date', str(start_date.date()),
        '--end_date', str(end_date.date()),
        '--raw_root', raw_root,
        '--out_dir', mosaic_root,
        '--quality_flag', str(quality_flag),
    ])


def ensure_regridded_files(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    grid_path: Path,
    mosaic_root: Path,
    regrid_root: Path,
    chunk_size: int,
    chunk_buffer: int,
) -> None:
    wanted = pd.date_range(start_date, end_date, freq='D')
    if all(daily_regrid_path(regrid_root, ts).exists() for ts in wanted):
        print(f'Regridded MODIS files already cover {start_date.date()} -> {end_date.date()}')
        return
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/regrid/main.py',
        '--target_grid', grid_path,
        '--src_dir', mosaic_root,
        '--target_dir', regrid_root,
        '--src_crs', MODIS_SINUSOIDAL_CRS,
        '--target_crs', 'EPSG:5070',
        '--chunk_size', str(chunk_size),
        '--chunk_buffer', str(chunk_buffer),
        '--start_date', str(start_date.date()),
        '--end_date', str(end_date.date()),
        '--skip_existing',
    ])


def validate_regridded_grid(
    regrid_root: Path,
    canonical_zarr: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    expected_paths = [daily_regrid_path(regrid_root, ts) for ts in pd.date_range(start_date, end_date, freq='D')]
    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing regridded MODIS files: sample={missing[:10]}')

    sample_path = expected_paths[0]
    with xr.open_dataset(sample_path, engine='netcdf4') as sample_ds:
        sample_x = sample_ds['x'].values
        sample_y = sample_ds['y'].values
    with xr.open_zarr(canonical_zarr, consolidated=False) as canonical_ds:
        canonical_x = canonical_ds['x'].values
        canonical_y = canonical_ds['y'].values
    if not np.array_equal(sample_x, canonical_x):
        raise ValueError(f'Regridded MODIS x coordinates do not match canonical zarr: {sample_path}')
    if not np.array_equal(sample_y, canonical_y):
        raise ValueError(f'Regridded MODIS y coordinates do not match canonical zarr: {sample_path}')
    print(f'Validated MODIS regrid coordinates against canonical zarr using {sample_path}')


def parse_daily_file_date(path: Path) -> pd.Timestamp | None:
    match = DAILY_DATE_PATTERN.search(path.name)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), format='%Y%m%d').normalize()


def parse_raw_file_date(path: Path) -> pd.Timestamp | None:
    parts = path.name.split('.')
    if len(parts) < 2 or not parts[1].startswith('A'):
        return None
    try:
        return pd.to_datetime(parts[1][1:], format='%Y%j').normalize()
    except ValueError:
        return None


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        if path == root:
            continue
        if not dirnames and not filenames:
            path.rmdir()


def prune_files_outside_window(root: Path, parser, retain_start: pd.Timestamp, retain_end: pd.Timestamp) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        date_value = parser(path)
        if date_value is None:
            continue
        if retain_start <= date_value <= retain_end:
            continue
        path.unlink()
        removed += 1
    prune_empty_dirs(root)
    return removed


def prune_source_artifacts(
    raw_root: Path,
    mosaic_root: Path,
    regrid_root: Path,
    retain_start: pd.Timestamp,
    retain_end: pd.Timestamp,
) -> None:
    raw_removed = prune_files_outside_window(raw_root, parse_raw_file_date, retain_start, retain_end)
    mosaic_removed = prune_files_outside_window(mosaic_root, parse_daily_file_date, retain_start, retain_end)
    regrid_removed = prune_files_outside_window(regrid_root, parse_daily_file_date, retain_start, retain_end)
    print(
        'Pruned MODIS source-side artifacts outside '
        f'{retain_start.date()} -> {retain_end.date()}: '
        f'raw={raw_removed}, mosaics={mosaic_removed}, regridded={regrid_removed}'
    )


def ensure_regridded_month(
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    grid_path: Path,
    raw_root: Path,
    mosaic_root: Path,
    regrid_root: Path,
    quality_flag: int,
    chunk_size: int,
    chunk_buffer: int,
):
    ensure_daily_mosaics(month_start, month_end, raw_root, mosaic_root, quality_flag)
    ensure_regridded_files(
        month_start,
        month_end,
        grid_path,
        mosaic_root,
        regrid_root,
        chunk_size,
        chunk_buffer,
    )


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
    diagnostic_plot = args.plots_dir / f'modis_interp_update_{month_stamp}_diagnostic.png'
    diagnostic_map = args.plots_dir / f'modis_interp_update_{month_stamp}_diagnostic_map.png'
    base_cmd = [
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
        '--diagnostic_plot_path', diagnostic_plot,
        '--diagnostic_map_plot_path', diagnostic_map,
    ]
    if staging_zarr.exists():
        _ensure_staging_matches_request(
            staging_zarr,
            refresh_start,
            month_end,
            args.canonical_zarr,
        )
        print(f'Resuming existing MODIS interpolation staging zarr: {staging_zarr}')
    else:
        run_cmd([
            *base_cmd,
            '--mode', 'init',
            '--overwrite_zarr',
        ])

    total_chunks = _staging_total_chunks(staging_zarr, args.xy_chunk_size)
    completed = _completed_chunk_count(staging_zarr)
    print(f'MODIS interpolation chunk markers: {completed}/{total_chunks}')

    attempt = 1
    while completed < total_chunks and attempt <= int(args.interpolation_array_max_retries):
        job_id = _submit_interpolation_array(args, base_cmd)
        ok = _wait_for_slurm_job(job_id, f'modis_interp_attempt_{attempt}')
        previous = completed
        completed = _completed_chunk_count(staging_zarr)
        print(f'MODIS interpolation chunk markers after attempt {attempt}: {completed}/{total_chunks}')
        if completed >= total_chunks:
            break
        if not ok and completed == previous:
            print(f'MODIS interpolation attempt {attempt} failed without marker progress')
        attempt += 1

    if completed < total_chunks:
        raise RuntimeError(
            f'MODIS interpolation incomplete after {args.interpolation_array_max_retries} attempts: '
            f'{completed}/{total_chunks} chunks complete'
        )

    run_cmd([
        *base_cmd,
        '--mode', 'finalize',
    ])
    run_cmd([
        *base_cmd,
        '--plot-only',
    ])
    return staging_zarr


def _ensure_staging_matches_request(
    staging_zarr: Path,
    refresh_start: pd.Timestamp,
    refresh_end: pd.Timestamp,
    canonical_zarr: Path,
):
    ds = xr.open_zarr(staging_zarr, consolidated=False)
    try:
        expected = pd.date_range(refresh_start, refresh_end, freq='D')
        actual = pd.to_datetime(ds['time'].values).normalize()
        if len(actual) != len(expected) or not np.array_equal(
            actual.values.astype('datetime64[ns]'),
            expected.values.astype('datetime64[ns]'),
        ):
            raise ValueError(
                f'Existing staging zarr has incompatible time coordinates: {staging_zarr}'
            )
        interp_vars = [name for name in ds.data_vars if name.endswith('_interp')]
        if not interp_vars:
            raise ValueError(f'Existing staging zarr has no *_interp variables: {staging_zarr}')
        with xr.open_zarr(canonical_zarr, consolidated=False) as canonical_ds:
            for coord_name in ['x', 'y']:
                if coord_name in ds.coords and coord_name in canonical_ds.coords:
                    if not np.array_equal(ds[coord_name].values, canonical_ds[coord_name].values):
                        raise ValueError(
                            f'Existing staging zarr has incompatible {coord_name} coordinates: '
                            f'{staging_zarr}'
                        )
    finally:
        ds.close()


def _staging_total_chunks(staging_zarr: Path, xy_chunk_size: int) -> int:
    ds = xr.open_zarr(staging_zarr, consolidated=False)
    try:
        y_size = int(ds.sizes['y'])
        x_size = int(ds.sizes['x'])
    finally:
        ds.close()
    return math.ceil(y_size / int(xy_chunk_size)) * math.ceil(x_size / int(xy_chunk_size))


def _chunk_status_dir(staging_zarr: Path) -> Path:
    return Path(f'{staging_zarr}.chunk_status')


def _legacy_chunk_status_dir(staging_zarr: Path) -> Path:
    return staging_zarr / '_chunk_status'


def _completed_chunk_count(staging_zarr: Path) -> int:
    marker_names = set()
    for status_dir in (_chunk_status_dir(staging_zarr), _legacy_chunk_status_dir(staging_zarr)):
        if status_dir.exists():
            marker_names.update(path.name for path in status_dir.glob('chunk_*.done'))
    return len(marker_names)


def _submit_interpolation_array(args, base_cmd) -> str:
    workers = int(args.interpolation_num_workers)
    if workers < 1:
        raise ValueError('interpolation_num_workers must be >= 1')
    export_items = {
        'INTERP_BASE_PATH': args.regrid_root,
        'INTERP_START_DATE': _cmd_value(base_cmd, '--start_date'),
        'INTERP_END_DATE': _cmd_value(base_cmd, '--end_date'),
        'INTERP_OUTPUT_ZARR': _cmd_value(base_cmd, '--output_zarr'),
        'INTERP_MAX_DAYS': args.max_interpolation_days,
        'INTERP_BUFFER_DAYS': args.buffer_days,
        'INTERP_XY_CHUNK_SIZE': args.xy_chunk_size,
        'INTERP_TIME_CHUNK_SIZE': args.time_chunk_size,
        'INTERP_NUM_WORKERS': workers,
    }
    export_arg = 'ALL,' + ','.join(f'{key}={value}' for key, value in export_items.items())
    cmd = [
        'sbatch',
        '--parsable',
        '--array', f'0-{workers - 1}',
        '--cpus-per-task', str(args.interpolation_worker_cpus),
        '--mem', str(args.interpolation_worker_mem),
        '--time', str(args.interpolation_worker_time),
        '--export', export_arg,
        INTERPOLATION_WORKER_SCRIPT,
    ]
    print('Submitting MODIS interpolation worker array:')
    print('  ' + ' '.join(str(part) for part in cmd))
    job_id = _run_capture(cmd)
    print(f'Submitted MODIS interpolation worker array {job_id}')
    return job_id


def _cmd_value(cmd, flag: str):
    for idx, value in enumerate(cmd):
        if str(value) == flag:
            return cmd[idx + 1]
    raise KeyError(f'{flag} not found in interpolation command')


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
    for coord_name in ['x', 'y']:
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


def _decoded_zarr_time_index(zarr_path: Path) -> pd.DatetimeIndex:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        return pd.DatetimeIndex(ds['time'].values).normalize()
    finally:
        ds.close()


def _encode_time_for_array(times: pd.DatetimeIndex, time_array):
    units = time_array.attrs.get('units')
    if not units:
        return times.values.astype(time_array.dtype, copy=False)
    match = re.match(r'^(\w+) since (.+)$', units)
    if match is None:
        raise ValueError(f'Unsupported time units for canonical MODIS zarr: {units}')
    unit_name, origin_text = match.groups()
    origin = pd.Timestamp(origin_text)
    deltas = times - origin
    unit_seconds = {
        'day': 86400,
        'days': 86400,
        'hour': 3600,
        'hours': 3600,
        'minute': 60,
        'minutes': 60,
        'second': 1,
        'seconds': 1,
    }
    if unit_name not in unit_seconds:
        raise ValueError(f'Unsupported time unit for canonical MODIS zarr: {unit_name}')
    encoded = deltas / pd.Timedelta(seconds=unit_seconds[unit_name])
    encoded = np.asarray(encoded, dtype=np.float64)
    if np.issubdtype(time_array.dtype, np.integer):
        rounded = np.rint(encoded)
        if not np.allclose(encoded, rounded, rtol=0, atol=1e-9):
            raise ValueError(f'Time values are not integral in canonical units: {units}')
        encoded = rounded
    return encoded.astype(time_array.dtype, copy=False)


def promote_staging_window(staging_zarr: Path, canonical_zarr: Path, refresh_start: pd.Timestamp, refresh_end: pd.Timestamp):
    if not canonical_zarr.exists():
        print(f'Canonical MODIS zarr missing; initializing from staging {staging_zarr}')
        shutil.copytree(staging_zarr, canonical_zarr)
        return 'initialized_from_staging'

    staging_root = zarr.open_group(str(staging_zarr), mode='r')
    canonical_root = zarr.open_group(str(canonical_zarr), mode='a')
    _ensure_static_coords_match(staging_root, canonical_root)

    staging_time = _decoded_zarr_time_index(staging_zarr)
    canonical_time = _decoded_zarr_time_index(canonical_zarr)
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
            if name == 'time':
                canonical_root[name][production_slice] = _encode_time_for_array(overlap_times, canonical_root[name])
            else:
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
                arr.resize((new_n,))
                arr[old_n:new_n] = _encode_time_for_array(append_times, arr)
            else:
                arr.resize((new_n, *arr.shape[1:]))
                arr[old_n:new_n] = staging_root[name][staging_append_slice]

    canonical_root.attrs['last_monthly_update'] = refresh_end.strftime('%Y-%m')
    canonical_root.attrs['last_refresh_start'] = str(refresh_start.date())
    return 'updated'


def main():
    args = apply_registry_defaults(parse_args())
    month_start, month_end = month_bounds(args.month)
    print(f'Updating MODIS for month {args.month}: {month_start.date()} -> {month_end.date()}')

    refresh_start = compute_refresh_start(args.canonical_zarr, month_start, args.max_interpolation_days)
    source_context_start = refresh_start - pd.Timedelta(days=int(args.buffer_days))
    source_context_end = month_end + pd.Timedelta(days=int(args.buffer_days))

    if args.check_only:
        if canonical_has_full_month(args.canonical_zarr, month_start, month_end):
            print(f'MODIS availability check for {args.month}: available=True reason=canonical_zarr_already_contains_month')
            raise SystemExit(0)
        available, reason = remote_month_available(source_context_start, source_context_end, args.grid_path)
        print(f'MODIS availability check for {args.month}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)

    if canonical_has_full_month(args.canonical_zarr, month_start, month_end):
        print(f'Canonical MODIS zarr already contains full month {args.month}; nothing to do')
        return

    ensure_raw_downloads(source_context_start, source_context_end, args.raw_root, args.grid_path)
    ensure_regridded_month(
        source_context_start,
        source_context_end,
        args.grid_path,
        args.raw_root,
        args.mosaic_root,
        args.regrid_root,
        args.quality_flag,
        args.regrid_chunk_size,
        args.regrid_chunk_buffer,
    )
    validate_regridded_grid(args.regrid_root, args.canonical_zarr, source_context_start, source_context_end)

    print(f'Refreshing canonical MODIS from {refresh_start.date()} through {month_end.date()} to preserve interpolation consistency')
    staging_zarr = build_staging_zarr(args, refresh_start, month_end)
    capture_zarr_rollback_from_env(
        target_zarr=args.canonical_zarr,
        label='modis_canonical',
        dim_name='time',
        window_start=refresh_start,
        window_end=month_end,
        reason='before_modis_month_promotion',
    )
    result = promote_staging_window(staging_zarr, args.canonical_zarr, refresh_start, month_end)
    print(f'MODIS monthly update complete for {args.month}: {result}')
    print(f'  staging_zarr={staging_zarr}')
    print(f'  canonical_zarr={args.canonical_zarr}')
    prune_source_artifacts(
        args.raw_root,
        args.mosaic_root,
        args.regrid_root,
        source_context_start,
        source_context_end,
    )
    if args.retain_staging_after_success:
        print(f'Retaining staging zarr after success: {staging_zarr}')


if __name__ == '__main__':
    main()
