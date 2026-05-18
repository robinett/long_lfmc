#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr
from zarr import consolidate_metadata


REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'
EXPECTED_VARIABLES = ['prcp', 'srad', 'swe', 'tmax', 'vp']
DATA_CHUNKS = (1, len(EXPECTED_VARIABLES), 512, 512)
LATLON_CHUNKS = (512, 512)
PRISM_WORKER = REPO_ROOT / 'data_processing/climate_low_latency/run_prism_range_worker.sbatch'
SNODAS_WORKER = REPO_ROOT / 'data_processing/climate_low_latency/run_snodas_range_worker.sbatch'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update the combined low-latency climate store from PRISM and SNODAS.'
    )
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=None)
    parser.add_argument('--regrid_root', type=Path, default=None)
    parser.add_argument('--out_zarr', type=Path, default=None)
    parser.add_argument('--append_coord_dir', type=Path, default=None)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    climate_cfg = registry.get('processing', {}).get('climate_low_latency', {})
    prism_cfg = registry.get('processing', {}).get('prism', {})
    snodas_cfg = registry.get('processing', {}).get('snodas', {})
    args.regrid_root = args.regrid_root or Path(climate_cfg['regrid_root'])
    args.out_zarr = args.out_zarr or Path(climate_cfg['zarr_path'])
    args.append_coord_dir = args.append_coord_dir or Path(climate_cfg['append_coord_dir'])
    args.prism_raw_root = Path(prism_cfg['raw_root'])
    args.prism_extracted_root = Path(prism_cfg['extracted_root'])
    args.prism_plots_dir = Path(prism_cfg['plots_dir'])
    args.prism_release_latency_days = int(prism_cfg['release_latency_days'])
    args.snodas_raw_root = Path(snodas_cfg['raw_root'])
    args.snodas_plots_dir = Path(snodas_cfg['plots_dir'])
    args.snodas_swe_product_token = str(snodas_cfg['swe_product_token'])
    if args.grid_path is None:
        args.grid_path = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
    return args


def parse_date_range(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f'end_date {end_date} is before start_date {start_date}')
    return start, end


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def run_capture(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    return subprocess.run([str(part) for part in cmd], check=True, capture_output=True, text=True)


def job_is_active(job_id: str) -> bool:
    result = subprocess.run(
        ['squeue', '-h', '-j', str(job_id), '-o', '%T'],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(states) > 0


def final_job_state(job_id: str) -> str:
    result = subprocess.run(
        ['sacct', '-j', str(job_id), '--format=State', '-n', '-P'],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.strip().split('|')[0] for line in result.stdout.splitlines() if line.strip()]
    if not states:
        return 'UNKNOWN'
    for state in states:
        if state not in {'COMPLETED'}:
            return state
    return 'COMPLETED'


def submit_sbatch_job(script_path: Path, env_vars: dict[str, str], label: str) -> str:
    env_arg = ','.join(['ALL'] + [f'{key}={value}' for key, value in env_vars.items()])
    result = run_capture(['sbatch', '--parsable', '--export=' + env_arg, script_path])
    job_id = result.stdout.strip()
    print(f'Submitted {label} job {job_id}')
    return job_id


def wait_for_jobs(job_ids: dict[str, str], label: str, poll_seconds: int = 30) -> None:
    pending = dict(job_ids)
    print(f'Waiting for {label} jobs to complete: {pending}')
    failed_states = {
        'BOOT_FAIL',
        'CANCELLED',
        'DEADLINE',
        'FAILED',
        'NODE_FAIL',
        'OUT_OF_MEMORY',
        'PREEMPTED',
        'TIMEOUT',
    }
    while pending:
        completed_items = []
        for item_label, job_id in pending.items():
            state = final_job_state(job_id)
            if state == 'COMPLETED':
                print(f'  {label} item={item_label} job={job_id} final_state={state}')
                completed_items.append(item_label)
                continue
            if state in failed_states:
                print(f'  {label} item={item_label} job={job_id} final_state={state}')
                raise RuntimeError(f'{label} job failed for item={item_label} job={job_id} state={state}')
            if job_is_active(job_id):
                print(
                    f'  {label} item={item_label} job={job_id} '
                    f'state=ACTIVE accounting_state={state}; sleeping'
                )
            else:
                print(
                    f'  {label} item={item_label} job={job_id} '
                    f'state={state}; waiting for terminal accounting state'
                )
        for item_label in completed_items:
            pending.pop(item_label, None)
        if pending:
            time.sleep(poll_seconds)


def update_sources(args):
    prism_script = REPO_ROOT / 'data_processing/prism/update_prism_range.py'
    snodas_script = REPO_ROOT / 'data_processing/snodas/update_snodas_range.py'
    common = [
        '--start_date', args.start_date,
        '--end_date', args.end_date,
        '--registry_path', args.registry_path,
        '--grid_path', args.grid_path,
    ]
    prism_cmd = [
        sys.executable,
        prism_script,
        *common,
        '--raw_root', args.prism_raw_root,
        '--extracted_root', args.prism_extracted_root,
        '--output_root', args.regrid_root,
        '--plots_dir', args.prism_plots_dir,
        '--release_latency_days', str(args.prism_release_latency_days),
    ]
    snodas_cmd = [
        sys.executable,
        snodas_script,
        *common,
        '--raw_root', args.snodas_raw_root,
        '--output_root', args.regrid_root,
        '--plots_dir', args.snodas_plots_dir,
        '--swe_product_token', args.snodas_swe_product_token,
    ]
    if args.check_only:
        prism_cmd.append('--check_only')
        snodas_cmd.append('--check_only')
        run_cmd(prism_cmd)
        run_cmd(snodas_cmd)
        return

    job_ids = {
        'prism': submit_sbatch_job(
            PRISM_WORKER,
            {
                'START_DATE': args.start_date,
                'END_DATE': args.end_date,
                'REGISTRY_PATH': str(args.registry_path),
                'GRID_PATH': str(args.grid_path),
                'RAW_ROOT': str(args.prism_raw_root),
                'EXTRACTED_ROOT': str(args.prism_extracted_root),
                'OUTPUT_ROOT': str(args.regrid_root),
                'PLOTS_DIR': str(args.prism_plots_dir),
                'RELEASE_LATENCY_DAYS': str(args.prism_release_latency_days),
                'OVERWRITE_EXISTING': '0',
            },
            'prism_range',
        ),
        'snodas': submit_sbatch_job(
            SNODAS_WORKER,
            {
                'START_DATE': args.start_date,
                'END_DATE': args.end_date,
                'REGISTRY_PATH': str(args.registry_path),
                'GRID_PATH': str(args.grid_path),
                'RAW_ROOT': str(args.snodas_raw_root),
                'OUTPUT_ROOT': str(args.regrid_root),
                'PLOTS_DIR': str(args.snodas_plots_dir),
                'SWE_PRODUCT_TOKEN': str(args.snodas_swe_product_token),
                'OVERWRITE_EXISTING': '0',
            },
            'snodas_range',
        ),
    }
    wait_for_jobs(job_ids, 'low_latency_source')


def target_file_path(regrid_root: Path, var_name: str, date_value: pd.Timestamp) -> Path:
    return (
        regrid_root
        / date_value.strftime('%Y')
        / f'daymet_v4_daily_na_{var_name}_{date_value.strftime("%Y%m%d")}_regridded.nc'
    )


def load_target_grid(grid_path: Path) -> xr.Dataset:
    return xr.open_dataset(grid_path)


def build_day_dataset(date_value: pd.Timestamp, regrid_root: Path, target_grid: xr.Dataset) -> xr.Dataset:
    data_stack = []
    for var_name in EXPECTED_VARIABLES:
        file_path = target_file_path(regrid_root, var_name, date_value)
        if not file_path.exists():
            raise FileNotFoundError(f'Missing low-latency daily file for {var_name} {date_value.date()}: {file_path}')
        with xr.open_dataset(file_path) as ds_var:
            data_stack.append(np.asarray(ds_var[var_name].values, dtype=np.float32))

    stacked = np.stack(data_stack, axis=0)[None, ...]
    ds = xr.Dataset(
        {
            'data': xr.DataArray(
                stacked,
                dims=('time', 'variable', 'y', 'x'),
                coords={
                    'time': [date_value.to_datetime64()],
                    'variable': EXPECTED_VARIABLES,
                    'y': target_grid['y'].values,
                    'x': target_grid['x'].values,
                },
                attrs={
                    'long_name': 'low-latency climate inputs',
                    'variable_order': ','.join(EXPECTED_VARIABLES),
                },
            )
        },
        coords={
            'time': ('time', [date_value.to_datetime64()]),
            'variable': ('variable', EXPECTED_VARIABLES),
            'y': target_grid['y'].values,
            'x': target_grid['x'].values,
            'lat': (('y', 'x'), np.asarray(target_grid['lat'].values, dtype=np.float64)),
            'lon': (('y', 'x'), np.asarray(target_grid['lon'].values, dtype=np.float64)),
        },
        attrs={
            'source': 'PRISM+SNODAS low-latency climate stack',
            'expected_variables': ','.join(EXPECTED_VARIABLES),
        },
    )
    return ds


def encoding_for_dataset():
    return {
        'data': {
            'chunks': DATA_CHUNKS,
            'dtype': 'float32',
        },
        'lat': {
            'chunks': LATLON_CHUNKS,
            'dtype': 'float64',
        },
        'lon': {
            'chunks': LATLON_CHUNKS,
            'dtype': 'float64',
        },
    }


def backup_incompatible_store(out_zarr: Path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = out_zarr.parent / f'{out_zarr.name}.invalid_{timestamp}'
    print(f'Backing up incompatible low-latency zarr: {out_zarr} -> {backup_path}')
    shutil.move(str(out_zarr), str(backup_path))
    return backup_path


def existing_store_info(out_zarr: Path):
    if not out_zarr.exists():
        return None
    try:
        ds = xr.open_zarr(out_zarr, consolidated=False)
    except Exception:
        backup_incompatible_store(out_zarr)
        return None
    try:
        variables = ds['variable'].values.tolist()
        times = pd.to_datetime(ds['time'].values).normalize()
        data_dims = ds['data'].dims
        data_shape = ds['data'].shape
        return {
            'variables': variables,
            'times': times,
            'data_dims': data_dims,
            'data_shape': data_shape,
        }
    except Exception:
        ds.close()
        backup_incompatible_store(out_zarr)
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def ensure_compatible_store(out_zarr: Path):
    info = existing_store_info(out_zarr)
    if info is None:
        return None
    if list(info['variables']) != EXPECTED_VARIABLES:
        backup_incompatible_store(out_zarr)
        return None
    if tuple(info['data_dims']) != ('time', 'variable', 'y', 'x'):
        backup_incompatible_store(out_zarr)
        return None
    return info


def truncate_store_before_date(out_zarr: Path, start_date: pd.Timestamp, info: dict):
    times = [pd.Timestamp(ts).normalize() for ts in info['times']]
    prefix_count = sum(1 for ts in times if ts < start_date)
    print(
        f'Truncating low-latency climate store before rebuild start '
        f'{start_date.date()}; keeping {prefix_count} time steps'
    )
    root = zarr.open_group(str(out_zarr), mode='a')
    root['time'].resize((prefix_count,))
    root['data'].resize(
        (
            prefix_count,
            root['data'].shape[1],
            root['data'].shape[2],
            root['data'].shape[3],
        )
    )


def append_range(args, start_date: pd.Timestamp, end_date: pd.Timestamp):
    target_grid = load_target_grid(args.grid_path)
    try:
        info = ensure_compatible_store(args.out_zarr)
        existing_dates = set()
        max_existing_date = None
        requested_dates = {pd.Timestamp(ts).normalize() for ts in pd.date_range(start_date, end_date, freq='D')}
        if info is not None:
            existing_dates = {pd.Timestamp(ts).normalize() for ts in info['times']}
            if existing_dates:
                max_existing_date = max(existing_dates)
                print(f'Existing low-latency climate store max date: {max_existing_date.date()}')
                if requested_dates.issubset(existing_dates):
                    print(
                        'Requested low-latency climate range is already fully present in the '
                        f'combined store; skipping rebuild for {start_date.date()} -> {end_date.date()}'
                    )
                    return
                if max_existing_date > end_date:
                    raise ValueError(
                        'Requested low-latency climate rebuild does not reach the current store tail; '
                        f'max existing date is {max_existing_date.date()} but requested end_date is {end_date.date()}'
                    )
                if start_date <= max_existing_date:
                    truncate_store_before_date(args.out_zarr, start_date, info)
                    existing_dates = {ts for ts in existing_dates if ts < start_date}
                    max_existing_date = max(existing_dates) if existing_dates else None

        dates = pd.date_range(start_date, end_date, freq='D')
        wrote_any = False
        for date_value in dates:
            date_norm = pd.Timestamp(date_value).normalize()
            if date_norm in existing_dates:
                print(f'[SKIP] Low-latency climate store already contains {date_norm.date()}')
                continue
            if max_existing_date is not None and date_norm < max_existing_date:
                raise ValueError(
                    f'Cannot append low-latency climate date {date_norm.date()} before existing max '
                    f'{max_existing_date.date()}; rebuild or use a fresh output store'
                )
            ds_day = build_day_dataset(date_norm, args.regrid_root, target_grid)
            try:
                if args.out_zarr.exists():
                    ds_day.to_zarr(
                        args.out_zarr,
                        mode='a',
                        append_dim='time',
                    )
                else:
                    ds_day.to_zarr(
                        args.out_zarr,
                        mode='w',
                        consolidated=False,
                        encoding=encoding_for_dataset(),
                    )
                print(f'Appended low-latency climate date {date_norm.date()} to {args.out_zarr}')
                wrote_any = True
            finally:
                ds_day.close()

        if args.out_zarr.exists():
            consolidate_metadata(str(args.out_zarr))
        if not wrote_any:
            print('No new low-latency climate dates needed append')
    finally:
        target_grid.close()


def main():
    args = apply_registry_defaults(parse_args())
    start_date, end_date = parse_date_range(args.start_date, args.end_date)
    print(f'Updating combined low-latency climate store for {start_date.date()} -> {end_date.date()}')
    update_sources(args)
    if args.check_only:
        print('Low-latency climate source checks passed')
        return
    append_range(args, start_date, end_date)
    print(f'Finished low-latency climate update for {start_date.date()} -> {end_date.date()}')


if __name__ == '__main__':
    main()
