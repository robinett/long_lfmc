#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from zarr import consolidate_metadata


REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'
EXPECTED_VARIABLES = ['prcp', 'srad', 'swe', 'tmax', 'vp']
DATA_CHUNKS = (1, len(EXPECTED_VARIABLES), 512, 512)
LATLON_CHUNKS = (512, 512)


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
    args.regrid_root = args.regrid_root or Path(climate_cfg['regrid_root'])
    args.out_zarr = args.out_zarr or Path(climate_cfg['zarr_path'])
    args.append_coord_dir = args.append_coord_dir or Path(climate_cfg['append_coord_dir'])
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


def update_sources(args):
    prism_script = REPO_ROOT / 'data_processing/prism/update_prism_range.py'
    snodas_script = REPO_ROOT / 'data_processing/snodas/update_snodas_range.py'
    common = [
        '--start_date', args.start_date,
        '--end_date', args.end_date,
        '--registry_path', args.registry_path,
        '--grid_path', args.grid_path,
    ]
    prism_cmd = [sys.executable, prism_script] + common
    snodas_cmd = [sys.executable, snodas_script] + common
    if args.check_only:
        prism_cmd.append('--check_only')
        snodas_cmd.append('--check_only')
    run_cmd(prism_cmd)
    run_cmd(snodas_cmd)


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


def append_range(args, start_date: pd.Timestamp, end_date: pd.Timestamp):
    target_grid = load_target_grid(args.grid_path)
    try:
        info = ensure_compatible_store(args.out_zarr)
        existing_dates = set()
        max_existing_date = None
        if info is not None:
            existing_dates = {pd.Timestamp(ts).normalize() for ts in info['times']}
            if existing_dates:
                max_existing_date = max(existing_dates)
                print(f'Existing low-latency climate store max date: {max_existing_date.date()}')

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
