#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from map_runtime_utils import (
    OUTPUT_DOMINANT_LANDCOVER_NAME,
    OUTPUT_LANDCOVER_YEAR_NAME,
    OUTPUT_MEAN_NAME,
    OUTPUT_QUALITY_FLAG_NAME,
    OUTPUT_STD_NAME,
)


TIME_VARS = [OUTPUT_MEAN_NAME, OUTPUT_STD_NAME, OUTPUT_QUALITY_FLAG_NAME]


def timestamped_message(message: str) -> str:
    return f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"


def get_args():
    parser = argparse.ArgumentParser(description='Promote staged LFMC map outputs into a production zarr store.')
    parser.add_argument('--staging_zarr', type=str, required=True)
    parser.add_argument('--production_zarr', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--mode', choices=['append_time_range', 'overwrite_time_range'], required=True)
    parser.add_argument('--tier', type=str, required=True)
    parser.add_argument('--metadata_dir', type=str, default=None)
    parser.add_argument('--initialize_if_missing', action='store_true')
    return parser.parse_args()


def _load_time_index(root) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(root['time'][:], dtype=np.int64))


def _load_landcover_years(root) -> np.ndarray:
    if OUTPUT_LANDCOVER_YEAR_NAME not in root:
        return np.asarray([], dtype=np.int32)
    return np.asarray(root[OUTPUT_LANDCOVER_YEAR_NAME][:], dtype=np.int32)


def _ensure_static_coords_match(staging_root, production_root):
    for coord_name in ['x', 'y']:
        if coord_name not in staging_root or coord_name not in production_root:
            raise ValueError(f'Missing coordinate {coord_name} in staging or production store')
        if not np.array_equal(staging_root[coord_name][:], production_root[coord_name][:]):
            raise ValueError(f'Coordinate mismatch for {coord_name}')
    for coord_name in ['lat', 'lon']:
        if coord_name in staging_root and coord_name in production_root:
            if not np.array_equal(staging_root[coord_name][:], production_root[coord_name][:]):
                raise ValueError(f'Coordinate mismatch for {coord_name}')


def _get_time_slice(index: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp) -> slice:
    positions = np.where((index >= start_date) & (index <= end_date))[0]
    if len(positions) == 0:
        raise ValueError(f'No time coordinates found between {start_date.date()} and {end_date.date()}')
    expected = np.arange(positions[0], positions[0] + len(positions))
    if not np.array_equal(positions, expected):
        raise ValueError('Selected time range is not contiguous in the zarr store')
    return slice(int(positions[0]), int(positions[-1]) + 1)


def _copy_landcover_years(staging_root, production_root):
    if OUTPUT_DOMINANT_LANDCOVER_NAME not in staging_root or OUTPUT_LANDCOVER_YEAR_NAME not in staging_root:
        return
    staging_years = _load_landcover_years(staging_root)
    if OUTPUT_DOMINANT_LANDCOVER_NAME not in production_root or OUTPUT_LANDCOVER_YEAR_NAME not in production_root:
        raise ValueError('Production store is missing yearly landcover datasets')
    production_years = _load_landcover_years(production_root)
    year_to_prod_idx = {int(year): idx for idx, year in enumerate(production_years.tolist())}
    landcover_arr = production_root[OUTPUT_DOMINANT_LANDCOVER_NAME]
    year_arr = production_root[OUTPUT_LANDCOVER_YEAR_NAME]
    for stage_idx, year in enumerate(staging_years.tolist()):
        year = int(year)
        if year in year_to_prod_idx:
            prod_idx = year_to_prod_idx[year]
            landcover_arr[prod_idx, :, :] = staging_root[OUTPUT_DOMINANT_LANDCOVER_NAME][stage_idx, :, :]
            continue
        old_n = int(year_arr.shape[0])
        new_n = old_n + 1
        year_arr.resize(new_n)
        landcover_arr.resize(new_n, landcover_arr.shape[1], landcover_arr.shape[2])
        year_arr[old_n:new_n] = np.asarray([year], dtype=np.int32)
        landcover_arr[old_n:new_n, :, :] = staging_root[OUTPUT_DOMINANT_LANDCOVER_NAME][stage_idx:stage_idx + 1, :, :]
        year_to_prod_idx[year] = old_n


def _append_time_range(staging_root, production_root, staging_slice: slice, production_time: pd.DatetimeIndex):
    staging_time = _load_time_index(staging_root)[staging_slice]
    if len(production_time) > 0 and staging_time[0] <= production_time[-1]:
        raise ValueError('append_time_range requires staged dates to be strictly newer than production time coverage')
    old_n = int(len(production_time))
    new_n = old_n + int(len(staging_time))
    production_root['time'].resize(new_n)
    production_root['time'][old_n:new_n] = np.asarray(staging_root['time'][staging_slice], dtype=np.int64)
    for var_name in TIME_VARS:
        arr = production_root[var_name]
        arr.resize(new_n, arr.shape[1], arr.shape[2]) if var_name != OUTPUT_QUALITY_FLAG_NAME else arr.resize(new_n)
        arr[old_n:new_n] = staging_root[var_name][staging_slice]


def _overwrite_time_range(staging_root, production_root, staging_slice: slice, production_slice: slice):
    staging_time = _load_time_index(staging_root)[staging_slice]
    production_time = _load_time_index(production_root)[production_slice]
    if not np.array_equal(staging_time.values.astype('datetime64[ns]'), production_time.values.astype('datetime64[ns]')):
        raise ValueError('Staging and production time coordinates differ for overwrite range')
    for var_name in TIME_VARS:
        production_root[var_name][production_slice] = staging_root[var_name][staging_slice]


def _write_metadata_record(metadata_dir: str, record: dict):
    os.makedirs(metadata_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(metadata_dir, f'production_update_{stamp}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(timestamped_message(f'Wrote promotion metadata to {path}'))


def main():
    args = get_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    staging_zarr = Path(args.staging_zarr)
    production_zarr = Path(args.production_zarr)
    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = str(production_zarr.parent / 'metadata')

    if not staging_zarr.exists():
        raise FileNotFoundError(f'Missing staging zarr: {staging_zarr}')

    if not production_zarr.exists():
        if not args.initialize_if_missing:
            raise FileNotFoundError(
                f'Missing production zarr: {production_zarr}. Use --initialize_if_missing to seed it from staging.'
            )
        print(timestamped_message(f'Initializing production store from staging: {production_zarr}'))
        shutil.copytree(staging_zarr, production_zarr)
        record = {
            'mode': args.mode,
            'tier': args.tier,
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date()),
            'staging_zarr': str(staging_zarr),
            'production_zarr': str(production_zarr),
            'status': 'completed_initialized_from_staging',
        }
        _write_metadata_record(metadata_dir, record)
        return

    staging_root = zarr.open_group(str(staging_zarr), mode='r')
    production_root = zarr.open_group(str(production_zarr), mode='a')
    _ensure_static_coords_match(staging_root, production_root)

    staging_time = _load_time_index(staging_root)
    production_time = _load_time_index(production_root)
    staging_slice = _get_time_slice(staging_time, start_date, end_date)

    if args.mode == 'append_time_range':
        _append_time_range(staging_root, production_root, staging_slice, production_time)
    else:
        production_slice = _get_time_slice(production_time, start_date, end_date)
        _overwrite_time_range(staging_root, production_root, staging_slice, production_slice)

    _copy_landcover_years(staging_root, production_root)
    production_root.attrs['last_promotion_mode'] = args.mode
    production_root.attrs['last_promotion_tier'] = args.tier
    production_root.attrs['last_promotion_at'] = dt.datetime.now().isoformat()

    record = {
        'mode': args.mode,
        'tier': args.tier,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'staging_zarr': str(staging_zarr),
        'production_zarr': str(production_zarr),
        'status': 'completed',
    }
    _write_metadata_record(metadata_dir, record)
    print(timestamped_message('Promotion completed successfully'))


if __name__ == '__main__':
    main()
