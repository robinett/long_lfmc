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
from low_latency_rollback import capture_zarr_rollback_from_env


TIME_VARS = [OUTPUT_MEAN_NAME, OUTPUT_STD_NAME, OUTPUT_QUALITY_FLAG_NAME]
TIME_UNIT_ALIASES = {
    'nanosecond': 'ns',
    'nanoseconds': 'ns',
    'ns': 'ns',
    'microsecond': 'us',
    'microseconds': 'us',
    'us': 'us',
    'millisecond': 'ms',
    'milliseconds': 'ms',
    'ms': 'ms',
    'second': 's',
    'seconds': 's',
    's': 's',
    'minute': 'm',
    'minutes': 'm',
    'hour': 'h',
    'hours': 'h',
    'day': 'D',
    'days': 'D',
}


def timestamped_message(message: str) -> str:
    return f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"


def get_args():
    parser = argparse.ArgumentParser(description='Promote staged LFMC map outputs into a production zarr store.')
    parser.add_argument('--staging_zarr', type=str, required=True)
    parser.add_argument('--production_zarr', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument(
        '--mode',
        choices=['append_time_range', 'overwrite_time_range', 'replace_tail_range'],
        required=True,
    )
    parser.add_argument('--tier', type=str, required=True)
    parser.add_argument('--metadata_dir', type=str, default=None)
    parser.add_argument('--initialize_if_missing', action='store_true')
    return parser.parse_args()


def _parse_time_units(units: str) -> tuple[str, pd.Timestamp]:
    parts = str(units).strip().split(' since ', 1)
    if len(parts) != 2:
        raise ValueError(f'Unsupported time units: {units}')
    unit = TIME_UNIT_ALIASES.get(parts[0].strip().lower())
    if unit is None:
        raise ValueError(f'Unsupported time unit in units: {units}')
    origin = pd.Timestamp(parts[1].strip()).tz_localize(None)
    return unit, origin


def _decode_time_array(time_arr) -> pd.DatetimeIndex:
    raw = np.asarray(time_arr[:])
    if np.issubdtype(raw.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(raw)).normalize()

    units = str(time_arr.attrs.get('units', '')).strip()
    if not units:
        return pd.DatetimeIndex(pd.to_datetime(np.asarray(raw, dtype=np.int64))).normalize()

    unit, origin = _parse_time_units(units)
    times = origin + pd.to_timedelta(np.asarray(raw, dtype=np.int64), unit=unit)
    return pd.DatetimeIndex(times).normalize()


def _encode_time_array(time_index: pd.DatetimeIndex, time_arr) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(time_index)).normalize()
    if np.issubdtype(time_arr.dtype, np.datetime64):
        return np.asarray(times.values, dtype=time_arr.dtype)

    units = str(time_arr.attrs.get('units', '')).strip()
    if not units:
        return np.asarray(times.values, dtype='datetime64[ns]').astype(np.int64).astype(time_arr.dtype)

    unit, origin = _parse_time_units(units)
    unit_ns = pd.Timedelta(1, unit=unit).value
    deltas = (
        np.asarray(times.values, dtype='datetime64[ns]') - np.datetime64(origin.to_datetime64(), 'ns')
    ).astype('timedelta64[ns]').astype(np.int64)
    if np.any(deltas % unit_ns != 0):
        raise ValueError(f'Time values are not exactly representable in production units: {units}')
    return (deltas // unit_ns).astype(time_arr.dtype)


def _write_time_values(time_arr, write_slice: slice, time_index: pd.DatetimeIndex):
    encoded = _encode_time_array(time_index, time_arr)
    time_arr[write_slice] = encoded
    observed = np.asarray(time_arr[write_slice])
    if not np.array_equal(observed, encoded):
        for offset, value in enumerate(encoded.tolist()):
            time_arr[int(write_slice.start) + offset] = value
        observed = np.asarray(time_arr[write_slice])
    if not np.array_equal(observed, encoded):
        raise RuntimeError(
            "Failed to persist production time coordinate values "
            f"for slice {write_slice}: expected={encoded.tolist()} observed={observed.tolist()}"
        )


def _load_time_index(root) -> pd.DatetimeIndex:
    return _decode_time_array(root['time'])


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
        year_arr.resize((new_n,))
        landcover_arr.resize((new_n, landcover_arr.shape[1], landcover_arr.shape[2]))
        year_arr[old_n:new_n] = np.asarray([year], dtype=np.int32)
        landcover_arr[old_n:new_n, :, :] = staging_root[OUTPUT_DOMINANT_LANDCOVER_NAME][stage_idx:stage_idx + 1, :, :]
        year_to_prod_idx[year] = old_n


def _range_starts(start: int, stop: int, step: int):
    return range(start, stop, max(1, int(step)))


def _resize_time_var(arr, new_n: int):
    if len(arr.shape) == 1:
        arr.resize((new_n,))
        return
    arr.resize((new_n, *arr.shape[1:]))


def _copy_time_var_chunkwise(
    staging_arr,
    production_arr,
    staging_slice: slice,
    production_slice: slice,
    var_name: str,
):
    if len(staging_arr.shape) == 1:
        t_chunk = int(staging_arr.chunks[0])
        n_blocks = max(1, int(np.ceil((staging_slice.stop - staging_slice.start) / t_chunk)))
        block_idx = 0
        for t0 in _range_starts(staging_slice.start, staging_slice.stop, t_chunk):
            t1 = min(staging_slice.stop, t0 + t_chunk)
            prod_t0 = production_slice.start + (t0 - staging_slice.start)
            prod_t1 = prod_t0 + (t1 - t0)
            production_arr[prod_t0:prod_t1] = staging_arr[t0:t1]
            block_idx += 1
            print(timestamped_message(f"{var_name}: copied block {block_idx}/{n_blocks}"))
        return

    t_chunk, y_chunk, x_chunk = [int(v) for v in staging_arr.chunks]
    t_starts = list(_range_starts(staging_slice.start, staging_slice.stop, t_chunk))
    y_starts = list(_range_starts(0, int(staging_arr.shape[1]), y_chunk))
    x_starts = list(_range_starts(0, int(staging_arr.shape[2]), x_chunk))
    total_blocks = len(t_starts) * len(y_starts) * len(x_starts)
    block_idx = 0
    for t0 in t_starts:
        t1 = min(staging_slice.stop, t0 + t_chunk)
        prod_t0 = production_slice.start + (t0 - staging_slice.start)
        prod_t1 = prod_t0 + (t1 - t0)
        for y0 in y_starts:
            y1 = min(int(staging_arr.shape[1]), y0 + y_chunk)
            for x0 in x_starts:
                x1 = min(int(staging_arr.shape[2]), x0 + x_chunk)
                production_arr[prod_t0:prod_t1, y0:y1, x0:x1] = staging_arr[t0:t1, y0:y1, x0:x1]
                block_idx += 1
                if block_idx == 1 or block_idx == total_blocks or block_idx % 200 == 0:
                    print(
                        timestamped_message(
                            f"{var_name}: copied block {block_idx}/{total_blocks}"
                        )
                    )


def _append_time_range(staging_root, production_root, staging_slice: slice, production_time: pd.DatetimeIndex):
    staging_time = _load_time_index(staging_root)[staging_slice]
    if len(production_time) > 0 and staging_time[0] <= production_time[-1]:
        raise ValueError('append_time_range requires staged dates to be strictly newer than production time coverage')
    old_n = int(len(production_time))
    new_n = old_n + int(len(staging_time))
    production_root['time'].resize((new_n,))
    _write_time_values(production_root['time'], slice(old_n, new_n), staging_time)
    production_slice = slice(old_n, new_n)
    for var_name in TIME_VARS:
        arr = production_root[var_name]
        _resize_time_var(arr, new_n)
        _copy_time_var_chunkwise(
            staging_arr=staging_root[var_name],
            production_arr=arr,
            staging_slice=staging_slice,
            production_slice=production_slice,
            var_name=var_name,
        )


def _replace_tail_range(
    staging_root,
    production_root,
    staging_slice: slice,
    production_time: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    staging_time = _load_time_index(staging_root)[staging_slice]
    if len(production_time) > 0 and production_time[-1] > end_date:
        raise ValueError(
            'replace_tail_range requires end_date to cover the current production tail; '
            f'production max is {production_time[-1].date()} but requested end_date is {end_date.date()}'
        )
    prefix_count = int(np.searchsorted(production_time.values, start_date.to_datetime64(), side='left'))
    if prefix_count < 0:
        prefix_count = 0
    for var_name in TIME_VARS:
        _resize_time_var(production_root[var_name], prefix_count)
    production_root['time'].resize((prefix_count,))

    new_n = prefix_count + int(len(staging_time))
    production_root['time'].resize((new_n,))
    _write_time_values(production_root['time'], slice(prefix_count, new_n), staging_time)
    production_slice = slice(prefix_count, new_n)
    for var_name in TIME_VARS:
        arr = production_root[var_name]
        _resize_time_var(arr, new_n)
        _copy_time_var_chunkwise(
            staging_arr=staging_root[var_name],
            production_arr=arr,
            staging_slice=staging_slice,
            production_slice=production_slice,
            var_name=var_name,
        )


def _overwrite_time_range(staging_root, production_root, staging_slice: slice, production_slice: slice):
    staging_time = _load_time_index(staging_root)[staging_slice]
    production_time = _load_time_index(production_root)[production_slice]
    if not np.array_equal(staging_time.values.astype('datetime64[ns]'), production_time.values.astype('datetime64[ns]')):
        raise ValueError('Staging and production time coordinates differ for overwrite range')
    for var_name in TIME_VARS:
        _copy_time_var_chunkwise(
            staging_arr=staging_root[var_name],
            production_arr=production_root[var_name],
            staging_slice=staging_slice,
            production_slice=production_slice,
            var_name=var_name,
        )


def _write_metadata_record(metadata_dir: str, record: dict):
    os.makedirs(metadata_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(metadata_dir, f'production_update_{stamp}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(timestamped_message(f'Wrote promotion metadata to {path}'))


def _consolidate_store_metadata(zarr_path: Path):
    zarr.consolidate_metadata(str(zarr_path))
    print(timestamped_message(f'Consolidated metadata for {zarr_path}'))


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
        _consolidate_store_metadata(production_zarr)
        return

    staging_root = zarr.open_group(str(staging_zarr), mode='r')
    production_root = zarr.open_group(str(production_zarr), mode='a')
    _ensure_static_coords_match(staging_root, production_root)

    staging_time = _load_time_index(staging_root)
    production_time = _load_time_index(production_root)
    staging_slice = _get_time_slice(staging_time, start_date, end_date)
    capture_zarr_rollback_from_env(
        target_zarr=production_zarr,
        label='production_lfmc',
        dim_name='time',
        window_start=start_date,
        window_end=end_date,
        reason=f'before_{args.tier}_promotion_{args.mode}',
    )
    staging_years = _load_landcover_years(staging_root)
    if len(staging_years) > 0:
        capture_zarr_rollback_from_env(
            target_zarr=production_zarr,
            label='production_lfmc_landcover',
            dim_name=OUTPUT_LANDCOVER_YEAR_NAME,
            window_start=int(np.min(staging_years)),
            window_end=int(np.max(staging_years)),
            reason=f'before_{args.tier}_landcover_promotion',
        )

    if args.mode == 'append_time_range':
        _append_time_range(staging_root, production_root, staging_slice, production_time)
    elif args.mode == 'replace_tail_range':
        _replace_tail_range(
            staging_root,
            production_root,
            staging_slice,
            production_time,
            start_date,
            end_date,
        )
    else:
        production_slice = _get_time_slice(production_time, start_date, end_date)
        _overwrite_time_range(staging_root, production_root, staging_slice, production_slice)

    _copy_landcover_years(staging_root, production_root)
    production_root.attrs['last_promotion_mode'] = args.mode
    production_root.attrs['last_promotion_tier'] = args.tier
    production_root.attrs['last_promotion_at'] = dt.datetime.now().isoformat()
    _consolidate_store_metadata(production_zarr)

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
