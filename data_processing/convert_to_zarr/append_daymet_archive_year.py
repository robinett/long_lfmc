#!/usr/bin/env python3

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import pandas as pd
import xarray as xr

from daymet_to_zarr_worker import (
    CAST_FLOAT32,
    COMP,
    DAYMET_VAR_WHITELIST,
    ENGINE,
    OUT,
    PARALLEL_OPEN,
    ROOT,
    WRITE_CHUNKS,
    load_or_build_month_index,
)
from zarr_build_utils import append_time, consolidate, open_time_batch, parse_daymet_regrid_filename, to_stacked_array, write_first


def maybe_fill_missing_leap_dec31(ds: xr.Dataset, year: str, month: str) -> xr.Dataset:
    if month != '12' or not calendar.isleap(int(year)):
        return ds
    if 'time' not in ds.coords:
        return ds

    times = pd.to_datetime(ds['time'].values)
    if len(times) == 0:
        return ds

    norm = pd.DatetimeIndex(times).normalize()
    dec30 = pd.Timestamp(f'{year}-12-30')
    dec31 = pd.Timestamp(f'{year}-12-31')
    if (norm == dec31).any():
        return ds
    dec30_idx = norm.get_indexer([dec30])
    if dec30_idx[0] < 0:
        raise ValueError(f'Leap-year December missing both Dec 30 and Dec 31 for {year}')

    fill = ds.isel(time=[int(dec30_idx[0])]).copy(deep=False)
    fill = fill.assign_coords(time=('time', [dec31.to_datetime64()]))
    print(f'Inserted synthetic {year}-12-31 from {year}-12-30')
    return xr.concat([ds, fill], dim='time').sortby('time')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Append one or more archive Daymet months/years into the canonical zarr store.')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out_zarr', type=Path, default=OUT)
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--start_year', type=int, default=None)
    parser.add_argument('--end_year', type=int, default=None)
    parser.add_argument('--start_month', type=int, default=1)
    parser.add_argument('--end_month', type=int, default=12)
    parser.add_argument('--coord_dir', type=Path, default=None)
    parser.add_argument('--rebuild_month_index', action='store_true')
    return parser.parse_args()


def selected_month_labels(month_index: dict[str, list[Path]], start_year: int, end_year: int, start_month: int, end_month: int) -> list[str]:
    labels: list[str] = []
    for ym in sorted(month_index.keys()):
        year_str, month_str = ym.split('-')
        year = int(year_str)
        month = int(month_str)
        if year < start_year or year > end_year:
            continue
        if year == start_year and month < start_month:
            continue
        if year == end_year and month > end_month:
            continue
        labels.append(ym)
    return labels


def existing_max_time(out_zarr: Path) -> pd.Timestamp | None:
    if not out_zarr.exists():
        return None
    ds = xr.open_zarr(out_zarr)
    times = pd.to_datetime(ds['time'].values)
    if len(times) == 0:
        return None
    max_time = pd.Timestamp(times.max()).normalize()
    print(f'Existing archive store max date: {max_time.date()}')
    return max_time


def month_time_bounds(files: list[Path]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    dates: list[pd.Timestamp] = []
    for file_path in files:
        _, raw_date = parse_daymet_regrid_filename(file_path)
        if raw_date is None:
            continue
        dates.append(pd.Timestamp(f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}'))
    if not dates:
        return None, None
    return min(dates), max(dates)


def main() -> None:
    args = parse_args()
    if args.year is not None:
        start_year = args.year
        end_year = args.year
    else:
        start_year = args.start_year
        end_year = args.end_year
    if start_year is None or end_year is None:
        raise SystemExit('Pass --year or both --start_year and --end_year')
    if end_year < start_year:
        raise SystemExit('end_year must be >= start_year')
    if not (1 <= args.start_month <= 12 and 1 <= args.end_month <= 12):
        raise SystemExit('start_month and end_month must be in 1..12')

    coord_dir = args.coord_dir or args.out_zarr.parent / 'append_daymet_coord'
    coord_dir.mkdir(parents=True, exist_ok=True)

    month_index, summary = load_or_build_month_index(coord_dir, args.root, rebuild=args.rebuild_month_index)
    labels = selected_month_labels(month_index, start_year, end_year, args.start_month, args.end_month)
    if not labels:
        raise SystemExit(f'No Daymet months found under {args.root} for requested range')

    print(f'Found {len(labels)} requested Daymet month batches: {labels[0]} -> {labels[-1]}')
    print(f'Daymet vars present in index: {summary.get("vars_seen", [])}')

    max_time = existing_max_time(args.out_zarr)
    appended_months = 0
    appended_days = 0
    skipped_without_open = 0

    for ym in labels:
        files = month_index[ym]
        year, month = ym.split('-')
        batch_min, batch_max = month_time_bounds(files)
        if batch_min is None or batch_max is None:
            print(f'Could not determine date bounds for batch {ym}; skipping')
            continue
        if max_time is not None and batch_max <= max_time:
            skipped_without_open += 1
            print(f'Batch {ym} already covered through {max_time.date()} based on filename dates; skipping without open')
            continue

        print(f'Opening Daymet batch {ym} with {len(files)} files')
        ds = open_time_batch(
            files,
            engine=ENGINE,
            parallel_open=PARALLEL_OPEN,
            cast_float32=CAST_FLOAT32,
            combine='nested',
            data_var_whitelist=DAYMET_VAR_WHITELIST,
        )
        ds = maybe_fill_missing_leap_dec31(ds, year, month)
        ds = ds.sortby('time')
        times = pd.to_datetime(ds['time'].values)
        if len(times) == 0:
            print(f'Batch {ym} is empty after open; skipping')
            continue

        if max_time is not None:
            keep_mask = times > max_time
            if not keep_mask.any():
                print(f'Batch {ym} already present through {max_time.date()}; skipping after open')
                continue
            if not keep_mask.all():
                first_new = pd.Timestamp(times[keep_mask][0]).date()
                print(f'Batch {ym} overlaps existing store; appending only dates >= {first_new}')
            ds = ds.isel(time=keep_mask)
            times = pd.to_datetime(ds['time'].values)

        arr = to_stacked_array(ds, WRITE_CHUNKS)
        if args.out_zarr.exists():
            append_time(arr, args.out_zarr)
        else:
            write_first(arr, args.out_zarr, compressor=COMP)

        max_time = pd.Timestamp(times.max()).normalize()
        appended_months += 1
        appended_days += len(times)
        print(f'Appended {len(times)} days from {ym}; new max date is {max_time.date()}')

    if args.out_zarr.exists():
        consolidate(args.out_zarr)
    print(
        'Finished Daymet archive append: '
        f'appended_months={appended_months}, appended_days={appended_days}, '
        f'skipped_without_open={skipped_without_open}, out_zarr={args.out_zarr}'
    )


if __name__ == '__main__':
    main()
