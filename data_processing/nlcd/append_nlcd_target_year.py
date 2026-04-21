#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc
from tqdm import tqdm
import zarr

RAW_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_2000_2024.zarr')
TARGET_GRID = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
OUT_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_target_grid_2000_2024.zarr')
BOX_SIZE = 500
COMP = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
CHUNKS = (500, 500, 1)

NLCD_CODE_TO_NAME = {
    11: 'water',
    12: 'water',
    21: 'developed',
    22: 'developed',
    23: 'developed',
    24: 'developed',
    31: 'barren',
    41: 'deciduous_forest',
    42: 'evergreen_forest',
    43: 'mixed_forest',
    52: 'shrub',
    71: 'grass',
    81: 'crops',
    82: 'crops',
    90: 'wetlands',
    95: 'wetlands',
}
CLASS_NAMES = [
    'barren',
    'crops',
    'deciduous_forest',
    'developed',
    'evergreen_forest',
    'grass',
    'mixed_forest',
    'other',
    'shrub',
    'water',
    'wetlands',
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MAX_CODE = max(NLCD_CODE_TO_NAME.keys())
CODE_TO_CLASS_INDEX = np.full(MAX_CODE + 1, CLASS_TO_INDEX['other'], dtype=np.int16)
for code, name in NLCD_CODE_TO_NAME.items():
    CODE_TO_CLASS_INDEX[code] = CLASS_TO_INDEX[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Append one annual target-grid NLCD fraction slice into the canonical NLCD target zarr store.')
    parser.add_argument('--raw_zarr', type=Path, default=RAW_ZARR)
    parser.add_argument('--target_grid', type=Path, default=TARGET_GRID)
    parser.add_argument('--out_zarr', type=Path, default=OUT_ZARR)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--box_size', type=int, default=BOX_SIZE)
    return parser.parse_args()


def validate_existing_target_store(out_zarr: Path) -> None:
    zarr_v3_meta = out_zarr / 'zarr.json'
    zarr_v2_meta = out_zarr / '.zgroup'
    if zarr_v3_meta.exists() and zarr_v2_meta.exists():
        raise ValueError(
            f'NLCD target zarr has mixed format metadata (both zarr.json and .zgroup): {out_zarr}. '
            'Rebuild the staged target store before appending.'
        )

    try:
        ds = xr.open_zarr(out_zarr, consolidated=False)
    except Exception as exc:
        raise ValueError(f'Failed to open existing NLCD target zarr cleanly: {out_zarr}') from exc

    dims = set(ds.dims)
    ds.close()
    if 'year' not in dims:
        raise ValueError(
            f"Existing NLCD target zarr does not expose the expected 'year' dimension: {out_zarr}. "
            'Rebuild the staged target store before appending.'
        )


def existing_years_and_dtype(out_zarr: Path) -> tuple[set[int], str | None]:
    if not out_zarr.exists():
        return set(), None
    validate_existing_target_store(out_zarr)
    ds = xr.open_zarr(out_zarr, consolidated=False)
    years = set(pd.to_datetime(ds['year'].values).year.astype(int).tolist())
    first_var = CLASS_NAMES[0]
    return years, str(ds[first_var].dtype)


def init_output_dataset(target_grid: xr.Dataset, year_value: pd.Timestamp, dtype: str) -> xr.Dataset:
    target_nlcd = target_grid.copy().drop_vars('random_vals').expand_dims(year=[year_value]).assign_coords(year=[year_value])
    fill = np.full((target_grid.sizes['y'], target_grid.sizes['x'], 1), np.nan, dtype=np.dtype(dtype))
    for name in CLASS_NAMES:
        target_nlcd[name] = (('y', 'x', 'year'), fill.copy())
    return target_nlcd


def box_bounds(target_grid: xr.Dataset, box_size: int) -> list[tuple[int, int, int, int]]:
    x_len = target_grid.sizes['x']
    y_len = target_grid.sizes['y']
    bounds = []
    for y_start in range(0, y_len, box_size):
        for x_start in range(0, x_len, box_size):
            bounds.append((y_start, min(y_start + box_size, y_len), x_start, min(x_start + box_size, x_len)))
    return bounds


def main() -> None:
    args = parse_args()
    years_present, target_dtype = existing_years_and_dtype(args.out_zarr)
    if args.year in years_present:
        print(f'NLCD target zarr already contains year {args.year}; skipping append')
        return
    if target_dtype is None:
        target_dtype = 'float64'

    year_ts = pd.Timestamp(f'{args.year}-01-01')
    raw_ds = xr.open_zarr(args.raw_zarr).sortby('x').sortby('y')
    raw_years = pd.to_datetime(raw_ds['time'].values).year.astype(int)
    matches = np.where(raw_years == args.year)[0]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one raw NLCD slice for year {args.year}, found {len(matches)}')
    raw_year = raw_ds.isel(time=int(matches[0]))

    target_grid = xr.open_dataset(args.target_grid).sortby('x').sortby('y')
    out_ds = init_output_dataset(target_grid, year_ts, target_dtype)

    x_res = float(np.abs(target_grid['x'].values[1] - target_grid['x'].values[0]))
    y_res = float(np.abs(target_grid['y'].values[1] - target_grid['y'].values[0]))
    bounds = box_bounds(target_grid, args.box_size)
    print(f'Processing NLCD target append for year {args.year} across {len(bounds)} boxes')

    for y0, y1, x0, x1 in tqdm(bounds, desc='NLCD boxes'):
        target_subset = target_grid.isel(y=slice(y0, y1), x=slice(x0, x1)).copy()
        valid_mask = ~np.isnan(target_subset['random_vals'].values)
        if np.all(~valid_mask):
            continue

        min_x = float(target_subset['x'].values.min())
        max_x = float(target_subset['x'].values.max())
        min_y = float(target_subset['y'].values.min())
        max_y = float(target_subset['y'].values.max())
        raw_subset = raw_year.sel(
            x=slice(min_x - x_res, max_x + x_res),
            y=slice(min_y - y_res, max_y + y_res),
        ).compute()

        raw_values = raw_subset['nlcd'].values
        raw_values = np.where(np.isnan(raw_values), -1, raw_values).astype(np.int32)
        raw_x = raw_subset['x'].values
        raw_y = raw_subset['y'].values
        x_t = target_subset['x'].values
        y_t = target_subset['y'].values
        if x_t.size < 2 or y_t.size < 2:
            continue

        dx = float(np.abs(x_t[1] - x_t[0]))
        dy = float(np.abs(y_t[1] - y_t[0]))
        x_edges = np.empty(x_t.size + 1)
        y_edges = np.empty(y_t.size + 1)
        x_edges[1:-1] = (x_t[:-1] + x_t[1:]) / 2.0
        y_edges[1:-1] = (y_t[:-1] + y_t[1:]) / 2.0
        x_edges[0] = x_t[0] - dx / 2.0
        x_edges[-1] = x_t[-1] + dx / 2.0
        y_edges[0] = y_t[0] - dy / 2.0
        y_edges[-1] = y_t[-1] + dy / 2.0

        fine_x_idx = np.searchsorted(x_edges, raw_x) - 1
        fine_y_idx = np.searchsorted(y_edges, raw_y) - 1
        fine_x_idx_2d, fine_y_idx_2d = np.meshgrid(fine_x_idx, fine_y_idx)
        inside = (
            (fine_x_idx_2d >= 0) & (fine_x_idx_2d < x_t.size) &
            (fine_y_idx_2d >= 0) & (fine_y_idx_2d < y_t.size)
        )
        if not np.any(inside):
            continue

        flat_coarse_idx = (fine_y_idx_2d[inside] * x_t.size) + fine_x_idx_2d[inside]
        lc = raw_values[inside]
        valid_lc = lc >= 0
        if not np.any(valid_lc):
            continue
        lc_valid = lc[valid_lc]
        flat_idx_valid = flat_coarse_idx[valid_lc]

        lc_idx = np.full(lc_valid.shape, CLASS_TO_INDEX['other'], dtype=np.int16)
        in_range = (lc_valid >= 0) & (lc_valid <= MAX_CODE)
        lc_idx[in_range] = CODE_TO_CLASS_INDEX[lc_valid[in_range]]

        n_coarse = x_t.size * y_t.size
        year_counts = np.zeros(n_coarse * len(CLASS_NAMES), dtype=np.int32)
        flat_idx = flat_idx_valid * len(CLASS_NAMES) + lc_idx
        np.add.at(year_counts, flat_idx, 1)
        year_counts = year_counts.reshape(y_t.size, x_t.size, len(CLASS_NAMES))
        year_counts = np.moveaxis(year_counts, -1, 0)

        box_totals = year_counts.sum(axis=0, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            box_fracs = year_counts / box_totals

        for class_idx, class_name in enumerate(CLASS_NAMES):
            arr = box_fracs[class_idx].astype(np.dtype(target_dtype), copy=False)
            arr[~valid_mask] = np.nan
            out_ds[class_name].loc[dict(y=y_t, x=x_t, year=[year_ts])] = arr[:, :, np.newaxis]

    data_to_write = out_ds[CLASS_NAMES]
    encoding = {
        name: {
            'compressor': COMP,
            'chunks': CHUNKS,
            'dtype': target_dtype,
        }
        for name in CLASS_NAMES
    }

    if args.out_zarr.exists():
        max_year = max(years_present)
        if args.year <= max_year:
            raise ValueError(f'Append script only supports new forward years; requested {args.year} but store max is {max_year}')
        print(f'Appending target-grid NLCD year {args.year} to existing store {args.out_zarr}')
        data_to_write.to_zarr(
            args.out_zarr,
            mode='a',
            append_dim='year',
            consolidated=False,
            zarr_format=2,
        )
    else:
        print(f'Creating target-grid NLCD store at {args.out_zarr} with first year {args.year}')
        out_ds.to_zarr(
            args.out_zarr,
            mode='w',
            consolidated=False,
            safe_chunks=False,
            zarr_format=2,
            encoding=encoding,
        )

    zarr.consolidate_metadata(str(args.out_zarr))
    print(f'Finished NLCD target append for year {args.year}: {args.out_zarr}')


if __name__ == '__main__':
    main()
