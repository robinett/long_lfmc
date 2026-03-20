#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
import xarray as xr
from numcodecs import Blosc
import zarr

from nlcd_to_zarr import open_lazy, year_from_name

RAW_DIR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_raw')
OUT_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_2000_2024.zarr')
COMP = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
WRITE_CHUNKS = (1, 2048, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Append one annual NLCD raster into the canonical raw NLCD zarr store.')
    parser.add_argument('--raw_dir', type=Path, default=RAW_DIR)
    parser.add_argument('--out_zarr', type=Path, default=OUT_ZARR)
    parser.add_argument('--year', type=int, required=True)
    return parser.parse_args()


def locate_year_file(raw_dir: Path, year: int) -> Path:
    matches = []
    for path_str in sorted(glob.glob(str(raw_dir / '*.tif'))):
        if year_from_name(path_str) == year:
            matches.append(Path(path_str))
    if not matches:
        raise FileNotFoundError(f'No NLCD tif found for year {year} in {raw_dir}')
    if len(matches) > 1:
        raise ValueError(f'Multiple NLCD tifs found for year {year}: {matches}')
    return matches[0]


def existing_years(out_zarr: Path) -> tuple[set[int], str | None]:
    if not out_zarr.exists():
        return set(), None
    ds = xr.open_zarr(out_zarr)
    years = set(pd.to_datetime(ds['time'].values).year.astype(int).tolist())
    return years, str(ds['nlcd'].dtype)


def main() -> None:
    args = parse_args()
    tif_path = locate_year_file(args.raw_dir, args.year)
    years_present, target_dtype = existing_years(args.out_zarr)
    if args.year in years_present:
        print(f'NLCD raw zarr already contains year {args.year}; skipping append')
        return

    if target_dtype is None:
        target_dtype = 'float64'

    print(f'Opening NLCD tif for year {args.year}: {tif_path}')
    da = open_lazy(str(tif_path)).astype(target_dtype).expand_dims(time=[pd.Timestamp(f'{args.year}-01-01')])
    ds = da.to_dataset().transpose('time', 'y', 'x')

    encoding = {
        'nlcd': {
            'compressor': COMP,
            'chunks': WRITE_CHUNKS,
            'dtype': target_dtype,
        }
    }

    if args.out_zarr.exists():
        max_year = max(years_present)
        if args.year <= max_year:
            raise ValueError(f'Append script only supports new forward years; requested {args.year} but store max is {max_year}')
        print(f'Appending NLCD raw year {args.year} to existing store {args.out_zarr}')
        ds.to_zarr(
            args.out_zarr,
            mode='a',
            append_dim='time',
            consolidated=False,
            zarr_format=2,
        )
    else:
        print(f'Creating NLCD raw store at {args.out_zarr} with first year {args.year}')
        ds.to_zarr(
            args.out_zarr,
            mode='w',
            consolidated=False,
            safe_chunks=False,
            zarr_format=2,
            encoding=encoding,
        )

    zarr.consolidate_metadata(str(args.out_zarr))
    print(f'Finished NLCD raw append for year {args.year}: {args.out_zarr}')


if __name__ == '__main__':
    main()
