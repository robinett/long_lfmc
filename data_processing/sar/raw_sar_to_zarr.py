#!/usr/bin/env python3

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm


DATE_RE = re.compile(r"s1_(\d{8})\.nc$")


def parse_date_from_path(path: str) -> pd.Timestamp:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse date from filename: {path}")
    return pd.Timestamp(m.group(1))


def add_time(ds: xr.Dataset, fp: str) -> xr.Dataset:
    t = np.datetime64(pd.to_datetime(parse_date_from_path(fp), format="%Y%m%d"))
    t = t.astype("datetime64[ns]")
    return ds.expand_dims(time=[t])


def filter_files(files, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        return files
    out = []
    for fp in files:
        dt = parse_date_from_path(fp)
        if start_date is not None and dt < start_date:
            continue
        if end_date is not None and dt > end_date:
            continue
        out.append(fp)
    return out


def existing_max_time(out_zarr: str):
    if not os.path.exists(out_zarr):
        return None
    ds = xr.open_zarr(out_zarr, consolidated=False)
    try:
        return pd.Timestamp(pd.to_datetime(ds["time"].values).max())
    finally:
        ds.close()


def build_parser():
    p = argparse.ArgumentParser(description="Convert daily SAR NetCDFs to a zarr store (append-capable).")
    p.add_argument(
        "--raw_dir",
        type=str,
        default="/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily_vv",
        help="Directory containing daily SAR NetCDF files s1_YYYYMMDD.nc",
    )
    p.add_argument(
        "--out_zarr",
        type=str,
        default="/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr",
        help="Output zarr path",
    )
    p.add_argument(
        "--var_name",
        type=str,
        default="vv_backscatter",
        help="Variable name to keep/filter in the daily NetCDF files",
    )
    p.add_argument(
        "--var_min",
        type=float,
        default=-30.0,
        help="Minimum valid backscatter value",
    )
    p.add_argument(
        "--var_max",
        type=float,
        default=5.0,
        help="Maximum valid backscatter value",
    )
    p.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Optional inclusive start date YYYY-MM-DD for raw daily files",
    )
    p.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Optional inclusive end date YYYY-MM-DD for raw daily files",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing zarr (only files newer than current max time are added)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing zarr and rebuild from selected files",
    )
    p.add_argument(
        "--consolidate_only",
        action="store_true",
        help="Only consolidate zarr metadata and exit",
    )
    return p


def main():
    args = build_parser().parse_args()

    raw_dir = args.raw_dir
    out_zarr = args.out_zarr
    var_name = args.var_name
    var_range = [args.var_min, args.var_max]
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    if args.consolidate_only:
        zarr.convenience.consolidate_metadata(out_zarr)
        print("Zarr metadata consolidated. Exiting.")
        return

    if args.append and args.overwrite:
        raise ValueError("Use only one of --append or --overwrite")

    if os.path.exists(out_zarr):
        if args.overwrite:
            print(f"Removing existing zarr for rebuild: {out_zarr}")
            shutil.rmtree(out_zarr)
        elif not args.append:
            raise FileExistsError(
                f"{out_zarr} exists. Use --append to add new files or --overwrite to rebuild."
            )

    files = sorted(glob.glob(f"{raw_dir}/s1_*.nc"))
    files = filter_files(files, start_date=start_date, end_date=end_date)

    if args.append and os.path.exists(out_zarr):
        max_dt = existing_max_time(out_zarr)
        if max_dt is not None:
            before = len(files)
            files = [fp for fp in files if parse_date_from_path(fp) > max_dt.normalize()]
            print(f"Append mode: existing zarr max time = {max_dt.date()} | selected {len(files)} of {before} files newer than max time")

    if not files:
        print("No files selected for processing. Exiting.")
        return

    first = not os.path.exists(out_zarr)
    print(f"Processing {len(files)} files into {out_zarr} (first_write={first})")
    print(f"raw_dir={raw_dir}")
    print(f"var_name={var_name}")

    for fp in tqdm(files, desc="Adding SAR file to zarr"):
        ds = xr.open_dataset(fp, cache=False)
        try:
            ds = ds.where((ds[var_name] >= var_range[0]) & (ds[var_name] <= var_range[1]))
            ds = add_time(ds, fp)
            ds = ds.chunk({"x": 512, "y": 512, "time": 1})

            if not first:
                ds = ds.drop_vars(["lat", "lon"], errors="ignore")

            mode = "w" if first else "a"
            if mode == "a":
                ds.to_zarr(out_zarr, mode=mode, append_dim="time")
            else:
                ds.to_zarr(out_zarr, mode=mode)
            first = False
        finally:
            ds.close()

    zarr.convenience.consolidate_metadata(out_zarr)
    print("Done:", out_zarr)


if __name__ == "__main__":
    main()
