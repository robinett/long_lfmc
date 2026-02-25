import argparse
from pathlib import Path
import re
import shutil

import pandas as pd
import xarray as xr
from numcodecs import Blosc
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from zarr_build_utils import (
    DEFAULT_COMP,
    preprocess_strip_attrs,
    open_time_batch,
    to_stacked_array,
    chunk_coords,
    write_first,
    append_time,
    consolidate,
    find_batches_by_year_or_month,
    files_for_batch,
)

# ---------- CONFIG ----------
ROOT = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_combined"
)

OUT = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_combined_all_vars.zarr"
)

# Keep identical chunking to your current script
WRITE_CHUNKS = {
    "time": 1,
    "variable": 14,   # one var-chunk (14 vars total)
    "y": 512,
    "x": 512,
}

CAST_FLOAT32   = True
ENGINE         = "netcdf4"
PARALLEL_OPEN  = False        # match your current choice
COMP           = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)
# ---------- /CONFIG ----------

DATE_RE = re.compile(r"(\d{8})")


def preprocess_modis_raw(ds: xr.Dataset) -> xr.Dataset:
    ds = preprocess_strip_attrs(ds)
    source = ds.encoding.get("source", "")
    match = DATE_RE.search(str(source))
    if match is None:
        raise ValueError(f"Could not parse date from source file path: {source}")
    time_val = pd.to_datetime(match.group(1), format="%Y%m%d")
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=("time", [time_val]))
    return ds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw MODIS daily NetCDF files to a single time-stacked Zarr."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT),
        help="Root directory containing raw MODIS daily files in YYYY/MM folders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT),
        help="Output Zarr store path.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=ENGINE,
        help="xarray engine for opening NetCDF files (e.g., netcdf4, h5netcdf).",
    )
    parser.add_argument(
        "--parallel_open",
        action="store_true",
        help="Enable parallel open in xarray.open_mfdataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Removing existing output store: {out}")
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()

    # Use year/month directories as batches (same effect as your list_month_dirs)
    batches = find_batches_by_year_or_month(root, patterns=(".nc4",))
    if len(batches) == 0:
        raise ValueError(f"No batches found under {root}")
    print(f"Found {len(batches)} batches")
    is_first = True

    batch_iter = tqdm(batches, desc="Batches", unit="batch") if tqdm is not None else batches
    for i, bdir in enumerate(batch_iter, start=1):
        files = files_for_batch(bdir, patterns=(".nc4",))
        if not files:
            continue
        if tqdm is not None:
            batch_iter.set_postfix_str(f"{bdir.name} ({len(files)} files)")
        print(f"Batch {i}/{len(batches)}: {bdir}  ({len(files)} files)")

        # Open + clean one batch (concat on time)
        ds = open_time_batch(
            files,
            engine=args.engine,
            parallel_open=args.parallel_open,
            cast_float32=CAST_FLOAT32,
            preprocess=preprocess_modis_raw,
        )

        # Stack → chunk → chunk coords (identical layout)
        arr = to_stacked_array(ds, WRITE_CHUNKS)
        arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])

        if is_first:
            write_first(arr, out, compressor=COMP)
            is_first = False
        else:
            append_time(arr, out)

        ds.close()

    consolidate(out)
    print("Done:", out)

if __name__ == "__main__":
    main()
