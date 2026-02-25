# daymet_to_zarr.py
from pathlib import Path
import calendar
import xarray as xr
from numcodecs import Blosc
import numpy as np
import pandas as pd
from zarr_build_utils import (
    DEFAULT_COMP,
    preprocess_strip_attrs,
    open_time_batch,
    to_stacked_array,
    chunk_coords,
    write_first,
    append_time,
    consolidate,
    scan_daymet_regrid_month_index,
)

# ---------- CONFIG ----------
ROOT = Path(
    "/scratch/users/trobinet/long_lfmc/"
    "final_lfmc/daymet/daymet_regrid"
)

OUT = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/"
    "daymet/daymet_all_vars.zarr"
)

# Daymet often has tmin/tmax/prcp/srad/vp/dayl … adjust if needed
VAR_WHITELIST = None
# e.g., VAR_WHITELIST = ["tmin", "tmax", "prcp", "srad", "vp", "dayl"]

WRITE_CHUNKS = {
    "time": 1,
    "variable": 9999,  # will clamp to actual var count
    "y": 512,
    "x": 512,
}

CAST_FLOAT32 = True
ENGINE = "h5netcdf"     # "netcdf4" also fine; set parallel_open=False for stability
PARALLEL_OPEN = False
COMP = DEFAULT_COMP     # Blosc(zstd)
DAYMET_VAR_WHITELIST = ["prcp", "srad", "swe", "tmax", "tmin", "vp"]

# ---------- /CONFIG ----------

def maybe_subset_vars(ds: xr.Dataset, keep=None) -> xr.Dataset:
    if keep is None:
        return ds
    missing = [v for v in keep if v not in ds.data_vars]
    if missing:
        print("Warning: missing variables:", missing)
    present = [v for v in keep if v in ds.data_vars]
    if not present:
        return ds
    return ds[present]


def maybe_fill_missing_leap_dec31(ds: xr.Dataset, year: str, month: str) -> xr.Dataset:
    if month != "12" or not calendar.isleap(int(year)):
        return ds
    if "time" not in ds.coords:
        return ds

    times = pd.to_datetime(ds["time"].values)
    if len(times) == 0:
        return ds

    norm = pd.DatetimeIndex(times).normalize()
    dec30 = pd.Timestamp(f"{year}-12-30")
    dec31 = pd.Timestamp(f"{year}-12-31")
    if (norm == dec31).any():
        return ds

    dec30_idx = np.where(norm == dec30)[0]
    if len(dec30_idx) == 0:
        raise ValueError(f"Leap-year December missing both Dec 30 and Dec 31 for {year}")

    fill = ds.isel(time=[int(dec30_idx[-1])]).copy(deep=False)
    fill = fill.assign_coords(time=("time", [dec31.to_datetime64()]))
    print(f"Inserted synthetic {year}-12-31 from {year}-12-30")
    return xr.concat([ds, fill], dim="time").sortby("time")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    month_index, summary = scan_daymet_regrid_month_index(ROOT)
    month_keys = sorted(month_index.keys())
    if not month_keys:
        raise ValueError(f"No year/month batches found under {ROOT}")
    print("Years found:", len(summary.get("years_seen", [])))
    print("Empty years:", summary.get("empty_years", []))
    print("Vars found:", summary.get("vars_seen", []))
    if summary.get("malformed_count", 0):
        print("Malformed filenames:", summary["malformed_count"])
        for ex in summary.get("malformed_examples", []):
            print("  malformed:", ex)
    is_first = True
    for ym in month_keys:
        year, month = ym.split("-")
        files = [Path(p) for p in month_index[ym]]
        if not files:
            continue
        print(f"Batch: {year}-{month}  ({len(files)} files)")

        # Open/clean one batch (year or month)
        ds = open_time_batch(
            files,
            engine=ENGINE,
            parallel_open=PARALLEL_OPEN,
            cast_float32=CAST_FLOAT32,
            preprocess=preprocess_strip_attrs,
            combine="by_coords",      # <— THIS fixes the NaNs
            data_var_whitelist=DAYMET_VAR_WHITELIST,
        )
        # Optional subset (if Daymet directory also contains derived vars, etc.)
        print('Subsetting variables...')
        ds = maybe_fill_missing_leap_dec31(ds, year, month)
        ds = maybe_subset_vars(ds, VAR_WHITELIST or DAYMET_VAR_WHITELIST)
        # Stack → chunk → chunk coords
        print('Stacking and chunking...')
        arr = to_stacked_array(ds, WRITE_CHUNKS)
        arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])
        # Write (first creates store+encoding; later appends)
        print('Writing to zarr...')
        if is_first:
            write_first(arr, OUT, compressor=COMP)
            is_first = False
        else:
            append_time(arr, OUT)
        ds.close()
    # Consolidate once at the end
    print('Consolidating metadata...')
    consolidate(OUT)
    print("Done:", OUT)

if __name__ == "__main__":
    main()
