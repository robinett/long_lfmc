# daymet_to_zarr.py
from pathlib import Path
import xarray as xr
from numcodecs import Blosc
import re
from typing import List, Tuple
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
    "/scratch/users/trobinet/long_lfmc/"
    "trent_datasets/daymet/daymet_regrid"   # <-- set to your Daymet root
)

OUT = Path(
    "/scratch/users/trobinet/long_lfmc/trent_datasets/"
    "daymet/daymet_all_vars.zarr"    # <-- output zarr
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

# ---------- /CONFIG ----------


MONTH_RE = re.compile(r"^\d{2}$")
YEAR_RE  = re.compile(r"^\d{4}$")

def find_month_keys_daymet(root: Path) -> List[Tuple[str, str]]:
    """
    Return sorted unique (year, month) pairs found anywhere under root/<var>/<year>/<month>.
    """
    keys = set()
    for var_dir in root.iterdir():
        if not var_dir.is_dir():
            continue
        for y_dir in var_dir.iterdir():
            if not (y_dir.is_dir() and YEAR_RE.match(y_dir.name)):
                continue
            for m_dir in y_dir.iterdir():
                if m_dir.is_dir() and MONTH_RE.match(m_dir.name):
                    keys.add((y_dir.name, m_dir.name))
    return sorted(keys)  # lexicographic => years then months

def files_for_month_daymet(root: Path,
                           year: str,
                           month: str,
                           patterns=(".nc", ".nc4")) -> List[Path]:
    """
    For a given (year, month), collect files from ALL variables:
      root/<var>/<year>/<month>/*.(nc|nc4)
    """
    out: List[Path] = []
    for var_dir in root.iterdir():
        ydir = var_dir / year
        mdir = ydir / month
        if not mdir.is_dir():
            continue
        for p in patterns:
            out.extend(sorted(mdir.glob(f"*{p}")))
    return out

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

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    month_keys = find_month_keys_daymet(ROOT)
    if not month_keys:
        raise ValueError(f"No year/month dirs found under {ROOT}")
    is_first = True
    for (year, month) in month_keys:
        files = files_for_month_daymet(ROOT, year, month, patterns=(".nc", ".nc4"))
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
        )
        # Optional subset (if Daymet directory also contains derived vars, etc.)
        print('Subsetting variables...')
        ds = maybe_subset_vars(ds, VAR_WHITELIST)
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
