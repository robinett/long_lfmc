from pathlib import Path
import xarray as xr
from numcodecs import Blosc

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
    "trent_datasets/modis/modis_regridded_gapfilled/"
    "quality_1/interpolated"
)

OUT = Path(
    "/scratch/users/trobinet/long_lfmc/trent_datasets/"
    "modis/modis_regridded_gapfilled/quality_1/interpolated/"
    "modis_all_vars.zarr"
)

# Keep identical chunking to your current script
WRITE_CHUNKS = {
    "time": 1,
    "variable": 14,   # one var-chunk (14 vars total)
    "y": 512,
    "x": 512,
}

CAST_FLOAT32   = True
ENGINE         = "h5netcdf"   # match your current choice
PARALLEL_OPEN  = False        # match your current choice
COMP           = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)
# ---------- /CONFIG ----------

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Use year/month directories as batches (same effect as your list_month_dirs)
    batches = find_batches_by_year_or_month(ROOT, patterns=(".nc4",))
    is_first = True

    for bdir in batches:
        files = files_for_batch(bdir, patterns=(".nc4",))
        if not files:
            continue
        print(f"Batch: {bdir}  ({len(files)} files)")

        # Open + clean one batch (concat on time)
        ds = open_time_batch(
            files,
            engine=ENGINE,
            parallel_open=PARALLEL_OPEN,
            cast_float32=CAST_FLOAT32,
            preprocess=preprocess_strip_attrs,
        )

        # Stack → chunk → chunk coords (identical layout)
        arr = to_stacked_array(ds, WRITE_CHUNKS)
        arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])

        if is_first:
            write_first(arr, OUT, compressor=COMP)
            is_first = False
        else:
            append_time(arr, OUT)

        ds.close()

    consolidate(OUT)
    print("Done:", OUT)

if __name__ == "__main__":
    main()
