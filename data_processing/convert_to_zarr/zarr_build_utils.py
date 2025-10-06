# zarr_build_utils.py
from pathlib import Path
import re
import pandas as pd
import numpy as np
import xarray as xr
from numcodecs import Blosc
import h5py  # pre-import to avoid race on parallel open
import sys
import os

# -------- Compression default (same as MODIS) --------
DEFAULT_COMP = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)

# -------- Attribute cleanup --------
def preprocess_strip_attrs(ds: xr.Dataset) -> xr.Dataset:
    drop_keys = ["history", "date_created", "creation_date",
                 "uuid", "references", "Conventions"]
    for v in ds.variables:
        for k in drop_keys:
            ds[v].attrs.pop(k, None)
    for k in drop_keys:
        ds.attrs.pop(k, None)
    return ds

# -------- Open a batch of files and concat along time --------
def open_time_batch(
    files,
    engine="h5netcdf",
    parallel_open=False,
    cast_float32=True,
    preprocess=preprocess_strip_attrs,
    combine="nested",   # <— add this
):
    import xarray as xr
    with xr.set_options(file_cache_maxsize=2):
        try:
            ds = xr.open_mfdataset(
                files,
                engine=engine,
                combine=combine,        # <— use requested mode
                concat_dim="time" if combine=="nested" else None,
                chunks={},
                parallel=parallel_open,
                decode_times=True,
                preprocess=preprocess,
            )
        except Exception:
            parts = []
            last_var_name = None
            first_iter = True
            for f,file in enumerate(files):
                this_var = str(file).split("/")[-1].split("_")[0]
                dsi = xr.open_dataset(
                    file,
                    engine=engine,
                    decode_times=True,
                    backend_kwargs={"invalid_netcdf": True, "phony_dims": "sort"},
                )
                if this_var != last_var_name:
                    print(f'finished {this_var}')
                    if not first_iter:
                        parts.append(this_ds)
                    this_ds = dsi
                    last_var_name = this_var
                    first_iter = False
                else:
                    this_ds = xr.concat([this_ds, dsi], dim="time")
                dsi.close()
            print(f'finished {this_var}')
            parts.append(this_ds)
            final_ds = xr.merge(parts, compat="no_conflicts",join="exact")
            for part in parts:
                part.close()
            print(final_ds)
            # check that there are values for every varaible
            for v in final_ds.data_vars:
                if np.all(final_ds[v].isnull()):
                    raise ValueError(f"Variable {v} is all NaN")
    # sort & de-dup time (same as before)
    print('sorting times')
    final_ds = final_ds.sortby("time")
    times = pd.to_datetime(final_ds["time"].values)
    mask = ~pd.Index(times).duplicated(keep="first")
    final_ds = final_ds.isel(time=mask)
    # normalize dtypes (same as before)
    if cast_float32:
        print('casting to float32')
        cast_map = {}
        for v in final_ds.data_vars:
            if final_ds[v].dtype.kind in ("f", "i", "u") and final_ds[v].dtype != "float32":
                cast_map[v] = "float32"
        if cast_map:
            final_ds = final_ds.astype(cast_map)

    for v in final_ds.data_vars:
        final_ds[v].encoding["_FillValue"] = None
    return final_ds


# -------- Stack to (time, variable, y, x) and rechunk --------
def to_stacked_array(ds: xr.Dataset, write_chunks: dict) -> xr.DataArray:
    var_names = sorted(ds.data_vars)   # stable order
    ds = ds[var_names]

    arr = ds.to_array(dim="variable", name="data").transpose(
        "time", "variable", "y", "x"
    )
    var_count = arr.sizes["variable"]
    target = {
        "time": write_chunks.get("time", 1),
        "variable": min(write_chunks.get("variable", var_count), var_count),
        "y": write_chunks.get("y", 512),
        "x": write_chunks.get("x", 512),
    }
    return arr.chunk(target)

# -------- Chunk lat/lon coords to match spatial tiles --------
def chunk_coords(ds_or_da, y=512, x=512):
    obj = ds_or_da
    if "lat" in obj.coords:
        obj.coords["lat"] = obj.coords["lat"].chunk({"y": y, "x": x})
    if "lon" in obj.coords:
        obj.coords["lon"] = obj.coords["lon"].chunk({"y": y, "x": x})
    return obj

# -------- Build encoding (chunks + compressor) --------
def zarr_encoding_for(arr: xr.DataArray, compressor=DEFAULT_COMP):
    chunks = tuple(c[0] for c in arr.data.chunks)
    return {"data": {"compressor": compressor, "chunks": chunks}}

# -------- Write helpers (first write uses encoding, append omits) --------
def write_first(arr: xr.DataArray, out: Path, compressor=DEFAULT_COMP):
    ds = arr.to_dataset(name="data")
    ds.to_zarr(
        out,
        mode="w",
        consolidated=False,
        encoding=zarr_encoding_for(arr, compressor),
        compute=True,
    )

def append_time(arr: xr.DataArray, out: Path):
    ds = arr.to_dataset(name="data")
    ds.to_zarr(
        out,
        mode="a",
        append_dim="time",
        consolidated=False,
    )

def consolidate(out: Path):
    import zarr
    zarr.consolidate_metadata(str(out))

# -------- Generic discovery helpers --------
def find_batches_by_year_or_month(root: Path, patterns=(".nc", ".nc4")):
    """
    Returns an ordered list of Path directories; each is a 'batch'.
    Preference:
      - If directories like root/YYYY/MM exist → return each YYYY/MM
      - Else if root/YYYY exists with yearly files → return each YYYY dir
      - Else → single batch: root
    """
    years = sorted([p for p in root.glob("[12][0-9][0-9][0-9]") if p.is_dir()])
    if years:
        # check if months exist
        months = []
        for y in years:
            found = [m for m in sorted(y.glob("[01][0-9]")) if list(_files_in(m, patterns))]
            if found:
                months.extend(found)
        if months:
            return months
        # else: at least year dirs exist; use each year as a batch
        year_batches = [y for y in years if list(_files_in(y, patterns))]
        if year_batches:
            return year_batches

    # fallback: single batch
    return [root]

def _files_in(folder: Path, patterns):
    files = []
    for ext in patterns:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)

def files_for_batch(batch_dir: Path, patterns=(".nc", ".nc4")):
    return _files_in(batch_dir, patterns)
