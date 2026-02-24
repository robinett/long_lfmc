import glob
import pandas as pd
import xarray as xr
import rioxarray as rxr
from numcodecs import Blosc
import shutil
import os
from dask.diagnostics import ProgressBar

def year_from_name(fp: str) -> int:
    for part in fp.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"No year found in {fp}")

def open_lazy(fp: str) -> xr.DataArray:
    # Keep integers by avoiding masked=True (which introduces NaNs/float)
    da = (
        rxr.open_rasterio(
            fp,
            masked=False,                      # <-- keep integer dtype
            chunks={"y": 2048, "x": 2048},
            cache=False,
            lock=False,
        )
        .squeeze("band", drop=True)
        .rename("nlcd")
    )
    # Ensure a defined nodata/fill value of 0 (NLCD background)
    if da.rio.nodata is None:
        da = da.rio.write_nodata(0)
    # Guarantee uint8 dtype (NLCD classes fit in 0..255)
    da = da.astype("uint8")
    return da


def main():
    in_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_raw"
    out_zarr = "/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_2000_2024.zarr"
    files = sorted(glob.glob(f"{in_dir}/*.tif"))
    if not files:
        raise SystemExit(f"No .tif files found in {in_dir}")
    years = [year_from_name(f) for f in files]
    time = pd.to_datetime([f"{y}-01-01" for y in years])
    arrays = [open_lazy(f) for f in files]
    # stack along time (still lazy)
    nlcd = xr.concat(
        arrays,
        dim=xr.DataArray(time, dims="time", name="time"),
        join="exact",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).chunk({"time": 1})  # spatial chunks already set
    print(nlcd)
    # Clean out any partial previous store to avoid mode conflicts
    if os.path.exists(out_zarr):
        shutil.rmtree(out_zarr)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    with ProgressBar():
        nlcd.to_dataset().to_zarr(
            out_zarr,
            mode="w",
            consolidated=True,
            safe_chunks=False,
            zarr_format=2,  # use v2; numcodecs.Blosc compressor is v2-style
            encoding={
                "nlcd": {
                    "compressor": compressor,     # v2 key
                    "chunks": (1, 2048, 2048),
                    "dtype": "uint8",
                }
            },
        )

if __name__ == "__main__":
    main()
