import os
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import shutil
import sys

def calc_ratio(sar_da, modis_ds, modis_name):
    #sar_da = sar_da.sel(time=slice(start_date, end_date))
    modis_da = (
        modis_ds["data"]
        .sel(variable=modis_name)
    )
    r = (sar_da / modis_da).squeeze(drop=True)
    # drop junk coords if present
    to_drop = [c for c in ["spatial_ref", "variable"] if c in r.coords]
    if to_drop:
        r = r.drop_vars(to_drop)
    return r

def write_var(name, da, out_store):
    ds = da.to_dataset(name=name)
    # chunk for safety
    ds = ds.chunk({"time": 1, "y": 512, "x": 512})
    with ProgressBar():
        ds.to_zarr(out_store, mode="a")

def main():
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"
    scratch_dir = "/scratch/users/trobinet/long_lfmc/trent_datasets"
    start = pd.Timestamp(2016, 4, 24)
    end   = pd.Timestamp(2023, 10, 3)
    daily_index = pd.date_range(start=start, end=end, freq='D')
    out_store = os.path.join(
        scratch_dir, "sar", "sar_500m_filled_ratios.zarr"
    )
    ## we need to delete the zarr store if it exists to avoid contamination
    #if os.path.exists(out_store):
    #    resp = input(f"{out_store} exists. Delete? [y/n] ")
    #    if resp.lower().startswith("y"):
    #        if out_store.endswith(".zarr"):
    #            shutil.rmtree(out_store)
    #        else:
    #            raise RuntimeError("Refusing to delete non-.zarr path")
    chunks = {"time": 1, "y": 512, "x": 512}
    vv_ds = xr.open_zarr(
        os.path.join(scratch_dir, "sar", "sar_500m_filled_vv.zarr"),
        chunks=chunks,
    )
    vv_ds = vv_ds.sel(time=slice(start, end))
    vv_ds = vv_ds.reindex(time=daily_index, method="ffill")
    #vv_ds = vv_da.to_dataset(name="vv_backscatter")
    vh_ds = xr.open_zarr(
        os.path.join(scratch_dir, "sar", "sar_500m_filled.zarr"),
        chunks=chunks,
    )
    vh_ds = vh_ds.sel(time=slice(start, end))
    vh_ds = vh_ds.reindex(time=daily_index, method="ffill")
    #vh_ds = vh_da.to_dataset(name="vh_backscatter")
    modis_ds = xr.open_zarr(
        os.path.join(
            scratch_dir, "modis", "modis_regridded_gapfilled",
            "quality_1", "interpolated", "modis_all_vars.zarr"
        ),
        chunks=chunks,
    )
    modis_ds = modis_ds.sel(time=slice(start, end))
    modis_ds = modis_ds.reindex(time=daily_index, method="ffill")
    #master_time = modis_ds.time.sel(time=slice(start, end))
    #vv_da = vv_ds["vv_backscatter"].reindex(time=master_time)
    #vh_da = vh_ds["vh_backscatter"].reindex(time=master_time)
    ## forward fill SAR onto MODIS daily grid
    #tchunk = 64
    #vv_da = vv_da.chunk({"time": tchunk, 'y': 512, 'x': 512})
    #vh_da = vh_da.chunk({"time": tchunk, 'y': 512, 'x': 512})
    #vv_da = vv_da.ffill("time", limit=10)
    #vh_da = vh_da.ffill("time", limit=10)
    #vv_ds = vv_da.to_dataset(name="vv_backscatter")
    #vh_ds = vh_da.to_dataset(name="vh_backscatter")
    #vv_ds = vv_ds.chunk({"time": 1, "y": 512, "x": 512})
    #vh_ds = vh_ds.chunk({"time": 1, "y": 512, "x": 512})
    # Initialize output with coords/time slice only (cheap)
    base = vv_ds.drop_vars(
        [v for v in vv_ds.data_vars]  # keep coords only
    )
    print(base)
    print(vv_ds)
    print(vh_ds)
    print(modis_ds)
    #with ProgressBar():
    #    # chunk for safety
    #    base = base.chunk({"time": 1, "y": 512, "x": 512})
    #    base.to_zarr(out_store, mode="w")
    ## vv - vh
    #print('Writing vv_minus_vh')
    #now = pd.Timestamp.now()
    #print(f'now: {now}')
    #write_var(
    #    "vv_minus_vh",
    #    (vv_ds["vv_backscatter"] - vh_ds["vh_backscatter"]),
    #    out_store
    #)
    #bands = [
    #    "Nadir_Reflectance_Band1_filled",
    #    "Nadir_Reflectance_Band2_filled",
    #    "Nadir_Reflectance_Band3_filled",
    #    "Nadir_Reflectance_Band4_filled",
    #    "Nadir_Reflectance_Band5_filled",
    #    "Nadir_Reflectance_Band6_filled",
    #    "Nadir_Reflectance_Band7_filled",
    #]
    #for i, b in enumerate(bands, start=1):
    #    print(f"Writing vv ratio for {b}")
    #    now = pd.Timestamp.now()
    #    print(f'now: {now}')
    #    write_var(
    #        f"vv_over_{i}",
    #        calc_ratio(vv_ds["vv_backscatter"], modis_ds, b),
    #        out_store
    #    )
    #    print(f"Writing vh ratio for {b}")
    #    write_var(
    #        f"vh_over_{i}",
    #        calc_ratio(vh_ds["vh_backscatter"], modis_ds, b),
    #        out_store
    #    )
    red = modis_ds['data'].sel(band="Nadir_Reflectance_Band1_filled")
    nir = modis_ds['data'].sel(band="Nadir_Reflectance_Band2_filled")
    ndvi = (nir - red) / (nir + red + 1e-10)
    sar_by_ndvi = (vv_ds['vv_backscatter'] - vh_ds['vh_backscatter']) * ndvi
    to_drop = [c for c in ["spatial_ref", "variable"] if c in sar_by_ndvi.coords]
    if to_drop:
        sar_by_ndvi = sar_by_ndvi.drop_vars(to_drop)
    write_var(
        "sar_by_ndvi",
        sar_by_ndvi,
        out_store
    )

if __name__ == "__main__":
    main()


