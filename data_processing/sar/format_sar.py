import os
import glob
import xarray as xr
import numpy as np
import re

def extract_date_from_filename(filename):
    """
    Extract date from filename assuming the date is in YYYY-MM-DD format
    somewhere in the filename.
    """
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return np.datetime64(match.group(1), 'ns')
    else:
        raise ValueError(f"Date not found in filename: {filename}")

def main():
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/'
    sar_files = sorted(glob.glob(os.path.join(
        scratch_dir,'regrid','*','*','*.nc4'
    )))
    print(f'Found {len(sar_files)} SAR files to process.')
    for f,fname in enumerate(sar_files):
        print(f'Processing file {f+1} of {len(sar_files)}: {fname}')
        ds = xr.open_dataset(fname, engine='netcdf4')
        date = extract_date_from_filename(fname)
        ds = ds.assign_coords(time=date)
        if f == 0:
            full_ds = ds
        else:
            full_ds = xr.concat((full_ds, ds), dim='time')
    # turn bands into variables, drop band dimension
    band_dim_names = ['VH', 'VV']
    # turn the band dimension into variables
    ds = full_ds['band_data'].to_dataset(dim='band')
    # rename using your band names
    ds = ds.rename({1: band_dim_names[0], 2: band_dim_names[1]})
    # compute VV − VH
    ds['vv_minus_vh'] = ds['VV'] - ds['VH']
    # if you also want to keep coordinates & attrs from original ds
    for c in ds.coords:
        if c not in ds.coords:
            ds = ds.assign_coords({c: ds[c]})
    # 1) Drop the leftover band coordinate/dimension (if present)
    ds = ds.drop_vars("band", errors="ignore")
    ds = ds.drop_dims("band", errors="ignore")
    # 2) Desired chunks
    desired_chunks = {"x": 256, "y": 256}
    if "time" in ds.dims:   # only add if time is a real dimension
        desired_chunks["time"] = 1
    ds = ds.chunk(desired_chunks)
    print(ds)
    # 3) Save to Zarr (all variables in one store)
    save_zarr_fname = "/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_formatted.zarr"
    os.makedirs(os.path.dirname(save_zarr_fname), exist_ok=True)
    ds.to_zarr(save_zarr_fname, mode="w")
    print(f"Saved to {save_zarr_fname}") 

if __name__ == "__main__":
    main()