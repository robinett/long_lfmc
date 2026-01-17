import glob
import re
import pandas as pd
import xarray as xr
import numpy as np
import sys
import os
import shutil
from tqdm import tqdm
import zarr

date_re = re.compile(r"s1_(\d{8})\.nc$")

def add_time(ds, fp):
    m = date_re.search(fp.split("/")[-1])
    t = np.datetime64(pd.to_datetime(m.group(1), format="%Y%m%d"))
    # make nanosecond precision
    t = t.astype("datetime64[ns]")
    ds = ds.expand_dims(time=[t])
    return ds

def main():
    raw_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily"
    out_zarr = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr"
    consolidate_only = False
    if consolidate_only:
        zarr.convenience.consolidate_metadata(out_zarr)
        print('Zarr metadata consolidated. Exiting.')
        sys.exit()
    # if out_zarr exists, get rid of it
    if os.path.exists(out_zarr):
        # make sure the user is cool
        response = input(f"{out_zarr} exists. Are you okay deleting it? (y/n) ")
        if response.lower() == "n":
            print("Exiting.")
            sys.exit()
        elif response.lower() == "y":
            shutil.rmtree(out_zarr)
        else:
            print('Unacceptable response. Exiting.')
            sys.exit()
    files = sorted(glob.glob(f"{raw_dir}/s1_*.nc"))
    #files = files[1000:2000] # while we are still finishing downloading all of our sar files
    first = True
    for i, fp in enumerate(tqdm(files, desc="Adding SAR file to zarr")):
        #print(i, len(files), fp)
        ds = xr.open_dataset(
            fp,
            cache=False,
        )
        # apply our range filtering to be reasonable
        vh_range = [-35.0, 0.0]
        ds = ds.where(
            (ds["vh_backscatter"] >= vh_range[0]) & (ds["vh_backscatter"] <= vh_range[1])
        )
        # chunk the dataset
        ds = add_time(ds, fp)
        ds = ds.chunk(
           {'x': 512, 'y': 512, 'time': 1}
        )
        # OPTIONAL: if lat/lon are identical every day, store once:
        if not first:
            ds = ds.drop_vars(["lat", "lon"], errors="ignore")
        mode = "w" if first else "a"
        if mode == "a":
            ds.to_zarr(out_zarr, mode=mode, append_dim="time")
        elif mode == "w":
            ds.to_zarr(out_zarr, mode=mode)
        ds.close()
        first = False
    # consolidate zarr
    zarr.convenience.consolidate_metadata(out_zarr)

if __name__ == "__main__":
    main()
