import xarray as xr
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys

def build_file_list(base_path,start_date,end_date):
    all_files = []
    for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1)):
        year = single_date.strftime('%Y')
        month = single_date.strftime('%m')
        day = single_date.strftime('%d')
        path = (
            Path(base_path) / year / month / f"modis_reflectance_{year}{month}{day}_regridded.nc4"
        )
        if path.exists():
            all_files.append(str(path))
    return all_files

def interpolate_with_mask(ds,max_gap=15):
    var_names = [var for var in ds.data_vars if 'time' in ds[var].dims]
    filled_ds = xr.Dataset(coords=ds.coords)
    for var_name in var_names:
        print('filling for variable:', var_name)
        da = ds[var_name]
        print('generating mask')
        all_nan_mask = da.isnull().all(dim='time')
        da_masked = da.where(~all_nan_mask)
        print('interpolating')
        interpolated = da_masked.interpolate_na(
            dim='time',
            method='linear',
            max_gap=pd.Timedelta(days=max_gap),
            limit=max_gap
        )
        print('sorting out what was filled')
        is_nan = da.isnull()
        was_filled = is_nan & interpolated.notnull()
        could_not_fill = is_nan & interpolated.isnull()
        filled_mask = xr.zeros_like(da,dtype=np.uint8)
        filled_mask = filled_mask.where(~was_filled, 1)
        filled_mask = filled_mask.where(~could_not_fill, 2)
        interpolated = interpolated.where(~all_nan_mask)
        print('creating final ds')
        filled_ds[f"{var_name}_filled"] = interpolated
        filled_ds[f"filled_{var_name[-1]}"] = filled_mask
    return filled_ds

def write_daily_outputs(filled_ds,start_date,end_date,output_base):
    date_range = pd.date_range(start=start_date,end=end_date)
    for t in date_range:
        print('writing output for:', t)
        t_str = t.strftime('%Y%m%d')
        year = t.strftime('%Y')
        month = t.strftime('%m')
        out_dir = Path(output_base) / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"modis_filled_{t_str}.nc4"
        daily_ds = filled_ds.where(
            filled_ds.time.dt.date == t.date(),
            drop=True
        )
        if daily_ds.time.size == 0:
            print(f"Warning: no data found for date {t.date()}, skipping")
            continue
        daily_ds.to_netcdf(out_path, engine='netcdf4')

def preprocess_add_time_dim(ds, filename):
    import re
    # Extract date from filename, e.g., 'modis_reflectance_20050531.nc4'
    match = re.search(r"(\d{8})", filename)
    if match:
        date_str = match.group(1)
        time_val = pd.to_datetime(date_str,format='%Y%m%d')
    else:
        raise ValueError(f"Date not found in filename: {filename}")
    # Expand scalar time to a 1D coordinate dimension 'time'
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=("time", [time_val]))
    return ds


def process_range(
    start_date,
    end_date,
    base_path,
    output_base,
    buffer_days=15,
    chunk_size=30,
    max_gap=15
):
    dt_start = datetime.strptime(start_date, '%Y-%m-%d')
    dt_end = datetime.strptime(end_date, '%Y-%m-%d')
    proc_start = dt_start - timedelta(days=buffer_days)
    proc_end = dt_end + timedelta(days=buffer_days)
    files = build_file_list(
        base_path, proc_start, proc_end
    )
    ds_all = None
    for i,f in enumerate(files,1):
        print(f"Processing file {i}/{len(files)}: {f}")
        ds = xr.open_dataset(f, engine='netcdf4')
        ds = preprocess_add_time_dim(ds, f)
        if ds_all is None:
            ds_all = ds
        else:
            ds_all = xr.concat([ds_all, ds], dim='time')
    test_date_range = pd.date_range(start=dt_start, end=dt_end)
    test_ds = ds_all.where(ds_all.time.dt.date == test_date_range[0].date(), drop=True)
    filled_ds = interpolate_with_mask(ds_all, max_gap=max_gap)
    write_daily_outputs(filled_ds, dt_start, dt_end, output_base)
    ds.close()
    filled_ds.close()

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(
        description="Interpolate MODIS data with sliding window processing."
    )
    arg_parse.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format."
    )
    arg_parse.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format."
    )
    #start_date = '2003-02-01'
    #end_date = '2023-12-15'
    start_date = arg_parse.parse_args().start_date
    end_date = arg_parse.parse_args().end_date
    base_path="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_regridded/quality_1"
    output_base="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_regridded_gapfilled/quality_1/interpolated"
    chunk_size = 100
    buffer_days = 14
    max_gap = 14
    # if chunk size is greater than date range, set it to date range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end_dt - start_dt).days + 1
    if chunk_size > (date_range + 2* buffer_days + 1):
        chunk_size = date_range + 2 * buffer_days + 1
    process_range(
        start_date,
        end_date,
        base_path,
        output_base,
        buffer_days,
        chunk_size,
        max_gap
    )
    # started job at 07:00pm
