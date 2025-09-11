import xarray as xr
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def build_file_list(base_path,start_date,end_date,buffer_days=15):
    all_files = []
    dt_start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=buffer_days)
    dt_end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=buffer_days)
    for single_date in (dt_start + timedelta(n) for n in range((dt_end - dt_start).days + 1)):
        year = single_date.strftime('%Y')
        month = single_date.strftime('%m')
        day = single_date.strftime('%d')
        path = (
            Path(base_path) / year / month / f"modis_reflectance_{year}{month}{day}.nc4"
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
        is_nan = da.isnull()
        interpolated = da.interpolate_na(
            dim='time', method='linear', max_gap=max_gap, limit=max_gap
        )
        was_filled = is_nan & nterpolated.notnull()
        could_not_fill = is_nan & interpolated.isnull()
        filled_mask = xr.zeros_like(da,dtype=np.uint8)
        filled_mask = filled_mask.where(~was_filled, 1)
        filled_mask = filled_mask.where(~could_not_fill, 2)
        filled_ds[f"{var_name}_filled"] = interpolated
        filled_ds[f"filled_{var_name[-1]}"] = filled_mask
    return filled_ds

def write_daily_outputs(filled_ds,start_date,end_date,output_base):
    date_range = pd.date_range(start=start_date, end=end_date)
    for t in date_range:
        print('writing output for:', t)
        t_str = t.strftime('%Y%m%d')
        year = t.strftime('%Y')
        month = t.strftime('%m')
        out_dir = Path(output_base) / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"modis_filled_{t_str}.nc4"
        daily_ds = filled_ds.sel(time=t)
        daily_ds.to_netcdf(out_path, engine='netcdf4')

def preprocess_add_time_dim(ds, filename):
    import re
    # Extract date from filename, e.g., 'modis_reflectance_20050531.nc4'
    match = re.search(r"(\d{8})", filename)
    if match:
        date_str = match.group(1)
        time_val = np.datetime64(date_str)
    else:
        raise ValueError(f"Date not found in filename: {filename}")

    # Expand scalar time to a 1D coordinate dimension 'time'
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=("time", [time_val]))
    return ds


def process_range_sliding(
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
    current_start = proc_start
    while current_start <= proc_end:
        current_end = min(current_start + timedelta(days=chunk_size - 1), proc_end)
        print('processing from', current_start, 'to', current_end)
        files = build_file_list(
            base_path, current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d'), buffer_days
        )
        if not files:
            print(f"No files found for range {current_start} to {current_end}. Skipping.")
            current_start += timedelta(days=chunk_size)
            continue
        ds_all = None
        for i,f in enumerate(files,1):
            print(f"Processing file {i}/{len(files)}: {f}")
            ds = xr.open_dataset(f, engine='netcdf4')
            ds = preprocess_add_time_dim(ds, f)
            if ds_all is None:
                ds_all = ds
            else:
                ds_all = xr.concat([ds_all, ds], dim='time')
        filled_ds = interpolate_with_mask(ds, max_gap=max_gap)
        write_start = max(current_start, dt_start)
        write_end = min(current_end, dt_end)
        write_daily_outputs(filled_ds, write_start.strftime('%Y-%m-%d'), write_end.strftime('%Y-%m-%d'), output_base)
        ds.close()
        filled_ds.close()
        current_start = current_end + timedelta(days=1)

if __name__ == "__main__":
    start_date = '2003-02-01'
    end_date = '2023-12-15'
    base_path="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_processed_daily_w_quality/quality_1"
    output_base="/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_gapfilled/quality_1/interpolated"
    process_range_sliding(
        start_date,
        end_date,
        base_path,
        output_base,
        buffer_days=15,
        chunk_size=5,
        max_gap=15
    )
