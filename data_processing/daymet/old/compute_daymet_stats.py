import datetime
import os
import xarray as xr
import pandas as pd
import sys
import numpy as np
import argparse
import warnings
import zarr

# replace your expand_dims(...) line with this helper:
def _as_bin(a, date, kept_offsets, win):
    # drop lingering 'variable' coord that will conflict later
    a = a.drop_vars("variable", errors="ignore")

    # rename time->bin and attach bin labels derived from true offsets
    labels = [f"{k}_{k+win-1}" for k in kept_offsets]
    a = a.rename(time="bin").assign_coords(bin=("bin", labels))

    # expand with ns-precision time coord
    a = a.expand_dims(time=[pd.Timestamp(date)])  # ns
    return a

def _as_time_dim(ds: xr.Dataset) -> np.datetime64:
    """Return a single np.datetime64 and ensure 'time' is a dim (size 1)."""
    if "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise ValueError("Input must have exactly one time step per write.")
        return pd.to_datetime(ds["time"].values[0]).to_datetime64()
    if "time" in ds.coords:
        tval = ds["time"].values
        if np.isscalar(tval):
            return pd.to_datetime(tval).to_datetime64()
        return pd.to_datetime(tval[0]).to_datetime64()
    return np.datetime64("NaT")

def _stack_all_vars_to_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Stack all data_vars into one DataArray 'data' with dims (var, time, y, x),
    so each Zarr chunk file contains all variables for that (time,y,x) block.
    """
    var_names = list(ds.data_vars)
    if not var_names:
        raise ValueError("No data variables found to write.")

    arrs = []
    for v in var_names:
        da = ds[v]
        if not all(d in da.dims for d in ("y", "x")):
            raise ValueError(f"{v} must have dims (y, x); found {da.dims}")
        da = da.expand_dims(var=[v])   # add a length-1 'var' dim
        arrs.append(da)

    data = xr.concat(arrs, dim="var")

    # Ensure a single time step as a dimension
    tval = _as_time_dim(ds)
    if "time" not in data.dims:
        data = data.expand_dims(time=[tval])

    out = xr.Dataset({"data": data})
    out = out.assign_coords(var=("var", var_names))
    return out

def write_or_append(stats_ds: xr.Dataset, save_zarr_fname: str):
    """
    Write as a single array 'data' with dims (var, time, y, x), chunked as:
      var = all variables, time = 1, y = 256, x = 256.
    Appends along 'time'.
    """
    ds = _stack_all_vars_to_data(stats_ds)

    n_vars = ds.sizes["var"]
    desired_chunks = {
        "var": n_vars,   # <- ALL variables in one chunk so each file contains all vars
        "time": 1,
        "y": 256,
        "x": 256,
    }

    # Dask chunking (controls write layout)
    ds = ds.chunk(desired_chunks)

    # Explicit on-disk chunking for 'data'
    ds["data"].encoding = {
        "chunks": (
            desired_chunks["var"],
            desired_chunks["time"],
            desired_chunks["y"],
            desired_chunks["x"],
        )
    }

    if not os.path.exists(save_zarr_fname):
        os.makedirs(os.path.dirname(save_zarr_fname), exist_ok=True)
        ds.to_zarr(save_zarr_fname, mode="w")
    else:
        # Sanity: variable list/order must match the existing store
        ds0 = xr.open_zarr(save_zarr_fname, chunks={})
        if "data" not in ds0:
            raise ValueError("Existing store missing 'data' variable.")
        old_vars = list(ds0["data"].coords["var"].values.astype(str))
        new_vars = list(ds["data"].coords["var"].values.astype(str))
        if old_vars != new_vars:
            raise ValueError(
                "Incoming variables differ from existing store (names or order).\n"
                f"Existing: {old_vars}\nNew: {new_vars}"
            )
        # Append along time
        ds.to_zarr(save_zarr_fname, mode="a", append_dim="time")

def compute_stats(
    start_date,
    end_date,
    generic_fname,
    save_zarr_fname,
    day_interval=5,
    max_days=180,
):
    '''
    stats that we are going to find:
        cumulative precip every n days
        max temp every n days
        min vapor pressure every n days
        max swe every n days
    '''
    daymet_ds = xr.open_zarr(generic_fname, chunks='auto')
    # set daymet times to midnight
    daymet_ds['time'] = pd.to_datetime(daymet_ds['time'].values).normalize()
    example_ds = daymet_ds.sel(time=start_date)
    days_to_sample = np.arange(0, max_days+1, day_interval)
    # get days to sample, we go in reverse order so that we can build up
    # our tracking arrays
    # initialize our tracker that tracks days since rain as zeros
    days_need_stats = pd.date_range(
        start=start_date,
        end=end_date,
        freq='D'
    )
    loaded_data = {}
    for d,date in enumerate(days_need_stats):
        print('computing stats for date: {}'.format(date))
        this_dates_to_sample = [
            date - datetime.timedelta(days=int(d))
            for d in days_to_sample
        ]
        # initialize our ds as the example but no variables
        stats_ds = example_ds.copy(deep=True)
        stats_ds = stats_ds.drop_vars(example_ds.data_vars)
        for s,sample_date in enumerate(this_dates_to_sample):
            sample_date_strf = sample_date.strftime('%Y%m%d')
            # print now for benchmarking
            print(pd.Timestamp.now())
            print('    sampling date:', sample_date)
            if sample_date_strf not in loaded_data.keys():
                print('re-loading')
                loaded_data[sample_date_strf] = {}
                # daymet doesn't have dec 31 on leap years, so use dec 30
                if (
                    (sample_date.month == 12)
                    and (sample_date.day == 31)
                    and (
                        (
                            (sample_date.year % 4 == 0 and sample_date.year % 100 != 0)
                            or (sample_date.year % 400 == 0)
                        )
                    )
                ):
                    print('    adjusting for leap year')
                    sel_date = sample_date - datetime.timedelta(days=1)
                else:
                    sel_date = sample_date
                loaded_data[sample_date_strf]['prcp'] = daymet_ds['data'].sel(
                    time=sel_date,
                    variable='prcp'
                ).values
                loaded_data[sample_date_strf]['tmax'] = daymet_ds['data'].sel(
                    time=sel_date,
                    variable='tmax'
                ).values
                loaded_data[sample_date_strf]['vp'] = daymet_ds['data'].sel(
                    time=sel_date,
                    variable='vp'
                ).values
                loaded_data[sample_date_strf]['swe'] = daymet_ds['data'].sel(
                    time=sel_date,
                    variable='swe'
                ).values
            this_cum_precip = loaded_data[sample_date_strf]['prcp']
            this_max_temp = loaded_data[sample_date_strf]['tmax']
            this_min_vp = loaded_data[sample_date_strf]['vp']
            this_max_swe = loaded_data[sample_date_strf]['swe']
            for r in range(1, day_interval):
                today = sample_date - datetime.timedelta(days=r)
                today_strf = today.strftime('%Y%m%d')
                print(f'        rolling back day {r}: {today}')
                if today_strf not in loaded_data.keys():
                    print('re-loading')
                    loaded_data[today_strf] = {}
                    # daymet doesn't have dec 31 on leap years, so use dec 30
                    if (
                        (today.month == 12)
                        and (today.day == 31)
                        and (
                            (
                                (today.year % 4 == 0 and today.year % 100 != 0)
                                or (today.year % 400 == 0)
                            )
                        )
                    ):
                        print('    adjusting for leap year')
                        sel_date = today - datetime.timedelta(days=1)
                    else:
                        sel_date = today
                    loaded_data[today_strf]['prcp'] = daymet_ds['data'].sel(
                        time=sel_date,
                        variable='prcp'
                    ).values
                    loaded_data[today_strf]['tmax'] = daymet_ds['data'].sel(
                        time=sel_date,
                        variable='tmax'
                    ).values
                    loaded_data[today_strf]['vp'] = daymet_ds['data'].sel(
                        time=sel_date,
                        variable='vp'
                    ).values
                    loaded_data[today_strf]['swe'] = daymet_ds['data'].sel(
                        time=sel_date,
                        variable='swe'
                    ).values
                this_cum_precip = this_cum_precip + loaded_data[today_strf]['prcp']
                this_max_temp = np.fmax(
                    this_max_temp,
                    loaded_data[today_strf]['tmax']
                )
                this_min_vp = np.fmin(
                    this_min_vp,
                    loaded_data[today_strf]['vp']
                )
                this_max_swe = np.fmax(
                    this_max_swe,
                    loaded_data[today_strf]['swe']
                )
            stats_ds[f'prcp_cum_{s*day_interval}d_{s*day_interval+day_interval-1}d'] = (
                (('y', 'x'), this_cum_precip)
            )
            stats_ds[f'tmax_max_{s*day_interval}d_{s*day_interval+day_interval-1}d'] = (
                (('y', 'x'), this_max_temp)
            )
            stats_ds[f'vp_min_{s*day_interval}d_{s*day_interval+day_interval-1}d'] = (
                (('y', 'x'), this_min_vp)
            )
            stats_ds[f'swe_max_{s*day_interval}d_{s*day_interval+day_interval-1}d'] = (
                (('y', 'x'), this_max_swe)
            )
        
        # check for dates that are too far gone and we can close out
        furthest_date_needed = date - pd.Timedelta(days=max_days) - pd.Timedelta(days=1)
        print('    furthest date needed:', furthest_date_needed)
        
        loaded_data_keys = list(loaded_data.keys())
        for d in loaded_data_keys:
            date_pd = pd.to_datetime(d)
            if date_pd < furthest_date_needed:
                print(f'    closing out date: {date_pd}')
                del loaded_data[d]
        print(stats_ds)
        # ensure NO stray 'variable' coord/dim
        if 'variable' in stats_ds.coords or 'variable' in stats_ds.dims:
            stats_ds = stats_ds.drop_vars('variable', errors='ignore')

        # make time a real dim, not scalar coord
        # (use ns precision to avoid the warning)
        stats_ds = stats_ds.expand_dims(time=[pd.Timestamp(date)])

        # optional but helpful: chunk with time=1 for appends
        stats_ds = stats_ds.chunk({'time': 1})

        # write / append
        write_or_append(stats_ds, save_zarr_fname)
    # final consolidation
    zarr.consolidate_metadata(save_zarr_fname)


if __name__ == "__main__":
    # start and end dates are passed to file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    start_date = pd.to_datetime(parser.parse_args().start_date)
    end_date = pd.to_datetime(parser.parse_args().end_date)
    max_days = 180
    day_interval = 5
    #start_date = pd.to_datetime('2010-01-01')
    #end_date = pd.to_datetime('2010-12-31')
    scratch_dir = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/daymet/'
    )
    generic_fname = os.path.join(
        scratch_dir,
        'daymet_all_vars.zarr'
    )
    start_year_strf = start_date.strftime('%Y')
    generic_save_fname = os.path.join(
        scratch_dir,
        'stats',
        f'stats_{start_year_strf}.zarr'
    )
    compute_stats(
        start_date,
        end_date,
        generic_fname,
        generic_save_fname,
        day_interval,
        max_days,
    )