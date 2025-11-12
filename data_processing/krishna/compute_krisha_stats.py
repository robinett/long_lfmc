import glob
import os
import xarray as xr
import re
import sys
import pandas as pd
import rioxarray
import numpy as np
import calendar

# Add the parent directory to the path to import plotting
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from shared import plotting

def nan_autocorr(x, lag=1, axis=-1):
    """
    Autocorrelation at integer `lag` along `axis`, ignoring NaNs.
    Returns a scalar per slice (i.e., input shape without `axis`).

    Parameters
    ----------
    x : np.ndarray
    lag : int >= 1
    axis : int

    Returns
    -------
    r : np.ndarray (x.shape with `axis` removed)
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    x = np.asarray(x, dtype=np.float64)
    # move target axis to the end
    x = np.swapaxes(x, axis, -1)

    if x.shape[-1] <= lag:
        # not enough samples for this lag
        return np.full(x.shape[:-1], np.nan, dtype=np.float64)

    x0 = x[..., :-lag]
    x1 = x[...,  lag:]

    mask = (~np.isnan(x0)) & (~np.isnan(x1))
    n = mask.sum(axis=-1)

    # early exit if no valid pairs anywhere
    if not np.any(n):
        return np.full(x.shape[:-1], np.nan, dtype=np.float64)

    # replace NaNs with 0 just for masked sums
    x0m = np.where(mask, x0, 0.0)
    x1m = np.where(mask, x1, 0.0)

    # means over valid pairs
    denom = np.where(n > 0, n, 1)
    mu0 = x0m.sum(axis=-1) / denom
    mu1 = x1m.sum(axis=-1) / denom

    # demean and apply mask
    d0 = np.where(mask, x0m - mu0[..., None], 0.0)
    d1 = np.where(mask, x1m - mu1[..., None], 0.0)

    # cov and variances over valid pairs
    cov = (d0 * d1).sum(axis=-1) / denom
    v0  = (d0 * d0).sum(axis=-1) / denom
    v1  = (d1 * d1).sum(axis=-1) / denom

    with np.errstate(divide='ignore', invalid='ignore'):
        r = cov / np.sqrt(v0 * v1)

    # guards: need ≥2 valid pairs and nonzero variance
    r = np.where((n >= 2) & (v0 > 0) & (v1 > 0), r, np.nan)
    return r

def nan_kurtosis(x, axis=0):
    """
    Compute excess kurtosis (Fisher definition) along a given axis,
    ignoring NaNs.

    Parameters
    ----------
    x : np.ndarray
        Input data array.
    axis : int, optional
        Axis along which to compute kurtosis. Default is 0.

    Returns
    -------
    kurtosis : np.ndarray
        Excess kurtosis (0 for normal). NaN where fewer than 4 valid
        samples exist or variance = 0.
    """
    x = np.asarray(x, dtype=np.float64)

    mean = np.nanmean(x, axis=axis, keepdims=True)
    dev = x - mean

    # Central moments
    m2 = np.nanmean(dev**2, axis=axis)
    m4 = np.nanmean(dev**4, axis=axis)

    # Sample count and guard conditions
    n = np.sum(~np.isnan(x), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        kurt = m4 / (m2**2) - 3.0  # excess kurtosis

    kurt = np.where((n >= 4) & (m2 > 0), kurt, np.nan)
    return kurt

def nan_skew(x, axis=0):
    """
    Compute skewness along a given axis, ignoring NaNs.
    Equivalent to Fisher-Pearson moment coefficient of skewness.

    Parameters
    ----------
    x : np.ndarray
        Input data array.
    axis : int, optional
        Axis along which to compute skewness. Default is 0.

    Returns
    -------
    skew : np.ndarray
        Skewness values along the specified axis.
        Returns NaN where fewer than 3 valid samples exist or variance = 0.
    """
    x = np.asarray(x, dtype=np.float64)

    # Mean and deviations
    mean = np.nanmean(x, axis=axis, keepdims=True)
    dev = x - mean

    # Central moments
    m2 = np.nanmean(dev**2, axis=axis)
    m3 = np.nanmean(dev**3, axis=axis)

    # Sample count and guard conditions
    n = np.sum(~np.isnan(x), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        skew = m3 / np.power(m2, 1.5)

    # Mask degenerate or insufficient cases
    skew = np.where((n >= 3) & (m2 > 0), skew, np.nan)
    return skew    
    print('computing skewness')

def main(krishna_files_path,krishna_plots_path,save_stats_path):
    # open all the files in the directory
    all_krishna_files = []
    all_krishna_dates = []
    for dirpath,dirnames,filenames in os.walk(krishna_files_path):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            all_krishna_files.append(full_path)
            this_date = extract_date_from_fname(file)
            all_krishna_dates.append(this_date)
    combined_ds = None
    #all_krishna_files = all_krishna_files[:5]
    for f,file in enumerate(all_krishna_files):
        print('adding file {} to dataset'.format(file))
        print('file {} of {}'.format(f+1,len(all_krishna_files)))
        ds = xr.open_dataset(file, engine='netcdf4')
        ds = ds.expand_dims(time=[all_krishna_dates[f]])
        if combined_ds is None:
            combined_ds = ds
        else:
            combined_ds = xr.concat(
                [combined_ds, ds],
                dim='time'
            )
    # compute our statistics of interest on this dataset
    print('computing statistics on dataset')
    print('computing num_obs')
    stats_ds = xr.Dataset({
        'num_obs': combined_ds['band_data'].count(dim='time'),
    })
    print('computing mean')
    stats_ds['retrieved_lfmc_mean'] = combined_ds['band_data'].mean(dim='time', skipna=True)
    print('computing std')
    stats_ds['retrieved_lfmc_std'] = combined_ds['band_data'].std(dim='time', skipna=True)
    print('computing min')
    stats_ds['retrieved_lfmc_min'] = combined_ds['band_data'].min(dim='time', skipna=True)
    print('computing max')
    stats_ds['retrieved_lfmc_max'] = combined_ds['band_data'].max(dim='time', skipna=True)
    # --- Monthly means: one variable per month ---
    print('computing monthly climatological means (Jan_mean, Feb_mean, ...)')
    monthly_clim = (
        combined_ds['band_data']
        .groupby('time.month')
        .mean(dim='time', skipna=True)
    )
    for i, month_name in enumerate(calendar.month_abbr[1:], start=1):  # Jan..Dec
        var_name = f'retrieved_{month_name}_mean'
        stats_ds[var_name] = monthly_clim.sel(month=i)
        stats_ds[var_name].attrs['description'] = f'Mean LFMC for {month_name} across all years'
    skewness_arr = np.zeros((stats_ds.dims['y'], stats_ds.dims['x'])) * np.nan
    kurtosis_arr = np.zeros((stats_ds.dims['y'], stats_ds.dims['x'])) * np.nan
    autocorr1_arr = np.zeros((stats_ds.dims['y'], stats_ds.dims['x'])) * np.nan
    autocorr2_arr = np.zeros((stats_ds.dims['y'], stats_ds.dims['x'])) * np.nan
    for i in range(stats_ds.dims['y']):
        print(f'Processing row {i+1} of {stats_ds.dims["y"]}')
        for j in range(stats_ds.dims['x']):
            pixel_time_series = combined_ds['band_data'][:, i, j].values
            if np.sum(~np.isnan(pixel_time_series)) > 2:
                skewness_arr[i, j] = nan_skew(pixel_time_series)
                kurtosis_arr[i, j] = nan_kurtosis(pixel_time_series)
                autocorr1_arr[i, j] = nan_autocorr(pixel_time_series, lag=1)
                autocorr2_arr[i, j] = nan_autocorr(pixel_time_series, lag=2)
    stats_ds['retrieved_lfmc_skewness'] = (('y', 'x'), skewness_arr)
    stats_ds['retrieved_lfmc_kurtosis'] = (('y', 'x'), kurtosis_arr)
    stats_ds['retrieved_lfmc_autocorr1'] = (('y', 'x'), autocorr1_arr)
    stats_ds['retrieved_lfmc_autocorr2'] = (('y', 'x'), autocorr2_arr)
    # save the dataset
    print('saving statistics dataset to {}'.format(save_stats_path))
    stats_ds.to_netcdf(save_stats_path, engine='netcdf4')
    #stats_ds = xr.open_dataset(save_stats_path, engine='netcdf4')
    # plot each of the variables
    for v,var in enumerate(stats_ds.data_vars):
        print(f'Plotting {var}...')
        print(np.nanmin(stats_ds[var].values))
        print(np.nanmax(stats_ds[var].values))
        plotting.plot_from_xarray(
            load_type='ds',
            type_obj=stats_ds,
            var=var,
            proj_in='EPSG:5070',
            proj_out='EPSG:5070',
            fname=os.path.join(krishna_plots_path, f'{var}_stats.png'),
            cmap='YlOrBr'
        )
    # set nan to -9999 and plot to get a view of nan values
    print('plotting where we have values')
    stats_ds['num_obs'] = stats_ds['num_obs'].where(stats_ds['num_obs'] == 0, other=-9999)
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=stats_ds,
        var='num_obs',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(krishna_plots_path, 'nan_viewing.png'),
        cmap='YlOrBr'
    )

def extract_date_from_fname(fname):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    if match:
        return pd.to_datetime(match.group(1))
    else:
        raise ValueError(f"Date not found in filename: {fname}")

if __name__ == "__main__":
    # Define the path to the Krishna files
    krishna_files_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/'
        'krishna_regrid'
    )
    krishna_plots_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/'
        'plots'
    )
    save_stats_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/'
        'stats/krishna_lfmc_statistics.nc4'
    )

    main(krishna_files_path,krishna_plots_path,save_stats_path)
