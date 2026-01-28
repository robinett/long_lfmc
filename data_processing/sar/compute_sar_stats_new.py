import glob
import os
import xarray as xr
import re
import sys
import pandas as pd
import rioxarray
import numpy as np
import calendar
from dask.diagnostics import ProgressBar
import shutil
from pyproj import Transformer
from scipy.stats import skew

# Add the parent directory to the path to import plotting
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from shared import plotting

def write_vars(ds_piece: xr.Dataset, zpath: str):
    ds_piece = ds_piece.chunk({"y": 512, "x": 512})
    mode = "w" if not os.path.exists(zpath) else "a"
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    ds_piece.to_zarr(zpath, mode=mode)

def main(ds_path,plots_path,save_stats_path):
    sar_ds = xr.open_zarr(ds_path, chunks='auto')
    ## timeseries plotting for debugging
    #trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    #sample_point = [-113.848216, 35.145574]
    #sample_point_5070 = trns.transform(*sample_point)
    #sar_point = sar_ds.sel(x=sample_point_5070[0], y=sample_point_5070[1], method="nearest")
    #sar_vals = sar_point['vh_backscatter'].values
    #sar_dates = sar_point['time'].values
    #plotting.plot_timeseries(
    #    sar_dates, sar_vals,
    #    xlabel='Date', ylabel='Backscatter (dB)',
    #    save_name=os.path.join(plots_path, 'sar_backscatter_timeseries_crazy_std.png'),
    #    title=f'{sample_point[0]}_{sample_point[1]}',
    #    #time_bound=[pd.Timestamp('2018-03-01'), pd.Timestamp('2018-03-15')]
    #)
    ## timeseries plotting for debugging
    #trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    #sample_point = [-118.976675, 37.715220]
    #sample_point_5070 = trns.transform(*sample_point)
    #sar_point = sar_ds.sel(x=sample_point_5070[0], y=sample_point_5070[1], method="nearest")
    #sar_vals = sar_point['vh_backscatter'].values
    #sar_dates = sar_point['time'].values
    #plotting.plot_timeseries(
    #    sar_dates, sar_vals,
    #    xlabel='Date', ylabel='Backscatter (dB)',
    #    save_name=os.path.join(plots_path, 'sar_backscatter_timeseries_normal_std.png'),
    #    title=f'{sample_point[0]}_{sample_point[1]}'
    #)
    # get rid of the save stats path if it already exists so that we aren't double-writing
    if os.path.exists(save_stats_path):
        # make sure the user is okay getting rid of the old file
        #response = input(f"{save_stats_path} exists. Are you okay deleting it? (y/n) ")
        response = 'y'
        if response.lower() == "n":
            print("Exiting.")
            sys.exit()
        elif response.lower() == "y":
            shutil.rmtree(save_stats_path)
        else:
            print('Unacceptable response. Exiting.')
            sys.exit()
    print('computing statistics on dataset')
    print('computing num_obs')
    with ProgressBar():
        num_obs = xr.Dataset({
            'num_obs_vh': sar_ds['vh_backscatter'].count(dim='time'),
        })
    print('writing num_obs')
    write_vars(num_obs, save_stats_path)
    print('plotting num_obs')
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=num_obs,
        var='num_obs_vh',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(plots_path, 'num_obs_vh.png'),
        cmap='YlOrBr'
    )
    # ---------------- mean ----------------
    print('computing means')
    means = xr.Dataset({
        'sar_vh_mean': sar_ds['vh_backscatter'].mean(dim='time', skipna=True),
    })
    print('writing means')
    write_vars(means, save_stats_path)
    print('plotting means')
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=means,
        var='sar_vh_mean',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(plots_path, 'sar_vh_mean.png'),
        cmap='YlOrBr'
    )
    print('computing seasonal amplitude')
    monthly_means = sar_ds['vh_backscatter'].groupby(
        'time.month'
    ).mean(dim='time', skipna=True)
    seasonal_amp = (
        monthly_means.max(dim='month', skipna=True)
        - monthly_means.min(dim='month', skipna=True)
    )
    seasonal_amp_ds = xr.Dataset({
        'sar_vh_seasonal_amp': seasonal_amp
    })
    print('writing seasonal amplitude')
    write_vars(seasonal_amp_ds, save_stats_path)
    print('plotting seasonal amplitude')
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=seasonal_amp_ds,
        var='sar_vh_seasonal_amp',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(plots_path, 'sar_vh_seasonal_amp.png'),
        cmap='YlOrBr'
    )
    print('computing annual harmonic stats')
    # fraction of variance explained by yearly seasonal cycle
    x = sar_ds['vh_backscatter']
    # 1) time in days relative to mean time (for stability)
    t_days = (
        (sar_ds['time'] - sar_ds['time'].mean('time'))
        / np.timedelta64(1, 'D')
    )
    # 2) annual harmonic basis
    omega = 2.0 * np.pi / 365.25
    c = xr.apply_ufunc(np.cos, omega * t_days)
    s = xr.apply_ufunc(np.sin, omega * t_days)
    # 3) center x (intercept handled cleanly)
    x_mean = x.mean('time', skipna=True)
    x0 = x - x_mean
    # 4) helper: covariance over time (robust to NaNs via skipna=True)
    def cov_time(a, b):
        return (a * b).mean('time', skipna=True)
    # 5) compute normal-equation pieces (all are 2D y,x fields)
    cc = cov_time(c, c)
    ss = cov_time(s, s)
    cs = cov_time(c, s)
    xc = cov_time(x0, c)
    xs = cov_time(x0, s)
    # 6) solve 2x2 system for a1,b1 per pixel:
    # [cc  cs][a1] = [xc]
    # [cs  ss][b1]   [xs]
    det = cc * ss - cs * cs
    # avoid divide-by-zero (e.g. degenerate time sampling)
    det = det.where(det != 0)
    a1 = ( xc * ss - xs * cs ) / det
    b1 = ( xs * cc - xc * cs ) / det
    annual_power = a1**2 + b1**2
    # 7) total power = variance of x over time (population variance is fine)
    total_power = x.var('time', skipna=True)
    annual_fraction = annual_power / total_power
    annual_fraction = annual_fraction.clip(0, 1)
    annual_stats = xr.Dataset({
        'sar_vh_annual_fraction': annual_fraction,
    })
    print('writing annual fraction')
    write_vars(annual_stats, save_stats_path)
    print('plotting annual fraction')
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=annual_stats,
        var='sar_vh_annual_fraction',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(plots_path, 'sar_vh_annual_fraction.png'),
        cmap='YlOrBr'
    )
    #print('computing jump statistics')
    #x = sar_ds['vh_backscatter']
    ## 1) absolute jump between consecutive observations
    ##dx = x.diff(dim='time').abs()
    #dx = np.abs(x.diff(dim='time'))
    ## 2) time delta in days between consecutive observations
    #dt_days = (
    #    sar_ds['time'].diff(dim='time')
    #    / np.timedelta64(1, 'D')
    #)
    ## dt_days is 1D (time-1). Broadcast will align automatically.
    #jump_rate = dx / dt_days
    ## guard against divide by zero / weird timestamps
    #jump_rate = jump_rate.where(dt_days > 0)
    #mean_abs_jump_rate = jump_rate.mean(dim='time', skipna=True)
    #jump_stats = xr.Dataset({
    #    'sar_vh_mean_abs_jump_rate_per_day': mean_abs_jump_rate
    #})
    #print('writing jump statistics')
    #write_vars(jump_stats, save_stats_path)
    #print('plotting jump statistics')
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=jump_stats,
    #    var='sar_vh_mean_abs_jump_rate_per_day',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vh_mean_abs_jump_rate_per_day.png'),
    #    cmap='YlOrBr'
    #)
    #print('computing skewness statistics')
    #skew_da = xr.apply_ufunc(
    #    skew,
    #    sar_ds['vh_backscatter'],
    #    kwargs={'nan_policy': 'omit', 'bias': False},
    #    vectorize=True,
    #    dask='parallelized',
    #    output_dtypes=[float],
    #)
    #skewness = xr.Dataset({
    #    'sar_vh_skewness': skew_da
    #})
    #print('writing skewness statistics')
    #write_vars(skewness, save_stats_path)
    #print('plotting skewness statistics')
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=skewness,
    #    var='sar_vh_skewness',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vh_skewness.png'),
    #    cmap='YlOrBr'
    #)

if __name__ == "__main__":
    # Define the path to the SAR files
    sar_files_path = (
        '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/'
        'sar_500m_full.zarr'
    )
    krishna_plots_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/'
        'plots'
    )
    save_stats_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/'
        'sar_stats_seasonal.zarr'
    )
    main(sar_files_path, krishna_plots_path, save_stats_path)
