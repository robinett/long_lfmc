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

def main(ds_path,plots_path,save_stats_path):
    # --- helper: write-or-append a small Dataset to Zarr ---
    def write_vars(ds_piece: xr.Dataset, zpath: str):
        ds_piece = ds_piece.chunk({"y": 512, "x": 512})
        mode = "w" if not os.path.exists(zpath) else "a"
        os.makedirs(os.path.dirname(zpath), exist_ok=True)
        ds_piece.to_zarr(zpath, mode=mode)

    # open source (unchanged)
    sar_ds = xr.open_zarr(ds_path, chunks='auto')
    #with ProgressBar():
    #    sar_ds.load()


    # timeseries plotting for debugging
    trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    sample_point = [-113.848216, 35.145574]
    sample_point_5070 = trns.transform(*sample_point)
    sar_point = sar_ds.sel(x=sample_point_5070[0], y=sample_point_5070[1], method="nearest")
    sar_vals = sar_point['vh_backscatter'].values
    sar_dates = sar_point['time'].values
    plotting.plot_timeseries(
        sar_dates, sar_vals,
        xlabel='Date', ylabel='Backscatter (dB)',
        save_name=os.path.join(plots_path, 'sar_backscatter_timeseries_crazy_std.png'),
        title=f'{sample_point[0]}_{sample_point[1]}',
        #time_bound=[pd.Timestamp('2018-03-01'), pd.Timestamp('2018-03-15')]
    )
    # timeseries plotting for debugging
    trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    sample_point = [-118.976675, 37.715220]
    sample_point_5070 = trns.transform(*sample_point)
    sar_point = sar_ds.sel(x=sample_point_5070[0], y=sample_point_5070[1], method="nearest")
    sar_vals = sar_point['vh_backscatter'].values
    sar_dates = sar_point['time'].values
    plotting.plot_timeseries(
        sar_dates, sar_vals,
        xlabel='Date', ylabel='Backscatter (dB)',
        save_name=os.path.join(plots_path, 'sar_backscatter_timeseries_normal_std.png'),
        title=f'{sample_point[0]}_{sample_point[1]}'
    )
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

    # ---------------- counts ----------------
    print('computing num_obs')
    num_obs = xr.Dataset({
        #'num_obs_vv': sar_ds['VV'].count(dim='time'),
        'num_obs_vh': sar_ds['vh_backscatter'].count(dim='time'),
        #'num_obs_vv_minus_vh': sar_ds['vv_minus_vh'].count(dim='time'),
    })
    write_vars(num_obs, save_stats_path)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=num_obs,
    #    var='num_obs_vh',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'num_obs_vh.png'),
    #    cmap='YlOrBr'
    #)

    # ---------------- mean ----------------
    print('computing means')
    means = xr.Dataset({
        #'sar_vv_mean': sar_ds['VV'].mean(dim='time', skipna=True),
        'sar_vh_mean': sar_ds['vh_backscatter'].mean(dim='time', skipna=True),
        #'sar_vv_minus_vh_mean': sar_ds['vv_minus_vh'].mean(dim='time', skipna=True),
    })
    write_vars(means, save_stats_path)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=means,
    #    var='sar_vv_mean',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vv_mean.png'),
    #    cmap='YlOrBr'
    #)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=means,
    #    var='sar_vh_mean',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vh_mean.png'),
    #    cmap='YlOrBr'
    #)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=means,
    #    var='sar_vv_minus_vh_mean',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vv_minus_vh_mean.png'),
    #    cmap='YlOrBr'
    #)

    # ---------------- std ----------------
    print('computing std')
    stds = xr.Dataset({
        #'sar_vv_std': sar_ds['VV'].std(dim='time', skipna=True),
        'sar_vh_std': sar_ds['vh_backscatter'].std(dim='time', skipna=True),
        #'sar_vv_minus_vh_std': sar_ds['vv_minus_vh'].std(dim='time', skipna=True),
    })
    write_vars(stds, save_stats_path)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=stds,
    #    var='sar_vh_std',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'sar_vh_std.png'),
    #    cmap='YlOrBr'
    #)

    # ---------------- min ----------------
    print('computing min')
    mins = xr.Dataset({
        #'sar_vv_min': sar_ds['VV'].min(dim='time', skipna=True),
        'sar_vh_min': sar_ds['vh_backscatter'].min(dim='time', skipna=True),
        #'sar_vv_minus_vh_min': sar_ds['vv_minus_vh'].min(dim='time', skipna=True),
    })
    write_vars(mins, save_stats_path)

    # ---------------- max ----------------
    print('computing max')
    maxs = xr.Dataset({
        #'sar_vv_max': sar_ds['VV'].max(dim='time', skipna=True),
        'sar_vh_max': sar_ds['vh_backscatter'].max(dim='time', skipna=True),
        #'sar_vv_minus_vh_max': sar_ds['vv_minus_vh'].max(dim='time', skipna=True),
    })
    write_vars(maxs, save_stats_path)

    ## ---------------- monthly climatology: VV ----------------
    #print('computing monthly climatological means for VV')
    #vv_monthly = sar_ds['VV'].groupby('time.month').mean(dim='time', skipna=True)
    #for i, month_name in enumerate(calendar.month_abbr[1:], start=1):
    #    print(f'  VV {month_name}')
    #    var = f'sar_vv_{month_name.lower()}_mean'
    #    write_vars(xr.Dataset({var: vv_monthly.sel(month=i)}), save_stats_path)

    # ---------------- monthly climatology: VH ----------------
    print('computing monthly climatological means for VH')
    vh_monthly = sar_ds['vh_backscatter'].groupby('time.month').mean(dim='time', skipna=True)
    for i, month_name in enumerate(calendar.month_abbr[1:], start=1):
        print(f'  vh {month_name}')
        var = f'sar_vh_{month_name.lower()}_mean'
        write_vars(xr.Dataset({var: vh_monthly.sel(month=i)}), save_stats_path)

    ## ---------------- monthly climatology: VV - VH ----------------
    #print('computing monthly climatological means for VV - VH')
    #dv_monthly = sar_ds['vv_minus_vh'].groupby('time.month').mean(dim='time', skipna=True)
    #for i, month_name in enumerate(calendar.month_abbr[1:], start=1):
    #    print(f'  VV-VH {month_name}')
    #    var = f'sar_vv_minus_vh_{month_name.lower()}_mean'
    #    write_vars(xr.Dataset({var: dv_monthly.sel(month=i)}), save_stats_path)

    # ---------------- pixelwise skew/kurt/ACF via apply_ufunc (unchanged) ----------------
    def _nan_stats_1d(a, lags=(1, 2)):
        x = np.asarray(a, dtype=np.float64)
        m = ~np.isnan(x)
        n = m.sum()
        if n <= 2:
            return (np.nan, np.nan, np.nan, np.nan)
        xv = x[m]
        v = xv.var(ddof=0)
        if v == 0.0:
            skew = 0.0
            kurt = -3.0
        else:
            mu = xv.mean()
            c3 = np.mean((xv - mu) ** 3)
            c4 = np.mean((xv - mu) ** 4)
            skew = c3 / (v ** 1.5)
            kurt = c4 / (v ** 2) - 3.0
        def acf_lag(k):
            if k >= xv.size:
                return np.nan
            a = xv[:-k] if k > 0 else xv
            b = xv[k:] if k > 0 else xv
            am, bm = a - a.mean(), b - b.mean()
            denom = np.sqrt((am*am).mean() * (bm*bm).mean())
            if denom == 0.0: return np.nan
            return (am*bm).mean() / denom
        return (skew, kurt, acf_lag(1), acf_lag(2))

    def _pack_stats(da): return _nan_stats_1d(da, lags=(1, 2))

    def reduce_all_stats(var_da):
        var_da = var_da.chunk({"time": -1})  # ensure single time chunk for core dim
        outs = xr.apply_ufunc(
            _pack_stats, var_da,
            input_core_dims=[["time"]],
            output_core_dims=[[], [], [], []],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[np.float32]*4,
            # or add: dask_gufunc_kwargs={"allow_rechunk": True}
        )
        skew, kurt, ac1, ac2 = outs
        return xr.Dataset(
            {"skew": skew, "kurtosis": kurt, "autocorr_lag1": ac1, "autocorr_lag2": ac2}
        )

    #vv = sar_ds["VV"]; vh = sar_ds["VH"]; dv = vv - vh
    vh = sar_ds["vh_backscatter"]

    #print('computing pixelwise stats for VV')
    #vv_stats = reduce_all_stats(vv).rename({
    #    "skew":"vv_skewness","kurtosis":"vv_kurtosis",
    #    "autocorr_lag1":"vv_autocorr1","autocorr_lag2":"vv_autocorr2"
    #})
    #write_vars(vv_stats, save_stats_path)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=vv_stats,
    #    var='vv_skewness',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'vv_skewness.png'),
    #    cmap='RdBu',
    #)

    print('computing pixelwise stats for vh')
    vh_stats = reduce_all_stats(vh).rename({
        "skew":"vh_skewness","kurtosis":"vh_kurtosis",
        "autocorr_lag1":"vh_autocorr1","autocorr_lag2":"vh_autocorr2"
    })
    write_vars(vh_stats, save_stats_path)
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=vh_stats,
        var='vh_skewness',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(plots_path, 'vh_skewness.png'),
        cmap='RdBu',
    )

    #print('computing pixelwise stats for VV - VH')
    #dv_stats = reduce_all_stats(dv).rename({
    #    "skew":"vv_minus_vh_skewness","kurtosis":"vv_minus_vh_kurtosis",
    #    "autocorr_lag1":"vv_minus_vh_autocorr1","autocorr_lag2":"vv_minus_vh_autocorr2"
    #})
    #write_vars(dv_stats, save_stats_path)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=dv_stats,
    #    var='vv_minus_vh_skewness',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(plots_path, 'vv_minus_vh_skewness.png'),
    #    cmap='RdBu',
    #)

    print(f"Saved (incrementally) to {save_stats_path}")
    #sar_ds = xr.open_zarr(ds_path, chunks='auto')
    ## compute our statistics of interest on this dataset
    #print('computing statistics on dataset')
    #print('computing num_obs')
    #stats_ds = xr.Dataset({
    #    'num_obs_vv': sar_ds['VV'].count(dim='time'),
    #})
    #stats_ds['num_obs_vh'] = sar_ds['VH'].count(dim='time')
    #stats_ds['num_obs_vv_minus_vh'] = sar_ds['vv_minus_vh'].count(dim='time')
    #print('computing means')
    #stats_ds['sar_vv_mean'] = sar_ds['VV'].mean(dim='time', skipna=True)
    #stats_ds['sar_vh_mean'] = sar_ds['VH'].mean(dim='time', skipna=True)
    #stats_ds['sar_vv_minus_vh_mean'] = sar_ds['vv_minus_vh'].mean(dim='time', skipna=True)
    #print('computing std')
    #stats_ds['sar_vv_std'] = sar_ds['VV'].std(dim='time', skipna=True)
    #stats_ds['sar_vh_std'] = sar_ds['VH'].std(dim='time', skipna=True)
    #stats_ds['sar_vv_minus_vh_std'] = sar_ds['vv_minus_vh'].std(dim='time', skipna=True)
    #print('computing min')
    #stats_ds['sar_vv_min'] = sar_ds['VV'].min(dim='time', skipna=True)
    #stats_ds['sar_vh_min'] = sar_ds['VH'].min(dim='time', skipna=True)
    #stats_ds['sar_vv_minus_vh_min'] = sar_ds['vv_minus_vh'].min(dim='time', skipna=True)
    #print('computing max')
    #stats_ds['sar_vv_max'] = sar_ds['VV'].max(dim='time', skipna=True)
    #stats_ds['sar_vh_max'] = sar_ds['VH'].max(dim='time', skipna=True)
    #stats_ds['sar_vv_minus_vh_max'] = sar_ds['vv_minus_vh'].max(dim='time', skipna=True)
    ## --- Monthly means: one variable per month ---
    #print('computing monthly climatological means (Jan_mean, Feb_mean, ...) for VV')
    #vv_monthly_clim = (
    #    sar_ds['VV']
    #    .groupby('time.month')
    #    .mean(dim='time', skipna=True)
    #)
    #for i, month_name in enumerate(calendar.month_abbr[1:], start=1):  # Jan..Dec
    #    print(f'  processing {month_name} for VV')
    #    month_name_fmt = month_name.lower()
    #    var_name = f'sar_vv_{month_name_fmt}_mean'
    #    stats_ds[var_name] = vv_monthly_clim.sel(month=i)
    #    stats_ds[var_name].attrs['description'] = f'Mean VV for {month_name} across all years'
    #print('computing monthly climatological means (Jan_mean, Feb_mean, ...) for VH')
    #vh_monthly_clim = (
    #    sar_ds['VH']
    #    .groupby('time.month')
    #    .mean(dim='time', skipna=True)
    #)
    #for i, month_name in enumerate(calendar.month_abbr[1:], start=1):  # Jan..Dec
    #    print(f'  processing {month_name} for VH')
    #    month_name_fmt = month_name.lower()
    #    var_name = f'sar_vh_{month_name_fmt}_mean'
    #    stats_ds[var_name] = vh_monthly_clim.sel(month=i)
    #    stats_ds[var_name].attrs['description'] = f'Mean VH for {month_name} across all years'
    #print('computing monthly climatological means (Jan_mean, Feb_mean, ...) for VV - VH')
    #vv_minus_vh_monthly_clim = (
    #    sar_ds['vv_minus_vh']
    #    .groupby('time.month')
    #    .mean(dim='time', skipna=True)
    #)
    #for i, month_name in enumerate(calendar.month_abbr[1:], start=1):  # Jan..Dec
    #    print(f'  processing {month_name} for VV - VH')
    #    month_name_fmt = month_name.lower()
    #    var_name = f'sar_vv_minus_vh_{month_name_fmt}_mean'
    #    stats_ds[var_name] = vv_minus_vh_monthly_clim.sel(month=i)
    #    stats_ds[var_name].attrs['description'] = f'Mean VV - VH for {month_name} across all years'
    ## 2) Pixelwise reducers (NaN-safe). vectorized over time axis only.
    #def _nan_stats_1d(a, lags=(1, 2)):
    #    # a: shape (T,), may contain NaNs
    #    x = np.asarray(a, dtype=np.float64)
    #    msk = ~np.isnan(x)
    #    n = msk.sum()
    #    if n <= 2:
    #        # not enough data
    #        return (np.nan, np.nan, np.nan, np.nan)

    #    xv = x[msk]
    #    n = xv.size
    #    if n <= 2:
    #        return (np.nan, np.nan, np.nan, np.nan)

    #    # moments
    #    mu = xv.mean()
    #    v = xv.var(ddof=0)
    #    if v == 0.0:
    #        skew = 0.0
    #        kurt = -3.0  # excess kurtosis of constant -> undefined; use -3?
    #    else:
    #        c3 = np.mean((xv - mu) ** 3)
    #        c4 = np.mean((xv - mu) ** 4)
    #        skew = c3 / (v ** 1.5)
    #        kurt = c4 / (v ** 2) - 3.0  # excess kurtosis

    #    # autocorr with NaNs: pairwise drop for each lag
    #    def acf_lag(k):
    #        if k >= n:
    #            return np.nan
    #        a = xv[:-k] if k > 0 else xv
    #        b = xv[k:] if k > 0 else xv
    #        if a.size == 0 or b.size == 0:
    #            return np.nan
    #        am = a - a.mean()
    #        bm = b - b.mean()
    #        denom = np.sqrt((am * am).mean() * (bm * bm).mean())
    #        if denom == 0.0:
    #            return np.nan
    #        return (am * bm).mean() / denom

    #    ac1 = acf_lag(lags[0])
    #    ac2 = acf_lag(lags[1])
    #    return (skew, kurt, ac1, ac2)

    #def _pack_stats(da):
    #    # da: DataArray(time)
    #    # returns tuple of scalars
    #    return _nan_stats_1d(da, lags=(1, 2))

    ## 3) Helper to run over (y,x) for one variable
    #def reduce_all_stats(var_da):
    #    var_da = var_da.chunk({"time": -1})  # single time chunk
    #    outs = xr.apply_ufunc(
    #        _pack_stats, var_da,
    #        input_core_dims=[["time"]],
    #        output_core_dims=[[], [], [], []],
    #        dask="parallelized",
    #        vectorize=True,
    #        output_dtypes=[np.float32]*4,
    #    )
    #    skew, kurt, ac1, ac2 = outs
    #    return xr.Dataset(
    #        {
    #            "skew": skew,
    #            "kurtosis": kurt,
    #            "autocorr_lag1": ac1,
    #            "autocorr_lag2": ac2,
    #        }
    #    )

    ## 4) Compute VV, VH, and VV-VH in one pass each (lazy)
    #vv = sar_ds["VV"]
    #vh = sar_ds["VH"]
    #vv_m_vh = vv - vh  # avoids extra reads if not already stored

    #vv_stats = reduce_all_stats(vv)
    #vh_stats = reduce_all_stats(vh)
    #dv_stats = reduce_all_stats(vv_m_vh)

    ## 5) Rename to final variables if you like
    #vv_stats = vv_stats.rename(
    #    {
    #        "skew": "vv_skewness",
    #        "kurtosis": "vv_kurtosis",
    #        "autocorr_lag1": "vv_autocorr1",
    #        "autocorr_lag2": "vv_autocorr2",
    #    }
    #)
    #vh_stats = vh_stats.rename(
    #    {
    #        "skew": "vh_skewness",
    #        "kurtosis": "vh_kurtosis",
    #        "autocorr_lag1": "vh_autocorr1",
    #        "autocorr_lag2": "vh_autocorr2",
    #    }
    #)
    #dv_stats = dv_stats.rename(
    #    {
    #        "skew": "vv_minus_vh_skewness",
    #        "kurtosis": "vv_minus_vh_kurtosis",
    #        "autocorr_lag1": "vv_minus_vh_autocorr1",
    #        "autocorr_lag2": "vv_minus_vh_autocorr2",
    #    }
    #)

    ## 6) Combine (still lazy). Compute() or to_zarr() when ready.
    #stats = xr.merge([vv_stats, vh_stats, dv_stats])
    #stats_ds = xr.merge([stats_ds, stats])
    ## save the dataset
    #print(stats_ds)
    #print('saving statistics dataset to {}'.format(save_stats_path))
    #desired_chunks = {"x": 256, "y": 256}
    #stats_ds = stats_ds.chunk(desired_chunks)
    #os.makedirs(os.path.dirname(save_stats_path), exist_ok=True)
    #stats_ds.to_zarr(save_stats_path, mode="w")
    #print(f"Saved to {save_stats_path}")
    ## plot each of the variables
    #for v,var in enumerate(stats_ds.data_vars):
    #    print(f'Plotting {var}...')
    #    print(np.nanmin(stats_ds[var].values))
    #    print(np.nanmax(stats_ds[var].values))
    #    plotting.plot_from_xarray(
    #        load_type='ds',
    #        type_obj=stats_ds,
    #        var=var,
    #        proj_in='EPSG:5070',
    #        proj_out='EPSG:5070',
    #        fname=os.path.join(krishna_plots_path, f'{var}_stats.png'),
    #        cmap='YlOrBr'
    #    )
    ## set nan to -9999 and plot to get a view of nan values
    #print('plotting where we have values')
    #stats_ds['num_obs'] = stats_ds['num_obs'].where(stats_ds['num_obs'] == 0, other=-9999)
    #plotting.plot_from_xarray(
    #    load_type='ds',
    #    type_obj=stats_ds,
    #    var='num_obs',
    #    proj_in='EPSG:5070',
    #    proj_out='EPSG:5070',
    #    fname=os.path.join(krishna_plots_path, 'nan_viewing.png'),
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
        '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/'
        'sar_stats_new.zarr'
    )
    main(sar_files_path, krishna_plots_path, save_stats_path)
