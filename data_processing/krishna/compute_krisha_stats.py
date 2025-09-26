import glob
import os
import xarray as xr
import re
import sys
import pandas as pd
import rioxarray
import numpy as np

# Add the parent directory to the path to import plotting
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from shared import plotting

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
    for f,file in enumerate(all_krishna_files):
        print('adding file {} to dataset'.format(file))
        print('file {} of {}'.format(f+1,len(all_krishna_files)))
        da = xr.open_dataarray(file, engine='netcdf4')
        da = da.expand_dims(time=[all_krishna_dates[f]])
        if combined_ds is None:
            combined_ds = da
        else:
            combined_ds = xr.concat(
                [combined_ds, da],
                dim='time'
            )
    # compute our statistics of interest on this dataset
    print('computing statistics on dataset')
    print('computing num_obs')
    stats_ds = xr.Dataset({
        'num_obs': combined_ds.count(dim='time'),
    })
    print('computing mean')
    stats_ds['retrieved_lfmc_mean'] = combined_ds.mean(dim='time', skipna=True)
    print('computing std')
    stats_ds['retrieved_lfmc_std'] = combined_ds.std(dim='time', skipna=True)
    print('computing min')
    stats_ds['retrieved_lfmc_min'] = combined_ds.min(dim='time', skipna=True)
    print('computing max')
    stats_ds['retrieved_lfmc_max'] = combined_ds.max(dim='time', skipna=True)  
    # computing seasonal means
    seasonal_means = combined_ds.groupby('time.season').mean(dim='time', skipna=True)
    for season in seasonal_means['season'].values:
        var_name = 'retrieved_lfmc_' + season.lower() + '_mean'
        stats_ds[var_name] = seasonal_means.sel(season=season)
    print('stats_ds')
    print(stats_ds)
    #print('computing p10')
    #stats_ds['p10'] = combined_ds.quantile(0.1, dim='time', skipna=True)
    #print('computing p90')
    #stats_ds['p90'] = combined_ds.quantile(0.9, dim='time', skipna=True)
    # need to squeeze the quantiles
    #stats_ds['p10'] = stats_ds['p10'].squeeze('quantile')
    #stats_ds['p90'] = stats_ds['p90'].squeeze('quantile')
    # plot each of these variables
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
