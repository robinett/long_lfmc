import datetime
import os
import xarray as xr
import pandas as pd
import sys
import numpy as np
import argparse
import warnings

def compute_stats(
    start_date,
    end_date,
    start_search_for_rain_date,
    generic_fname,
    generic_save_fname
):
    '''
    stats that we are going to find:
        days since last rain
        max precip in last 14 days
        rolling precip in last 14 days
        max temp in last 14 days
        rolling temp in last 14 days
        min water vapor pressure in last 14 days
        rolling water vapor pressure in last 14 days
    '''
    # since these are all stats that will be relatively recent in time, we need
    # to fill out things that have happened
    fourteen_days_before = start_date - datetime.timedelta(days=14)
    seven_days_before = start_date - datetime.timedelta(days=7)
    # get first day dataset as an example
    example_ds_name = generic_fname.format(
        var='prcp',
        year=start_date.year,
        month=start_date.month,
        day=start_date.day
    )
    example_ds = xr.open_dataset(example_ds_name, engine='netcdf4')
    # initialize our ds as the example but no variables
    stats_ds = example_ds.copy(deep=True)
    stats_ds = stats_ds.drop_vars(example_ds.data_vars)
    # initialize our variables
    days_since_rain = np.zeros(
        (stats_ds.y.size, stats_ds.x.size),
    )
    prcp_last_14_days = np.zeros(
        (stats_ds.y.size, stats_ds.x.size, 14)
    )
    temp_last_14_days = np.zeros(
        (stats_ds.y.size, stats_ds.x.size, 14)
    )
    watervp_last_14_days = np.zeros(
        (stats_ds.y.size, stats_ds.x.size, 14)
    )
    current_date = start_search_for_rain_date
    while current_date < start_date:
        print('pre-processing date: {}'.format(current_date))
        # generate the filename for this date
        precip_fname = generic_fname.format(
            var='prcp',
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        # open the dataset
        precip_ds = xr.open_dataset(precip_fname, engine='netcdf4')
        # calculate days since last rain
        prcp = precip_ds['prcp'].values
        # if the first day, set correct nan positions
        if current_date == start_search_for_rain_date:
            nan_idx = np.isnan(prcp)
            days_since_rain[nan_idx] = np.nan
        has_rain = prcp > 0
        days_since_rain[has_rain] = 0
        days_since_rain[~has_rain] += 1
        # preserve nans in the correct locations
        prcp[nan_idx] = np.nan
        days_to_start = (start_date - current_date).days
        if days_to_start <= 14:
            temp_fname = generic_fname.format(
                var='tmax',
                year=current_date.year,
                month=current_date.month,
                day=current_date.day
            )
            temp_ds = xr.open_dataset(temp_fname, engine='netcdf4')
            watervp_fname = generic_fname.format(
                var='vp',
                year=current_date.year,
                month=current_date.month,
                day=current_date.day
            )
            watervp_ds = xr.open_dataset(watervp_fname, engine='netcdf4')
            # fill in the prcp last 14 days
            prcp_last_14_days[:, :, days_to_start-1] = prcp
            temp_last_14_days[:, :, days_to_start-1] = temp_ds['tmax'].values
            watervp_last_14_days[:, :, days_to_start-1] = watervp_ds['vp'].values
        current_date += datetime.timedelta(days=1)
    while current_date <= end_date:
        # if it's a leap day we don't have a file and need to add nan to our
        # tracking arrays
        if (
            current_date.month == 12 and current_date.day == 31 and
            pd.Timestamp(current_date).is_leap_year
        ):
            print('skipping leap day: {}'.format(current_date))
            prcp_last_14_days = np.roll(
                prcp_last_14_days, shift=1, axis=2
            )
            temp_last_14_days = np.roll(
                temp_last_14_days, shift=1, axis=2
            )
            watervp_last_14_days = np.roll(
                watervp_last_14_days, shift=1, axis=2
            )
            prcp_last_14_days[:, :, 0] = np.nan
            temp_last_14_days[:, :, 0] = np.nan
            watervp_last_14_days[:, :, 0] = np.nan
            days_since_rain[:,:] += 1
            current_date += datetime.timedelta(days=1)
            continue
        print('processing date: {}'.format(current_date))
        # generate our dataset for this date
        stats_ds = example_ds.copy(deep=True)
        stats_ds = stats_ds.drop_vars(example_ds.data_vars)
        # move all information back one day in current arrays
        prcp_last_14_days = np.roll(prcp_last_14_days, shift=1, axis=2)
        temp_last_14_days = np.roll(temp_last_14_days, shift=1, axis=2)
        watervp_last_14_days = np.roll(watervp_last_14_days, shift=1, axis=2)
        # add today's information
        prcp_fname = generic_fname.format(
            var='prcp',
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        prcp_ds = xr.open_dataset(prcp_fname, engine='netcdf4')
        prcp = prcp_ds['prcp'].values
        prcp_last_14_days[:, :, 0] = prcp
        has_rain = prcp > 0
        days_since_rain[has_rain] = 0
        days_since_rain[~has_rain] += 1
        # can't be more than 59 days since rain
        days_since_rain[days_since_rain > 59] = 59
        days_since_rain[nan_idx] = np.nan
        temp_fname = generic_fname.format(
            var='tmax',
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        temp_ds = xr.open_dataset(temp_fname, engine='netcdf4')
        temp = temp_ds['tmax'].values
        temp_last_14_days[:, :, 0] = temp
        watervp_fname = generic_fname.format(
            var='vp',
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        watervp_ds = xr.open_dataset(watervp_fname, engine='netcdf4')
        watervp = watervp_ds['vp'].values
        watervp_last_14_days[:, :, 0] = watervp
        # calculate the stats that we want
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats_ds['days_since_rain'] = (('y', 'x'), days_since_rain)
            stats_ds['max_precip_14_days'] = (
                ('y', 'x'),
                prcp_last_14_days.max(axis=2)
            )
            stats_ds['rolling_precip_14_days'] = (
                ('y', 'x'),
                np.nanmean(prcp_last_14_days, axis=2)
            )
            stats_ds['max_temp_14_days'] = (
                ('y', 'x'),
                np.nanmax(temp_last_14_days, axis=2)
            )
            stats_ds['rolling_temp_14_days'] = (
                ('y', 'x'),
                np.nanmean(temp_last_14_days, axis=2)
            )
            stats_ds['min_watervp_14_days'] = (
                ('y', 'x'),
                np.nanmin(watervp_last_14_days, axis=2)
            )
            stats_ds['rolling_watervp_14_days'] = (
                ('y', 'x'),
                np.nanmean(watervp_last_14_days, axis=2)
            )
        # save the dataset. make sure directory exists
        save_stats_path = generic_save_fname.format(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        os.makedirs(
            os.path.dirname(save_stats_path), exist_ok=True
        )
        stats_ds.to_netcdf(save_stats_path, engine='netcdf4')
        current_date += datetime.timedelta(days=1)



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
    start_date = datetime.datetime.strptime(
        parser.parse_args().start_date,
        '%Y-%m-%d'
    ).date()
    end_date = datetime.datetime.strptime(
        parser.parse_args().end_date,
        '%Y-%m-%d'
    ).date()
    #start_date = datetime.date(2003, 3, 1)
    #end_date = datetime.date(2023, 12, 31)
    start_search_for_rain_date = start_date - datetime.timedelta(days=59)
    scratch_dir = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/daymet/'
        '/daymet_regrid/'
    )
    generic_fname = os.path.join(
        scratch_dir,
        '{var}',
        '{year}',
        '{month:02d}',
        '{var}_{year}_{month:02d}_{day:02d}_regridded.nc'
    )
    generic_save_fname = os.path.join(
        scratch_dir,
        '..',
        'stats',
        '{year}',
        '{month:02d}',
        'stats_{year}_{month:02d}_{day:02d}_regridded.nc'
    )
    compute_stats(
        start_date,
        end_date,
        start_search_for_rain_date,
        generic_fname,
        generic_save_fname
    )
