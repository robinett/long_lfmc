import datetime
import copy
import pandas as pd
import os
import sys
import numpy as np
import xarray as xr
from pyproj import Transformer
import calendar

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
import plotting as plot

def compile_data(
    start_date,
    end_date,
    feature_info,
    label_info,
    days_to_include,
    out_dir
):
    # get all of the dates that we need to compile (all the dates that we have
    # labels for)
    label_names = list(label_info['dirs'].keys())
    for l,label_name in enumerate(label_names):
        this_labs = pd.read_csv(label_info['dirs'][label_name])
        # add a column that notes the source of the label
        this_labs['source'] = label_name
        # filter by date
        this_labs['date'] = pd.to_datetime(this_labs['date'])
        this_labs = this_labs[
            (this_labs['date'] >= pd.to_datetime(start_date,utc=True)) &
            (this_labs['date'] <= pd.to_datetime(end_date,utc=True))
        ]
        if l == 0:
            labels = this_labs
        else:
            labels = labels.append(this_labs, ignore_index=True)
    # print getting the dates for which we need to open a file
    label_dates = labels['date'].to_numpy()
    explicit_dates = [0 for _ in range(len(label_dates))]
    for d,date in enumerate(label_dates):
        datetime_datetime = datetime.datetime.strptime(
            str(date), '%Y-%m-%d %H:%M:%S+00:00'
        )
        datetime_date = datetime_datetime.date()
        explicit_dates[d] = datetime_date
    # all data dates has the potential to just be every date. so we don't have
    # to keep having to append, let's just initialize to that length.
    possible_days = (end_date - start_date).days + 1 + np.max(days_to_include)
    all_data_dates = ['none' for _ in range(possible_days)]
    date_idx = 0
    for date in explicit_dates:
        # if the date is not in the all_data_dates, then we need to add it
        if date not in all_data_dates:
            all_data_dates[date_idx] = date
            date_idx += 1
    # get all of the dates that we possible need to open. This is the dates
    # that we have labels for as 
    for day_offset in days_to_include:
        print('working for day offset:', day_offset)
        # get the date that we need to open
        offset_dates = [
            date - datetime.timedelta(days=day_offset)
            for date in explicit_dates
        ]
        # if the date is not in the explicit dates, then we need to add it
        for date in offset_dates:
            if date not in all_data_dates:
                all_data_dates[date_idx] = date
                date_idx += 1
    # get rid of all the 'none' values
    all_data_dates_cleared = [date for date in all_data_dates if date != 'none']
    # sort the dates
    all_data_dates_cleared.sort(reverse=True)
    # go through the features and compile the data
    feature_source_names = list(feature_info['dirs'].keys())
    static_sources = []
    dynamic_sources = []
    for f_source in feature_source_names:
        this_time_type = feature_info['type'][f_source]
        if 'static' in this_time_type:
            static_sources.append(f_source)
        elif 'temporal' in this_time_type:
            dynamic_sources.append(f_source)
    # let's get the labels too
    label_source_names = list(label_info['dirs'].keys())
    label_names = []
    label_sources = []
    for l_source in label_source_names:
        for var in label_info['vars'][l_source]:
            if var not in label_names:
                # we only want to add the variable name once
                label_names.append(var)
                label_names.append('source')
    # we are going to store all of our information in a dictionary of
    # dictionaries that we will then turn into a pandas dataframe at the end.
    compiled_data = {}
    # finally, were going to be doing a lot of transforming here. Let's just
    # make this now
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    # loop over each day. this goes in reverse order to make some things easier
    for d,day in enumerate(all_data_dates_cleared):
        print('working on day:', day)
        todays_data_needs = []
        # get the lat/lon coordinates for the observations that we have on this
        # day. these observations need their own new entry in the dictionary of
        # {day}_{lat}_{lon}
        day_str = day.strftime('%Y-%m-%d')
        labels_today = labels[
            (labels['date'].dt.year == day.year) &
            (labels['date'].dt.month == day.month) &
            (labels['date'].dt.day == day.day)
        ]
        # initialize the dictionary for this day
        print('working for static variables')
        for idx in labels_today.index:
            # get the lat/lon for this variable
            lat = labels_today['latitude'].loc[idx]
            lon = labels_today['longitude'].loc[idx]
            this_name = (
                day_str + '_' +
                str(lat) + '_' +
                str(lon)
            )
            compiled_data[this_name] = {}
            # add the label for this day
            for ln,label_name in enumerate(label_names):
                compiled_data[this_name][label_name] = (
                    labels_today[label_name].loc[idx]
                )
            # add this to the tracker of the data we need to get
            todays_data_needs.append(copy.deepcopy(this_name))
            # add the static data while we have the lat/lons ready to go
            # loop over each static source
            for s_source in static_sources:
                this_ds = xr.open_dataset(
                    feature_info['dirs'][s_source],
                    engine='netcdf4'
                )
                ds_bounds = this_ds.rio.bounds()
                # get the variable names for this source
                this_vars = feature_info['vars'][s_source]
                # loop over each variable in that source
                for v,var in enumerate(this_vars):
                    # we need to convert to the common resolution of EPSG:5070
                    x,y = transformer.transform(lon, lat)
                    # make sure not outside the bounds of our dataset
                    if (x < ds_bounds[0] or x > ds_bounds[2] or
                        y < ds_bounds[1] or y > ds_bounds[3]):
                        # this lat/lon is outside the bounds of our dataset
                        # so we will just fill it with NaNs
                        for name in todays_data_needs:
                            this_data = np.nan
                    else:
                        # extract the data for this lat lon
                        this_data = this_ds[var].sel(
                            x=x, y=y, method='nearest'
                        ).values
                    # add this data to the dictionary
                    compiled_data[this_name][var] = this_data
                this_ds.close()
            # add site, lat, and lon to our dataset as well because we will
            # need this to inform train/test split later
            compiled_data[this_name]['site_name'] = labels_today[
                'site_name'
            ].loc[idx]
            compiled_data[this_name]['latitude'] = lat
            compiled_data[this_name]['longitude'] = lon
            compiled_data[this_name]['date'] = day_str
        # get the lat/lon coordinates for the labels that need information from
        # this day to meet their t-d requirement.
        for o,off in enumerate(days_to_include):
            this_offset_date = day + datetime.timedelta(days=off)
            this_off_day_str = this_offset_date.strftime('%Y-%m-%d')
            labels_this_off = labels[
                (labels['date'].dt.year == this_offset_date.year) &
                (labels['date'].dt.month == this_offset_date.month) &
                (labels['date'].dt.day == this_offset_date.day)
            ]
            for idx in labels_this_off.index:
                this_name = (
                    this_off_day_str + '_' +
                    str(labels_this_off['latitude'].loc[idx]) + '_' +
                    str(labels_this_off['longitude'].loc[idx])
                )
                todays_data_needs.append(copy.deepcopy(this_name))
        # loop over each source file
        for d_source in dynamic_sources:
            # open the file for that source
            this_fname = feature_info['dirs'][d_source]
            this_full_dir = this_fname.format(
                year=day.year,
                month=day.month,
                day=day.day
            )
            # because daymet makes no sense in how they deal with leap years,
            # we need to check and note if this is the 31st of december in a
            # leap year in a daymet source
            #if (
            #    'daymet' in d_source and
            #    day.month == 12 and
            #    day.day == 31 and
            #    calendar.isleap(day.year)
            #):
            #    daymet_leap = True
            #    fake_day = day - datetime.timedelta(days=1)
            #    this_full_dir = this_fname.format(
            #        year=fake_day.year,
            #        month=fake_day.month,
            #        day=fake_day.day
            #    )
            #else:
            #    daymet_leap = False
            this_ds = xr.open_dataset(this_full_dir,engine='netcdf4')
            #if d_source == 'modis':
            #    this_ds_vals = this_ds['Nadir_Reflectance_Band1'].values
            #    not_nans = np.where(~np.isnan(this_ds_vals))
            #    print(not_nans)
            #    sys.exit()
            #    #plot.plot_from_xarray(
            #    #    'ds',this_ds,'Nadir_Reflectance_Band1',
            #    #    'EPSG:5070','EPSG:5070',
            #    #    (
            #    #        '/scratch/users/trobinet/long_lfmc/trent_datasets/'
            #    #        'compiled/plots/test_modis.png'
            #    #    )
            #    #)
            #    #sys.exit()
            ds_bounds = this_ds.rio.bounds()
            this_vars = feature_info['vars'][d_source]
            # loop over each variable in that file
            for v,var in enumerate(this_vars):
                print('working on dynamic data for variable:', var)
                # get the unique lat lons
                unique_coords = set()
                for name in todays_data_needs:
                    _, lat_str,lon_str = name.split('_')
                    lat = float(lat_str)
                    lon = float(lon_str)
                    unique_coords.add((lat, lon))
                # loop over the unique lat lots
                for lat, lon in unique_coords:
                    # we need to conver to the common resolution of EPSG:5070
                    x,y = transformer.transform(lon, lat)
                    # make sure not outside the bounds of our dataset
                    if (x < ds_bounds[0] or x > ds_bounds[2] or
                        y < ds_bounds[1] or y > ds_bounds[3]):
                        # this lat/lon is outside the bounds of our dataset
                        # so we will just fill it with NaNs
                        for name in todays_data_needs:
                            this_data = np.nan
                    else:
                        # extract the data for this lat lon
                        this_data = this_ds[var].sel(
                            x=x, y=y, method='nearest'
                        ).values
                    # if this is a daymet source and we are on the 31st of
                    # december in a leap year, then we need to set the data to
                    # NaN for the 31st of december
                    #if daymet_leap:
                    #    this_data = np.nan
                    # add this information to the relevant places in the
                    # dictionary
                    target_pair = f"{lat}_{lon}"
                    matches = [
                        name for name in todays_data_needs if name.endswith(target_pair)
                    ]
                    for m,ma in enumerate(matches):
                        match_date, _, _ = ma.split('_')
                        match_date_dtm = datetime.datetime.strptime(
                            match_date, '%Y-%m-%d'
                        )
                        # how many days is this from our current date?
                        days_from_today = (match_date_dtm.date() - day).days
                        this_key_name = '{var}_day_minus_{days_from_today}'.format(
                            var=var, days_from_today=days_from_today
                        )
                        # add the data to the dictionary
                        compiled_data[ma][this_key_name] = this_data
            # close the file
            this_ds.close()
    # turn the dictionary into a pandas dataframe
    compiled_df = pd.DataFrame.from_dict(compiled_data, orient='index')
    print(compiled_df)
    # save the dataframe to a csv file
    compiled_df.to_csv(
        out_dir,
        index_label='day_lat_lon'
    )

def merge_compiled_data():
    pass
