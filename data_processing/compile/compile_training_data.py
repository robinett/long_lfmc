import datetime
import copy
import pandas as pd
import os
import sys
import numpy as np
import xarray as xr
from pyproj import Transformer
import calendar
from collections import defaultdict
import dask

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
import plotting as plot

def nearest_indices_unsorted(coord, vals):
    coord = np.asarray(coord)
    vals  = np.asarray(vals)
    diffs = np.abs(coord[None, :] - vals[:, None])
    return diffs.argmin(axis=1)

def compile_data(
    start_date,
    end_date,
    feature_info,
    label_info,
    days_to_include,
    out_dir,
    num_rs_samples=0.0 # will include num_nfmd_samples * factor random samples from RS data
):
    # get all of the dates that we need to compile (all the dates that we have
    # labels for)
    label_names = list(label_info['dirs'].keys())
    for l,label_name in enumerate(label_names):
        print('working on label source:', label_name)
        this_labs = pd.read_csv(label_info['dirs'][label_name])
        #if label_name == 'nfmd':
        #    this_labs = this_labs.sample(n=200, random_state=42)  # shuffle
        # add a column that notes the source of the label
        this_labs['source'] = label_name
        # filter by date
        this_labs['date'] = pd.to_datetime(this_labs['date'], utc=True)
        this_labs = this_labs[
            (this_labs['date'] >= pd.to_datetime(start_date,utc=True)) &
            (this_labs['date'] <= pd.to_datetime(end_date,utc=True))
        ]
        # if this is in-situ data, we need to record how many samples we have
        # if this is remote sensing data, take the first n columns
        if num_rs_samples > 0.0 and label_name == 'rs':
            if len(this_labs) > num_rs_samples:
                this_labs = this_labs.iloc[:num_rs_samples]
            else:
                print(
                    f"Warning: requested {num_rs_samples} samples from remote sensing"
                    f" data, but only {len(this_labs)} available."
                )
                print("Using all available samples.")
        elif num_rs_samples == 0.0 and label_name == 'rs':
            this_labs = this_labs.iloc[0:0]  # 0 rows
        if l == 0:
            labels = this_labs
        else:
            labels = pd.concat([labels, this_labs], ignore_index=True)
    print('labels:')
    print(labels)
    print('figuring out when we need to open files...')
    # get the range of dates we could possibly need to open files for
    first_date_possible = start_date - datetime.timedelta(
        days=max(days_to_include)
    )
    last_date_possible = end_date
    all_possible_dates = pd.date_range(
        first_date_possible,
        last_date_possible,
        freq='D'
    )
    # map each of these possible dates to two lists: the first is the index
    # of the lfmc sample that needs info from this date, the second is the 
    # day offset (0 is current day, 1 is previous day, etc.)
    date_to_needed = {}
    for date in all_possible_dates:
        date_to_needed[date.date()] = {
            'indices':[],
            'offsets':[]
        }
    # go through each of the labels and note which dates they need info from
    label_dates = labels['date'].to_numpy()
    for i,date in enumerate(label_dates):
        datetime_datetime = datetime.datetime.strptime(
            str(date), '%Y-%m-%d %H:%M:%S+00:00'
        )
        datetime_date = datetime_datetime.date()
        for day_offset in days_to_include:
            needed_date = datetime_date - datetime.timedelta(days=day_offset)
            if needed_date in date_to_needed:
                date_to_needed[needed_date]['indices'].append(i)
                date_to_needed[needed_date]['offsets'].append(day_offset)
    # get all of the information that we need from our static datasets
    # first, get the unique list of lat/lon coordinates that we need and 
    # map them to the indices of the labels that need them
    coord_to_indices = {}
    coord_to_dates = {}
    unique_coords = []
    for i in range(len(labels)):
        lat = labels['latitude'].loc[i]
        lon = labels['longitude'].loc[i]
        date = labels['date'].loc[i]
        coord_str = f"{lat}_{lon}"
        if coord_str not in coord_to_indices:
            coord_to_indices[coord_str] = []
            unique_coords.append((lat, lon))
        coord_to_indices[coord_str].append(i)
        if coord_str not in coord_to_dates:
            coord_to_dates[coord_str] = []
        coord_to_dates[coord_str].append(date)
    # now, go through each of the static datasets and extract the information
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    feature_source_names = list(feature_info['dirs'].keys())
    static_sources = []
    nolag_sources = []
    dynamic_sources = []
    for f_source in feature_source_names:
        this_time_type = feature_info['type'][f_source]
        if 'static' in this_time_type:
            static_sources.append(f_source)
        elif 'temporal' in this_time_type:
            dynamic_sources.append(f_source)
        elif 'nolag' in this_time_type:
            nolag_sources.append(f_source)
    for s_source in static_sources:
        if feature_info['dirs'][s_source].endswith('.nc') or \
           feature_info['dirs'][s_source].endswith('.nc4'):
            this_ds = xr.open_dataset(
                feature_info['dirs'][s_source],
                engine='netcdf4'
            ).load()
        elif feature_info['dirs'][s_source].endswith('.zarr'):
            this_ds = xr.open_zarr(
                feature_info['dirs'][s_source],
                chunks='auto'
            )
        else:
            raise ValueError(
                f"Unrecognized file format for static dataset: "
                f"{feature_info['dirs'][s_source]}"
            )
        ds_bounds = this_ds.rio.bounds()
        xs, ys = transformer.transform(
            [lon for (lat,lon) in unique_coords],
            [lat for (lat,lon) in unique_coords]
        )
        # get the index of the unique coords that are outside the bounds of
        # our dataset
        outside_mask = []
        for i,(lat,lon) in enumerate(unique_coords):
            x,y = transformer.transform(lon, lat)
            if (x < ds_bounds[0] or x > ds_bounds[2] or
                y < ds_bounds[1] or y > ds_bounds[3]):
                outside_mask.append(i)
        outside_mask = np.array(outside_mask)
        # get the data for each of the variables in this dataset
        this_vars = feature_info['vars'][s_source]
        for v,var in enumerate(this_vars):
            to_labels = np.zeros(labels.shape[0]) + np.nan
            print('selecting static data for variable:', var)
            x_idx = nearest_indices_unsorted(this_ds['x'].values, xs)
            y_idx = nearest_indices_unsorted(this_ds['y'].values, ys)
            this_var_data = this_ds[var].isel(
                y=('points', y_idx), x=('points', x_idx)
            ).values
            this_var_data = this_var_data.squeeze()
            # assign NaNs for those outside the bounds
            this_var_data = np.where(
                np.isin(np.arange(len(this_var_data)), outside_mask),
                np.nan,
                this_var_data
            )
            print('adding to labels for variable:', var)
            # add to the labels
            for i,(lat,lon) in enumerate(unique_coords):
                coord_str = f"{lat}_{lon}"
                this_indices = coord_to_indices[coord_str]
                for idx in coord_to_indices[coord_str]:
                    to_labels[idx] = this_var_data[i]
            if 'onehot' in feature_info['type'][s_source]:
                # convert to one-hot encoding
                all_ds_vals = np.unique(this_ds[var].values)
                all_ds_vals = all_ds_vals[~np.isnan(all_ds_vals)]
                for uv in all_ds_vals:
                    this_onehot = copy.deepcopy(to_labels)
                    this_onehot = np.where(this_onehot == uv, 1, 0)
                    new_var_name = f"{var}_{int(uv)}"
                    labels[new_var_name] = this_onehot
            else:
                labels[var] = to_labels
    # edge case for the daymet stats which has a temporal dimension but 
    # does not need to be lagged
    for n_source in nolag_sources:
        nolag_data = {}
        for var in feature_info['vars'][n_source]:
            nolag_data[var] = np.zeros(labels.shape[0]) + np.nan
        this_ds = xr.open_zarr(
            feature_info['dirs'][n_source].format(year=start_date.year),
            chunks='auto'
        )
        ds_bounds = this_ds.rio.bounds()
        xs, ys = transformer.transform(
            [lon for (lat,lon) in unique_coords],
            [lat for (lat,lon) in unique_coords]
        )
        outside_mask = []
        for i,(lat,lon) in enumerate(unique_coords):
            x,y = transformer.transform(lon, lat)
            if (x < ds_bounds[0] or x > ds_bounds[2] or
                y < ds_bounds[1] or y > ds_bounds[3]):
                outside_mask.append(i)
        outside_mask = np.array(outside_mask)
        this_vars = feature_info['vars'][n_source]
        for i,row in labels.iterrows():
            if i % 100 == 0:
                print(
                    'working on nolag source:', n_source, 'for label', i, 
                    'of', len(labels)
                )
            lat = row['latitude']
            lon = row['longitude']
            date = row['date'].date()
            x,y = transformer.transform(lon, lat)
            x_idx = nearest_indices_unsorted(this_ds['x'].values, [x])[0]
            y_idx = nearest_indices_unsorted(this_ds['y'].values, [y])[0]
            this_data = this_ds['data'].sel(
                time=np.datetime64(date),
                var=this_vars,
            ).isel(
                y=y_idx, x=x_idx
            ).values
            this_data = this_data.squeeze()
            # assign NaN if outside bounds
            if (x < ds_bounds[0] or x > ds_bounds[2] or
                y < ds_bounds[1] or y > ds_bounds[3]):
                this_data = this_data + np.nan
            for v,var in enumerate(this_vars):
                nolag_data[var][i] = this_data[v]
        for var in feature_info['vars'][n_source]:
            labels = labels.copy()
            labels[var] = nolag_data[var]
    # now do the same for the dynamic datasets
    to_labels = {}
    for d_source in dynamic_sources:
        for var in feature_info['vars'][d_source]:
            for lag in days_to_include:
                var_lag_name = f"{var}_lag_{lag}d"
                to_labels[var_lag_name] = np.zeros(labels.shape[0]) + np.nan
    for d_source in dynamic_sources:
        this_ds = xr.open_zarr(
            feature_info['dirs'][d_source],
            chunks='auto'
        )
        # normalize to midnight
        this_ds['time'] = this_ds.indexes['time'].normalize()
        ds_bounds = this_ds.rio.bounds()
        # go through each of the dates that we need to open
        for date in date_to_needed:
            print('working on dynamic source:', d_source, 'for date:', date)
            # if there are no indices that need info from this date, skip it
            if len(date_to_needed[date]['indices']) == 0:
                continue
            # get the x/y and lag of labels that need info from this date
            lats = np.array([])
            lons = np.array([])
            lags = np.array([])
            for i,idx in enumerate(date_to_needed[date]['indices']):
                lat = labels['latitude'].loc[idx]
                lon = labels['longitude'].loc[idx]
                lats = np.append(lats, lat)
                lons = np.append(lons, lon)
                lag = date_to_needed[date]['offsets'][i]
                lags = np.append(lags, lag)
            # convert to x/y
            xs, ys = transformer.transform(lons, lats)
            # extract for this date/vars/xs/ys
            this_vars = feature_info['vars'][d_source]
            try:
                sub = this_ds.sel(
                    time=np.datetime64(date),
                    variable=this_vars,
                )
            except KeyError:
                time_to_try = date - pd.Timedelta(days=1)
                sub = this_ds.sel(
                    time=np.datetime64(time_to_try),
                    variable=this_vars,
                )
            x_idx = nearest_indices_unsorted(this_ds['x'].values, xs)
            y_idx = nearest_indices_unsorted(this_ds['y'].values, ys)
            all_data = sub['data'].isel(
                y=('points', y_idx), x=('points', x_idx)
            ).values
            for i,idx in enumerate(date_to_needed[date]['indices']):
                lag = lags[i]
                for v,var in enumerate(this_vars):
                    var_lag_name = f"{var}_lag_{int(lag)}d"
                    data_to_add = all_data[v][i]
                    to_labels[var_lag_name][idx] = data_to_add
    # add all of the dynamic variables to the labels dataframe
    labels = pd.concat([labels, pd.DataFrame(to_labels, index=labels.index)], axis=1)
    #labels = labels.assign(**to_labels)
    print(labels)
    # save the dataframe to a csv file
    labels.to_csv(
        out_dir,
        index=False
    )

def merge_compiled_data():
    pass
