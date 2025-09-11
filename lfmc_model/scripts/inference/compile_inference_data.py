import pandas as pd
import sys
from pyproj import Transformer
import xarray as xr
import datetime
import multiprocessing as mp
from datetime import timedelta
import numpy as np

def compile_inference_data(
    points,
    start_date,
    end_date,
    lag_days,
    feature_info
):
    # get a list of the dates that we need to sort through. It needs to be
    # daily end date through start date while going back before the start date
    # enough to get all lag days
    dates = pd.date_range(
        end=pd.to_datetime(end_date),
        start=pd.to_datetime(start_date) - pd.Timedelta(days=max(lag_days)),
        freq='D'
    )
    # sort dates from last to first
    dates = dates[::-1]
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
    # store all info in a dictionary that gets turned into a pandas DataFrame
    compiled_data = {}
    # going to need to transform between the lat/lon given and the x and y for
    # our model. make the transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    # loop over dates. go in reverse order
    percent_done_counter = 0
    for d,day in enumerate(dates):
        percent_done = (d + 1) / len(dates) * 100
        if percent_done > percent_done_counter:
            print('percent done:', round(percent_done, 2), '%')
            percent_done_counter += 10
        todays_data_needs = []
        # get a lat/lon/date listing for this date for each point
        for l,(lon,lat) in enumerate(points):
            todays_name = f"{day.strftime('%Y-%m-%d')}_{lat}_{lon}"
            compiled_data[todays_name] = {}
            # add to the tracker of the data that we need to extract
            todays_data_needs.append(todays_name)
            # the x,y so do that now
            x,y = transformer.transform(lon,lat)
            # go ahead and add the static data for this point. going to need
            # but only if within our dates of interest
            if day >= start_date:
                for s_source in static_sources:
                    this_ds = xr.open_dataset(
                        feature_info['dirs'][s_source],
                        engine='netcdf4'
                    )
                    ds_bounds = this_ds.rio.bounds()
                    # get the variable names that we need here
                    this_vars = feature_info['vars'][s_source]
                    # loop over each variable
                    for v,var in enumerate(this_vars):
                        # raise an error if the requested point is outside of the
                        # bounds of our grid
                        if (x < ds_bounds[0] or x > ds_bounds[2] or
                            y < ds_bounds[1] or y > ds_bounds[3]):
                            raise ValueError(
                                f"Point {lat},{lon} is outside of the bounds "
                                f"of the static source {s_source}."
                            )
                        this_data = this_ds[var].sel(
                            x=x, y=y, method='nearest'
                        ).values
                        if np.ndim(this_data) == 0:
                            this_data = this_data.item()
                        else:
                            this_data = this_data.flat[0]
                        # we need to figure out why we are getting so many nans
                        # here. throw and error and print stuff if this has nans
                        compiled_data[todays_name][var] = this_data
                    this_ds.close()
                compiled_data[todays_name]['latitude'] = lat
                compiled_data[todays_name]['longitude'] = lon
                compiled_data[todays_name]['date'] = day.strftime('%Y-%m-%d')
        # get the names of the dates for the previous days that also need data
        # from today to fulfill their lag requirements
        for o,off in enumerate(lag_days):
            this_offset_date = day + pd.Timedelta(days=off)
            if (this_offset_date <= end_date):
                this_offset_date_str = this_offset_date.strftime('%Y-%m-%d')
                for l,(lon,lat) in enumerate(points):
                    todays_name = f"{this_offset_date_str}_{lat}_{lon}"
                    if todays_name not in todays_data_needs:
                        todays_data_needs.append(todays_name)
        # loop over each dynamic source to add to the relevant needy days
        for d_source in dynamic_sources:
            this_fname = feature_info['dirs'][d_source]
            this_full_dir = this_fname.format(
                year=day.year,
                month=day.month,
                day=day.day
            )
            this_ds = xr.open_dataset(
                this_full_dir,
                engine='netcdf4'
            )
            this_vars = feature_info['vars'][d_source]
            # loop over each variable that we need for that file
            for v,var in enumerate(this_vars):
                # loop over our lat/lons
                for l,(lon,lat) in enumerate(points):
                    this_data = this_ds[var].sel(
                        x=x, y=y, method='nearest'
                    ).values
                    # add this to the relevant locations in the dictionary
                    target_pair = f"{lat}_{lon}"
                    matches = [
                        name for name in todays_data_needs if name.endswith(target_pair)
                    ]
                    for m,ma in enumerate(matches):
                        match_date,_,_ = ma.split('_')
                        match_date_dtm = datetime.datetime.strptime(
                            match_date, '%Y-%m-%d'
                        )
                        # how many days from our current date?
                        days_from_today = (match_date_dtm - day).days
                        this_key_name = '{var}_day_minus_{days_from_today}'.format(
                            var=var,
                            days_from_today=days_from_today
                        )
                        # add to the dictionary
                        if match_date_dtm >= start_date:
                            compiled_data[ma][this_key_name] = this_data
            this_ds.close()
    # convert the dictionary to a pandas DataFrame
    compiled_df = pd.DataFrame.from_dict(compiled_data, orient='index')
    return compiled_df

def run_parallel_chunks(
    points, start_date, end_date, lag_days,
    feature_info, max_processes=None,min_chunk_days=30
):
    available_cores = mp.cpu_count()
    if max_processes is not None:
        available_cores = min(max_processes, available_cores)
    total_days = (end_date - start_date).days + 1
    optimal_chunk_days = max(
        total_days // available_cores, min_chunk_days
    )
    print('Running on', available_cores, 'processes')
    print('Optimal chunk size:', optimal_chunk_days, 'days')
    # Generate date chunks
    chunks = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=optimal_chunk_days - 1), end_date)
        chunks.append((chunk_start, chunk_end, points, lag_days, feature_info))
        chunk_start = chunk_end + timedelta(days=1)

    # Run chunks in parallel
    with mp.Pool(processes=available_cores) as pool:
        results = pool.map(run_single_chunk, chunks)

    # Concatenate and return
    final_df = pd.concat(results).sort_values(by='date')
    # sort by the index
    final_df = final_df.sort_index()
    print(final_df)
    return final_df

def run_single_chunk(args):
    chunk_start, chunk_end, points, lag_days, feature_info = args
    return compile_inference_data(points, chunk_start, chunk_end, lag_days, feature_info)













