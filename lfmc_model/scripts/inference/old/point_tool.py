import argparse
import os
import sys
import re
import torch
import numpy as np
import datetime
import pandas as pd

sys.path.append(os.path.abspath(
    os.path.join('..','..','models','transformer')
))
sys.path.append(os.path.abspath(
    os.path.join('..','..','utils')
))
sys.path.append(os.path.abspath(
    os.path.join('..','data')
))
sys.path.append(os.path.abspath(
    os.path.join('..','training')
))


from transformer_model import TimeSeriesTransformer
import compile_inference_data
import build_dataset
import utils
import train

def point_tool(
    locations_to_run,
    locations_names,
    checkpoint_fname,
    start_date,
    end_date,
    lag_days,
    feature_info,
    static_features,
    dynamic_features,
    norm_df_fname,
    out_fname,
    plot_timeseries_fname
):
    print("Compiling inference data...")
    # compile the inference data for the locations and dates specified
    infer_df = compile_inference_data.run_parallel_chunks(
        locations_to_run,
        start_date,
        end_date,
        lag_days,
        feature_info
    )
    #infer_df = compile_inference_data.compile_inference_data(
    #    locations_to_run,
    #    start_date,
    #    end_date,
    #    lag_days,
    #    feature_info
    #)
    infer_df.replace('nan', np.nan, inplace=True)
    suspicious_cols = [
        col for col in infer_df.columns
        if infer_df[col].dropna().astype(str).head(50).map(
            build_dataset.is_list_like_string).any()
    ]
    for col in suspicious_cols:
        infer_df[col] = infer_df[col].astype(str).map(build_dataset.parse_list_first)
    infer_df.to_csv('compiled_inference_data.csv', index=True)
    #infer_df = pd.read_csv(
    #    'compiled_inference_data.csv',
    #    index_col=0
    #)
    # check if Krishn'as data has nans. If so we need to raise an error
    columns_with_nans = infer_df.columns[
        infer_df.isna().any()
    ].tolist()
    using_retrieved = False
    for feat in dynamic_features:
        if 'retrieved' in feat:
            using_retrieved = True
            break
    for col in columns_with_nans:
        if 'retrieved' in col and using_retrieved:
            raise ValueError(
                f"Column {col} has NaN values, which is not allowed for inference."
            )
    # get the dates that are remaining
    #final_dates = infer_df['date'].unique()
    # keep only the columns of interest
    actual_dynamic_features = []
    for feature in dynamic_features:
        this_feature_give = feature + '_day_minus_'
        this_col = infer_df.columns[
                infer_df.columns.str.contains(
                    this_feature_give, case=False, regex=False
                )
            ]
        if len(this_col) > 0:
            for adf in this_col:
                actual_dynamic_features.append(adf)
    cols_to_use = set(static_features + actual_dynamic_features)
    cols_dont_use = set(infer_df.columns) - cols_to_use
    print("Dropping columns that are not needed for inference...")
    infer_df = infer_df.drop(
        columns=list(cols_dont_use)
    )
    # get the rows where we have any nan
    nan_rows = infer_df.isna().any(axis=1)
    # get rid of dates that have information that couldn't be gap filled
    infer_df = infer_df.dropna()
    # also drop from final_dates
    #final_dates = final_dates[~nan_rows]
    # normalize the data
    print("Normalizing data...")
    if infer_df.isna().any().any():
        # print the columns that have nans
        nan_cols = infer_df.columns[infer_df.isna().any()].tolist()
        print(f"Columns with NaN values: {nan_cols}")
        raise ValueError(
            "There are NaN values in the inference data, please check the input data."
        )
    # before we normalize, get the columns corresponding to each location
    loc_idxs = []
    for i, loc in enumerate(locations_to_run):
        # get the index where lat/lon correspond to this location
        loc_idx = infer_df[
            (infer_df['latitude'] == loc[1]) &
            (infer_df['longitude'] == loc[0])
        ].index.tolist()
        loc_idxs.append(loc_idx)
    norm_df = pd.read_csv(norm_df_fname)
    for col in infer_df.columns:
        mean = norm_df.loc[
            norm_df['feature'] == col, 'mean'
        ].values[0]
        std = norm_df.loc[
            norm_df['feature'] == col, 'std'
        ].values[0]
        infer_df[col] = (infer_df[col] - mean) / std
    # find where there is nan in infer_df
    if infer_df.isna().any().any():
        # print the columns that have nans
        nan_cols = infer_df.columns[infer_df.isna().any()].tolist()
        print(f"Columns with NaN values: {nan_cols}")
        raise ValueError(
            "There are NaN values in the inference data, please check the input data."
        )

    print("creating tensor")
    model_info_str = checkpoint_fname.split('/')[-2]
    model_info_list = model_info_str.split('_')
    model_type = model_info_list[0]
    # get a tensor for each location
    loc_tensors = []
    loc_dates = []
    for loc_idx in loc_idxs:
        # get the data for this location
        loc_df = infer_df.loc[loc_idx]
        loc_index = loc_df.index
        this_loc_dates = [
            l.split('_')[0] for l in loc_index
        ]
        loc_dates.append(this_loc_dates)
        # get the X and y tensors for this location
        infer_X, _ = build_dataset.get_X_y_from_df(
            loc_df,
            static_features,
            None,
            np.array(lag_days, dtype=np.float32),
            model_type
        )
        loc_tensors.append(infer_X)
    print("Instantiating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load up the correct model based on the string passed and send it to
    # device
    if model_type == 'transformer':
        model_params = parse_transformer_fname(model_info_str)
        this_model = TimeSeriesTransformer(
            input_dim = infer_X.shape[2],
            d_model = model_params['depth'],
            nhead = model_params['num_heads'],
            num_layers = model_params['num_layers'],
            dim_feedforward = model_params['ff_dim'],
            dropout = model_params['dropout'],
        ).to(device)
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    print("Loading model...")
    state_dict = torch.load(
        checkpoint_fname, map_location=device, weights_only=True
    )
    this_model.load_state_dict(state_dict)
    this_model.eval()
    print("Running inference...")
    #infer_out_df = pd.DataFrame({
    #    'date':list(final_dates)
    #})
    norm_df = norm_df.set_index('feature')
    with torch.no_grad():
        for i, loc_tensor in enumerate(loc_tensors):
            infer_out = this_model(loc_tensor.to(device))
            print('infer_out.shape:', infer_out.shape)
            this_loc = locations_to_run[i]
            this_loc_str = str(this_loc[0]) + '_' + str(this_loc[1]) + '_vals'
            this_loc_dates_str = str(this_loc[0]) + '_' + str(this_loc[1]) + '_dates'
            # denormalize the output
            infer_out_denorm = train.denormalize(
                infer_out,
                'lfmc',
                norm_df
            )
            if i == 0:
                infer_out_df = pd.DataFrame({
                    this_loc_str: infer_out_denorm.cpu().numpy().flatten(),
                    this_loc_dates_str: loc_dates[i]
                })
            else:
                infer_out_df[this_loc_str] = infer_out_denorm.cpu().numpy().flatten()
                infer_out_df[this_loc_dates_str] = loc_dates[i]
    print(infer_out_df)
    # save the output to a csv file
    infer_out_df.to_csv(out_fname, index=False)
    ## plot the output
    #utils.plotting.plot_multiple_timeseries_from_df(
    #    infer_out_df,
    #    'date',
    #    'date',
    #    'LFMC (%)',
    #    plot_timeseries_fname
    #)



def parse_transformer_fname(s):
    pattern = (
        r"transformer_spatial_(\d{8})_(\d{8})_y_([^_]+)_x_([^_]+)_"
        r"d(\d+)_n(\d+)_l(\d+)_df(\d+)_"
        r"dr([\d.]+)_bs(\d+)_lr([\d.]+)_ep(\d+)"
    )
    match = re.match(pattern, s)
    if not match:
        raise ValueError("String format not recognized.")
    return {
        "start_date": match.group(1),
        "end_date": match.group(2),
        "target": match.group(3),
        "input": match.group(4),
        "depth": int(match.group(5)),
        "num_heads": int(match.group(6)),
        "num_layers": int(match.group(7)),
        "ff_dim": int(match.group(8)),
        "dropout": float(match.group(9)),
        "batch_size": int(match.group(10)),
        "learning_rate": float(match.group(11)),
        "epochs": int(match.group(12)),
    }


if __name__ == "__main__":
    scratch_dir = '/scratch/users/trobinet'
    oak_dir = '/oak/stanford/groups/konings/trobinet'
    input_data_base_dir = os.path.join(
        oak_dir,
        'long_lfmc',
        'trent_datasets'
    )
    # information on the inputs that we need to compile the data
    features = {
        'dirs':{
            'daymet_prcp':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/prcp/{year:04d}/{month:02d}',
                'prcp_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'daymet_srad':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/srad/{year:04d}/{month:02d}',
                'srad_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'daymet_swe':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/swe/{year:04d}/{month:02d}',
                'swe_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'daymet_tmax':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/tmax/{year:04d}/{month:02d}',
                'tmax_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'daymet_tmin':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/tmin/{year:04d}/{month:02d}',
                'tmin_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'daymet_vpd':os.path.join(
                input_data_base_dir,
                'daymet/daymet_regrid/vp/{year:04d}/{month:02d}',
                'vp_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
            'modis':os.path.join(
                input_data_base_dir,
                'modis/modis_regridded_gapfilled/quality_1/interpolated',
                '{year:04d}/{month:02d}',
                'modis_filled_{year:04d}{month:02d}{day:02d}.nc4'
            ),
            'static':os.path.join(
                input_data_base_dir,
                'static/static_features_500m_epsg5070_float32.nc'
            ),
            'krishna_stats':os.path.join(
                input_data_base_dir,
                'krishna/stats/krishna_lfmc_statistics.nc4'
            ),
            'daymet_stats':os.path.join(
                input_data_base_dir,
                'daymet/stats/{year:04d}/{month:02d}/'
                'stats_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            ),
        },
        'vars':{
            'daymet_prcp': ['prcp'],
            'daymet_srad': ['srad'],
            'daymet_swe': ['swe'],
            'daymet_tmax': ['tmax'],
            'daymet_tmin': ['tmin'],
            'daymet_vpd': ['vp'],
            'modis': [
                'Nadir_Reflectance_Band1_filled',
                'Nadir_Reflectance_Band2_filled',
                'Nadir_Reflectance_Band3_filled',
                'Nadir_Reflectance_Band4_filled',
                'Nadir_Reflectance_Band5_filled',
                'Nadir_Reflectance_Band6_filled',
                'Nadir_Reflectance_Band7_filled',
                'filled_1',
                'filled_2',
                'filled_3',
                'filled_4',
                'filled_5',
                'filled_6',
                'filled_7'
            ],
            'static':[
                'slope',
                'elevation',
                'canopy_height',
                'forest_cover',
                'clay',
                'sand'
            ],
            'krishna_stats':[
                'retrieved_lfmc_mean',
                'retrieved_lfmc_std',
                'retrieved_lfmc_min',
                'retrieved_lfmc_max',
                'retrieved_lfmc_djf_mean',
                'retrieved_lfmc_mam_mean',
                'retrieved_lfmc_jja_mean',
                'retrieved_lfmc_son_mean'
            ],
            'daymet_stats':[
                'days_since_rain',
                'max_precip_14_days',
                'rolling_precip_14_days',
                'max_temp_14_days',
                'rolling_temp_14_days',
                'min_watervp_14_days',
                'rolling_watervp_14_days'
            ]
        },
        'type':{
            'daymet_dayl':'spatial_temporal',
            'daymet_prcp':'spatial_temporal',
            'daymet_srad':'spatial_temporal',
            'daymet_swe':'spatial_temporal',
            'daymet_tmax':'spatial_temporal',
            'daymet_tmin':'spatial_temporal',
            'daymet_vpd':'spatial_temporal',
            'modis':'spatial_temporal',
            'static':'spatial_static',
            'krishna_stats':'spatial_static',
            'daymet_stats':'spatial_temporal'
        }
    }
    # more fillable things
    # lat/lon locations to run the model
    locations_to_run = [
        [-109.74912126210066, 37.52479545326746],
        [-123.902019186624,40.70751983652574],
        [-123.95508870065137,41.31605322667633],
        [-121.81754259257885,39.68558803915657],
        [-111.94232277804926,33.46599425471452]
    ]
    # locations names for saving output and plots in a meaningful way
    locations_names = [
        'test_location_1',
        'test_location_2',
        'test_location_3',
        'test_location_4',
        'test_location_5'
    ]
    static_features = [
        'slope','elevation','canopy_height','forest_cover',
        'clay','sand','latitude','longitude'
        #'retrieved_lfmc_mean',
        #'retrieved_lfmc_std','retrieved_lfmc_min','retrieved_lfmc_max',
        #'retrieved_lfmc_djf_mean','retrieved_lfmc_mam_mean',
        #'retrieved_lfmc_jja_mean','retrieved_lfmc_son_mean'
    ]
    lagged_features = [
        'srad','prcp','swe','tmax','tmin','vp',
        'Nadir_Reflectance_Band1_filled',
        'Nadir_Reflectance_Band2_filled',
        'Nadir_Reflectance_Band3_filled',
        'Nadir_Reflectance_Band4_filled',
        'Nadir_Reflectance_Band5_filled',
        'Nadir_Reflectance_Band6_filled',
        'Nadir_Reflectance_Band7_filled'
    ]
    # what is the model type?
    model_type = 'transformer'
    # which model should we use?
    checkpoint_fname = os.path.join(
        scratch_dir,
        'long_lfmc',
        'trent_datasets',
        'lfmc_model',
        'checkpoints',
        (
            'transformer_spatial_20030101_20231231_y_Insitu_x_'
            'ModisDaymetStaticLatlon_d32_n2_l2_df64_'
            'dr0.2_bs64_lr0.0001_ep200'
        ),
        'best_model.pth'
    )
    norm_df_fname = os.path.join(
        scratch_dir,
        'long_lfmc',
        'trent_datasets',
        'lfmc_model',
        'data',
        'norm_df',
        (
            'transformer_spatial_20030101_20231231_y_Insitu_x_'
            'ModisDaymetStaticLatlon.csv'
        )
    )
    out_fname = os.path.join(
        oak_dir,
        'long_lfmc',
        'trent_datasets',
        'lfmc_model',
        'outputs',
        'predictions',
        'test_preds.csv'
    )
    plot_timeseries_fname = os.path.join(
        oak_dir,
        'long_lfmc',
        'trent_datasets',
        'lfmc_model',
        'outputs',
        'viz',
        'test_preds_plot.png'
    )
    # what are the start and end dates for the model?
    start_date = datetime.datetime(2021, 5, 1)
    end_date = datetime.datetime(2021, 12, 1)
    # how many lag days should we use?
    lag_days = [
        0,1,2,3,4,7,10,15,20,25,30
    ]
    point_tool(
        locations_to_run,
        locations_names,
        checkpoint_fname,
        start_date,
        end_date,
        lag_days,
        features,
        static_features,
        lagged_features,
        norm_df_fname,
        out_fname,
        plot_timeseries_fname
    )
