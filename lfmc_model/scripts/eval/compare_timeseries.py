import os
from compare_models_at_sites import get_site_error, site_analysis
import sys
import pandas as pd
import json
import xarray as xr
import numpy as np
import glob
import torch
import re

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..','..')
sys.path.append(os.path.join(project_root,'lfmc_model','scripts','inference'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from point_tool_new import build_tensors, run_model_forward
from transformer_multitask_longclimate import LFMCTransformer
from transformer_multitask_longclimate_uncertainty import LFMCTransformer as LFMCTransformerUncertainty
import plotting

def compare_models_at_site(
    site,
    this_model_site_error,
    model_gen_names,
    scratch_dir,
    model_dirs,
    model_tasks,
    model_types,
    input_data_names,
    comparison_type
):
        lat_float = float(site.split('_')[0])
        lon_float = float(site.split('_')[1])
        site_fmt = f"{lat_float:.5f}_{lon_float:.5f}"
        this_fold = this_model_site_error[site]['fold']
        this_folds = np.repeat(this_fold, len(model_gen_names))
        this_site_lfmc = get_site_preds(
            site,
            this_model_site_error,
            scratch_dir,
            model_gen_names,
            model_dirs,
            this_folds,
            model_tasks,
            model_types,
            input_data_names
        )
        print(this_site_lfmc)
        dates = []
        preds = []
        labels = []
        for model_name in model_gen_names:
            this_preds = this_site_lfmc[this_site_lfmc['source'] == model_name]
            this_lfmc = this_preds['vals']
            this_dates = this_preds['dates']
            preds.append(this_lfmc)
            dates.append(this_dates)
            labels.append(model_name)
        preds.append(this_site_lfmc[this_site_lfmc['source'] == 'true']['vals'])
        dates.append(this_site_lfmc[this_site_lfmc['source'] == 'true']['dates'])
        labels.append('true')
        #preds.append(this_model_site_error[site]['predictions'])
        #dates.append(this_site_lfmc[this_site_lfmc['source'] == 'true']['dates'])
        #labels.append('pred_orig')
        #linestyles = ['' if 'true' in label or 'pred_orig' in label else '-' for label in labels]
        linestyles = []
        markers = ['o' if 'true' in label or 'pred_orig' in label else '' for label in labels]
        for label in labels:
            if 'true' in label or 'pred_orig' in label:
                linestyles.append('')
            elif 'nll' in label:
                linestyles.append('--')
            else:
                linestyles.append('-')
        save_path = os.path.join(
            scratch_dir,
            'outputs',
            'model_comparisons',
            f'comparison_timeseries_{site_fmt}_{comparison_type}.png'
        )
        plotting.plot_multiple_timeseries(dates, preds, labels, linestyles, markers,save_path)

def epoch_num(p):
    m = re.search(r"model_epoch(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1

def run_point_tool(locs,start_dates,end_dates,var_names_path,norm_params_path,checkpoint_path,model_num_tasks,model_type):
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/'
    # where to save the outputs of the model
    # just take the first location for now for testing
    #site_info = site_info.iloc[:1]
    # information that we load from the model
    with open(var_names_path) as f:
        var_names = json.load(f)
    with open(norm_params_path) as f:
        norm_params = json.load(f)
    # lets lay out where the varaibles are that we are going to need to find
    # location of possible long input variables
    var_locs = {
        'daymet':[
            'prcp','srad','swe','tmax','vp'
        ],
        'modis':[
            'Nadir_Reflectance_Band1_filled',
            'Nadir_Reflectance_Band2_filled',
            'Nadir_Reflectance_Band3_filled',
            'Nadir_Reflectance_Band4_filled',
            'Nadir_Reflectance_Band5_filled',
            'Nadir_Reflectance_Band6_filled',
            'Nadir_Reflectance_Band7_filled'
        ],
        'static':[
            'slope',
            'elevation',
            'canopy_height',
            'clay',
            'sand'
        ],
        'climate_zone':[
            'climate_zone_1','climate_zone_2','climate_zone_3',
            'climate_zone_4','climate_zone_5','climate_zone_6',
            'climate_zone_7','climate_zone_8','climate_zone_9',
            'climate_zone_10','climate_zone_11','climate_zone_12',
            'climate_zone_13','climate_zone_14','climate_zone_15',
            'climate_zone_16','climate_zone_17','climate_zone_18',
            'climate_zone_19','climate_zone_20','climate_zone_21',
            'climate_zone_22','climate_zone_23','climate_zone_24',
            'climate_zone_25','climate_zone_26','climate_zone_27',
            'climate_zone_28','climate_zone_29',
        ],
        'landcover_frac':[
            'barren',
            'crops',
            'deciduous_forest',
            'developed',
            'evergreen_forest',
            'grass',
            'mixed_forest',
            'other',
            'shrub',
            'water',
            'wetlands'
        ]
    }
    print('opening datasets...')
    dss = {
        'daymet': xr.open_zarr(
            os.path.join(oak_dir, 'daymet/daymet_all_vars.zarr'),
            consolidated=False
        ),
        'modis': xr.open_zarr(
            os.path.join(
                oak_dir,
                'modis/modis_regridded_gapfilled/quality_1/interpolated/modis_all_vars.zarr'
            )
        ),
        'static': xr.open_dataset(
            os.path.join(oak_dir, 'static', 'static_features_500m_epsg5070_float32.nc')
        ),
        'climate_zone': xr.open_dataset(
            os.path.join(oak_dir, 'climate_zones', 'climate_zone_per_pixel_westUS.nc4')
        ),
        'landcover_frac': xr.open_zarr(
            os.path.join(oak_dir, 'nlcd', 'nlcd_target_grid_2003_2023.zarr')
        ),
    }
    short_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ]
    long_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,
        81,82,83,84,85,86,87,88,89,90,
        91,92,93,94,95,96,97,98,99,100,
        101,102,103,104,105,106,107,108,109,110,
        111,112,113,114,115,116,117,118,119,120,
        121,122,123,124,125,126,127,128,129,130,
        131,132,133,134,135,136,137,138,139,140,
        141,142,143,144,145,146,147,148,149,150,
        151,152,153,154,155,156,157,158,159,160,
        161,162,163,164,165,166,167,168,169,170,
        171,172,173,174,175,176,177,178,179,180,
    ]
    # locations of possible static input variables
    short_tensor, long_tensor, static_tensor, info_df = build_tensors(
        locs,
        start_dates,
        end_dates,
        var_names,
        var_locs,
        dss,
        short_lag_days,
        long_lag_days,
        norm_params,
    )
    ## large section for debuggin why we couldn't reproduce results for a long time...
    ## e.g. checking point tool tensors vs. training tensors
    #short_example_tensor = os.path.join(
    #    os.path.abspath(os.path.dirname(var_names_path)),
    #    'X_short.pt'
    #)
    #long_example_tensor = os.path.join(
    #    os.path.abspath(os.path.dirname(var_names_path)),
    #    'X_long.pt'
    #)
    #static_example_tensor = os.path.join(
    #    os.path.abspath(os.path.dirname(var_names_path)),
    #    'X_static.pt'
    #)
    #example_info = os.path.join(
    #    os.path.abspath(os.path.dirname(var_names_path)),
    #    'info.csv'
    #)
    #short_example_tensor = torch.load(short_example_tensor)
    #long_example_tensor = torch.load(long_example_tensor)
    #static_example_tensor = torch.load(static_example_tensor)
    #example_info = pd.read_csv(example_info)
    #example_info['date'] = pd.to_datetime(example_info['date']).dt.normalize().dt.tz_localize(None)
    #idx = example_info[
    #    (example_info['longitude']  == locs[0][0]) &
    #    (example_info['latitude']   == locs[0][1]) &
    #    (example_info['date']       == start_dates[0])
    #].index.tolist()[0]
    #short_example_tensor = short_example_tensor[idx,:,:]
    #long_example_tensor = long_example_tensor[idx,:,:]
    #static_example_tensor = static_example_tensor[idx,:,:]
    #our_test_short_tensor = short_tensor[0,:,:]
    #our_test_long_tensor = long_tensor[0,:,:]
    #our_test_static_tensor = static_tensor[0,:,:]
    #for v,var in enumerate(var_names['short_vars']):
    #    this_mean = norm_params['train_short_mean'][v]
    #    this_std = norm_params['train_short_std'][v]
    #    short_example_tensor[:,v] = (short_example_tensor[:,v] - this_mean) / this_std
    #for v,var in enumerate(var_names['long_vars']):
    #    this_mean = norm_params['train_long_mean'][v]
    #    this_std = norm_params['train_long_std'][v]
    #    long_example_tensor[:,v] = (long_example_tensor[:,v] - this_mean) / this_std
    #for v,var in enumerate(var_names['static_vars']):
    #    if (
    #        'barren' in var or
    #        'crops' in var or
    #        'forest' in var or
    #        'developed' in var or
    #        'grass' in var or
    #        'other' in var or
    #        'shrub' in var or
    #        'water' in var or
    #        'wetlands' in var or 
    #        'climate_zone' in var
    #    ):
    #        continue
    #    this_mean = norm_params['train_static_mean'][v]
    #    this_std = norm_params['train_static_std'][v]
    #    static_example_tensor[:,v] = (static_example_tensor[:,v] - this_mean) / this_std
    #short_model = np.asarray(short_example_tensor)
    #short_ours = np.asarray(our_test_short_tensor)
    #short_diff = np.abs(short_model - short_ours)
    #short_bad = np.where((short_diff > 1e-6) | np.isnan(short_ours))
    #long_model = np.asarray(long_example_tensor)
    #long_ours = np.asarray(our_test_long_tensor)
    #long_diff = np.abs(long_model - long_ours)
    #long_bad = np.where((long_diff > 1e-6) | np.isnan(long_ours))
    #static_model = np.asarray(static_example_tensor)
    #static_ours = np.asarray(our_test_static_tensor)
    #static_diff = np.abs(static_model - static_ours)
    #static_bad = np.where((static_diff > 1e-6) | np.isnan(static_ours))
    #print(short_bad)
    #print(long_bad)
    #print(static_bad)
    ##print(short_model)
    ##print(short_ours)
    ##print(long_model)
    ##print(long_ours)
    ##print(np.where(np.isnan(long_ours)))
    ##print(np.where(np.isnan(long_model)))
    ##print(static_model)
    ##print(static_ours)
    ##sys.exit()
    ## if these aren't empty then we have a problem
    #if short_bad[0].size > 0:
    #    raise ValueError("Short tensor mismatch")
    #if long_bad[0].size > 0:
    #    raise ValueError("Long tensor mismatch")
    #if static_bad[0].size > 0:
    #    raise ValueError("Static tensor mismatch")
    preds_df = run_model_forward(
        short_tensor,
        long_tensor,
        static_tensor,
        info_df,
        checkpoint_path,
        norm_params,
        model_task_weights=model_num_tasks,
        model_type=model_type
    )
    return preds_df

def get_site_preds(
    site,
    site_error,
    scratch_dir,
    model_names,
    model_dirs,
    model_folds,
    model_tasks,
    model_types,
    input_data_names
):
    this_lat = float(site.split('_')[0])
    this_lon = float(site.split('_')[1])
    this_site = site_error[site]
    this_trues = this_site['true_values']
    this_trues_dates = pd.to_datetime(this_site['dates'])
    this_trues_dates = this_trues_dates.tz_localize(None)
    start_date = this_trues_dates.min() - pd.DateOffset(days=60)
    end_date = this_trues_dates.max() + pd.DateOffset(days=60)
    # if longer than 3 years, just show three years
    if start_date < pd.Timestamp('2004-01-01'):
        start_date = pd.Timestamp('2004-01-01')
    if (end_date - start_date).days > 365 * 3:
        end_date = start_date + pd.DateOffset(days=365 * 3)
        trues_df = pd.DataFrame({
            'vals':this_trues,
            'dates':this_trues_dates
        })
        trues_df_trimmed = trues_df[(trues_df['dates'] >= start_date) & (trues_df['dates'] <= end_date)]
        this_trues = trues_df_trimmed['vals'].values
        this_trues_dates = trues_df_trimmed['dates'].values
    #start_date = this_trues_dates.min()
    #end_date = this_trues_dates.max()
    print(start_date)
    print(end_date)
    source = np.repeat('true', len(this_trues_dates))
    # normalize to 00:00:00
    out_df = pd.DataFrame({
        'dates':this_trues_dates,
        'vals':this_trues,
        'source':source
    })
    start_date = start_date.normalize()
    end_date = end_date.normalize()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    for m,this_model_name in enumerate(model_names):
        this_model_dir = model_dirs[m]
        this_var_names_path = os.path.join(
            scratch_dir,'inputs',input_data_names[m],'var_names.json'
        )
        this_norm_params_path = os.path.join(
            this_model_dir,
            f'fold_{model_folds[m]}',
            'norm_params.json'
        )
        this_epochs = glob.glob(
            os.path.join(this_model_dir,f'fold_{model_folds[m]}','model_epoch*.pt')
        )
        #this_epochs = sorted(this_epochs)
        this_epochs = sorted(this_epochs, key=epoch_num)
        this_checkpoint_path = this_epochs[-1]
        this_model_num_tasks = model_tasks[m]
        this_model_type = model_types[m]
        this_model_preds = run_point_tool(
            locs=[[this_lon,this_lat]],
            start_dates=[start_date],
            end_dates=[end_date],
            var_names_path=this_var_names_path,
            norm_params_path=this_norm_params_path,
            checkpoint_path=this_checkpoint_path,
            model_num_tasks=this_model_num_tasks,
            model_type=this_model_type
        )
        this_out = pd.DataFrame({
            'dates': all_dates,
            'vals': this_model_preds['lfmc_pred'],
            'source': np.repeat(this_model_name, len(all_dates))
        })
        out_df = pd.concat([out_df, this_out], ignore_index=True)
    return out_df

def main():
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
    model_gen_dirs = [
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_base',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_multitask_5_1',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_base_nll',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_multitask_5_1_nll',
    ]
    model_1_gen_name = 'base'
    model_2_gen_name = 'multitask_5_1'
    model_3_gen_name = 'base_nll'
    model_4_gen_name = 'multitask_5_1_nll'
    # settings for the best sarstats model
    dms = [32,32,32,32]
    nh = [1,1,1,1]
    nl = [2,2,2,2]
    df = [64,64,64,64]
    do = [0.15,0.15,0.15,0.15]
    bs = [128,128,128,128]
    lr = [5e-4,5e-4,5e-4,5e-4]
    warmup = [502,2458,502,2458]
    wd = [1e-4,1e-4,1e-4,1e-4]
    iobs = [30638,30638,30638,30638]
    vvobs = [0,0,0,0]
    vhobs = [0,119237,0,119237]
    dmlong = [32,64,64,32]
    nhlong = [1,2,2,1]
    nllong = [2,3,3,2]
    dflong = [64,128,128,64]
    outlong = [32,32,32,32]
    model_1_name = (
        f'transformer_dm{dms[0]}_nh{nh[0]}_nl{nl[0]}_df{df[0]}_do{do[0]}'
        f'_bs{bs[0]}_lr{lr[0]}_warmup{warmup[0]}_wd{wd[0]}'
        f'_iobs{iobs[0]}_vvobs{vvobs[0]}_vhobs{vhobs[0]}'
        f'_dmlong{dmlong[0]}_nhlong{nhlong[0]}_nllong{nllong[0]}'
        f'_dflong{dflong[0]}_outlong{outlong[0]}_basic'
    )
    model_2_name = (
        f'transformer_dm{dms[1]}_nh{nh[1]}_nl{nl[1]}_df{df[1]}_do{do[1]}'
        f'_bs{bs[1]}_lr{lr[1]}_warmup{warmup[1]}_wd{wd[1]}'
        f'_iobs{iobs[1]}_vvobs{vvobs[1]}_vhobs{vhobs[1]}'
        f'_dmlong{dmlong[1]}_nhlong{nhlong[1]}_nllong{nllong[1]}'
        f'_dflong{dflong[1]}_outlong{outlong[1]}_basic'
    )
    model_3_name = (
        f'transformer_dm{dms[2]}_nh{nh[2]}_nl{nl[2]}_df{df[2]}_do{do[2]}'
        f'_bs{bs[2]}_lr{lr[2]}_warmup{warmup[2]}_wd{wd[2]}'
        f'_iobs{iobs[2]}_vvobs{vvobs[2]}_vhobs{vhobs[2]}'
        f'_dmlong{dmlong[2]}_nhlong{nhlong[2]}_nllong{nllong[2]}'
        f'_dflong{dflong[2]}_outlong{outlong[2]}_basic'
    )
    model_4_name = (
        f'transformer_dm{dms[3]}_nh{nh[3]}_nl{nl[3]}_df{df[3]}_do{do[3]}'
        f'_bs{bs[3]}_lr{lr[3]}_warmup{warmup[3]}_wd{wd[3]}'
        f'_iobs{iobs[3]}_vvobs{vvobs[3]}_vhobs{vhobs[3]}'
        f'_dmlong{dmlong[3]}_nhlong{nhlong[3]}_nllong{nllong[3]}'
        f'_dflong{dflong[3]}_outlong{outlong[3]}_basic'
    )
    model_1_dir = os.path.join(model_gen_dirs[0], model_1_name)
    model_2_dir = os.path.join(model_gen_dirs[1], model_2_name)
    model_3_dir = os.path.join(model_gen_dirs[2], model_3_name)
    model_4_dir = os.path.join(model_gen_dirs[3], model_4_name)
    model_tasks = [1,2,1,2]
    model_types = ['standard','standard','uncertainty','uncertainty']
    #final_epochs = [
    #    'model_epoch5.pt',
    #    'model_epoch4.pt',
    #    'model_epoch5.pt',
    #    'model_epoch8.pt'
    #]
    input_data_names = ['news1_base','news1_multitask','news1_base','news1_multitask']
    model_gen_names = [model_1_gen_name, model_2_gen_name, model_3_gen_name, model_4_gen_name]
    model_dirs = [model_1_dir, model_2_dir, model_3_dir, model_4_dir]
    model_1_site_error = get_site_error(model_1_dir)
    model_2_site_error = get_site_error(model_2_dir)
    model_3_site_error = get_site_error(model_3_dir)
    model_4_site_error = get_site_error(model_4_dir)
    model_1_2_comparison_df = site_analysis(
        model_1_site_error,
        model_2_site_error,
        model_1_gen_name,
        model_2_gen_name,
        make_plots = False
    )
    model_3_4_comparison_df = site_analysis(
        model_3_site_error,
        model_4_site_error,
        model_3_gen_name,
        model_4_gen_name,
        make_plots = False
    )
    # get the 5 sites where base is far better than multitask
    model_1_2_comparison_df = model_1_2_comparison_df[model_1_2_comparison_df['num_measurements'] > 10]
    rmse_diffs_1_2 = model_1_2_comparison_df['rmse_diff']
    best_sites_1_df = rmse_diffs_1_2.nlargest(5)
    best_sites_1 = rmse_diffs_1_2.nlargest(5).index.tolist()
    print('best sites for model 1:')
    print(best_sites_1_df)
    for s,site in enumerate(best_sites_1):
        compare_models_at_site(
            site,
            model_1_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'base_best'
        )
    # get the 5 sites where multitask is far better than base
    best_sites_2 = rmse_diffs_1_2.nsmallest(5).index.tolist()
    for s,site in enumerate(best_sites_2):
        compare_models_at_site(
            site,
            model_2_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'multitask_best'
        )
    # get 5 sites where base and multitask are similar
    similar_sites_1_2 = rmse_diffs_1_2.abs().nsmallest(5).index.tolist()
    for s,site in enumerate(similar_sites_1_2):
        compare_models_at_site(
            site,
            model_1_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'base_multitask_similar'
        )
    # get 5 sites where base_nll is far better than multitask_nll
    model_3_4_comparison_df = model_3_4_comparison_df[model_3_4_comparison_df['num_measurements'] > 10]
    rmse_diffs_3_4 = model_3_4_comparison_df['rmse_diff']
    best_sites_3 = rmse_diffs_3_4.nlargest(5).index.tolist()
    for s,site in enumerate(best_sites_3):
        compare_models_at_site(
            site,
            model_3_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'base_nll_best'
        )
    # get 5 sites where multitask_nll is far better than base_nll
    best_sites_4 = rmse_diffs_3_4.nsmallest(5).index.tolist()
    for s,site in enumerate(best_sites_4):
        compare_models_at_site(
            site,
            model_4_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'multitask_nll_best'
        )
    # get 5 sites where multitask_nll and base_nll are similar
    similar_sites_3_4 = rmse_diffs_3_4.abs().nsmallest(5).index.tolist()
    for s,site in enumerate(similar_sites_3_4):
        compare_models_at_site(
            site,
            model_3_site_error,
            model_gen_names,
            scratch_dir,
            model_dirs,
            model_tasks,
            model_types,
            input_data_names,
            'base_nll_multitask_nll_similar'
        )

if __name__ == "__main__":
    main()