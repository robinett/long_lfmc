
import os
import json
import sys
import pandas as pd
import torch
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
from pyproj import transformer

here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(here, '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

import plotting

nlcd_dict = {
    11:'water',
    12:'water',
    21:'developed',
    22:'developed',
    23:'developed',
    24:'developed',
    31:'barren',
    41:'deciduous_forest',
    42:'evergreen_forest',
    43:'mixed_forest',
    52:'shrub',
    71:'grass',
    81:'crops',
    82:'crops',
    90:'wetlands',
    95:'wetlands'
}

def get_landcover_df(
    model_dir,
    land_cover_single_ds,
    land_cover_frac_ds,
    nfmd_df,
):
    Transformer = transformer.Transformer.from_crs(
        'epsg:4326',
        'epsg:5070',
        always_xy=True
    )
    with open(os.path.join(model_dir,'fold_info.json')) as f:
        fold_info = json.load(f)
    folds = list(fold_info.keys())
    # we want a dataframe that is:
    #     each prediction
    #     each obs
    #     the measured land cover
    #     the dominant land cover from satellite perspective
    #        hetero if no land cover > 75%
    #     #the fraction of the dominant land cover
    landcover_errors = {}
    obs_num = 0
    # get all the landcover types that we have fractions for
    land_covers = list(land_cover_frac_ds.data_vars)
    for f,fold in enumerate(folds):
        print(f'Evaluating fold {f+1}/{len(folds)}')
        test_info_path = os.path.join(model_dir, f'fold_{fold}', 'test_info.csv')
        test_info = pd.read_csv(test_info_path)
        test_data_path = os.path.join(model_dir, f'fold_{fold}', 'test_outputs.pth')
        test_data = torch.load(test_data_path, weights_only=False)
        test_preds_insitu = test_data['lfmc_preds']
        #test_preds_insitu_std = test_data['lfmc_std']
        #test_preds_vv = test_data['vv_preds']
        #test_preds_vv_std = test_data['vv_std']
        #test_preds_vh = test_data['vh_preds']
        #test_preds_vh_std = test_data['vh_std']
        test_true_insitu = test_data['lfmc_true']
        #test_true_vv = test_data['vv_true']
        #test_true_vh = test_data['vh_true']
        # pre-store the satellite info for each site to speed up processing
        # get all the insitu preds to compare
        test_info = test_info[test_info['source']=='nfmd'].reset_index(drop=True)
        # get unique lat/lon combinations
        # just take the first 50 of everything for testing
        test_info = test_info.iloc[0:50]
        test_true_insitu = test_true_insitu[0:50]
        test_preds_insitu = test_preds_insitu[0:50]
        
        unique_sites = test_info[['latitude','longitude']].drop_duplicates().values
        site_lc = {}
        for s,site in enumerate(unique_sites):
            print(f'Pre-loading land cover for site {s+1}/{len(unique_sites)}')
            lat = site[0]
            lon = site[1]
            site_x, site_y = Transformer.transform(
                lon,
                lat
            )
            this_site_lcs = land_cover_single_ds.sel(
                x=site_x,
                y=site_y,
                method='nearest'
            ).load()
            site_lc[f'{lat}_{lon}'] = this_site_lcs
        # loop over each observation to get the relevant information
        for r,row in test_info.iterrows():
            # match to an observation from nfmc
            this_lat = row['latitude']
            this_lon = row['longitude']
            this_date = pd.to_datetime(row['date'])
            nfmd_mask = (
                (nfmd_df['latitude']==this_lat) &
                (nfmd_df['longitude']==this_lon) &
                (pd.to_datetime(nfmd_df['date'])==this_date)
            )
            this_nfmd_obs = nfmd_df[nfmd_mask]
            print(this_nfmd_obs['latitude'].values)
            print(this_nfmd_obs['longitude'].values)
            print(this_nfmd_obs['date'].values)
            print(this_nfmd_obs['lfmc'].values)
            print(this_nfmd_obs['site_id'].values)
            print(this_nfmd_obs['site_name'].values)
            sys.exit() 
        
        
        sys.exit()
        # get the error at each site
        for s,site in enumerate(unique_sites):
            print(f'Analyzing site {s+1}/{len(unique_sites)}')
            lat = site[0]
            lon = site[1]
            site_mask = (
                (test_info_insitu['latitude']==lat) &
                (test_info_insitu['longitude']==lon)
            )
            site_indices = test_info_insitu[site_mask].index.values
            site_dates = pd.to_datetime(test_info_insitu[site_mask]['date'])
            site_dates = site_dates.dt.year.astype(str) + '-01-01'
            site_dates = pd.to_datetime(site_dates)
            site_true = test_true_insitu[site_indices]
            site_preds = test_preds_insitu[site_indices]
            site_num_measurements = len(site_true)
            site_x, site_y = Transformer.transform(
                lon,
                lat
            )
            this_site_lcs = land_cover_single_ds.sel(
                x=site_x,
                y=site_y,
                method='nearest'
            )
            this_site_lc_info = {}
            for d,date in enumerate(site_dates):
                if date.year in this_site_lc_info:
                    this_class_name = this_site_lc_info[date.year]['class_name']
                    this_frac_today_vals = this_site_lc_info[date.year]['frac_vals']
                else:
                    this_class = int(this_site_lcs['nlcd'].sel(time=date).values)
                    this_class_name = nlcd_dict.get(this_class, 'unknown')
                    this_site_lc_info[date.year] = {}
                    this_site_lc_info[date.year]['class_name'] = this_class_name
                    this_frac = land_cover_frac_ds.sel(
                        x=site_x,
                        y=site_y,
                        method='nearest'
                    )
                    this_frac_today = this_frac.sel(year=date)
                    this_frac_today_vals = this_frac_today.to_array().values
                    this_site_lc_info[date.year]['frac_vals'] = this_frac_today_vals
                this_pred = site_preds[d]
                this_true = site_true[d]
                # get the maximum
                max_frac_idx = np.argmax(this_frac_today_vals)
                max_frac_lc = land_covers[max_frac_idx]
                max_frac_percent = this_frac_today_vals[max_frac_idx]
                # is the max the same land cover as @ the location?
                same_class = (max_frac_lc == this_class_name)
                if this_class_name not in landcover_errors:
                    landcover_errors[this_class_name] = {
                        'true_values': [],
                        'predictions': [],
                        'num_measurements': 0,
                        'site_classes': [],
                        'max_fraction_landcovers': [],
                        'max_fraction_percents': [],
                        'same_class_flags': []
                    }
                landcover_errors[this_class_name]['true_values'].append(this_true)
                landcover_errors[this_class_name]['predictions'].append(this_pred)
                landcover_errors[this_class_name]['num_measurements'] += 1
                landcover_errors[this_class_name]['site_classes'].append(this_class_name)
                landcover_errors[this_class_name]['max_fraction_landcovers'].append(max_frac_lc)
                landcover_errors[this_class_name]['max_fraction_percents'].append(max_frac_percent)
                landcover_errors[this_class_name]['same_class_flags'].append(same_class)
    # 





def get_site_error(
    model_dir,
    site_errors=None
):
    with open(os.path.join(model_dir,'fold_info.json')) as f:
        fold_info = json.load(f)
    folds = list(fold_info.keys())
    for f,fold in enumerate(folds):
        print(f'Evaluating fold {f+1}/{len(folds)}')
        test_info_path = os.path.join(model_dir, f'fold_{fold}', 'test_info.csv')
        test_info = pd.read_csv(test_info_path)
        test_data_path = os.path.join(model_dir, f'fold_{fold}', 'test_outputs.pth')
        test_data = torch.load(test_data_path, weights_only=False)
        test_preds_insitu = test_data['lfmc_preds']
        test_preds_insitu_std = test_data['lfmc_std']
        test_preds_vv = test_data['vv_preds']
        test_preds_vv_std = test_data['vv_std']
        test_preds_vh = test_data['vh_preds']
        test_preds_vh_std = test_data['vh_std']
        test_true_insitu = test_data['lfmc_true']
        test_true_vv = test_data['vv_true']
        test_true_vh = test_data['vh_true']
        # get all the insitu preds to compare
        test_info_insitu = test_info[test_info['source']=='nfmd'].reset_index(drop=True)
        # get unique lat/lon combinations
        unique_sites = test_info_insitu[['latitude','longitude']].drop_duplicates().values
        # get the error at each site
        if site_errors is None:
            site_errors = {}
        for s,site in enumerate(unique_sites):
            lat = site[0]
            lon = site[1]
            site_mask = (
                (test_info_insitu['latitude']==lat) &
                (test_info_insitu['longitude']==lon)
            )
            site_indices = test_info_insitu[site_mask].index.values
            site_true = test_true_insitu[site_indices]
            site_preds = test_preds_insitu[site_indices]
            site_num_measurements = len(site_true)
            site_rmse = np.sqrt(
                np.mean(
                    (site_true - site_preds)**2
                )
            )
            if site_num_measurements < 2:
                site_r2 = float('nan')
            else:
                site_r2 = r2_score(
                    site_true,
                    site_preds
                )
            site_key = f'{lat}_{lon}'
            site_errors[site_key] = {
                'num_measurements': site_num_measurements,
                'true_values': site_true.tolist(),
                'predictions': site_preds.tolist(),
                'rmse': site_rmse,
                'r2': site_r2
            }
    return site_errors

def site_analysis(
    model_1_site_error,
    model_2_site_error,
    model_1_gen_name,
    model_2_gen_name,
    make_plots=True
):
    Transformer = transformer.Transformer.from_crs(
        'epsg:4326',
        'epsg:5070',
        always_xy=True
    )
    # get per-site differences in rmse and r2
    site_keys = list(model_1_site_error.keys())
    comparison_dict = {}
    for site_key in site_keys:
        site_lat = site_key.split('_')[0]
        site_lon = site_key.split('_')[1]
        site_x, site_y = Transformer.transform(
            float(site_lon),
            float(site_lat)
        )
        model_1_rmse = model_1_site_error[site_key]['rmse']
        model_1_r2 = model_1_site_error[site_key]['r2']
        model_2_rmse = model_2_site_error[site_key]['rmse']
        model_2_r2 = model_2_site_error[site_key]['r2']
        comparison_dict[site_key] = {
            'latitude': site_lat,
            'longitude': site_lon,
            'model_1_rmse': model_1_rmse,
            'model_1_r2': model_1_r2,
            'model_2_rmse': model_2_rmse,
            'model_2_r2': model_2_r2,
            'rmse_diff': model_2_rmse - model_1_rmse,
            'r2_diff': model_2_r2 - model_1_r2,
            'num_measurements': model_1_site_error[site_key]['num_measurements']
        }
    # turn into DataFrame for easier analysis
    comparison_df = pd.DataFrame.from_dict(comparison_dict, orient='index')
    print(comparison_df)
    if not make_plots:
        return comparison_df
    # correlation between r2 and rmse for each model by site
    plotting.generic_scatter(
        comparison_df.model_1_rmse.values,
        comparison_df.model_2_rmse.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_{model_2_gen_name}_vs_{model_1_gen_name}.png'
        ),
        xlabel=f'Model 1 ({model_1_gen_name}) RMSE',
        ylabel=f'Model 2 ({model_2_gen_name}) RMSE'
    )
    plotting.generic_scatter(
        comparison_df.model_1_r2.values,
        comparison_df.model_2_r2.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'r2_{model_2_gen_name}_vs_{model_1_gen_name}.png'
        ),
        xlabel=f'Model 1 ({model_1_gen_name}) R2',
        ylabel=f'Model 2 ({model_2_gen_name}) R2',
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.1)
    )
    
    # test correlation between num measurements and performance
    plotting.generic_scatter(
        comparison_df.num_measurements.values,
        comparison_df.model_1_rmse.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'num_measurements_vs_rmse_{model_1_gen_name}.png'
        ),
        xlabel='Number of Measurements',
        ylabel='RMSE'
    )
    # test correlation between rmse/r2 and differences in model performance
    plotting.generic_scatter(
        comparison_df.model_1_rmse.values,
        np.abs(comparison_df.rmse_diff.values),
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_{model_2_gen_name}_minus_{model_1_gen_name}.png'
        ),
        xlabel=f'Model 1 ({model_1_gen_name}) RMSE',
        ylabel=f'Model 2 minus Model 1 RMSE Difference',
        xlim=(0.0,75.0)
    )
    plotting.generic_scatter(
        #np.clip(comparison_df.model_1_r2.values, 0.0, 1.0),
        #np.clip(np.abs(comparison_df.r2_diff.values), 0.0, 1.0),
        comparison_df.model_1_r2.values,
        np.abs(comparison_df.r2_diff.values),
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'r2_{model_2_gen_name}_minus_{model_1_gen_name}.png'
        ),
        xlabel=f'Model 1 ({model_1_gen_name}) R2',
        ylabel=f'Model 2 minus Model 1 R2 Difference',
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.1)
    )
    
    # make spatial plots of the differences
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_map_{model_1_gen_name}.png'
        ),
        colors=comparison_df.model_1_rmse.values,
        s_min=5,
        s_max=50,
        cbar_lim=(0,50),
        cmap='winter'
    )
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'r2_map_{model_1_gen_name}.png'
        ),
        colors=comparison_df.model_1_r2.values,
        s_min=5,
        s_max=50,
        cbar_lim=(0,1),
        cmap='winter'
    )
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_map_{model_2_gen_name}.png'
        ),
        colors=comparison_df.model_2_rmse.values,
        s_min=5,
        s_max=50,
        cbar_lim=(0,50),
        cmap='winter'
    )
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'r2_map_{model_2_gen_name}.png'
        ),
        colors=comparison_df.model_2_r2.values,
        s_min=5,
        s_max=50,
        cbar_lim=(0,1),
        cmap='winter'
    )
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_map_{model_2_gen_name}_minus_{model_1_gen_name}.png'
        ),
        colors=comparison_df.rmse_diff.values,
        s_min=5,
        s_max=50,
        cbar_lim=(-10,10),
        cmap='PiYG'
    )
    plotting.map_points(
        comparison_df.longitude.values,
        comparison_df.latitude.values,
        comparison_df.num_measurements.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'r2_map_{model_2_gen_name}_minus_{model_1_gen_name}.png'
        ),
        colors=comparison_df.r2_diff.values,
        s_min=5,
        s_max=50,
        cbar_lim=(-0.5,0.5),
        cmap='PiYG'
    )
    return comparison_df

def landcover_analysis():
    pass

def main():
    # analysis settings
    analyze_at_sites = False
    analyze_by_landcover = True
    # model settings
    model_gen_dirs = [
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/base',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/sarstats_onlyandminimal',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/sarmultitask_vhonly_gradnorm',
    ]
    model_1_gen_name = 'base'
    model_2_gen_name = 'sarstats_onlyandminimal'
    model_3_gen_name = 'sarmultitask_vhonly_gradnorm'
    # settings for the best sarstats model
    dms = [64,64,32]
    nh = [2,2,1]
    nl = [3,3,2]
    df = [128,128,64]
    do = [0.15,0.15,0.15]
    bs = [128,128,128]
    lr = [1e-4,1e-4,5e-4]
    warmup = [554,554,1227]
    wd = [1e-4,1e-4,1e-4]
    iobs = [33806,33806,33806]
    vvobs = [0,0,0]
    vhobs = [0,0,41024]
    dmlong = [128,256,128]
    nhlong = [4,8,4]
    nllong = [4,5,4]
    dflong = [256,512,256]
    outlong = [32,64,32]
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
    model_1_dir = os.path.join(model_gen_dirs[0], model_1_name)
    model_2_dir = os.path.join(model_gen_dirs[1], model_2_name)
    model_3_dir = os.path.join(model_gen_dirs[2], model_3_name)
    # get the information for model 1
    if analyze_at_sites:
        model_1_site_error = get_site_error(model_1_dir)
        model_2_site_error = get_site_error(model_2_dir)
        model_3_site_error = get_site_error(model_3_dir)
        model_1_2_comparision_df = site_analysis(
            model_1_site_error,
            model_2_site_error,
            model_1_gen_name,
            model_2_gen_name,
            make_plots=False
        )
        model_1_3_comparision_df = site_analysis(
            model_1_site_error,
            model_3_site_error,
            model_1_gen_name,
            model_3_gen_name,
            make_plots=False
        )
        # other things where we don't want to run a full analysis...
        plotting.generic_scatter(
            model_1_2_comparision_df.rmse_diff.values,
            model_1_3_comparision_df.rmse_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'rmse_diff_from_base_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} RMSE Difference from Base Model',
            ylabel=f'{model_3_gen_name} RMSE Difference from Base Model'
        )
        plotting.generic_scatter(
            model_1_2_comparision_df.r2_diff.values,
            model_1_3_comparision_df.r2_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'r2_diff_from_base_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} R2 Difference from Base Model',
            ylabel=f'{model_3_gen_name} R2 Difference from Base Model',
            xlim=(-1.1,1.1),
            ylim=(-1.1,1.1),
            corrclip=[-1.0,1.0]
        )
    if analyze_by_landcover:
        # load up the land cover dataset so that we can get land cover at site
        land_cover_single_ds = xr.open_zarr(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/nlcd/nlcd_2003_2023.zarr'
        )
        land_cover_breakdown_ds = xr.open_zarr(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/nlcd/nlcd_target_grid_2003_2023.zarr'
        )
        nfmd_df = pd.read_csv(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/nfmd/nfmd_processed.csv'
        )
        model_1_landcover_df = get_landcover_df(
            model_1_dir,
            land_cover_single_ds,
            land_cover_breakdown_ds,
            nfmd_df
        )
        model_2_landcover_df = get_landcover_df(
            model_2_dir,
            land_cover_single_ds,
            land_cover_breakdown_ds,
            nfmd_df
        )
        model_3_landcover_df = get_landcover_df(
            model_3_dir,
            land_cover_single_ds,
            land_cover_breakdown_ds,
            nfmd_df
        )


if __name__ == "__main__":
    main()