
import os
import json
import sys
import time
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

def analyze_landcover_of_sites(
    sites,
    landcover_df,
    location_group_name
):
    # plot hte landcover distribution of these sites
    landcover_dist_dict = {}
    for key,name in nlcd_dict.items():
        landcover_dist_dict[name] = 0.0
    for r,row in sites.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        lat  = round(lat,5)
        lon = round(lon,5)
        # trim to 6 decimal places to avoid floating point issues
        lat = round(lat,6)
        site_mask = (
            (round(landcover_df['lat'],5)==lat) &
            (round(landcover_df['lon'],5)==lon)
        )
        if r == 0:
            site_lc_df = landcover_df[site_mask].reset_index(drop=True)
        else:
            site_lc_df = pd.concat(
                [site_lc_df, landcover_df[site_mask].reset_index(drop=True)],
                ignore_index=True
            )
    # map of these sites
    # get the unique site locations
    unique_sites = sites[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
    plotting.map_points(
        unique_sites['longitude'].values,
        unique_sites['latitude'].values,
        np.repeat(1.0, len(unique_sites)),
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'site_locations_{location_group_name}.png'
        ),
    )
    # mean lfmc of these sites
    mean_lfmc_all = landcover_df['obs'].mean()
    mean_lfmc_here = site_lc_df['obs'].mean()
    print('For location group:', location_group_name)
    print(f'Mean LFMC at all sites: {mean_lfmc_all}')
    print(f'Mean LFMC at {location_group_name} sites: {mean_lfmc_here}')
    # std of lfmc at these sites
    std_lfmc_all = landcover_df['obs'].std()
    std_lfmc_here = site_lc_df['obs'].std()
    print(f'Std LFMC at all sites: {std_lfmc_all}')
    print(f'Std LFMC at {location_group_name} sites: {std_lfmc_here}')
    # number of measurements
    num_measurements_all = len(landcover_df)
    num_sites_all = len(landcover_df[['lat','lon']].drop_duplicates())
    avg_measurements_all = num_measurements_all / num_sites_all
    num_measurements_here = len(site_lc_df)
    num_sites_here = len(site_lc_df[['lat','lon']].drop_duplicates())
    avg_measurements_here = num_measurements_here / num_sites_here
    print(f'Num measurements per site at all sites: {avg_measurements_all}')
    print(f'Num measurements per site at {location_group_name} sites: {avg_measurements_here}')
    # dominant satellite landcover of these sites
    sat_lc_counts = site_lc_df['satellite_landcover'].value_counts(normalize=True)
    plotting.bar_plot(
        sat_lc_counts.index,
        sat_lc_counts.values,
        xlabel='Landcover Type',
        ylabel='Proportion',
        save_path=os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'dominant_satellite_landcover_{location_group_name}.png'
        ),
    )
    # measured landcover of these sites
    measured_lc_counts = site_lc_df['measured_landcover'].value_counts(normalize=True)
    plotting.bar_plot(
        measured_lc_counts.index,
        measured_lc_counts.values,
        xlabel='Landcover Type',
        ylabel='Proportion',
        save_path=os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'measured_landcover_{location_group_name}.png'
        ),
    )
    # satellite heterogenieity of these sites
    # kde of max landcover fraction satellite
    plotting.kde_plot(
        [site_lc_df['highest_lc_perc'].values],
        'Max Satellite Land Cover Fraction',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'max_satellite_land_cover_fraction_{location_group_name}.png'
        ),
        ylimit=[0,4.0],
    )
    # representativeness of sampled land cover at these sites
    # kde of measured landcover fraction at these sites
    plotting.kde_plot(
        [site_lc_df['measured_lc_perc'].values],
        'Measured Land Cover Fraction',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'measured_land_cover_fraction_{location_group_name}.png'
        ),
        ylimit=[0,3.0],
    )

def get_landcover_df(
    model_dir,
    land_cover_single_ds,
    land_cover_frac_ds,
    nfmd_df,
    site_lc_df_path
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
    #        hetero if no land cover > 66%
    #     #the fraction of the dominant land cover
    landcover_errors = {}
    # get all the landcover types that we have fractions for
    land_covers = list(land_cover_frac_ds.data_vars)
    site_lc_dict = {}
    final_vars = [
        'lat',
        'lon',
        'date',
        'obs',
        'pred',
        'measured_landcover',
        'satellite_landcover',
        'highest_lc_perc',
        'measured_lc_perc'
    ]
    final_var_types = [
        float,
        float,
        str,
        float,
        float,
        str,
        str,
        float,
        float
    ]
    # get the number of observations that we are going to have so that we can pre-allocate
    num_obs = 0
    for f,fold in enumerate(folds):
        print(f'Counting observations in fold {f+1}/{len(folds)}')
        test_info_path = os.path.join(model_dir, f'fold_{fold}', 'test_info.csv')
        test_info = pd.read_csv(test_info_path)
        test_info = test_info[test_info['source']=='nfmd'].reset_index(drop=True)
        num_obs += len(test_info)
    for var in final_vars:
        site_lc_dict[var] = [np.nan for _ in range(num_obs)]
    num_obs_filled = 0
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
        #test_info = test_info.iloc[0:50]
        #test_true_insitu = test_true_insitu[0:50]
        #test_preds_insitu = test_preds_insitu[0:50]
        unique_sites = test_info[['latitude','longitude']].drop_duplicates().values
        site_lc_preload = {}
        for s,site in enumerate(unique_sites):
            print(f'Pre-loading land cover for site {s+1}/{len(unique_sites)}')
            lat = site[0]
            lon = site[1]
            site_x, site_y = Transformer.transform(
                lon,
                lat
            )
            this_site_lcs = land_cover_frac_ds.sel(
                x=site_x,
                y=site_y,
                method='nearest'
            ).load()
            site_lc_preload[f'{lat}_{lon}'] = this_site_lcs
        # loop over each observation to get the relevant information
        for r,row in test_info.iterrows():
            if r % 100 == 0:
                print(f'Processing observation {r+1}/{len(test_info)}')
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
            # make sure that we only have one observation
            if len(this_nfmd_obs) != 1:
                print('Warning: multiple or no NFMD observations found for site/date!')
                print(f'Lat: {this_lat}, Lon: {this_lon}, Date: {this_date}, Num Obs: {len(this_nfmd_obs)}')
                continue
            this_obs = test_true_insitu[r]
            this_pred = test_preds_insitu[r]
            this_measured_landcover = this_nfmd_obs['landcover'].values[0]
            this_sat_lcs = site_lc_preload[f'{this_lat}_{this_lon}'].sel(
                year=pd.Timestamp(this_date.year,1,1)
            )
            highest_lc_perc = this_sat_lcs.to_array().max().values
            if highest_lc_perc < 0.66:
                this_sat_lc = 'heterogeneous'
            else:
                max_idx = this_sat_lcs.to_array().argmax().values
                this_sat_lc = land_covers[max_idx]
            try:
                measured_lc_perc = this_sat_lcs[this_measured_landcover].values
            except KeyError:
                measured_lc_perc = 0.0
            site_lc_dict['lat'][num_obs_filled] = this_lat
            site_lc_dict['lon'][num_obs_filled] = this_lon
            site_lc_dict['date'][num_obs_filled] = this_date
            site_lc_dict['obs'][num_obs_filled] = this_obs
            site_lc_dict['pred'][num_obs_filled] = this_pred
            site_lc_dict['measured_landcover'][num_obs_filled] = this_measured_landcover
            site_lc_dict['satellite_landcover'][num_obs_filled] = this_sat_lc
            site_lc_dict['highest_lc_perc'][num_obs_filled] = highest_lc_perc
            site_lc_dict['measured_lc_perc'][num_obs_filled] = measured_lc_perc
            num_obs_filled += 1
    fold_lc_df = pd.DataFrame.from_dict(site_lc_dict)
    fold_lc_df = fold_lc_df.iloc[0:num_obs_filled].reset_index(drop=True)
    # save out the land cover dataframe for later analysis
    fold_lc_df.to_csv(site_lc_df_path, index=False)
    return fold_lc_df

def plot_landcover_analysis(
    model_1_landcover_df,
    model_2_landcover_df,
    model_3_landcover_df,
    model_1_gen_name,
    model_2_gen_name,
    model_3_gen_name
):
    # plot the land cover performance as reported by the sample
    lc_types = model_1_landcover_df.measured_landcover.unique()
    lc_rmse_dict = {}
    lc_rmse_dfs = []
    all_model_dfs = [
        model_1_landcover_df,
        model_2_landcover_df,
        model_3_landcover_df
    ]
    all_model_names = [
        model_1_gen_name,
        model_2_gen_name,
        model_3_gen_name
    ]
    for m,model in enumerate(all_model_dfs):
        lc_rmse_dict[m+1] = {}
        for lc in lc_types:
            lc_mask = (model.measured_landcover == lc)
            lc_obs = model.obs[lc_mask].values
            lc_preds = model.pred[lc_mask].values
            model_lc_rmse = np.sqrt(
                np.mean(
                    (lc_obs - lc_preds)**2
                )
            )
            lc_r2 = r2_score(
                lc_obs,
                lc_preds
            )
            lc_rmse_dict[m+1][lc] = {
                'rmse': model_lc_rmse,
                'r2': lc_r2,
                'num_samples': len(lc_obs)
            }
    plotting.bar_plot(
        lc_types,
        np.array([
            [lc_rmse_dict[1][lc]['rmse'] for lc in lc_types],
            [lc_rmse_dict[2][lc]['rmse'] for lc in lc_types],
            [lc_rmse_dict[3][lc]['rmse'] for lc in lc_types]
        ]).T,
        'Land Cover Type',
        'RMSE (%)',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_rmse_comparison_{model_1_gen_name}_vs_{model_2_gen_name}_vs_{model_3_gen_name}.png'
        ),
        label_with_n=True,
        sample_counts=np.array([
            [lc_rmse_dict[1][lc]['num_samples'] for lc in lc_types],
            [lc_rmse_dict[2][lc]['num_samples'] for lc in lc_types],
            [lc_rmse_dict[3][lc]['num_samples'] for lc in lc_types]
        ]).T,
        subcategory_labels=[model_1_gen_name, model_2_gen_name, model_3_gen_name]
    )
    # now plot a matrix of performance with landcover as satellite vs measured
    sat_lc_types = model_1_landcover_df.satellite_landcover.unique()
    performance_matrices = {}
    for m,model in enumerate(all_model_dfs):
        performance_matrix = np.zeros((len(lc_types), len(sat_lc_types)))
        for i,measured_lc in enumerate(lc_types):
            for j,sat_lc in enumerate(sat_lc_types):
                lc_mask = (
                    (model.measured_landcover == measured_lc) &
                    (model.satellite_landcover == sat_lc)
                )
                lc_obs = model.obs[lc_mask].values
                lc_preds = model.pred[lc_mask].values
                if len(lc_obs) == 0:
                    performance_matrix[i,j] = np.nan
                else:
                    lc_rmse = np.sqrt(
                        np.mean(
                            (lc_obs - lc_preds)**2
                        )
                    )
                    performance_matrix[i,j] = lc_rmse
        performance_matrices[m+1] = performance_matrix
        this_model_name =  all_model_names[m]
        plotting.heatmap(
            performance_matrix,
            sat_lc_types,
            lc_types,
            'Satellite Land Cover',
            'Measured Land Cover',
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'landcover_performance_matrix_{this_model_name}.png'
            ),
            cbar_label='RMSE (%)',
            vmin=0,
            vmax=50
        )
    # make a heatmap that is the difference between model 1 and model 3
    performance_matrix_diff = performance_matrices[1] - performance_matrices[3]
    plotting.heatmap(
        performance_matrix_diff,
        sat_lc_types,
        lc_types,
        'Satellite Land Cover',
        'Measured Land Cover',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_performance_matrix_diff_{model_1_gen_name}_vs_{model_3_gen_name}.png'
        ),
        cbar_label='RMSE (%)',
        vmin=-10,
        vmax=10,
        cmap_name='PiYG'
    )
    # bin by the % of the satellite land cover
    # plot rmse for each of these bins
    bin_delineations = np.array([
        [0.0,0.1],
        [0.1,0.2],
        [0.2,0.3],
        [0.3,0.4],
        [0.4,0.5],
        [0.5,0.6],
        [0.6,0.7],
        [0.7,0.8],
        [0.8,0.9],
        [0.9,1.0]
    ])
    ulbs_used = []
    bin_rmse_dict = {}
    for m,model in enumerate(all_model_dfs):
        bin_rmse_dict[m+1] = {}
        lower_limits = bin_delineations[:,0]
        upper_limits = bin_delineations[:,1]
        for u,ulb in enumerate(upper_limits):
            bin_mask = (
                (model.highest_lc_perc > lower_limits[u]) &
                (model.highest_lc_perc <= upper_limits[u])
            )
            bin_obs = model.obs[bin_mask].values
            bin_preds = model.pred[bin_mask].values
            if len(bin_obs) == 0:
                continue
            bin_rmse = np.sqrt(
                np.mean(
                    (bin_obs - bin_preds)**2
                )
            )
            bin_r2 = r2_score(
                bin_obs,
                bin_preds
            )
            bin_rmse_dict[m+1][ulb] = {
                'rmse': bin_rmse,
                'r2': bin_r2,
                'num_samples': len(bin_obs)
            }
            if ulb not in ulbs_used:
                ulbs_used.append(ulb)
    plotting.bar_plot(
        [str(ulb) for ulb in ulbs_used],
        np.array([
            [bin_rmse_dict[1][ulb]['rmse'] for ulb in ulbs_used],
            [bin_rmse_dict[2][ulb]['rmse'] for ulb in ulbs_used],
            [bin_rmse_dict[3][ulb]['rmse'] for ulb in ulbs_used]
        ]).T,
        'Max Measured Land Cover Fraction',
        'RMSE (%)',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'satellite_landcover_bin_rmse_comparison_{model_1_gen_name}_vs_{model_2_gen_name}_vs_{model_3_gen_name}.png'
        ),
        label_with_n=True,
        sample_counts=np.array([
            [bin_rmse_dict[1][ulb]['num_samples'] for ulb in ulbs_used],
            [bin_rmse_dict[2][ulb]['num_samples'] for ulb in ulbs_used],
            [bin_rmse_dict[3][ulb]['num_samples'] for ulb in ulbs_used]
        ]).T,
        subcategory_labels=[model_1_gen_name, model_2_gen_name, model_3_gen_name]
    )
    # bin by the % of the measured land cover
    # plot rmse for each of these bins
    bin_delineations = np.array([
        [0.0,0.1],
        [0.1,0.2],
        [0.2,0.3],
        [0.3,0.4],
        [0.4,0.5],
        [0.5,0.6],
        [0.6,0.7],
        [0.7,0.8],
        [0.8,0.9],
        [0.9,1.0]
    ])
    ulbs_used = []
    bin_rmse_dict = {}
    for m,model in enumerate(all_model_dfs):
        bin_rmse_dict[m+1] = {}
        lower_limits = bin_delineations[:,0]
        upper_limits = bin_delineations[:,1]
        for u,ulb in enumerate(upper_limits):
            bin_mask = (
                (model.measured_lc_perc > lower_limits[u]) &
                (model.measured_lc_perc <= upper_limits[u])
            )
            bin_obs = model.obs[bin_mask].values
            bin_preds = model.pred[bin_mask].values
            if len(bin_obs) == 0:
                continue
            bin_rmse = np.sqrt(
                np.mean(
                    (bin_obs - bin_preds)**2
                )
            )
            bin_r2 = r2_score(
                bin_obs,
                bin_preds
            )
            bin_rmse_dict[m+1][ulb] = {
                'rmse': bin_rmse,
                'r2': bin_r2,
                'num_samples': len(bin_obs)
            }
            if ulb not in ulbs_used:
                ulbs_used.append(ulb)
    plotting.bar_plot(
        [str(ulb) for ulb in ulbs_used],
        np.array([
            [bin_rmse_dict[1][ulb]['rmse'] for ulb in ulbs_used],
            [bin_rmse_dict[2][ulb]['rmse'] for ulb in ulbs_used],
            [bin_rmse_dict[3][ulb]['rmse'] for ulb in ulbs_used]
        ]).T,
        'Max Satellite Land Cover Fraction',
        'RMSE (%)',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'measured_landcover_bin_rmse_comparison_{model_1_gen_name}_vs_{model_2_gen_name}_vs_{model_3_gen_name}.png'
        ),
        label_with_n=True,
        sample_counts=np.array([
            [bin_rmse_dict[1][ulb]['num_samples'] for ulb in ulbs_used],
            [bin_rmse_dict[2][ulb]['num_samples'] for ulb in ulbs_used],
            [bin_rmse_dict[3][ulb]['num_samples'] for ulb in ulbs_used]
        ]).T,
        subcategory_labels=[model_1_gen_name, model_2_gen_name, model_3_gen_name]
    )
    
    
    agree_lc_rmse_dict = {}
    for m,model in enumerate(all_model_dfs):
        agree_lc_rmse_dict[m+1] = {}
        agree_mask = (model.measured_landcover == model.satellite_landcover)
        agree_df = model[agree_mask].reset_index(drop=True)
        agree_lc_types = agree_df.measured_landcover.unique()
        for lc in agree_lc_types:
            lc_mask = (agree_df.measured_landcover == lc)
            lc_obs = agree_df.obs[lc_mask].values
            lc_preds = agree_df.pred[lc_mask].values
            lc_rmse = np.sqrt(
                np.mean(
                    (lc_obs - lc_preds)**2
                )
            )
            lc_r2 = r2_score(
                lc_obs,
                lc_preds
            )
            agree_lc_rmse_dict[m+1][lc] = {
                'rmse': lc_rmse,
                'r2': lc_r2,
                'num_samples': len(lc_obs)
            }
    # bar plot of rmse by land cover
    plotting.bar_plot(
        agree_lc_types,
        np.array([
            [agree_lc_rmse_dict[1][lc]['rmse'] for lc in agree_lc_types],
            [agree_lc_rmse_dict[2][lc]['rmse'] for lc in agree_lc_types],
            [agree_lc_rmse_dict[3][lc]['rmse'] for lc in agree_lc_types]
        ]).T,
        'Land Cover Type',
        'RMSE (%)',
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_agreement_rmse_comparison_{model_1_gen_name}_vs_{model_2_gen_name}_vs_{model_3_gen_name}.png'
        ),
        label_with_n=True,
        sample_counts=np.array([
            [agree_lc_rmse_dict[1][lc]['num_samples'] for lc in agree_lc_types],
            [agree_lc_rmse_dict[2][lc]['num_samples'] for lc in agree_lc_types],
            [agree_lc_rmse_dict[3][lc]['num_samples'] for lc in agree_lc_types]
        ]).T,
        subcategory_labels=[model_1_gen_name, model_2_gen_name, model_3_gen_name]
    )
    
    
    
    #plotting.bar_plot(
    #    agree_lc_types,
    #    [agree_lc_rmse_df.loc[lc,'rmse'] for lc in agree_lc_types],
    #    'Land Cover Type',
    #    'RMSE (%)',
    #    os.path.join(
    #        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
    #        f'landcover_agreement_rmse_{model_gen_name}.png'
    #    ),
    #    label_with_n=True,
    #    sample_counts=[agree_lc_rmse_df.loc[lc,'num_samples'] for lc in agree_lc_types]
    #)


def get_site_error(
    model_dir,
    site_errors=None,
    progress_label=None,
):
    start_time = time.time()
    model_name = os.path.basename(os.path.normpath(model_dir))
    progress_prefix = f"[site_error] {progress_label}" if progress_label else f"[site_error] {model_name}"
    print(f"{progress_prefix}: loading fold outputs from {model_name}")
    with open(os.path.join(model_dir,'fold_info.json')) as f:
        fold_info = json.load(f)
    folds = list(fold_info.keys())
    for f,fold in enumerate(folds):
        print(f"{progress_prefix}: fold {f+1}/{len(folds)} ({fold})")
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
            site_dates = test_info_insitu.loc[site_indices, 'date'].values
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
                site_var = float('nan')
            else:
                site_r2 = r2_score(
                    site_true,
                    site_preds
                )
                site_var = np.var(site_true)
            site_key = f'{lat}_{lon}'
            site_errors[site_key] = {
                'num_measurements': site_num_measurements,
                'true_values': site_true.tolist(),
                'predictions': site_preds.tolist(),
                'dates': site_dates.tolist(),
                'rmse': site_rmse,
                'r2': site_r2,
                'var': site_var,
                'fold': fold
            }
    elapsed_s = time.time() - start_time
    print(
        f"{progress_prefix}: complete with {len(site_errors)} sites "
        f"in {elapsed_s:.1f}s"
    )
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
        model_1_fold = model_1_site_error[site_key]['fold']
        try:
            model_2_rmse = model_2_site_error[site_key]['rmse']
            model_2_r2 = model_2_site_error[site_key]['r2']
            model_2_fold = model_2_site_error[site_key]['fold']
        except KeyError:
            model_2_rmse = float('nan')
            model_2_r2 = float('nan')
            model_2_fold = float('nan')
        comparison_dict[site_key] = {
            'latitude': site_lat,
            'longitude': site_lon,
            'model_1_rmse': model_1_rmse,
            'model_1_r2': model_1_r2,
            'model_2_rmse': model_2_rmse,
            'model_2_r2': model_2_r2,
            'rmse_diff': model_2_rmse - model_1_rmse,
            'r2_diff': model_2_r2 - model_1_r2,
            'num_measurements': model_1_site_error[site_key]['num_measurements'],
            'var': model_1_site_error[site_key]['var'],
            'model_1_fold': model_1_fold,
            'model_2_fold': model_2_fold
        }
    # turn into DataFrame for easier analysis
    comparison_df = pd.DataFrame.from_dict(comparison_dict, orient='index')
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
        ylabel=f'Model 2 ({model_2_gen_name}) RMSE',
        line_to_plot='one_to_one'
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
        line_to_plot='one_to_one',
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1)
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
        ylabel='RMSE',
        line_to_plot='one_to_one'
    )
    # test correlation between rmse/r2 and differences in model performance
    plotting.generic_scatter(
        comparison_df.model_1_rmse.values,
        comparison_df.rmse_diff.values,
        os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'rmse_{model_2_gen_name}_minus_{model_1_gen_name}_vs_{model_1_gen_name}.png'
        ),
        xlabel=f'{model_1_gen_name} RMSE',
        ylabel=f'{model_2_gen_name} minus {model_1_gen_name} RMSE Difference',
        color_array=np.ones_like(comparison_df.model_1_rmse.values),
        alpha=0.8,
        s=22,
        fontsize=20
        #xlim=(0.0,75.0)
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
    analyze_at_sites = True
    analyze_by_landcover = True
    ## model settings
    #model_gen_dirs = [
    #    '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_base',
    #    '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_stats_nomonths',
    #    '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_multitask_5_1',
    #]
    #model_1_gen_name = 'base'
    #model_2_gen_name = 'sarstats_nomonths'
    #model_3_gen_name = 'multitask_5_1'
    ## settings for the best sarstats model
    #dms = [32,64,32]
    #nh = [1,2,1]
    #nl = [2,3,2]
    #df = [64,128,64]
    #do = [0.15,0.15,0.15]
    #bs = [128,128,128]
    #lr = [5e-4,5e-4,5e-4]
    #warmup = [502,501,2458]
    #wd = [1e-4,1e-4,1e-4]
    #iobs = [30638,30565,30638]
    #vvobs = [0,0,0]
    #vhobs = [0,0,119237]
    #dmlong = [32,128,64]
    #nhlong = [1,4,2]
    #nllong = [2,4,3]
    #dflong = [64,256,128]
    #outlong = [32,64,32]
    
    
    model_gen_dirs = [
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_base_nll',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_stats_nomonths_nll',
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/news1_multitask_5_1_nll',
    ]
    model_1_gen_name = 'base_nll'
    model_2_gen_name = 'sarstats_nomonths_nll'
    model_3_gen_name = 'multitask_5_1_nll'
    # settings for the best sarstats model
    dms = [64,32,128]
    nh = [2,1,4]
    nl = [3,2,4]
    df = [128,64,256]
    do = [0.15,0.15,0.15]
    bs = [128,128,128]
    lr = [5e-4,5e-4,1e-4]
    warmup = [502,501,2458]
    wd = [1e-4,1e-4,1e-4]
    iobs = [30638,30565,30638]
    vvobs = [0,0,0]
    vhobs = [0,0,119237]
    dmlong = [256,128,32]
    nhlong = [8,4,1]
    nllong = [5,4,2]
    dflong = [512,256,64]
    outlong = [64,64,32]
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
        all_1s_true = []
        all_1s_pred = []
        all_3s_true = []
        all_3s_pred = []
        for site in model_1_site_error.keys():
            all_1s_true.extend(model_1_site_error[site]['true_values'])
            all_1s_pred.extend(model_1_site_error[site]['predictions'])
        for site in model_3_site_error.keys():
            all_3s_true.extend(model_3_site_error[site]['true_values'])
            all_3s_pred.extend(model_3_site_error[site]['predictions'])
        all_1s_abserrs = np.abs(np.array(all_1s_true) - np.array(all_1s_pred))
        all_3s_abserrs = np.abs(np.array(all_3s_true) - np.array(all_3s_pred))
        ## filter anything larger than 100
        #all_1s_abserrs = all_1s_abserrs[all_1s_abserrs < 100]
        #all_3s_abserrs = all_3s_abserrs[all_3s_abserrs < 100]
        plotting.kde_plot(
            [all_1s_abserrs, all_3s_abserrs],
            [f'{model_1_gen_name} Absolute Errors', f'{model_3_gen_name} Absolute Errors'],
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'rmse_distribution_{model_1_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel='RMSE',
            ylabel='Density',
        )
        sys.exit()
        model_1_2_comparison_df = site_analysis(
            model_1_site_error,
            model_2_site_error,
            model_1_gen_name,
            model_2_gen_name,
            make_plots=True
        )
        model_1_3_comparison_df = site_analysis(
            model_1_site_error,
            model_3_site_error,
            model_1_gen_name,
            model_3_gen_name,
            make_plots=True
        )
        # other things where we don't want to run a full analysis...
        plotting.generic_scatter(
            model_1_2_comparison_df.rmse_diff.values,
            model_1_3_comparison_df.rmse_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'rmse_diff_from_base_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} RMSE Difference from Base Model',
            ylabel=f'{model_3_gen_name} RMSE Difference from Base Model'
        )
        plotting.generic_scatter(
            model_1_2_comparison_df.r2_diff.values,
            model_1_3_comparison_df.r2_diff.values,
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
            '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/nlcd/nlcd_2003_2023.zarr'
        )
        land_cover_breakdown_ds = xr.open_zarr(
            '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/nlcd/nlcd_target_grid_2003_2023.zarr'
        )
        nfmd_df = pd.read_csv(
            '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/nfmd/nfmd_processed.csv'
        )
        site_lc_df_path_model_1 = os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_analysis_{model_1_gen_name}.csv'
        )
        site_lc_df_path_model_2 = os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_analysis_{model_2_gen_name}.csv'
        )
        site_lc_df_path_model_3 = os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
            f'landcover_analysis_{model_3_gen_name}.csv'
        )
        rebuild_lc_dfs = False
        if rebuild_lc_dfs:
            model_1_landcover_df = get_landcover_df(
                model_1_dir,
                land_cover_single_ds,
                land_cover_breakdown_ds,
                nfmd_df,
                site_lc_df_path_model_1
            )
            model_2_landcover_df = get_landcover_df(
                model_2_dir,
                land_cover_single_ds,
                land_cover_breakdown_ds,
                nfmd_df,
                site_lc_df_path_model_2
            )
            model_3_landcover_df = get_landcover_df(
                model_3_dir,
                land_cover_single_ds,
                land_cover_breakdown_ds,
                nfmd_df,
                site_lc_df_path_model_3
            )
        model_1_landcover_df = pd.read_csv(site_lc_df_path_model_1)
        model_2_landcover_df = pd.read_csv(site_lc_df_path_model_2)
        model_3_landcover_df = pd.read_csv(site_lc_df_path_model_3)
        # now what do we actaully want to do with these?
        plot_landcover_analysis(
            model_1_landcover_df,
            model_2_landcover_df,
            model_3_landcover_df,
            model_1_gen_name,
            model_2_gen_name,
            model_3_gen_name
        )
        #plot_landcover_analysis(
        #    model_1_landcover_df,
        #    model_1_gen_name
        #)
        #plot_landcover_analysis(
        #    model_2_landcover_df,
        #    model_2_gen_name
        #)
        #plot_landcover_analysis(
        #    model_3_landcover_df,
        #    model_3_gen_name
        #)
        # do some comparison of the sites that seem to be influenced by microwaves
        # both positive and negative
        sarstats_help_df = (
            model_1_2_comparison_df[
                model_1_2_comparison_df.rmse_diff < -5.0
            ]
        )
        sarstats_help_sites = sarstats_help_df[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
        sarstats_hurt_df = (
            model_1_2_comparison_df[
                model_1_2_comparison_df.rmse_diff > 5.0
            ]
        )
        sarstats_hurt_sites = sarstats_hurt_df[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
        sarmultitask_help_df = (
            model_1_3_comparison_df[
                model_1_3_comparison_df.rmse_diff < -5.0
            ]
        )
        sarmultitask_help_sites = sarmultitask_help_df[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
        sarmultitask_hurt_df = (
            model_1_3_comparison_df[
                model_1_3_comparison_df.rmse_diff > 5.0
            ]
        )
        sarmultitask_hurt_sites = sarmultitask_hurt_df[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
        #microwaves_help_sites = pd.merge(
        #    sarstats_help_sites,
        #    sarmultitask_help_sites,
        #    on=['latitude','longitude'],
        #    how='inner'
        #)
        #microwaves_hurt_sites = pd.merge(
        #    sarstats_hurt_sites,
        #    sarmultitask_hurt_sites,
        #    on=['latitude','longitude'],
        #    how='inner'
        #)
        microwaves_help_sites = sarmultitask_help_sites
        microwaves_hurt_sites = sarmultitask_hurt_sites
        help_vs_hurt = pd.concat(
            [microwaves_help_sites, microwaves_hurt_sites],
            axis=0
        ).reset_index(drop=True)
        help_vs_hurt['type'] = [0]*len(microwaves_help_sites) + [1]*len(microwaves_hurt_sites)
        plotting.map_points(
            help_vs_hurt.longitude.values,
            help_vs_hurt.latitude.values,
            np.ones(len(help_vs_hurt)),
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'microwave_help_vs_hurt_sites_{model_2_gen_name}_and_{model_3_gen_name}.png'
            ),
            colors=help_vs_hurt['type'].values,
            cbar_lim=(-0.5,1.5),
            cmap='bwr',
            s_min=150,
            s_max=150
        )
        analyze_landcover_of_sites(
            microwaves_help_sites,
            model_1_landcover_df,
            "microwaves_help"
        )
        analyze_landcover_of_sites(
            microwaves_hurt_sites,
            model_1_landcover_df,
            "microwaves_hurt"
        )
        # try coloring this plot by different thigs...
        site_rmses = []
        for site_key in model_1_2_comparison_df.index:
            site_rmses.append(model_1_site_error[site_key]['rmse'])
        model_1_site_error_array = np.array(site_rmses)
        plotting.generic_scatter(
            model_1_2_comparison_df.rmse_diff.values,
            model_1_3_comparison_df.rmse_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'rmse_diff_from_base_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} RMSE Difference from Base Model',
            ylabel=f'{model_3_gen_name} RMSE Difference from Base Model',
            color_array=model_1_site_error_array,
            alpha=0.8,
            s=22,
            cbar_range=[0,50],
            cbar_label=f'{model_1_gen_name} RMSE',
            fontsize=20
        )
        plotting.generic_scatter(
            model_1_2_comparison_df.rmse_diff.values,
            model_1_3_comparison_df.rmse_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'rmse_diff_from_base_nocolor_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} RMSE Difference from Base Model',
            ylabel=f'{model_3_gen_name} RMSE Difference from Base Model',
            alpha=0.8,
            s=4,
            color_array=np.array([0.2]*len(model_1_2_comparison_df)),
            line_to_plot='one_to_one'
        )
        site_r2s = []
        for site_key in model_1_2_comparison_df.index:
            site_r2s.append(model_1_site_error[site_key]['r2'])
        model_1_site_r2_array = np.array(site_r2s)
        plotting.generic_scatter(
            model_1_2_comparison_df.r2_diff.values,
            model_1_3_comparison_df.r2_diff.values,
            os.path.join(
                '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/model_comparisons/',
                f'r2_diff_from_base_{model_2_gen_name}_vs_{model_3_gen_name}.png'
            ),
            xlabel=f'{model_2_gen_name} R² Difference from Base Model',
            ylabel=f'{model_3_gen_name} R² Difference from Base Model',
            color_array=model_1_site_r2_array,
            alpha=0.8,
            s=4,
            #xlim=(-1.1,1.1),
            #ylim=(-1.1,1.1),
            #corrclip=[-1.1,1.1],
            cbar_range=[0,1],
            cbar_label=f'{model_1_gen_name} R²',
            line_to_plot='one_to_one'
        )


if __name__ == "__main__":
    main()
