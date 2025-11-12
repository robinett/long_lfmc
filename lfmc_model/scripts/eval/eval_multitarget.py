import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(proj_root, 'lfmc_model','utils'))

from plotting import pred_obs_scatter,map_points

def main():
    # model settings
    d_model = 64
    nhead = 2
    num_layers = 2
    dim_feedforward = 128
    dropout = 0.2
    batch_size = 128
    lr = 1e-4
    warmup_steps = 400
    adam_weight_decay = 1e-4
    remote_sensing_factor = 1.0
    rs_obs = 0
    dm_long = 128
    nhead_long = 4
    num_layers_long = 4
    dim_feedforward_long = 256
    dout_long = 64
    # get model name
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}_rsf{remote_sensing_factor}_rsobs{rs_obs}_'
        f'dmlong{dm_long}_nhlong{nhead_long}_nllong{num_layers_long}'
        f'_dflong{dim_feedforward_long}_outlong{dout_long}_sarstats'
    )
    # set up directories
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs'
    model_dir = os.path.join(save_dir, this_model_name)
    # get the folds
    with open(os.path.join(model_dir,'fold_info.json'), 'r') as f:
        fold_info = json.load(f)
    folds = list(fold_info.keys())
    all_val_preds_insitu = np.array([])
    all_val_true_insitu = np.array([])
    all_val_preds_vv = np.array([])
    all_val_true_vv = np.array([])
    all_val_preds_vh = np.array([])
    all_val_true_vh = np.array([])
    all_test_preds_insitu = np.array([])
    all_test_true_insitu = np.array([])
    all_test_preds_vv = np.array([])
    all_test_true_vv = np.array([])
    all_test_preds_vh = np.array([])
    all_test_true_vh = np.array([])
    for f,fold in enumerate(folds):
        print(f'Evaluating fold {f+1}/{len(folds)}: {fold}')
        print(f'This is fold {fold} out of {len(folds) + 1} folds')
        val_info_path = os.path.join(model_dir, f'fold_{fold}', 'val_info.csv')
        val_info = pd.read_csv(val_info_path)
        val_data_path = os.path.join(model_dir, f'fold_{fold}', 'val_outputs.pth')
        val_data = torch.load(val_data_path, weights_only=False)
        test_info_path = os.path.join(model_dir, f'fold_{fold}', 'test_info.csv')
        test_info = pd.read_csv(test_info_path)
        test_data_path = os.path.join(model_dir, f'fold_{fold}', 'test_outputs.pth')
        test_data = torch.load(test_data_path, weights_only=False)
        # get the preds and true
        val_preds_insitu = val_data['lfmc_insitu_preds']
        val_preds_insitu_std = val_data['lfmc_insitu_std']
        val_preds_vv = val_data['lfmc_vv_preds']
        val_preds_vv_std = val_data['lfmc_vv_std']
        val_preds_vh = val_data['lfmc_vh_preds']
        val_preds_vh_std = val_data['lfmc_vh_std']
        val_true_insitu = val_data['lfmc_insitu_true']
        val_true_vv = val_data['lfmc_vv_true']
        val_true_vh = val_data['lfmc_vh_true']
        test_preds_insitu = test_data['lfmc_insitu_preds']
        test_preds_insitu_std = test_data['lfmc_insitu_std']
        test_preds_vv = test_data['lfmc_vv_preds']
        test_preds_vv_std = test_data['lfmc_vv_std']
        test_preds_vh = test_data['lfmc_vh_preds']
        test_preds_vh_std = test_data['lfmc_vh_std']
        test_true_insitu = test_data['lfmc_insitu_true']
        test_true_vv = test_data['lfmc_vv_true']
        test_true_vh = test_data['lfmc_vh_true']
        # compute metrics
        val_insitu_mae = np.mean(np.abs(val_preds_insitu - val_true_insitu))
        val_insitu_rmse = np.sqrt(np.mean((val_preds_insitu - val_true_insitu)**2))
        val_insitu_r2 = r2_score(val_true_insitu, val_preds_insitu)
        val_insitu_n = len(val_true_insitu)
        val_vv_mae = np.mean(np.abs(val_preds_vv - val_true_vv))
        val_vv_rmse = np.sqrt(np.mean((val_preds_vv - val_true_vv)**2))
        val_vh_mae = np.mean(np.abs(val_preds_vh - val_true_vh))
        val_vh_rmse = np.sqrt(np.mean((val_preds_vh - val_true_vh)**2))
        if np.isnan(val_preds_insitu).all() or np.isnan(val_true_insitu).all():
            val_insitu_r2 = np.nan
        else:
            val_insitu_r2 = r2_score(val_true_insitu, val_preds_insitu)
        if np.isnan(val_preds_vv).all() or np.isnan(val_true_vv).all():
            val_vv_r2 = np.nan
            val_vv_n = 0
        else:
            val_vv_r2 = r2_score(val_true_vv, val_preds_vv)
            val_vv_n = len(val_true_vv)
        if np.isnan(val_preds_vh).all() or np.isnan(val_true_vh).all():
            val_vh_r2 = np.nan
            val_vh_n = 0
        else:
            val_vh_r2 = r2_score(val_true_vh, val_preds_vh)
            val_vh_n = len(val_true_vh)
        test_insitu_mae = np.mean(np.abs(test_preds_insitu - test_true_insitu))
        test_insitu_rmse = np.sqrt(np.mean((test_preds_insitu - test_true_insitu)**2))
        if np.isnan(test_preds_insitu).all() or np.isnan(test_true_insitu).all():
            test_insitu_r2 = np.nan
        else:
            test_insitu_r2 = r2_score(test_true_insitu, test_preds_insitu)
        test_insitu_n = len(test_true_insitu)
        test_vv_mae = np.mean(np.abs(test_preds_vv - test_true_vv))
        test_vv_rmse = np.sqrt(np.mean((test_preds_vv - test_true_vv)**2))
        test_vh_mae = np.mean(np.abs(test_preds_vh - test_true_vh))
        test_vh_rmse = np.sqrt(np.mean((test_preds_vh - test_true_vh)**2))
        if np.isnan(test_preds_vv).all() or np.isnan(test_true_vv).all():
            test_vv_r2 = np.nan
            test_vv_n = 0
        else:
            test_vv_r2 = r2_score(test_true_vv, test_preds_vv)
            test_vv_n = len(test_true_vv)
        if np.isnan(test_preds_vh).all() or np.isnan(test_true_vh).all():
            test_vh_r2 = np.nan
            test_vh_n = 0
        else:
            test_vh_r2 = r2_score(test_true_vh, test_preds_vh)
            test_vh_n = len(test_true_vh)
        ## make the plots
        #plot_path = os.path.join(model_dir, f'fold_{fold}', 'val_insitu_pred_obs_scatter.png')
        #pred_obs_scatter(
        #    val_preds_insitu,
        #    val_true_insitu,
        #    plot_path,
        #    mae=val_insitu_mae,
        #    rmse=val_insitu_rmse,
        #    r2=val_insitu_r2,
        #    n=val_insitu_n
        #)
        #plot_path = os.path.join(model_dir, f'fold_{fold}', 'val_rs_pred_obs_scatter.png')
        #pred_obs_scatter(
        #    val_preds_rs,
        #    val_true_rs,
        #    plot_path,
        #    mae=val_rs_mae,
        #    rmse=val_rs_rmse,
        #    r2=val_rs_r2,
        #    n=val_rs_n
        #)
        #plot_path = os.path.join(model_dir, f'fold_{fold}', 'test_insitu_pred_obs_scatter.png')
        #pred_obs_scatter(
        #    test_preds_insitu,
        #    test_true_insitu,
        #    plot_path,
        #    mae=test_insitu_mae,
        #    rmse=test_insitu_rmse,
        #    r2=test_insitu_r2,
        #    n=test_insitu_n
        #)
        #plot_path = os.path.join(model_dir, f'fold_{fold}', 'test_rs_pred_obs_scatter.png')
        #pred_obs_scatter(
        #    test_preds_rs,
        #    test_true_rs,
        #    plot_path,
        #    mae=test_rs_mae,
        #    rmse=test_rs_rmse,
        #    r2=test_rs_r2,
        #    n=test_rs_n
        #)
        # add data to overall
        all_val_preds_insitu = np.concatenate((all_val_preds_insitu, val_preds_insitu))
        all_val_true_insitu = np.concatenate((all_val_true_insitu, val_true_insitu))
        if val_vv_n > 0:
            all_val_preds_vv = np.concatenate((all_val_preds_vv, val_preds_vv))
            all_val_true_vv = np.concatenate((all_val_true_vv, val_true_vv))
        if val_vh_n > 0:
            all_val_preds_vh = np.concatenate((all_val_preds_vh, val_preds_vh))
            all_val_true_vh = np.concatenate((all_val_true_vh, val_true_vh))
        all_test_preds_insitu = np.concatenate((all_test_preds_insitu, test_preds_insitu))
        all_test_true_insitu = np.concatenate((all_test_true_insitu, test_true_insitu))
        if test_vv_n > 0:
            all_test_preds_vv = np.concatenate((all_test_preds_vv, test_preds_vv))
            all_test_true_vv = np.concatenate((all_test_true_vv, test_true_vv))
        if test_vh_n > 0:
            all_test_preds_vh = np.concatenate((all_test_preds_vh, test_preds_vh))
            all_test_true_vh = np.concatenate((all_test_true_vh, test_true_vh))
        if f == 0:
            val_info_all = val_info
            test_info_all = test_info
        else:
            val_info_all = pd.concat([val_info_all, val_info], ignore_index=True)
            test_info_all = pd.concat([test_info_all, test_info], ignore_index=True)
    # compute overall metrics
    overall_val_insitu_mae = np.mean(np.abs(all_val_preds_insitu - all_val_true_insitu))
    overall_val_insitu_rmse = np.sqrt(np.mean((all_val_preds_insitu - all_val_true_insitu)**2))
    overall_val_insitu_r2 = r2_score(all_val_true_insitu, all_val_preds_insitu)
    overall_val_insitu_n = len(all_val_true_insitu)
    overall_test_insitu_mae = np.mean(np.abs(all_test_preds_insitu - all_test_true_insitu))
    overall_test_insitu_rmse = np.sqrt(np.mean((all_test_preds_insitu - all_test_true_insitu)**2))
    overall_test_insitu_r2 = r2_score(all_test_true_insitu, all_test_preds_insitu)
    overall_test_insitu_n = len(all_test_true_insitu)
    overall_val_vv_mae = np.mean(np.abs(all_val_preds_vv - all_val_true_vv))
    overall_val_vv_rmse = np.sqrt(np.mean((all_val_preds_vv - all_val_true_vv)**2))
    overall_val_vh_mae = np.mean(np.abs(all_val_preds_vh - all_val_true_vh))
    overall_val_vh_rmse = np.sqrt(np.mean((all_val_preds_vh - all_val_true_vh)**2))
    if len(all_val_true_vv) == 0:
        overall_val_vv_r2 = np.nan
        overall_val_vv_n = 0
    else:
        overall_val_vv_r2 = r2_score(all_val_true_vv, all_val_preds_vv)
        overall_val_vv_n = len(all_val_true_vv)
    if len(all_val_true_vh) == 0:
        overall_val_vh_r2 = np.nan
        overall_val_vh_n = 0
    else:
        overall_val_vh_r2 = r2_score(all_val_true_vh, all_val_preds_vh)
        overall_val_vh_n = len(all_val_true_vh)
    overall_test_vv_mae = np.mean(np.abs(all_test_preds_vv - all_test_true_vv))
    overall_test_vv_rmse = np.sqrt(np.mean((all_test_preds_vv - all_test_true_vv)**2))
    overall_test_vh_mae = np.mean(np.abs(all_test_preds_vh - all_test_true_vh))
    overall_test_vh_rmse = np.sqrt(np.mean((all_test_preds_vh - all_test_true_vh)**2))
    if len(all_test_true_vv) == 0:
        overall_test_vv_r2 = np.nan
        overall_test_vv_n = 0
    else:
        overall_test_vv_r2 = r2_score(all_test_true_vv, all_test_preds_vv)
        overall_test_vv_n = len(all_test_true_vv)
    if len(all_test_true_vh) == 0:
        overall_test_vh_r2 = np.nan
        overall_test_vh_n = 0
    else:
        overall_test_vh_r2 = r2_score(all_test_true_vh, all_test_preds_vh)
        overall_test_vh_n = len(all_test_true_vh)
    print('Overall Validation Metrics - In Situ:')
    print(f'MAE: {overall_val_insitu_mae:.3f}, RMSE: {overall_val_insitu_rmse:.3f}, R2: {overall_val_insitu_r2:.3f}')
    print('Overall Test Metrics - In Situ:')
    print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}')
    print('Overall Validation Metrics - VV:')
    print(f'MAE: {overall_val_vv_mae:.3f}, RMSE: {overall_val_vv_rmse:.3f}, R2: {overall_val_vv_r2:.3f}')
    print('Overall Validation Metrics - VH:')
    print(f'MAE: {overall_val_vh_mae:.3f}, RMSE: {overall_val_vh_rmse:.3f}, R2: {overall_val_vh_r2:.3f}')
    print('Overall Test Metrics - In Situ:')
    print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}')
    print('Overall Test Metrics - VV:')
    print(f'MAE: {overall_test_vv_mae:.3f}, RMSE: {overall_test_vv_rmse:.3f}, R2: {overall_test_vv_r2:.3f}')
    print('Overall Test Metrics - VH:')
    print(f'MAE: {overall_test_vh_mae:.3f}, RMSE: {overall_test_vh_rmse:.3f}, R2: {overall_test_vh_r2:.3f}')
    # make overall plots
    overall_val_insitu_plot_path = os.path.join(model_dir, 'overall_val_insitu_pred_obs_scatter.png')
    pred_obs_scatter(
        all_val_preds_insitu,
        all_val_true_insitu,
        overall_val_insitu_plot_path,
        mae=overall_val_insitu_mae,
        rmse=overall_val_insitu_rmse,
        r2=overall_val_insitu_r2,
        n=overall_val_insitu_n
    )
    overall_test_insitu_plot_path = os.path.join(model_dir, 'overall_test_insitu_pred_obs_scatter.png')
    pred_obs_scatter(
        all_test_preds_insitu,
        all_test_true_insitu,
        overall_test_insitu_plot_path,
        mae=overall_test_insitu_mae,
        rmse=overall_test_insitu_rmse,
        r2=overall_test_insitu_r2,
        n=overall_test_insitu_n
    )
    #if overall_val_rs_n > 0:
    #    overall_val_rs_plot_path = os.path.join(model_dir, 'overall_val_rs_pred_obs_scatter.png')
    #    pred_obs_scatter(
    #        all_val_preds_rs,
    #        all_val_true_rs,
    #        overall_val_rs_plot_path,
    #        mae=overall_val_rs_mae,
    #        rmse=overall_val_rs_rmse,
    #        r2=overall_val_rs_r2,
    #        n=overall_val_rs_n
    #    )
    #    overall_test_rs_plot_path = os.path.join(model_dir, 'overall_test_rs_pred_obs_scatter.png')
    #    pred_obs_scatter(
    #        all_test_preds_rs,
    #        all_test_true_rs,
    #        overall_test_rs_plot_path,
    #        mae=overall_test_rs_mae,
    #        rmse=overall_test_rs_rmse,
    #        r2=overall_test_rs_r2,
    #        n=overall_test_rs_n
    #    )
    # compute and make plots for space vs time tradeoff
    print('Computing space vs time tradeoff metrics and plots...')
    test_info_insitu = test_info_all[test_info_all['source'] == 'nfmd'].copy()
    #test_info_rs = test_info_all[test_info_all['source'] == 'rs'].copy()
    # get the lat/lon pairs (sites)
    test_lats_insitu = test_info_insitu['latitude'].values
    test_lons_insitu = test_info_insitu['longitude'].values
    test_insitu_sites = np.column_stack((test_lats_insitu, test_lons_insitu))
    test_insitu_unique_sites = np.unique(test_insitu_sites, axis=0)
    for us, (lat, lon) in enumerate(test_insitu_unique_sites):
        site_mask = (test_lats_insitu == lat) & (test_lons_insitu == lon)
        this_site_preds = all_test_preds_insitu[site_mask]
        this_site_true = all_test_true_insitu[site_mask]
        # get the averages
        this_site_pred_mean = np.mean(this_site_preds)
        this_site_true_mean = np.mean(this_site_true)
        this_site_preds_anom = this_site_preds - this_site_pred_mean
        this_site_true_anom = this_site_true - this_site_true_mean
        if us == 0:
            all_test_preds_insitu_anom = this_site_preds_anom
            all_test_true_insitu_anom = this_site_true_anom
            all_test_preds_insitu_means = np.array([this_site_pred_mean])
            all_test_true_insitu_means = np.array([this_site_true_mean])
        else:
            all_test_preds_insitu_anom = np.concatenate(
                (all_test_preds_insitu_anom, this_site_preds_anom)
            )
            all_test_true_insitu_anom = np.concatenate(
                (all_test_true_insitu_anom, this_site_true_anom)
            )
            all_test_preds_insitu_means = np.concatenate(
                (all_test_preds_insitu_means, np.array([this_site_pred_mean]))
            )
            all_test_true_insitu_means = np.concatenate(
                (all_test_true_insitu_means, np.array([this_site_true_mean]))
            )
    ## do the same for remote
    #test_lats_rs = test_info_rs['latitude'].values
    #test_lons_rs = test_info_rs['longitude'].values
    #test_rs_sites = np.column_stack((test_lats_rs, test_lons_rs))
    #test_rs_unique_sites = np.unique(test_rs_sites, axis=0)
    #for us, (lat, lon) in enumerate(test_rs_unique_sites):
    #    site_mask = (test_lats_rs == lat) & (test_lons_rs == lon)
    #    this_site_preds = all_test_preds_rs[site_mask]
    #    this_site_true = all_test_true_rs[site_mask]
    #    # get the averages
    #    this_site_pred_mean = np.mean(this_site_preds)
    #    this_site_true_mean = np.mean(this_site_true)
    #    this_site_preds_anom = this_site_preds - this_site_pred_mean
    #    this_site_true_anom = this_site_true - this_site_true_mean
    #    if us == 0:
    #        all_test_preds_rs_anom = this_site_preds_anom
    #        all_test_true_rs_anom = this_site_true_anom
    #        all_test_preds_rs_means = np.array([this_site_pred_mean])
    #        all_test_true_rs_means = np.array([this_site_true_mean])
    #    else:
    #        all_test_preds_rs_anom = np.concatenate(
    #            (all_test_preds_rs_anom, this_site_preds_anom)
    #        )
    #        all_test_true_rs_anom = np.concatenate(
    #            (all_test_true_rs_anom, this_site_true_anom)
    #        )
    #        all_test_preds_rs_means = np.concatenate(
    #            (all_test_preds_rs_means, np.array([this_site_pred_mean]))
    #        )
    #        all_test_true_rs_means = np.concatenate(
    #            (all_test_true_rs_means, np.array([this_site_true_mean]))
    #        )
    # compute metrics for anomalies
    overall_test_insitu_anom_mae = np.mean(np.abs(all_test_preds_insitu_anom - all_test_true_insitu_anom))
    overall_test_insitu_anom_rmse = np.sqrt(np.mean((all_test_preds_insitu_anom - all_test_true_insitu_anom)**2))
    overall_test_insitu_anom_r2 = r2_score(all_test_true_insitu_anom, all_test_preds_insitu_anom)
    overall_test_insitu_anom_n = len(all_test_true_insitu_anom)
    print('Overall Test Metrics - In Situ Anomalies:')
    print(f'MAE: {overall_test_insitu_anom_mae:.3f}, RMSE: {overall_test_insitu_anom_rmse:.3f}, R2: {overall_test_insitu_anom_r2:.3f}')
    # make overall plots for anomalies
    overall_test_insitu_anom_plot_path = os.path.join(model_dir, 'overall_test_insitu_anom_pred_obs_scatter.png')
    pred_obs_scatter(
        all_test_preds_insitu_anom,
        all_test_true_insitu_anom,
        overall_test_insitu_anom_plot_path,
        mae=overall_test_insitu_anom_mae,
        rmse=overall_test_insitu_anom_rmse,
        r2=overall_test_insitu_anom_r2,
        n=overall_test_insitu_anom_n
    )
    #if np.isnan(test_rs_r2):
    #    overall_test_rs_anom_mae = np.nan
    #    overall_test_rs_anom_rmse = np.nan
    #    overall_test_rs_anom_r2 = np.nan
    #    overall_test_rs_anom_n = 0
    #else:
    #    overall_test_rs_anom_mae = np.mean(np.abs(all_test_preds_rs_anom - all_test_true_rs_anom))
    #    overall_test_rs_anom_rmse = np.sqrt(np.mean((all_test_preds_rs_anom - all_test_true_rs_anom)**2))
    #    overall_test_rs_anom_r2 = r2_score(all_test_true_rs_anom, all_test_preds_rs_anom)
    #    overall_test_rs_anom_n = len(all_test_true_rs_anom)
    #    print('Overall Test Metrics - Remote Sensing Anomalies:')
    #    print(f'MAE: {overall_test_rs_anom_mae:.3f}, RMSE: {overall_test_rs_anom_rmse:.3f}, R2: {overall_test_rs_anom_r2:.3f}')
    #    overall_test_rs_anom_plot_path = os.path.join(model_dir, 'overall_test_rs_anom_pred_obs_scatter.png')
    #    pred_obs_scatter(
    #        all_test_preds_rs_anom,
    #        all_test_true_rs_anom,
    #        overall_test_rs_anom_plot_path,
    #        mae=overall_test_rs_anom_mae,
    #        rmse=overall_test_rs_anom_rmse,
    #        r2=overall_test_rs_anom_r2,
    #        n=overall_test_rs_anom_n
    #    )
    # compute metrics for means
    overall_test_insitu_mean_mae = np.mean(np.abs(all_test_preds_insitu_means - all_test_true_insitu_means))
    overall_test_insitu_mean_rmse = np.sqrt(np.mean((all_test_preds_insitu_means - all_test_true_insitu_means)**2))
    overall_test_insitu_mean_r2 = r2_score(all_test_true_insitu_means, all_test_preds_insitu_means)
    overall_test_insitu_mean_n = len(all_test_true_insitu_means)
    print('Overall Test Metrics - In Situ Means:')
    print(f'MAE: {overall_test_insitu_mean_mae:.3f}, RMSE: {overall_test_insitu_mean_rmse:.3f}, R2: {overall_test_insitu_mean_r2:.3f}')
    # make overall plots for means
    overall_test_insitu_mean_plot_path = os.path.join(model_dir, 'overall_test_insitu_mean_pred_obs_scatter.png')
    pred_obs_scatter(
        all_test_preds_insitu_means,
        all_test_true_insitu_means,
        overall_test_insitu_mean_plot_path,
        mae=overall_test_insitu_mean_mae,
        rmse=overall_test_insitu_mean_rmse,
        r2=overall_test_insitu_mean_r2,
        n=overall_test_insitu_mean_n
    )
    sys.exit()
    if np.isnan(test_rs_r2):
        overall_test_rs_mean_mae = np.nan
        overall_test_rs_mean_rmse = np.nan
        overall_test_rs_mean_r2 = np.nan
        overall_test_rs_mean_n = 0
    else:
        overall_test_rs_mean_mae = np.mean(np.abs(all_test_preds_rs_means - all_test_true_rs_means))
        overall_test_rs_mean_rmse = np.sqrt(np.mean((all_test_preds_rs_means - all_test_true_rs_means)**2))
        overall_test_rs_mean_r2 = r2_score(all_test_true_rs_means, all_test_preds_rs_means)
        overall_test_rs_mean_n = len(all_test_true_rs_means)
        print('Overall Test Metrics - Remote Sensing Means:')
        print(f'MAE: {overall_test_rs_mean_mae:.3f}, RMSE: {overall_test_rs_mean_rmse:.3f}, R2: {overall_test_rs_mean_r2:.3f}')
        overall_test_rs_mean_plot_path = os.path.join(model_dir, 'overall_test_rs_mean_pred_obs_scatter.png')
        pred_obs_scatter(
            all_test_preds_rs_means,
            all_test_true_rs_means,
            overall_test_rs_mean_plot_path,
            mae=overall_test_rs_mean_mae,
            rmse=overall_test_rs_mean_rmse,
            r2=overall_test_rs_mean_r2,
            n=overall_test_rs_mean_n
        )
    # plot the locations of where we trained
    test_lats_insitu = test_info_insitu['latitude'].values
    test_lons_insitu = test_info_insitu['longitude'].values
    test_insitu_sites = np.column_stack((test_lats_insitu, test_lons_insitu))
    test_insitu_unique_sites = np.unique(test_insitu_sites, axis=0)
    map_points(
        test_insitu_unique_sites[:,1],
        test_insitu_unique_sites[:,0],
        np.ones(test_insitu_unique_sites.shape[0]),
        os.path.join(model_dir, 'test_insitu_site_locations.png'),
    )
    test_lats_rs = test_info_rs['latitude'].values
    test_lons_rs = test_info_rs['longitude'].values
    test_rs_sites = np.column_stack((test_lats_rs, test_lons_rs))
    test_rs_unique_sites = np.unique(test_rs_sites, axis=0)
    map_points(
        test_rs_unique_sites[:,1],
        test_rs_unique_sites[:,0],
        np.ones(test_rs_unique_sites.shape[0]),
        os.path.join(model_dir, 'test_rs_site_locations.png'),
    )

if __name__ == "__main__":
    main()