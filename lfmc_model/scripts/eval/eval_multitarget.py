import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys
from rich.console import Console
from rich.table import Table
import warnings

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(proj_root, 'lfmc_model','utils'))

from plotting import pred_obs_scatter,map_points

warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning
)

def main():
    # perform analysis across all models in a dir, since these are now batched out
    base_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/old_models/sarmultitask'
    load_only = False
    model_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    relevant_cols = [
        'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',
        'long_d_model', 'long_nhead', 'long_num_layers', 'long_dim_feedforward', 'long_out',
        'test_insitu_r2', 'test_insitu_rmse', 'test_vv_r2', 'test_vh_r2'
    ]
    if load_only:
        csv = pd.read_csv(
            os.path.join(
                base_dir, 'model_summary_results.csv'
            )
        )
        csv_sub = csv[relevant_cols]
        csv_sub = csv_sub.sort_values(by='test_insitu_rmse', ascending=True)
        console = Console()
        console.print(csv_sub)
        sys.exit()
    # get the model information for each dir
    d_models = []
    nheads = []
    n_layers = []
    dim_ffs = []
    dropouts = []
    bss = []
    lrs = []
    warmups = []
    wds = []
    iobs = []
    vvobs = []
    vhobs = []
    long_d_models = []
    long_nheads = []
    long_n_layers = []
    long_dim_ffs = []
    long_outs = []
    model_r2s = []
    model_rmses = []
    model_r2s_anom = []
    model_rmses_anom = []
    model_r2s_mean = []
    model_rmses_mean = []
    model_r2s_vv = []
    model_r2s_vh = []
    for m,model_dir in enumerate(model_dirs):
        model_name = model_dir.split('/')[-1]
        d_models.append(int(model_name.split('transformer_dm')[1].split('_')[0]))
        nheads.append(int(model_name.split('nh')[1].split('_')[0]))
        n_layers.append(int(model_name.split('nl')[1].split('_')[0]))
        dim_ffs.append(int(model_name.split('df')[1].split('_')[0]))
        dropouts.append(float(model_name.split('do')[1].split('_')[0]))
        bss.append(int(model_name.split('bs')[1].split('_')[0]))
        lrs.append(float(model_name.split('lr')[1].split('_')[0]))
        warmups.append(int(model_name.split('warmup')[1].split('_')[0]))
        wds.append(float(model_name.split('wd')[1].split('_')[0]))
        iobs.append(int(model_name.split('iobs')[1].split('_')[0]))
        vvobs.append(int(model_name.split('vvobs')[1].split('_')[0]))
        vhobs.append(int(model_name.split('vhobs')[1].split('_')[0]))
        long_d_models.append(int(model_name.split('dmlong')[1].split('_')[0]))
        long_nheads.append(int(model_name.split('nhlong')[1].split('_')[0]))
        long_n_layers.append(int(model_name.split('nllong')[1].split('_')[0]))
        long_dim_ffs.append(int(model_name.split('dflong')[1].split('_')[0]))
        long_outs.append(int(model_name.split('outlong')[1].split('_')[0]))
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
            print(f'Evaluating fold {f+1}/{len(folds)} for model {m+1}/{len(model_dirs)}')
            val_info_path = os.path.join(model_dir, f'fold_{fold}', 'val_info.csv')
            val_info = pd.read_csv(val_info_path)
            val_data_path = os.path.join(model_dir, f'fold_{fold}', 'val_outputs.pth')
            val_data = torch.load(val_data_path, weights_only=False)
            test_info_path = os.path.join(model_dir, f'fold_{fold}', 'test_info.csv')
            test_info = pd.read_csv(test_info_path)
            test_data_path = os.path.join(model_dir, f'fold_{fold}', 'test_outputs.pth')
            test_data = torch.load(test_data_path, weights_only=False)
            # get the preds and true
            val_preds_insitu = val_data['lfmc_preds']
            val_preds_insitu_std = val_data['lfmc_std']
            val_preds_vv = val_data['vv_preds']
            val_preds_vv_std = val_data['vv_std']
            val_preds_vh = val_data['vh_preds']
            val_preds_vh_std = val_data['vh_std']
            val_true_insitu = val_data['lfmc_true']
            val_true_vv = val_data['vv_true']
            val_true_vh = val_data['vh_true']
            test_preds_insitu = test_data['lfmc_preds']
            test_preds_insitu_std = test_data['lfmc_std']
            test_preds_vv = test_data['vv_preds']
            test_preds_vv_std = test_data['vv_std']
            test_preds_vh = test_data['vh_preds']
            test_preds_vh_std = test_data['vh_std']
            test_true_insitu = test_data['lfmc_true']
            test_true_vv = test_data['vv_true']
            test_true_vh = test_data['vh_true']
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
            try:
                test_insitu_n = len(test_true_insitu)
            except:
                test_insitu_n = 0
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
            if val_insitu_n > 0:
                all_val_preds_insitu = np.concatenate((all_val_preds_insitu, val_preds_insitu))
                all_val_true_insitu = np.concatenate((all_val_true_insitu, val_true_insitu))
            if val_vv_n > 0:
                all_val_preds_vv = np.concatenate((all_val_preds_vv, val_preds_vv))
                all_val_true_vv = np.concatenate((all_val_true_vv, val_true_vv))
            if val_vh_n > 0:
                all_val_preds_vh = np.concatenate((all_val_preds_vh, val_preds_vh))
                all_val_true_vh = np.concatenate((all_val_true_vh, val_true_vh))
            if test_insitu_n > 0:
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
        #print('Overall Validation Metrics - In Situ:')
        #print(f'MAE: {overall_val_insitu_mae:.3f}, RMSE: {overall_val_insitu_rmse:.3f}, R2: {overall_val_insitu_r2:.3f}')
        #print('Overall Test Metrics - In Situ:')
        #print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}')
        #print('Overall Validation Metrics - VV:')
        #print(f'MAE: {overall_val_vv_mae:.3f}, RMSE: {overall_val_vv_rmse:.3f}, R2: {overall_val_vv_r2:.3f}')
        #print('Overall Validation Metrics - VH:')
        #print(f'MAE: {overall_val_vh_mae:.3f}, RMSE: {overall_val_vh_rmse:.3f}, R2: {overall_val_vh_r2:.3f}')
        print('Overall Test Metrics - In Situ:')
        print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}')
        #print('Overall Test Metrics - VV:')
        #print(f'MAE: {overall_test_vv_mae:.3f}, RMSE: {overall_test_vv_rmse:.3f}, R2: {overall_test_vv_r2:.3f}')
        #print('Overall Test Metrics - VH:')
        #print(f'MAE: {overall_test_vh_mae:.3f}, RMSE: {overall_test_vh_rmse:.3f}, R2: {overall_test_vh_r2:.3f}')
        # keep track of these across models
        model_r2s.append(overall_test_insitu_r2)
        model_rmses.append(overall_test_insitu_rmse)
        model_r2s_vv.append(overall_test_vv_r2)
        model_r2s_vh.append(overall_test_vh_r2)
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
        model_r2s_anom.append(overall_test_insitu_anom_r2)
        model_rmses_anom.append(overall_test_insitu_anom_rmse)
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
        model_r2s_mean.append(overall_test_insitu_mean_r2)
        model_rmses_mean.append(overall_test_insitu_mean_rmse)
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
        #if np.isnan(test_rs_r2):
        #    overall_test_rs_mean_mae = np.nan
        #    overall_test_rs_mean_rmse = np.nan
        #    overall_test_rs_mean_r2 = np.nan
        #    overall_test_rs_mean_n = 0
        #else:
        #    overall_test_rs_mean_mae = np.mean(np.abs(all_test_preds_rs_means - all_test_true_rs_means))
        #    overall_test_rs_mean_rmse = np.sqrt(np.mean((all_test_preds_rs_means - all_test_true_rs_means)**2))
        #    overall_test_rs_mean_r2 = r2_score(all_test_true_rs_means, all_test_preds_rs_means)
        #    overall_test_rs_mean_n = len(all_test_true_rs_means)
        #    print('Overall Test Metrics - Remote Sensing Means:')
        #    print(f'MAE: {overall_test_rs_mean_mae:.3f}, RMSE: {overall_test_rs_mean_rmse:.3f}, R2: {overall_test_rs_mean_r2:.3f}')
        #    overall_test_rs_mean_plot_path = os.path.join(model_dir, 'overall_test_rs_mean_pred_obs_scatter.png')
        #    pred_obs_scatter(
        #        all_test_preds_rs_means,
        #        all_test_true_rs_means,
        #        overall_test_rs_mean_plot_path,
        #        mae=overall_test_rs_mean_mae,
        #        rmse=overall_test_rs_mean_rmse,
        #        r2=overall_test_rs_mean_r2,
        #        n=overall_test_rs_mean_n
        #    )
        ## plot the locations of where we trained
        #test_lats_insitu = test_info_insitu['latitude'].values
        #test_lons_insitu = test_info_insitu['longitude'].values
        #test_insitu_sites = np.column_stack((test_lats_insitu, test_lons_insitu))
        #test_insitu_unique_sites = np.unique(test_insitu_sites, axis=0)
        #map_points(
        #    test_insitu_unique_sites[:,1],
        #    test_insitu_unique_sites[:,0],
        #    np.ones(test_insitu_unique_sites.shape[0]),
        #    os.path.join(model_dir, 'test_insitu_site_locations.png'),
        #)
        #test_lats_rs = test_info_rs['latitude'].values
        #test_lons_rs = test_info_rs['longitude'].values
        #test_rs_sites = np.column_stack((test_lats_rs, test_lons_rs))
        #test_rs_unique_sites = np.unique(test_rs_sites, axis=0)
        #map_points(
        #    test_rs_unique_sites[:,1],
        #    test_rs_unique_sites[:,0],
        #    np.ones(test_rs_unique_sites.shape[0]),
        #    os.path.join(model_dir, 'test_rs_site_locations.png'),
        #)
    # summarize all model results
    summary_df = pd.DataFrame({
        'model_dir': model_dirs,
        'd_model': d_models,
        'nhead': nheads,
        'num_layers': n_layers,
        'dim_feedforward': dim_ffs,
        'dropout': dropouts,
        'batch_size': bss,
        'learning_rate': lrs,
        'warmup_steps': warmups,
        'weight_decay': wds,
        'insitu_obs': iobs,
        'vv_obs': vvobs,
        'vh_obs': vhobs,
        'long_d_model': long_d_models,
        'long_nhead': long_nheads,
        'long_num_layers': long_n_layers,
        'long_dim_feedforward': long_dim_ffs,
        'long_out': long_outs,
        'test_insitu_r2': model_r2s,
        'test_insitu_rmse': model_rmses,
        'test_insitu_r2_anom': model_r2s_anom,
        'test_insitu_rmse_anom': model_rmses_anom,
        'test_insitu_r2_mean': model_r2s_mean,
        'test_insitu_rmse_mean': model_rmses_mean,
        'test_vv_r2': model_r2s_vv,
        'test_vh_r2': model_r2s_vh
    })
    # save dataframe
    summary_csv_path = os.path.join(base_dir, 'model_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    # print this as a nice table
    # sort by lowest rmse (lowest @ top)
    summary_df = summary_df[relevant_cols]
    summary_df = summary_df.sort_values(by='test_insitu_rmse', ascending=True)
    print('Model Summary Results:')
    console = Console()
    console.print(summary_df)

if __name__ == "__main__":
    main()