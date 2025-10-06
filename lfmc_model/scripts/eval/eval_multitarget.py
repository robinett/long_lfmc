import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(proj_root, 'lfmc_model','utils'))

from plotting import pred_obs_scatter 

def main():
    # model settings
    d_model = 96
    nhead = 2
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    batch_size = 128
    lr = 1e-4
    warmup_steps = 2000
    adam_weight_decay = 0.01   
    remote_sensing_factor = 1.0
    # get model name
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}_rsf{remote_sensing_factor}_70k'
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
    all_val_preds_rs = np.array([])
    all_val_true_rs = np.array([])
    all_test_preds_insitu = np.array([])
    all_test_true_insitu = np.array([])
    all_test_preds_rs = np.array([])
    all_test_true_rs = np.array([])
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
        val_preds_rs = val_data['lfmc_rs_preds']
        val_preds_rs_std = val_data['lfmc_rs_std']
        val_true_insitu = val_data['lfmc_insitu_true']
        val_true_rs = val_data['lfmc_rs_true']
        test_preds_insitu = test_data['lfmc_insitu_preds']
        test_preds_insitu_std = test_data['lfmc_insitu_std']
        test_preds_rs = test_data['lfmc_rs_preds']
        test_preds_rs_std = test_data['lfmc_rs_std']
        test_true_insitu = test_data['lfmc_insitu_true']
        test_true_rs = test_data['lfmc_rs_true']
        # compute metrics
        val_insitu_mae = np.mean(np.abs(val_preds_insitu - val_true_insitu))
        val_insitu_rmse = np.sqrt(np.mean((val_preds_insitu - val_true_insitu)**2))
        val_insitu_r2 = r2_score(val_true_insitu, val_preds_insitu)
        val_insitu_n = len(val_true_insitu)
        val_rs_mae = np.mean(np.abs(val_preds_rs - val_true_rs))
        val_rs_rmse = np.sqrt(np.mean((val_preds_rs - val_true_rs)**2))
        val_rs_r2 = r2_score(val_true_rs, val_preds_rs)
        val_rs_n = len(val_true_rs)
        test_insitu_mae = np.mean(np.abs(test_preds_insitu - test_true_insitu))
        test_insitu_rmse = np.sqrt(np.mean((test_preds_insitu - test_true_insitu)**2))
        test_insitu_r2 = r2_score(test_true_insitu, test_preds_insitu)
        test_insitu_n = len(test_true_insitu)
        test_rs_mae = np.mean(np.abs(test_preds_rs - test_true_rs))
        test_rs_rmse = np.sqrt(np.mean((test_preds_rs - test_true_rs)**2))
        test_rs_r2 = r2_score(test_true_rs, test_preds_rs)
        test_rs_n = len(test_true_rs)
        # make the plots
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'val_insitu_pred_obs_scatter.png')
        pred_obs_scatter(
            val_preds_insitu,
            val_true_insitu,
            plot_path,
            mae=val_insitu_mae,
            rmse=val_insitu_rmse,
            r2=val_insitu_r2,
            n=val_insitu_n
        )
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'val_rs_pred_obs_scatter.png')
        pred_obs_scatter(
            val_preds_rs,
            val_true_rs,
            plot_path,
            mae=val_rs_mae,
            rmse=val_rs_rmse,
            r2=val_rs_r2,
            n=val_rs_n
        )
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'test_insitu_pred_obs_scatter.png')
        pred_obs_scatter(
            test_preds_insitu,
            test_true_insitu,
            plot_path,
            mae=test_insitu_mae,
            rmse=test_insitu_rmse,
            r2=test_insitu_r2,
            n=test_insitu_n
        )
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'test_rs_pred_obs_scatter.png')
        pred_obs_scatter(
            test_preds_rs,
            test_true_rs,
            plot_path,
            mae=test_rs_mae,
            rmse=test_rs_rmse,
            r2=test_rs_r2,
            n=test_rs_n
        )
        # add data to overall
        all_val_preds_insitu = np.concatenate((all_val_preds_insitu, val_preds_insitu))
        all_val_true_insitu = np.concatenate((all_val_true_insitu, val_true_insitu))
        all_val_preds_rs = np.concatenate((all_val_preds_rs, val_preds_rs))
        all_val_true_rs = np.concatenate((all_val_true_rs, val_true_rs))
        all_test_preds_insitu = np.concatenate((all_test_preds_insitu, test_preds_insitu))
        all_test_true_insitu = np.concatenate((all_test_true_insitu, test_true_insitu))
        all_test_preds_rs = np.concatenate((all_test_preds_rs, test_preds_rs))
        all_test_true_rs = np.concatenate((all_test_true_rs, test_true_rs))    
    # compute overall metrics
    ovall_val_insitu_mae = np.mean(np.abs(all_val_preds_insitu - all_val_true_insitu))
    overall_val_insitu_rmse = np.sqrt(np.mean((all_val_preds_insitu - all_val_true_insitu)**2))
    overall_val_insitu_r2 = r2_score(all_val_true_insitu, all_val_preds_insitu)
    overall_val_insitu_n = len(all_val_true_insitu)
    overall_test_insitu_mae = np.mean(np.abs(all_test_preds_insitu - all_test_true_insitu))
    overall_test_insitu_rmse = np.sqrt(np.mean((all_test_preds_insitu - all_test_true_insitu)**2))
    overall_test_insitu_r2 = r2_score(all_test_true_insitu, all_test_preds_insitu)
    overall_test_insitu_n = len(all_test_true_insitu)
    overall_val_rs_mae = np.mean(np.abs(all_val_preds_rs - all_val_true_rs))
    overall_val_rs_rmse = np.sqrt(np.mean((all_val_preds_rs - all_val_true_rs)**2))
    overall_val_rs_r2 = r2_score(all_val_true_rs, all_val_preds_rs)
    overall_val_rs_n = len(all_val_true_rs)
    overall_test_rs_mae = np.mean(np.abs(all_test_preds_rs - all_test_true_rs))
    overall_test_rs_rmse = np.sqrt(np.mean((all_test_preds_rs - all_test_true_rs)**2))
    overall_test_rs_r2 = r2_score(all_test_true_rs, all_test_preds_rs)
    overall_test_rs_n = len(all_test_true_rs)
    print('Overall Validation Metrics - In Situ:')
    print(f'MAE: {ovall_val_insitu_mae:.3f}, RMSE: {overall_val_insitu_rmse:.3f}, R2: {overall_val_insitu_r2:.3f}')
    print('Overall Test Metrics - In Situ:')
    print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}')
    print('Overall Validation Metrics - Remote Sensing:')
    print(f'MAE: {overall_val_rs_mae:.3f}, RMSE: {overall_val_rs_rmse:.3f}, R2: {overall_val_rs_r2:.3f}')
    print('Overall Test Metrics - Remote Sensing:')
    print(f'MAE: {overall_test_rs_mae:.3f}, RMSE: {overall_test_rs_rmse:.3f}, R2: {overall_test_rs_r2:.3f}')
    # make overall plots
    overall_val_insitu_plot_path = os.path.join(model_dir, 'overall_val_insitu_pred_obs_scatter.png')
    pred_obs_scatter(
        all_val_preds_insitu,
        all_val_true_insitu,
        overall_val_insitu_plot_path,
        mae=ovall_val_insitu_mae,
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
    overall_val_rs_plot_path = os.path.join(model_dir, 'overall_val_rs_pred_obs_scatter.png')
    pred_obs_scatter(
        all_val_preds_rs,
        all_val_true_rs,
        overall_val_rs_plot_path,
        mae=overall_val_rs_mae,
        rmse=overall_val_rs_rmse,
        r2=overall_val_rs_r2,
        n=overall_val_rs_n
    )
    overall_test_rs_plot_path = os.path.join(model_dir, 'overall_test_rs_pred_obs_scatter.png')
    pred_obs_scatter(
        all_test_preds_rs,
        all_test_true_rs,
        overall_test_rs_plot_path,
        mae=overall_test_rs_mae,
        rmse=overall_test_rs_rmse,
        r2=overall_test_rs_r2,
        n=overall_test_rs_n
    )

if __name__ == "__main__":
    main()