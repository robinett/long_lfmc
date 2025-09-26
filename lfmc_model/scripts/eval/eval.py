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
    d_model = 32
    nhead = 1
    num_layers = 2
    dim_feedforward = 64
    dropout = 0.1
    batch_size = 128
    lr = 1e-4
    warmup_steps = 200
    adam_weight_decay = 0.01   
    # get model name
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}'
    )
    # set up directories
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs'
    model_dir = os.path.join(save_dir, this_model_name)
    # get the folds
    with open(os.path.join(model_dir,'fold_info.json'), 'r') as f:
        fold_info = json.load(f)
    folds = list(fold_info.keys())
    all_val_preds = np.array([])
    all_val_true = np.array([])
    all_test_preds = np.array([])
    all_test_true = np.array([])
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
        val_preds = val_data['preds']
        val_true = val_data['true']
        test_preds = test_data['preds']
        test_true = test_data['true']
        # compute metrics
        val_mae = np.mean(np.abs(val_preds - val_true))
        val_rmse = np.sqrt(np.mean((val_preds - val_true)**2))
        val_r2 = r2_score(val_true, val_preds)
        test_mae = np.mean(np.abs(test_preds - test_true))
        test_rmse = np.sqrt(np.mean((test_preds - test_true)**2))
        test_r2 = r2_score(test_true, test_preds)
        # make the plots
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'val_pred_obs_scatter.png')
        pred_obs_scatter(
            val_preds,
            val_true,
            plot_path,
            mae=val_mae,
            rmse=val_rmse,
            r2=val_r2
        )
        plot_path = os.path.join(model_dir, f'fold_{fold}', 'test_pred_obs_scatter.png')
        pred_obs_scatter(
            test_preds,
            test_true,
            plot_path,
            mae=test_mae,
            rmse=test_rmse,
            r2=test_r2
        )
        # add data to overall
        all_val_preds = np.concatenate((all_val_preds, val_preds))
        all_val_true = np.concatenate((all_val_true, val_true))
        all_test_preds = np.concatenate((all_test_preds, test_preds))
        all_test_true = np.concatenate((all_test_true, test_true))
    # compute overall metrics
    overall_val_mae = np.mean(np.abs(all_val_preds - all_val_true))
    overall_val_rmse = np.sqrt(np.mean((all_val_preds - all_val_true)**2))
    overall_val_r2 = r2_score(all_val_true, all_val_preds)
    overall_test_mae = np.mean(np.abs(all_test_preds - all_test_true))
    overall_test_rmse = np.sqrt(np.mean((all_test_preds - all_test_true)**2))
    overall_test_r2 = r2_score(all_test_true, all_test_preds)
    print('Overall Validation Metrics:')
    print(f'MAE: {overall_val_mae:.3f}, RMSE: {overall_val_rmse:.3f}, R2: {overall_val_r2:.3f}')
    print('Overall Test Metrics:')
    print(f'MAE: {overall_test_mae:.3f}, RMSE: {overall_test_rmse:.3f}, R2: {overall_test_r2:.3f}')
    # make overall plots
    overall_val_plot_path = os.path.join(model_dir, 'overall_val_pred_obs_scatter.png')
    pred_obs_scatter(
        all_val_preds,
        all_val_true,
        overall_val_plot_path,
        mae=overall_val_mae,
        rmse=overall_val_rmse,
        r2=overall_val_r2
    )
    overall_test_plot_path = os.path.join(model_dir, 'overall_test_pred_obs_scatter.png')
    pred_obs_scatter(
        all_test_preds,
        all_test_true,
        overall_test_plot_path,
        mae=overall_test_mae,
        rmse=overall_test_rmse,
        r2=overall_test_r2
    )


if __name__ == "__main__":
    main()