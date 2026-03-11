import os
import json
import re
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

from plotting import pred_obs_scatter,map_points,generic_scatter,generic_hexbin

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

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

def gaussian_nll(preds,targets,var):
    return 0.5 * (((preds - targets) ** 2) / var + np.log(var) + np.log(2.0 * np.pi))


def parse_model_name_value(model_name, tag, cast_fn):
    match = re.search(rf"{re.escape(tag)}([^_]+)", model_name)
    if match is None:
        return np.nan
    try:
        return cast_fn(match.group(1))
    except Exception:
        return np.nan


def parse_first_task_weight(model_name):
    # Expected pattern from training naming: ..._firstweight_tw0<value>
    match = re.search(r"_firstweight_tw0([0-9.]+)", model_name)
    if match is None:
        return np.nan
    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def compute_site_month_y2y_metrics(test_info_insitu, preds_insitu, true_insitu):
    """Compute year-to-year site-month deviation MAE metrics."""
    metrics = {
        'test_insitu_y2y_mae_obs_centered_abs': np.nan,
        'test_insitu_y2y_mae_obs_centered_pct': np.nan,
        'test_insitu_y2y_mae_source_centered_abs': np.nan,
        'test_insitu_y2y_mae_source_centered_pct': np.nan,
        'test_insitu_y2y_pct_captured_obs_centered': np.nan,
        'test_insitu_y2y_pct_captured_source_centered': np.nan,
        'test_insitu_y2y_n': 0,
        'test_insitu_y2y_groups': 0
    }
    if len(test_info_insitu) == 0 or len(preds_insitu) == 0 or len(true_insitu) == 0:
        return metrics

    aligned_n = min(len(test_info_insitu), len(preds_insitu), len(true_insitu))
    if aligned_n == 0:
        return metrics

    if aligned_n != len(test_info_insitu) or aligned_n != len(preds_insitu) or aligned_n != len(true_insitu):
        print(
            'Warning: alignment mismatch for y2y metrics; '
            f'using first {aligned_n} rows.'
        )

    eval_df = test_info_insitu.iloc[:aligned_n].copy()
    eval_df['pred'] = np.asarray(preds_insitu[:aligned_n], dtype=float)
    eval_df['true'] = np.asarray(true_insitu[:aligned_n], dtype=float)
    eval_df['date'] = pd.to_datetime(eval_df['date'], errors='coerce')

    valid_mask = (
        eval_df['date'].notna()
        & eval_df['site_id'].notna()
        & np.isfinite(eval_df['pred'])
        & np.isfinite(eval_df['true'])
    )
    eval_df = eval_df.loc[valid_mask].copy()
    if len(eval_df) == 0:
        return metrics

    eval_df['year'] = eval_df['date'].dt.year
    eval_df['month'] = eval_df['date'].dt.month

    grp_cols = ['site_id', 'month']
    group_counts = eval_df.groupby(grp_cols).size().rename('n_obs')
    unique_year_counts = eval_df.groupby(grp_cols)['year'].nunique().rename('n_years')
    group_stats = pd.concat([group_counts, unique_year_counts], axis=1).reset_index()
    valid_groups = group_stats[
        (group_stats['n_obs'] >= 20) & (group_stats['n_years'] >= 3)
    ][grp_cols]
    if len(valid_groups) == 0:
        return metrics

    eval_df = eval_df.merge(valid_groups, on=grp_cols, how='inner')
    if len(eval_df) == 0:
        return metrics

    eval_df['obs_mean'] = eval_df.groupby(grp_cols)['true'].transform('mean')
    eval_df['pred_mean'] = eval_df.groupby(grp_cols)['pred'].transform('mean')

    true_dev_obs = eval_df['true'] - eval_df['obs_mean']
    pred_dev_obs = eval_df['pred'] - eval_df['obs_mean']
    pred_dev_pred = eval_df['pred'] - eval_df['pred_mean']

    metrics['test_insitu_y2y_mae_obs_centered_abs'] = np.mean(np.abs(true_dev_obs - pred_dev_obs))
    metrics['test_insitu_y2y_mae_source_centered_abs'] = np.mean(np.abs(true_dev_obs - pred_dev_pred))

    eps = 1e-8
    obs_den_ok = np.abs(eval_df['obs_mean']) > eps
    pred_den_ok = np.abs(eval_df['pred_mean']) > eps

    if obs_den_ok.any():
        true_pct_obs = true_dev_obs[obs_den_ok] / eval_df.loc[obs_den_ok, 'obs_mean']
        pred_pct_obs = pred_dev_obs[obs_den_ok] / eval_df.loc[obs_den_ok, 'obs_mean']
        metrics['test_insitu_y2y_mae_obs_centered_pct'] = np.mean(np.abs(true_pct_obs - pred_pct_obs))

    source_pct_mask = obs_den_ok & pred_den_ok
    if source_pct_mask.any():
        true_pct_source = true_dev_obs[source_pct_mask] / eval_df.loc[source_pct_mask, 'obs_mean']
        pred_pct_source = pred_dev_pred[source_pct_mask] / eval_df.loc[source_pct_mask, 'pred_mean']
        metrics['test_insitu_y2y_mae_source_centered_pct'] = np.mean(np.abs(true_pct_source - pred_pct_source))

    true_dev_obs_np = true_dev_obs.to_numpy(dtype=float)
    pred_dev_obs_np = pred_dev_obs.to_numpy(dtype=float)
    pred_dev_pred_np = pred_dev_pred.to_numpy(dtype=float)
    denom = np.sum(np.square(true_dev_obs_np))
    if denom > eps:
        numer_obs = np.sum(np.square(pred_dev_obs_np - true_dev_obs_np))
        numer_source = np.sum(np.square(pred_dev_pred_np - true_dev_obs_np))
        metrics['test_insitu_y2y_pct_captured_obs_centered'] = 100.0 * (1.0 - (numer_obs / denom))
        metrics['test_insitu_y2y_pct_captured_source_centered'] = 100.0 * (1.0 - (numer_source / denom))

    metrics['test_insitu_y2y_n'] = int(len(eval_df))
    metrics['test_insitu_y2y_groups'] = int(len(valid_groups))
    return metrics


def main():
    # perform analysis across all models in a dir, since these are now batched out
    base_dir = '/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_365'
    load_only = True
    model_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    relevant_cols = [
        'd_model', #'nhead', 'num_layers', 'dim_feedforward', 'dropout', 
        'learning_rate',
        'first_task_weight',
        'long_d_model', #'long_nhead', 'long_num_layers', 'long_dim_feedforward', 
        'long_out',
        'test_insitu_r2', 'test_insitu_rmse', 
        'test_insitu_r2_mean', 'test_insitu_r2_anom',
        'test_insitu_y2y_pct_captured_source_centered',
        'test_vv_r2', 'test_vh_r2',
    ]
    #sort_col = 'test_insitu_r2'
    sort_col = 'test_insitu_y2y_pct_captured_source_centered'
    if load_only:
        csv = pd.read_csv(
            os.path.join(
                base_dir, 'model_summary_results.csv'
            )
        )
        display_cols = [c for c in relevant_cols if c in csv.columns]
        csv_sub = csv[display_cols]
        csv_sub = csv_sub.sort_values(by=sort_col, ascending=True)
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
    first_task_weights = []
    model_r2s = []
    model_rmses = []
    model_nlls = []
    model_r2s_anom = []
    model_rmses_anom = []
    model_r2s_mean = []
    model_rmses_mean = []
    model_y2y_mae_obs_centered_abs = []
    model_y2y_mae_obs_centered_pct = []
    model_y2y_mae_source_centered_abs = []
    model_y2y_mae_source_centered_pct = []
    model_y2y_pct_captured_obs_centered = []
    model_y2y_pct_captured_source_centered = []
    model_y2y_n = []
    model_y2y_groups = []
    model_r2s_vv = []
    model_r2s_vh = []
    evaluated_model_dirs = []
    required_fold_files = [
        'val_info.csv',
        'val_outputs.pth',
        'test_info.csv',
        'test_outputs.pth'
    ]
    for m,model_dir in enumerate(model_dirs):
        model_name = model_dir.split('/')[-1]
        d_model = parse_model_name_value(model_name, 'transformer_dm', int)
        nhead = parse_model_name_value(model_name, 'nh', int)
        n_layer = parse_model_name_value(model_name, 'nl', int)
        dim_ff = parse_model_name_value(model_name, 'df', int)
        dropout = parse_model_name_value(model_name, 'do', float)
        batch_size = parse_model_name_value(model_name, 'bs', int)
        learning_rate = parse_model_name_value(model_name, 'lr', float)
        warmup = parse_model_name_value(model_name, 'warmup', int)
        weight_decay = parse_model_name_value(model_name, 'wd', float)
        insitu_obs = parse_model_name_value(model_name, 'iobs', int)
        vv_obs = parse_model_name_value(model_name, 'vvobs', int)
        vh_obs = parse_model_name_value(model_name, 'vhobs', int)
        long_d_model = parse_model_name_value(model_name, 'dmlong', int)
        long_nhead = parse_model_name_value(model_name, 'nhlong', int)
        long_n_layer = parse_model_name_value(model_name, 'nllong', int)
        long_dim_ff = parse_model_name_value(model_name, 'dflong', int)
        long_out = parse_model_name_value(model_name, 'outlong', int)
        first_task_weight = parse_first_task_weight(model_name)
        # get the folds
        fold_info_path = os.path.join(model_dir, 'fold_info.json')
        if not os.path.exists(fold_info_path):
            print(f'Skipping model {model_name}: missing fold_info.json (likely in progress).')
            continue
        with open(fold_info_path, 'r') as f:
            fold_info = json.load(f)
        folds = list(fold_info.keys())
        missing_artifacts = []
        for fold in folds:
            fold_dir = os.path.join(model_dir, f'fold_{fold}')
            for required_file in required_fold_files:
                required_path = os.path.join(fold_dir, required_file)
                if not os.path.exists(required_path):
                    missing_artifacts.append(required_path)
        if len(missing_artifacts) > 0:
            print(
                f'Skipping model {model_name}: missing fold outputs '
                f'({len(missing_artifacts)} files, likely in progress).'
            )
            continue
        all_val_preds_insitu = np.array([])
        all_val_true_insitu = np.array([])
        all_val_preds_vv = np.array([])
        all_val_true_vv = np.array([])
        all_val_preds_vh = np.array([])
        all_val_true_vh = np.array([])
        all_test_preds_insitu = np.array([])
        all_test_true_insitu = np.array([])
        all_test_preds_insitu_std = np.array([])
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
            if (test_preds_insitu_std != 0).all():
                test_insitu_nll = gaussian_nll(test_preds_insitu, test_true_insitu, (test_preds_insitu_std ** 2))
            else:
                test_insitu_nll = np.nan
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
                all_test_preds_insitu_std = np.concatenate((all_test_preds_insitu_std, test_preds_insitu_std))
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
        individual_test_insitu_err = np.abs(all_test_preds_insitu - all_test_true_insitu)
        overall_test_insitu_r2 = r2_score(all_test_true_insitu, all_test_preds_insitu)
        overall_test_insitu_n = len(all_test_true_insitu)
        overall_test_insitu_nll = np.mean(gaussian_nll(all_test_preds_insitu, all_test_true_insitu, (all_test_preds_insitu_std ** 2)))
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
        print(f'MAE: {overall_test_insitu_mae:.3f}, RMSE: {overall_test_insitu_rmse:.3f}, R2: {overall_test_insitu_r2:.3f}, NLL: {overall_test_insitu_nll:.3f}')
        #print('Overall Test Metrics - VV:')
        #print(f'MAE: {overall_test_vv_mae:.3f}, RMSE: {overall_test_vv_rmse:.3f}, R2: {overall_test_vv_r2:.3f}')
        #print('Overall Test Metrics - VH:')
        #print(f'MAE: {overall_test_vh_mae:.3f}, RMSE: {overall_test_vh_rmse:.3f}, R2: {overall_test_vh_r2:.3f}')
        # keep track of these across models
        model_r2s.append(overall_test_insitu_r2)
        model_rmses.append(overall_test_insitu_rmse)
        model_nlls.append(overall_test_insitu_nll)
        model_r2s_vv.append(overall_test_vv_r2)
        model_r2s_vh.append(overall_test_vh_r2)
        evaluated_model_dirs.append(model_dir)
        d_models.append(d_model)
        nheads.append(nhead)
        n_layers.append(n_layer)
        dim_ffs.append(dim_ff)
        dropouts.append(dropout)
        bss.append(batch_size)
        lrs.append(learning_rate)
        warmups.append(warmup)
        wds.append(weight_decay)
        iobs.append(insitu_obs)
        vvobs.append(vv_obs)
        vhobs.append(vh_obs)
        long_d_models.append(long_d_model)
        long_nheads.append(long_nhead)
        long_n_layers.append(long_n_layer)
        long_dim_ffs.append(long_dim_ff)
        long_outs.append(long_out)
        first_task_weights.append(first_task_weight)
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
        overall_test_insitu_std_plot_path = os.path.join(model_dir, 'overall_test_insitu_pred_obs_std_scatter.png')
        generic_hexbin(
            individual_test_insitu_err / all_test_true_insitu,
            all_test_preds_insitu_std / all_test_true_insitu,
            overall_test_insitu_std_plot_path,
            xlabel='Normalized error',
            ylabel='Normalized standard deviation',
            line_to_plot='one_to_one',
            xlim=[0,1],
            ylim=[0,1],
            #cbarlim=[0,350]
        )
        #generic_scatter(
        #    individual_test_insitu_err,
        #    all_test_preds_insitu_std,
        #    overall_test_insitu_std_plot_path,
        #    line_to_plot='correlation'
        #)
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
        y2y_metrics = compute_site_month_y2y_metrics(
            test_info_insitu=test_info_insitu,
            preds_insitu=all_test_preds_insitu,
            true_insitu=all_test_true_insitu
        )
        print('Overall Test Metrics - In Situ Year-to-Year Site-Month Variability:')
        print(
            f"Variability captured (source-centered, %): "
            f"{y2y_metrics['test_insitu_y2y_pct_captured_source_centered']:.3f}, "
            f"N: {y2y_metrics['test_insitu_y2y_n']}, "
            f"Groups: {y2y_metrics['test_insitu_y2y_groups']}"
        )
        model_y2y_mae_obs_centered_abs.append(y2y_metrics['test_insitu_y2y_mae_obs_centered_abs'])
        model_y2y_mae_obs_centered_pct.append(y2y_metrics['test_insitu_y2y_mae_obs_centered_pct'])
        model_y2y_mae_source_centered_abs.append(y2y_metrics['test_insitu_y2y_mae_source_centered_abs'])
        model_y2y_mae_source_centered_pct.append(y2y_metrics['test_insitu_y2y_mae_source_centered_pct'])
        model_y2y_pct_captured_obs_centered.append(y2y_metrics['test_insitu_y2y_pct_captured_obs_centered'])
        model_y2y_pct_captured_source_centered.append(y2y_metrics['test_insitu_y2y_pct_captured_source_centered'])
        model_y2y_n.append(y2y_metrics['test_insitu_y2y_n'])
        model_y2y_groups.append(y2y_metrics['test_insitu_y2y_groups'])
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
    if len(evaluated_model_dirs) == 0:
        print('No complete model directories found. Nothing to summarize.')
        return
    # summarize all model results
    summary_df = pd.DataFrame({
        'model_dir': evaluated_model_dirs,
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
        'first_task_weight': first_task_weights,
        'test_insitu_r2': model_r2s,
        'test_insitu_rmse': model_rmses,
        'test_insitu_nll': model_nlls,
        'test_insitu_r2_anom': model_r2s_anom,
        'test_insitu_rmse_anom': model_rmses_anom,
        'test_insitu_r2_mean': model_r2s_mean,
        'test_insitu_rmse_mean': model_rmses_mean,
        'test_insitu_y2y_mae_obs_centered_abs': model_y2y_mae_obs_centered_abs,
        'test_insitu_y2y_mae_obs_centered_pct': model_y2y_mae_obs_centered_pct,
        'test_insitu_y2y_mae_source_centered_abs': model_y2y_mae_source_centered_abs,
        'test_insitu_y2y_mae_source_centered_pct': model_y2y_mae_source_centered_pct,
        'test_insitu_y2y_pct_captured_obs_centered': model_y2y_pct_captured_obs_centered,
        'test_insitu_y2y_pct_captured_source_centered': model_y2y_pct_captured_source_centered,
        'test_insitu_y2y_n': model_y2y_n,
        'test_insitu_y2y_groups': model_y2y_groups,
        'test_vv_r2': model_r2s_vv,
        'test_vh_r2': model_r2s_vh
    })
    # save dataframe
    summary_csv_path = os.path.join(base_dir, 'model_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    # print this as a nice table
    # sort by lowest rmse (lowest @ top)
    summary_df = summary_df[relevant_cols]
    summary_df = summary_df.sort_values(by=sort_col, ascending=True)
    print('Model Summary Results:')
    console = Console()
    console.print(summary_df)

if __name__ == "__main__":
    main()
