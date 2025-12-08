import os
import glob
import torch
import numpy as np
import sys
import json
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','scripts','training'))

from transformer_multitask_longclimate import LFMCTransformer
from train_multitarget_longweather_vvvh import load_data, GaussianNLLLoss, run_model

def make_permuted_loader(
    base_loader,feature_idx,pos_in_loader=2
):
    base_ds = base_loader.dataset
    tensors = list(base_ds.tensors)
    x_static = tensors[pos_in_loader]
    N = x_static.shape[0]
    perm = torch.randperm(N)
    x_static_perm = x_static.clone()
    x_static_perm[:, :, feature_idx] = x_static_perm[perm][:, :, feature_idx]
    tensors[pos_in_loader] = x_static_perm
    permuted_ds = TensorDataset(*tensors)
    permuted_loader = DataLoader(
        permuted_ds,
        batch_size=base_loader.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return permuted_loader


def permutation_importance(
    model,loader,device,feature_names,criterion
):
    # get the model loss on the original data
    (
        _,
        original_loss,
        original_loss_insitu,
        original_loss_vv,
        original_loss_vh,
        _,_,_,_,_,_,_,_,_,_
    ) = run_model(
        model,
        loader,
        device,
        criterion,
        num_tasks=1,
        task_weight_type='default'
    )
    print(f'Original Loss: {original_loss}')
    print(f'Original Insitu Loss: {original_loss_insitu}')
    print(f'Original VV Loss: {original_loss_vv}')
    print(f'Original VH Loss: {original_loss_vh}')
    importance = {}
    for name in feature_names:
        importance[name] = {
            'loss_increase': 0.0,
            'insitu_loss_increase': 0.0,
        }
    for f,feature in enumerate(feature_names):
        print(f'Permuting feature {feature} ({f+1}/{len(feature_names)})')
        perm_loader = make_permuted_loader(
            loader,feature_idx=f,pos_in_loader=2
        )
        (
            _,
            permuted_loss,
            permuted_loss_insitu,
            _,
            _,
            _,_,_,_,_,_,_,_,_,_
        ) = run_model(
            model,
            perm_loader,
            device,
            criterion,
            num_tasks=1,
            task_weight_type='default'
        )
        loss_increase = permuted_loss - original_loss
        insitu_loss_increase = permuted_loss_insitu - original_loss_insitu
        percent_loss_increase = (loss_increase / original_loss) * 100.0
        percent_insitu_loss_increase = (insitu_loss_increase / original_loss_insitu) * 100.0
        importance[feature]['loss_increase'] = loss_increase
        importance[feature]['insitu_loss_increase'] = insitu_loss_increase
        importance[feature]['percent_loss_increase'] = percent_loss_increase
        importance[feature]['percent_insitu_loss_increase'] = percent_insitu_loss_increase
        print(
            f'Feature: {feature}, '
            f'Percent Loss Increase: {percent_loss_increase}, '
            f'Percent Insitu Loss Increase: {percent_insitu_loss_increase}'
        )
    return importance

def main():
    # data settings
    data_dir = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs_sarstats_onlyandminimal'
    )
    # model settings
    model_out_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/sarstats_onlyandminimal'
    dm = 64
    nh = 2
    nl = 3
    df = 128
    do = 0.15
    bs = 128
    lr = 1e-4
    warmup = 554
    wd = 1e-4
    iobs = 33806
    vvobs = 0
    vhobs = 0
    dmlong = 256
    nhlong = 8
    nllong = 5
    dflong = 512
    outlong = 64
    model_name = (
        f'transformer_dm{dm}_nh{nh}_nl{nl}_df{df}_do{do}'
        f'_bs{bs}_lr{lr}_warmup{warmup}_wd{wd}'
        f'_iobs{iobs}_vvobs{vvobs}_vhobs{vhobs}'
        f'_dmlong{dmlong}_nhlong{nhlong}_nllong{nllong}'
        f'_dflong{dflong}_outlong{outlong}_basic'
    )
    # if the anaysis has already been run, just load it, print it, and go
    load_csv = False
    out_df_path = os.path.join(
        model_out_dir,
        "feature_importance.csv"
    )
    if load_csv:
        importance_df = pd.read_csv(out_df_path)
        print(importance_df)
        return
    # load datasets
    datasets = load_data(data_dir)
    # load var names
    with open(os.path.join(data_dir,'var_names.json')) as f:
        var_names = json.load(f)
    # load normalization stat
    norm_path = os.path.join(
        data_dir,
        model_out_dir,
        model_name,
        'fold_9998',
        'norm_params.json'
    )
    with open(norm_path) as f:
        norm_params = json.load(f)
    # normalize short data
    idx = 0
    for v,var in enumerate(var_names['short_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var or
            'zone' in var or
            'barren' in var or
            'crops' in var or
            'forest' in var or
            'developed' in var or
            'grass' in var or
            'other' in var or
            'shrub' in var or
            'water' in var or
            'wetlands' in var
        ):
            continue
        mean = norm_params['train_short_mean'][idx]
        std = norm_params['train_short_std'][idx]
        datasets[0][:, :, v] = (datasets[0][:, :, v] - mean) / std
        idx += 1
    # normalize long data
    idx = 0
    for v,var in enumerate(var_names['long_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var or
            'zone' in var or
            'barren' in var or
            'crops' in var or
            'forest' in var or
            'developed' in var or
            'grass' in var or
            'other' in var or
            'shrub' in var or
            'water' in var or
            'wetlands' in var
        ):
            continue
        mean = norm_params['train_long_mean'][idx]
        std = norm_params['train_long_std'][idx]
        datasets[1][:, :, v] = (datasets[1][:, :, v] - mean) / std
        idx += 1
    # normalize static data
    idx = 0
    for v,var in enumerate(var_names['static_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var or
            'zone' in var or
            'barren' in var or
            'crops' in var or
            'forest' in var or
            'developed' in var or
            'grass' in var or
            'other' in var or
            'shrub' in var or
            'water' in var or
            'wetlands' in var
        ):
            continue
        mean = norm_params['train_static_mean'][idx]
        std = norm_params['train_static_std'][idx]
        if mean == 0:
            idx += 1
            continue
        if std == 0 or np.isnan(std) or np.isinf(std):
            print(norm_params)
            sys.exit()
            raise ValueError(f'Standard deviation is {std} for static var {var}')
        if np.isnan(mean) or np.isinf(mean):
            raise ValueError(f'Mean is {mean} for static var {var}')
        datasets[2][:, :, v] = (datasets[2][:, :, v] - mean) / std
        idx += 1
    # normalize y data
    Y = datasets[3]
    source = datasets[4]
    lfmc_mean = norm_params['lfmc_mean']
    lfmc_std = norm_params['lfmc_std']
    vv_mean = norm_params['vv_mean']
    vv_std = norm_params['vv_std']
    vh_mean = norm_params['vh_mean']
    vh_std = norm_params['vh_std']
    Y[source == 0] = (Y[source == 0] - lfmc_mean) / lfmc_std
    Y[source == 1] = (Y[source == 1] - vv_mean) / vv_std
    Y[source == 2] = (Y[source == 2] - vh_mean) / vh_std
    datasets[3] = Y
    # create dataloaders    
    dataset = TensorDataset(
        datasets[0],
        datasets[1],
        datasets[2],
        datasets[3],
        datasets[4]
    )
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
    )
    # load the model
    short_input_dim = datasets[0].shape[-1]
    long_input_dim = datasets[1].shape[-1]
    static_input_dim = datasets[2].shape[-1]
    saved_models = glob.glob(
        os.path.join(
            model_out_dir,
            model_name,
            'fold_9998',
            'model_epoch*.pt'
        )
    )
    saved_models.sort()
    latest_model_path = saved_models[-1]
    model = LFMCTransformer(
        short_input_dim=short_input_dim,
        long_input_dim=long_input_dim,
        static_input_dim=static_input_dim,
        d_model=dm,
        nhead=nh,
        num_layers=nl,
        dim_feedforward=df,
        dropout=do,
        num_queries=2,
        long_d_model=dmlong,
        long_nhead=nhlong,
        long_num_layers=nllong,
        long_dim_feedforward=dflong,
        long_out_dim=outlong,
        num_task_weights=1
    )
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()
    # get our device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # get the loss that we are going to use to evaluate sensitivity
    criterion = GaussianNLLLoss(reduction='mean')
    # get the importance of each feature
    static_vars = var_names['static_vars']
    # get rid of the features that don't exist
    importance = permutation_importance(
        model,loader,device,static_vars,criterion
    )
    importance_df = pd.DataFrame(importance).T
    importance_df.to_csv(out_df_path)
    # sort by overall importance and print nicely
    importance_df = importance_df.sort_values(
        by='percent_loss_increase',
        ascending=False
    )
    print(importance_df)

if __name__ == "__main__":
    main()