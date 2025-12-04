import os
import glob
import torch
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))

from transformer_multitask_longclimate import LFMCTransformer



def eval_loss(
    model,loader,criterion,device
):
    loss = 0.0
    return loss

def permutation_importance(
    model,loader,device,feature_names
):
    pass

def main():
    # get the model
    model_out_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs/base'
    dm = 32
    nh = 1
    nl = 2
    df = 64
    do = 0.15
    bs = 128
    lr = 5e-4
    warmup = 554
    wd = 1e-4
    iobs = 33806
    vvobs = 0
    vhobs = 0
    dmlong = 256
    nhlong = 4
    nllong = 4
    dflong = 256
    outlong = 32
    model_name = (
        f'transformer_dm{dm}_nh{nh}_nl{nl}_df{df}_do{do}'
        f'_bs{bs}_lr{lr}_warmup{warmup}_wd{wd}'
        f'_iobs{iobs}_vvobs{vvobs}_vhobs{vhobs}'
        f'_dmlong{dmlong}_nhlong{nhlong}_nllong{nllong}'
        f'_dflong{dflong}_outlong{outlong}_basic'
    )
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
        long_output_dim=outlong,
        num_task_weights=1
    )
        

    
    
    
    
    # get the data loader

    # get the feature names

    # get the importance of each feature

    # print importance

if __name__ == "__main__":
    main()