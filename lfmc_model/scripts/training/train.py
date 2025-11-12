import os
import sys
import copy
import json
import shutil
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_model import LFMCTransformer
import plotting

import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",
    category=UserWarning,
)

def load_data(center_data_dir):
    # load all the center data
    X_daily = torch.load(
        os.path.join(center_data_dir, 'X_daily.pt'),
        weights_only=False
    )
    X_static = torch.load(
        os.path.join(center_data_dir, 'X_static.pt'),
        weights_only=False
    )
    Y_lfmc = torch.load(
        os.path.join(center_data_dir, 'Y.pt'),
        weights_only=False
    )
    # load the center info
    center_info = pd.read_csv(os.path.join(center_data_dir, 'info.csv'))
    all_center_data = [
        X_daily, X_static, Y_lfmc, center_info
    ]
    return all_center_data

class EarlyStopping:
    def __init__(
        self,
        patience=5,
        rmse_delta=0.001,
        pr_auc_delta=0.001,
        best_score_delta=0.0001
    ):
        self.patience = patience
        self.rmse_delta = rmse_delta
        self.best_rmse = float('inf')
        self.rmse_counter = 0
        self.early_stop = False
        self.save_model = False
        self.first_epoch = True  # to save the first epoch model

    def __call__(self,val_rmse):
        if self.first_epoch:
            print("First epoch, saving model")
            self.save_model = True
            self.first_epoch = False
            self.last_saved_rmse = val_rmse
        elif val_rmse < self.last_saved_rmse - self.rmse_delta:
            print("New best model found!")
            print(f"RMSE improved from {self.last_saved_rmse} to {val_rmse}")
            self.save_model = True
            self.last_saved_rmse = val_rmse
            self.rmse_counter = 0
        else:
            print("No improvement in model performance.")
            print(f"Current RMSE: {val_rmse}, Best RMSE: {self.last_saved_rmse}")
            self.save_model = False
            self.rmse_counter += 1
        print(
            f"EarlyStopping counter: rmse {self.rmse_counter}"
        )
        if self.rmse_counter >= self.patience:
            self.early_stop = True

def create_site_split(
    data_info: pd.DataFrame,
    desired_sample_size: int,
    seed=42,
    used_sites=None,
):
    # only create for in-situ sites
    data_info = data_info[data_info['source'] == 'nfmd']
    # use explicit date/latitude/longitude columns
    parts = data_info[["date", "latitude", "longitude"]].copy()
    parts = parts.rename(columns={"latitude": "lat", "longitude": "lon"})
    parts["lat"] = pd.to_numeric(parts["lat"], errors="coerce")
    parts["lon"] = pd.to_numeric(parts["lon"], errors="coerce")
    parts = parts.dropna(subset=["lat", "lon"])
    if parts.empty:
        raise ValueError("No valid lat/lon rows after parsing.")

    # count obs per site
    counts = (
        parts.groupby(["lat", "lon"])
        .size()
        .reset_index(name="n")
    )

    # filter out used_sites if provided
    if used_sites is not None and len(used_sites) > 0:
        counts = counts[
            ~counts.apply(
                lambda r: (float(r["lat"]), float(r["lon"])) in used_sites,
                axis=1
            )
        ].reset_index(drop=True)

    if counts.empty:
        return []

    # shuffle sites reproducibly
    rng = np.random.default_rng(seed)
    idx = np.arange(len(counts))
    rng.shuffle(idx)
    counts = counts.iloc[idx].reset_index(drop=True)

    # accumulate sites until we hit desired total obs
    val_locs = []
    total = 0
    for _, row in counts.iterrows():
        val_locs.append((float(row["lat"]), float(row["lon"])))
        total += int(row["n"])
        if total >= desired_sample_size:
            break

    # if we couldn't meet desired_sample_size, just return all
    if not val_locs:
        return []
    return val_locs

def run_model(
    model,
    loader,
    device,
    loss_fn=None,
    train_model=False,
    optimizer=None,
    warmup_steps=0,
    global_step=0,
    warmup_start_lr=None,
    warmup_end_lr=None
):
    pbar = tqdm.tqdm(
        loader,
        desc='Batch'
    )
    # tracking paraphanalia
    n_samples = 0.0
    running_loss = 0.0
    preds = []
    for Xd_b,Xs_b,Y_b in pbar:
        # move data to device
        Xd_b = Xd_b.to(device=device, dtype=torch.float32)
        Xs_b = Xs_b.to(device=device, dtype=torch.float32)
        Y_b = Y_b.to(device=device, dtype=torch.float32)
        if train_model:
            this_preds = model(Xd_b, Xs_b)
        else:
            with torch.no_grad():
                this_preds = model(Xd_b, Xs_b)
        preds.append(this_preds)
        if loss_fn is not None:
            loss = loss_fn(this_preds, Y_b)
            running_loss += loss.item()
            n_samples += Y_b.size(0)
        if train_model:
            if global_step < warmup_steps:
                this_t = global_step / warmup_steps
                lr = warmup_start_lr * ((warmup_end_lr / warmup_start_lr) ** this_t)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
    # calculate running loss
    if loss_fn is not None and n_samples > 0:
        running_loss /= n_samples
    else:
        running_loss = None
    if len(preds) > 0:
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    else:
        preds = None
    return(
        model, running_loss, preds, global_step
    )

def train_fold_k(
    model,
    save_dir,
    data,
    fold_test_locs,
    var_names,
    device,
    optimizer,
    scheduler,
    early_stopping,
    batch_size,
    max_epochs,
    warmup_steps,
    warmup_start_lr,
    val_split,
    plot_distributions=False
):
    this_fold_num = fold_test_locs[0]
    this_locs = np.array(fold_test_locs[1])
    fold_save_dir = os.path.join(
        save_dir,
        f'fold_{this_fold_num}'
    )
    if not os.path.exists(fold_save_dir):
        os.makedirs(fold_save_dir)
    # split out the test data
    daily_data = data[0]
    static_data = data[1]
    lfmc_insitu = data[2]
    info = data[3]
    parts = info[['date', 'latitude', 'longitude']].copy()
    parts = parts.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    idx = []
    fold_test_lats = this_locs[:, 0]
    fold_test_lons = this_locs[:, 1]
    for i in range(len(parts)):
        this_lat = float(parts.iloc[i]['lat'])
        this_lon = float(parts.iloc[i]['lon'])
        if this_lat in fold_test_lats and this_lon in fold_test_lons:
            idx.append(i)
    test_data_mask = np.array([False]*len(info))
    test_data_mask[idx] = True
    test_daily_data = daily_data[test_data_mask]
    test_static_data = static_data[test_data_mask]
    test_lfmc_insitu = lfmc_insitu[test_data_mask]
    test_info = info[test_data_mask]
    remaining_daily_data = daily_data[~test_data_mask]
    remaining_static_data = static_data[~test_data_mask]
    remaining_lfmc_insitu = lfmc_insitu[~test_data_mask]
    remaining_info = info[~test_data_mask]
    # split out the validation data from the remaining data
    # select sites to remove until we reach the val_split proportion
    total_obs = len(remaining_lfmc_insitu)
    desired_val_obs = int(total_obs * val_split)
    val_locs = create_site_split(
        remaining_info,
        desired_val_obs
    )
    val_locs = np.array(val_locs)
    val_lats = val_locs[:,0]
    val_lons = val_locs[:,1]
    # --- 1) Parse lat/lon from remaining_info['day_lat_lon'] ---
    # day_lat_lon format: 'YYYY-MM-DD_<lat>_<lon>'
    parts = remaining_info[['date', 'latitude', 'longitude']].copy()
    parts = parts.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    # columns: [date, lat, lon]
    info_lats = parts['lat'].astype(float).to_numpy()
    info_lons = parts['lon'].astype(float).to_numpy()
    # --- 2) Build (N,2) and (M,2) pairs on the same device/dtype as your tensors ---
    # choose a reference tensor to inherit device/dtype
    dtype = remaining_daily_data.dtype
    info_pairs = torch.tensor(
        np.column_stack([info_lats, info_lons]),
        dtype=dtype, device=device
    )  # (N, 2)
    val_pairs = torch.tensor(
        np.column_stack([val_lats, val_lons]),
        dtype=dtype, device=device
    ).reshape(-1, 2)  # (M, 2)
    # --- 3) Membership mask: info (lat,lon) belongs to ANY of the given val pairs ---
    # use small atol to avoid float jitter
    atol = torch.tensor(1e-6, dtype=dtype, device=device)
    # broadcast compare: (N,1,2) vs (1,M,2) -> (N,M,2)
    diff = (info_pairs[:, None, :] - val_pairs[None, :, :]).abs()
    match_nm2 = diff <= atol
    # row matches any val pair if BOTH lat and lon match for some column m
    val_mask = match_nm2.all(dim=2).any(dim=1)  # (N,) bool
    val_mask_cpu = val_mask.detach().cpu().numpy()
    # --- 4) Split remaining_info (DataFrame) ---
    val_info   = remaining_info.iloc[val_mask.detach().cpu().numpy()]
    train_info = remaining_info.iloc[(~val_mask).detach().cpu().numpy()]
    # --- 5) Split all aligned tensors ---
    # If you have multiple tensors, put them in a dict for convenience:
    tensors = {
        "daily":  remaining_daily_data,
        "static": remaining_static_data,
        "lfmc":   remaining_lfmc_insitu,
        # add more here if needed, all must be (N, ...) and aligned to remaining_info
    }
    val_tensors   = {k: v[val_mask_cpu]   for k, v in tensors.items()}
    train_tensors = {k: v[~val_mask_cpu]  for k, v in tensors.items()}
    # (Optionally expose individual variables as before)
    val_daily_data   = val_tensors["daily"]
    val_static_data  = val_tensors["static"]
    val_lfmc_insitu  = val_tensors["lfmc"]
    train_daily_data  = train_tensors["daily"]
    train_static_data = train_tensors["static"]
    train_lfmc_insitu = train_tensors["lfmc"]
    train_lfmc_insitu = train_lfmc_insitu.squeeze()
    val_lfmc_insitu = val_lfmc_insitu.squeeze()
    test_lfmc_insitu = test_lfmc_insitu.squeeze()
    # --- 6) Quick sanity print ---
    print(f"Total rows: {len(info)} | "
          f"Test: {int(test_data_mask.sum())} | "
          f"Val: {int(val_mask.sum())} | "
          f"Train: {int((~val_mask).sum())}")
    if plot_distributions:
        print('Plotting feature distributions for train/val/test splits')
        plot_save_dir = os.path.join(fold_save_dir, 'plots')
        daily_to_plot = [
            train_daily_data,
            val_daily_data,
            test_daily_data
        ]
        static_to_plot = [
            train_static_data,
            val_static_data,
            test_static_data
        ]
        lfmc_to_plot = [
            train_lfmc_insitu,
            val_lfmc_insitu,
            test_lfmc_insitu
        ]
        plot_feature_distributions(
            daily_to_plot,
            static_to_plot,
            lfmc_to_plot,
            var_names,
            plot_save_dir
        )
    print('Normalizing the data')
    train_daily_mean = np.nanmean(train_daily_data, axis=(0,1))
    train_daily_std = np.nanstd(train_daily_data, axis=(0,1))
    train_static_mean = np.nanmean(train_static_data, axis=(0,1))
    train_static_std = np.nanstd(train_static_data, axis=(0,1))
    y_mean = np.nanmean(train_lfmc_insitu)
    y_std = np.nanstd(train_lfmc_insitu)
    for v,var in enumerate(var_names['daily_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var or
            'zone' in var
        ):
            continue
        train_daily_data[:,:,v] = (train_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
        val_daily_data[:,:,v] = (val_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
        test_daily_data[:,:,v] = (test_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
    for v,var in enumerate(var_names['static_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var or
            'zone' in var
        ):
            continue
        train_static_data[:,:,v] = (train_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        val_static_data[:,:,v] = (val_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        test_static_data[:,:,v] = (test_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
    train_lfmc_insitu = (train_lfmc_insitu - y_mean) / y_std
    val_lfmc_insitu = (val_lfmc_insitu - y_mean) / y_std
    test_lfmc_insitu = (test_lfmc_insitu - y_mean) / y_std
    # save the normalization parameters for later use
    norm_params = {
        'train_daily_mean': train_daily_mean.tolist(),
        'train_daily_std': train_daily_std.tolist(),
        'train_static_mean': train_static_mean.tolist(),
        'train_static_std': train_static_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist()
    }
    # save the normalization parameters to disk
    with open(os.path.join(fold_save_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f)
    # create the datasets and dataloaders
    train_dataset = TensorDataset(
        train_daily_data,
        train_static_data,
        train_lfmc_insitu
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_dataset = TensorDataset(
        val_daily_data,
        val_static_data,
        val_lfmc_insitu
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_dataset = TensorDataset(
        test_daily_data,
        test_static_data,
        test_lfmc_insitu
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    # set up the loss functions
    criterion = nn.MSELoss()
    # make sure that we have the warmup end lr
    warmup_end_lr = optimizer.param_groups[0]['lr']
    # set up the things that we need to track
    train_loss = []
    train_loss_d = []
    train_loss_p = []
    val_loss = []
    val_loss_d = []
    val_loss_p = []
    global_step = 0
    for epoch in range(1,max_epochs):
        print(f'Fold {this_fold_num}, Epoch {epoch}/{max_epochs}')
        model.train()
        (
            model,
            this_train_loss,
            this_train_preds,
            global_step
        ) = run_model(
            model,
            train_loader,
            device,
            criterion,
            train_model=True,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            global_step=global_step,
            warmup_start_lr=warmup_start_lr,
            warmup_end_lr=warmup_end_lr
        )
        train_loss.append(this_train_loss)
        print(f'Training loss: {this_train_loss:.4f}')
        scheduler.step()
        # run the validation
        model.eval()
        (
            model,
            this_val_loss,
            this_val_preds,
            _
        ) = run_model(
            model,
            val_loader,
            device,
            criterion,
            train_model=False
        )
        val_loss.append(this_val_loss)
        print(f'Validation loss: {this_val_loss:.4f}')
        # calculate metrics of interes
        # MAE
        # denorm our preds and truths
        this_val_preds_denorm = this_val_preds * y_std + y_mean
        val_lfmc_insitu_denorm = val_lfmc_insitu.numpy() * y_std + y_mean
        val_mae = np.mean(np.abs(this_val_preds_denorm - val_lfmc_insitu_denorm))
        print(f'Validation MAE: {val_mae:.4f}')
        # R2
        val_r2 = r2_score(val_lfmc_insitu_denorm, this_val_preds_denorm)
        print(f'Validation R2: {val_r2:.4f}')
        val_rmse = np.sqrt(np.mean((this_val_preds_denorm - val_lfmc_insitu_denorm) ** 2))
        print(f'Validation RMSE: {val_rmse:.4f}')
        # check early stopping
        early_stopping(val_rmse)
        if early_stopping.save_model:
            print('New best model, saving...')
            model_save_path = os.path.join(
                fold_save_dir,
                f'model_epoch{epoch}.pt'
            )
            torch.save(model.state_dict(), model_save_path)
            best_epoch = copy.deepcopy(epoch)
        if early_stopping.early_stop:
            print('Early stopping triggered, ending training')
            break
    # re-load the best model and compute test statistics
    print('Training Complete, loading best model for testing')
    state = torch.load(
        os.path.join(
            fold_save_dir,
            f'model_epoch{best_epoch}.pt'
        ),
        weights_only=False,
        map_location=device
        
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    # re-run val so we save the best metrics
    print('re-running val with best model')
    (
        model,
        val_loss,
        val_preds,
        _
    ) = run_model(
        model,
        val_loader,
        device,
        criterion,
        train_model=False
    )
    # denorm
    val_preds_denorm = val_preds * y_std + y_mean
    val_lfmc_insitu_denorm = val_lfmc_insitu.numpy() * y_std + y_mean
    val_mae = np.mean(np.abs(val_preds_denorm - val_lfmc_insitu_denorm))
    val_r2 = r2_score(val_lfmc_insitu_denorm, val_preds_denorm)
    val_rmse = np.sqrt(np.mean((val_preds_denorm - val_lfmc_insitu_denorm) ** 2))
    print(f'Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation R2: {val_r2:.4f}, Validation RMSE: {val_rmse:.4f}')
    # run the test
    print('running test with best model')
    (
        _,
        test_loss,
        test_preds,
        _
    ) = run_model(
        model,
        test_loader,
        device,
        criterion,
        train_model=False
    )
    # denorm
    if test_preds is None:
        test_preds_denorm = np.array([])
        test_lfmc_insitu_denorm = np.array([])
        test_mae = np.nan
        test_r2 = np.nan
        test_rmse = np.nan
    else:
        test_preds_denorm = test_preds * y_std + y_mean
        test_lfmc_insitu_denorm = test_lfmc_insitu.numpy() * y_std + y_mean
        test_mae = np.mean(np.abs(test_preds_denorm - test_lfmc_insitu_denorm))
        test_r2 = r2_score(test_lfmc_insitu_denorm, test_preds_denorm)
        test_rmse = np.sqrt(np.mean((test_preds_denorm - test_lfmc_insitu_denorm) ** 2))
        print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}')
    # save the outputs
    torch.save(
        {
            'loss':train_loss
        },
        os.path.join(fold_save_dir,'train_outputs.pth')
    )
    torch.save(
        {
            'loss':val_loss,
            'preds':val_preds_denorm,
            'true':val_lfmc_insitu_denorm
        },
        os.path.join(fold_save_dir,'val_outputs.pth')
    )
    if test_preds is not None:
        torch.save(
            {
                'loss':test_loss,
                'preds':test_preds_denorm,
                'true':test_lfmc_insitu_denorm
            },
            os.path.join(fold_save_dir,'test_outputs.pth')
        )
    train_info.to_csv(
        os.path.join(fold_save_dir,'train_info.csv'),
        index=False
    )
    val_info.to_csv(
        os.path.join(fold_save_dir,'val_info.csv'),
        index=False
    )
    test_info.to_csv(
        os.path.join(fold_save_dir,'test_info.csv'),
        index=False
    )


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    # configs
    # directories, etc.
    input_data_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs_30daysweather_nokrishnastats'
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs'
    # training settings
    batch_size = 128
    max_epochs = 100
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available, using CPU. This will be slow!')
    warmup_steps = 250
    base_lr = lr
    warmup_start_lr = 1e-6
    val_split = 0.2
    adam_weight_decay = 1e-4
    patience = 8
    # model hyperparameters (go back to the github and get what I deleted here)
    d_model = 16
    nhead = 1
    num_layers = 2
    dim_feedforward = 32
    dropout = 0.2
    # load the data
    datasets = load_data(input_data_dir)
    var_names = json.load(
        open(os.path.join(input_data_dir, 'var_names.json'), 'r')
    )
    # get the input dims that we are working with to build the model
    short_input_dim = datasets[0].shape[-1]
    static_input_dim = datasets[1].shape[-1]
    # set up the save directories
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}_nostats_forreal'
    )
    full_save_dir = os.path.join(save_dir, this_model_name)
    #if os.path.exists(full_save_dir):
    #    print(
    #        f"WARNING: {full_save_dir} already exists "
    #        "and will be overwritten!"
    #    )
    #    resp = input("Do you want to continue? [y/N]: ").strip().lower()
    #    if resp == "y":
    #        shutil.rmtree(full_save_dir)
    #    else:
    #        print("Aborting to avoid overwrite.")
    #        sys.exit(1)
    if os.path.exists(full_save_dir):
        shutil.rmtree(full_save_dir)
    os.makedirs(full_save_dir)
    # build the folds by location
    daily_data = datasets[0]
    info = datasets[3]
    n_folds = 10
    total_obs = daily_data.shape[0]
    desired_obs_per_fold = total_obs / n_folds
    fold_locs = {}
    used_sites = []
    for fold in range(n_folds):
        this_locs = create_site_split(
            info,
            desired_obs_per_fold,
            used_sites=used_sites
        )
        fold_locs[fold + 1] = this_locs
        used_sites.extend(this_locs)
    # save this fold info
    with open(os.path.join(full_save_dir, 'fold_info.json'), 'w') as f:
        json.dump(fold_locs, f)
    # train this fold
    for fold, locs in enumerate(fold_locs.items()):
        print(f'Training fold {fold+1}/{n_folds} with {len(locs[1])} locations held out for testing')
        # build the model
        model = LFMCTransformer(
            short_input_dim=short_input_dim,
            static_input_dim=static_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=2
        ).to(device)
        # build the optimizer
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                ('bias' in name) or
                ('norm' in name.lower()) or
                ('bn' in name.lower())
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': adam_weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}
            ],
            lr=lr
        )
        # build the scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        # build early stopping
        early_stopping = EarlyStopping(patience=patience)
        # train on this fold
        train_fold_k(
            model,
            full_save_dir,
            datasets,
            locs,
            var_names,
            device,
            optimizer,
            scheduler,
            early_stopping,
            batch_size,
            max_epochs,
            warmup_steps,
            warmup_start_lr,
            val_split
        )
    # one final version of the model trained on all the data
    print('Training final model on all data')
    model = LFMCTransformer(
        short_input_dim=short_input_dim,
        static_input_dim=static_input_dim,
        d_model=d_model,   
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=2
    ).to(device)
    # build the optimizer
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            ('bias' in name) or
            ('norm' in name.lower()) or
            ('bn' in name.lower())
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': adam_weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ],
        lr=lr
    )
    # build the scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=25,
        eta_min=1e-6
    )
    # build early stopping
    early_stopping = EarlyStopping(patience=patience)
    # train on this fold
    locs = (9998, [(-9998.0,-9998.0)])  # dummy value to indicate training on all data
    train_fold_k(
        model,
        full_save_dir,
        datasets,
        locs,
        var_names,
        device,
        optimizer,
        scheduler,
        early_stopping,
        batch_size,
        max_epochs,
        warmup_steps,
        warmup_start_lr,
        val_split
    )

            

if __name__ == "__main__":
    main()