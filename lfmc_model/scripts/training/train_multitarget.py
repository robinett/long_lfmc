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
from sklearn.neighbors import BallTree
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_model import LFMCTransformer
from transformer_model_multitask import LFMCTransformer as LFMCTransformerMultiTask
import plotting

import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",
    category=UserWarning,
)

import numpy as np
import torch
from sklearn.neighbors import BallTree  # pip install scikit-learn


import numpy as np

def fuse_gaussians(mu_i, logv_i, mu_r, logv_r,
                      min_var=1e-6, max_logv=10.0,
                      w_i=1.0, w_r=1.0):
    """
    Precision-weighted fusion of two Gaussian heads.
    Inputs are NumPy arrays (same shape).
      mu_i, logv_i: in-situ head mean/log-var
      mu_r, logv_r: RS head mean/log-var
    w_i, w_r: optional precision scalers (<=1 to downweight)
    Returns:
      mu_star, logv_star (NumPy arrays)
    """
    # clamp log-variance for stability
    logv_i = np.minimum(logv_i, max_logv)
    logv_r = np.minimum(logv_r, max_logv)

    v_i = np.maximum(np.exp(logv_i), min_var)
    v_r = np.maximum(np.exp(logv_r), min_var)

    tau_i = (w_i / v_i)
    tau_r = (w_r / v_r)
    tau_sum = tau_i + tau_r

    mu_star = (tau_i * mu_i + tau_r * mu_r) / tau_sum
    v_star  = 1.0 / tau_sum
    logv_star = np.log(v_star)
    return mu_star, logv_star


def mask_location_data_fast(
    keep_locs,            # np.array (K,2) [lat, lon] in degrees
    daily_data,           # torch.Tensor [N, ...]
    static_data,          # torch.Tensor [N, ...]
    lfmc_insitu,          # torch.Tensor [N, ...]
    source,               # torch.Tensor [N]
    info,                 # pd.DataFrame with 'latitude','longitude'
    masking_radius_m=600.0,
    match_tolerance_m=5.0,   # how close is "same location"
    round_decimals=6,        # snap to grid to avoid FP drift
):
    # Edge cases
    if len(info) == 0 or len(keep_locs) == 0:
        # nothing to keep or nothing to act on → everything is remaining
        empty = slice(0, 0)
        return (daily_data[empty], static_data[empty], lfmc_insitu[empty], source[empty], info.iloc[:0],
                daily_data, static_data, lfmc_insitu, source, info)

    # Prepare coordinates
    all_lats = np.asarray(info["latitude"], dtype=float)
    all_lons = np.asarray(info["longitude"], dtype=float)

    keep_locs = np.asarray(keep_locs, dtype=float)
    keep_locs = np.round(keep_locs, round_decimals)
    keep_locs = np.unique(keep_locs, axis=0)  # dedup

    rad = np.deg2rad
    X_points = np.c_[rad(all_lats), rad(all_lons)]
    X_keep   = np.c_[rad(keep_locs[:, 0]), rad(keep_locs[:, 1])]

    # BallTree on keep locations with haversine metric
    R_earth_m = 6_371_000.0
    radius_rad   = masking_radius_m / R_earth_m
    match_rad    = max(match_tolerance_m / R_earth_m, 1e-12)  # tiny > 0
    tree = BallTree(X_keep, metric="haversine")

    # Points exactly at keep locations (within small tolerance) → KEPT
    kept_counts = tree.query_radius(X_points, r=match_rad, count_only=True)
    mask_kept = kept_counts > 0

    # Points within masking radius of any keep location (but not exact) → DELETE
    neigh_counts = tree.query_radius(X_points, r=radius_rad, count_only=True)
    within_radius = neigh_counts > 0
    mask_delete = within_radius & ~mask_kept

    # Remaining = not kept and not deleted
    mask_remaining = ~(mask_kept | mask_delete)

    # To tensors
    m_kept = torch.from_numpy(mask_kept)
    m_rem  = torch.from_numpy(mask_remaining)

    # Slice outputs
    kept_daily_data   = daily_data[m_kept]
    kept_static_data  = static_data[m_kept]
    kept_lfmc_insitu  = lfmc_insitu[m_kept]
    kept_source       = source[m_kept]
    kept_info         = info.loc[mask_kept]

    remaining_daily_data   = daily_data[m_rem]
    remaining_static_data  = static_data[m_rem]
    remaining_lfmc_insitu  = lfmc_insitu[m_rem]
    remaining_source       = source[m_rem]
    remaining_info         = info.loc[mask_remaining]

    return (
        kept_daily_data, kept_static_data, kept_lfmc_insitu, kept_source, kept_info,
        remaining_daily_data, remaining_static_data, remaining_lfmc_insitu, remaining_source, remaining_info
    )


def mask_location_data(
    test_locs,
    daily_data,
    static_data,
    lfmc_insitu,
    source,
    info,
    rs_masking_radius_m=600.0
):
    all_lats = info["latitude"].astype(float).to_numpy()
    all_lons = info["longitude"].astype(float).to_numpy()
    # ensure source is a numpy array of ints
    source_np = source.numpy().astype(int)
    fold_test_lats = np.asarray(test_locs[:,0])
    fold_test_lons = np.asarray(test_locs[:,1])
    # radius of the earth in meters
    R = 6371000.0
    # check points within 250m
    lat1 = np.deg2rad(all_lats)[:, None]
    lon1 = np.deg2rad(all_lons)[:, None]
    lat2 = np.deg2rad(fold_test_lats)[None, :]
    lon2 = np.deg2rad(fold_test_lons)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    dist_m = R * c
    min_dist_m = np.min(dist_m, axis=1)
    within_masking_dist = min_dist_m <= rs_masking_radius_m
    # build indices
    test_idx = np.nonzero(within_masking_dist & (source_np == 1))[0].tolist()
    delete_idx = np.nonzero(within_masking_dist & (source_np == 0))[0].tolist()
    N = len(info)
    mask_test = np.zeros(N, dtype=bool)
    mask_delete = np.zeros(N, dtype=bool)
    mask_test[test_idx] = True
    mask_delete[delete_idx] = True
    mask_keep = ~(mask_test | mask_delete)
    m_test = torch.as_tensor(
        mask_test,
        dtype=torch.bool,
    )
    m_keep = torch.as_tensor(
        mask_keep,
        dtype=torch.bool,
    )
    test_daily_data = daily_data[m_test]
    test_static_data = static_data[m_test]
    test_lfmc_insitu = lfmc_insitu[m_test]
    test_source = source[m_test]
    test_info = info.iloc[mask_test]
    remaining_daily_data = daily_data[m_keep]
    remaining_static_data = static_data[m_keep]
    remaining_lfmc_insitu = lfmc_insitu[m_keep]
    remaining_source = source[m_keep]
    remaining_info = info.iloc[mask_keep]
    return (
        test_daily_data, test_static_data, test_lfmc_insitu, test_source, test_info,
        remaining_daily_data, remaining_static_data, remaining_lfmc_insitu, remaining_source, remaining_info
    )

class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        """
        Gaussian Negative Log-Likelihood loss.

        Args:
            reduction: 'mean' | 'sum' | 'none'
            eps: floor for variance to avoid div by 0
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.eps = eps
        self.log2pi = math.log(2.0 * math.pi)

    def forward(self, mu, log_var, target, mask=None):
        """
        Args:
            mu: (B,) predicted mean
            log_var: (B,) predicted log variance
            target: (B,) ground truth
            mask: optional boolean mask for valid entries
        """
        if mask is None:
            mask = torch.isfinite(target)

        mu = mu[mask]
        log_var = log_var[mask]
        target = target[mask]

        if target.numel() == 0:
            # no valid targets
            return mu.new_tensor(0.0, requires_grad=True)

        var = torch.exp(log_var).clamp_min(self.eps)
        nll = 0.5 * ((target - mu) ** 2 / var + log_var + self.log2pi)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:  # 'none'
            return nll

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
    source = torch.load(
        os.path.join(center_data_dir, 'source.pt'),
        weights_only=False
    )
    # load the center info
    center_info = pd.read_csv(os.path.join(center_data_dir, 'info.csv'))
    all_center_data = [
        X_daily, X_static, Y_lfmc, source, center_info
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
    desired_insitu_sample_size: int,
    desired_rs_sample_size: int,
    seed: int = 42,
    used_sites=None,
    round_decimals: int = 6,
):
    # split sources
    insitu = data_info[data_info['source'] == 'nfmd']
    rs = data_info[data_info['source'] == 'rs']

    # clean/standardize lat/lon once
    def clean(df):
        out = df[['date', 'latitude', 'longitude']].copy()
        out = out.rename(columns={'latitude': 'lat',
                                  'longitude': 'lon'})
        out['lat'] = pd.to_numeric(out['lat'],
                                   errors='coerce')
        out['lon'] = pd.to_numeric(out['lon'],
                                   errors='coerce')
        out = out.dropna(subset=['lat', 'lon'])
        # optional: snap to grid to avoid tiny fp diffs
        out['lat'] = out['lat'].round(round_decimals)
        out['lon'] = out['lon'].round(round_decimals)
        return out

    insitu = clean(insitu)
    rs = clean(rs)
    if insitu.empty and rs.empty:
        return []

    # group counts per site (lat, lon)
    insitu_counts = (insitu.groupby(['lat', 'lon'])
                     .size()
                     .rename('n')
                     .reset_index())
    rs_counts = (rs.groupby(['lat', 'lon'])
                 .size()
                 .rename('n')
                 .reset_index())

    # fast exclude used_sites (as set of tuples)
    if used_sites:
        # apply same rounding to used_sites keys
        us = {(round(float(lat), round_decimals),
               round(float(lon), round_decimals))
              for (lat, lon) in used_sites}
        if not insitu_counts.empty:
            mi_i = pd.MultiIndex.from_frame(
                insitu_counts[['lat', 'lon']]
            )
            mask_i = ~mi_i.isin(us)
            insitu_counts = insitu_counts.loc[mask_i]
        if not rs_counts.empty:
            mi_r = pd.MultiIndex.from_frame(
                rs_counts[['lat', 'lon']]
            )
            mask_r = ~mi_r.isin(us)
            rs_counts = rs_counts.loc[mask_r]

    # shuffle sites reproducibly
    rng = np.random.default_rng(seed)
    if not insitu_counts.empty:
        insitu_counts = insitu_counts.iloc[
            rng.permutation(len(insitu_counts))
        ].reset_index(drop=True)
    if not rs_counts.empty:
        rs_counts = rs_counts.iloc[
            rng.permutation(len(rs_counts))
        ].reset_index(drop=True)

    # pick minimum number of sites needed to hit
    # desired obs using cumsum + searchsorted
    def pick_sites(df, goal):
        if df.empty or goal <= 0:
            return []
        csum = df['n'].to_numpy().cumsum()
        # index of last site needed
        k = np.searchsorted(csum, goal, side='left')
        k = min(k, len(df) - 1)
        take = df.iloc[:k+1][['lat', 'lon']]
        return list(map(tuple, take.to_numpy()))

    val_i = pick_sites(insitu_counts,
                       desired_insitu_sample_size)
    val_r = pick_sites(rs_counts,
                       desired_rs_sample_size)

    # combine; if you want to prevent duplicates,
    # make it a set then back to list:
    val_locs = val_i + val_r
    # val_locs = list(dict.fromkeys(val_i + val_r))

    return val_locs


#def create_site_split(
#    data_info: pd.DataFrame,
#    desired_insitu_sample_size: int,
#    desired_rs_sample_size: int,
#    seed=42,
#    used_sites=None,
#):
#    # only create for in-situ sites
#    insitu_data_info = data_info[data_info['source'] == 'nfmd'].reset_index(drop=True)
#    rs_data_info = data_info[data_info['source'] == 'rs'].reset_index(drop=True)
#    # use explicit date/latitude/longitude columns
#    insitu_parts = insitu_data_info[["date", "latitude", "longitude"]].copy()
#    insitu_parts = insitu_parts.rename(columns={"latitude": "lat", "longitude": "lon"})
#    insitu_parts["lat"] = pd.to_numeric(insitu_parts["lat"], errors="coerce")
#    insitu_parts["lon"] = pd.to_numeric(insitu_parts["lon"], errors="coerce")
#    insitu_parts = insitu_parts.dropna(subset=["lat", "lon"])
#    if insitu_parts.empty:
#        raise ValueError("No valid lat/lon rows after parsing.")
#    rs_parts = rs_data_info[["date", "latitude", "longitude"]].copy()
#    rs_parts = rs_parts.rename(columns={"latitude": "lat", "longitude": "lon"})
#    rs_parts["lat"] = pd.to_numeric(rs_parts["lat"], errors="coerce")
#    rs_parts["lon"] = pd.to_numeric(rs_parts["lon"], errors="coerce")
#    rs_parts = rs_parts.dropna(subset=["lat", "lon"])
#    if rs_parts.empty:
#        raise ValueError("No valid lat/lon rows after parsing.")
#    # count obs per site
#    insitu_counts = (
#        insitu_parts.groupby(["lat", "lon"])
#        .size()
#        .reset_index(name="n")
#    )
#    rs_counts = (
#        rs_parts.groupby(["lat", "lon"])
#        .size()
#        .reset_index(name="n")
#    )
#    # filter out used_sites if provided
#    if used_sites is not None and len(used_sites) > 0:
#        insitu_counts = insitu_counts[
#            ~insitu_counts.apply(
#                lambda r: (float(r["lat"]), float(r["lon"])) in used_sites,
#                axis=1
#            )
#        ].reset_index(drop=True)
#        rs_counts = rs_counts[
#            ~rs_counts.apply(
#                lambda r: (float(r["lat"]), float(r["lon"])) in used_sites,
#                axis=1
#            )
#        ].reset_index(drop=True)
#    # shuffle sites reproducibly
#    rng = np.random.default_rng(seed)
#    idx = np.arange(len(insitu_counts))
#    rng.shuffle(idx)
#    insitu_counts = insitu_counts.iloc[idx].reset_index(drop=True)
#    idx = np.arange(len(rs_counts))
#    rng.shuffle(idx)
#    rs_counts = rs_counts.iloc[idx].reset_index(drop=True)
#    # accumulate sites until we hit desired total obs
#    val_locs = []
#    total_insitu = 0
#    total_rs = 0
#    for _, row in insitu_counts.iterrows():
#        val_locs.append((float(row["lat"]), float(row["lon"])))
#        total_insitu += int(row["n"])
#        if total_insitu >= desired_insitu_sample_size:
#            break
#    for _, row in rs_counts.iterrows():
#        val_locs.append((float(row["lat"]), float(row["lon"])))
#        total_rs += int(row["n"])
#        if total_rs >= desired_rs_sample_size:
#            break
#    # if we couldn't meet desired_sample_size, just return all
#    if not val_locs:
#        return []
#    return val_locs


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
    warmup_end_lr=None,
    lambda_rs=0.9
):
    pbar = tqdm.tqdm(
        loader,
        desc='Batch'
    )
    # tracking paraphanalia
    n_samples_tot = 0.0
    n_i_tot = 0.0
    n_rs_tot = 0.0
    running_loss = 0.0
    running_loss_insitu = 0.0
    running_loss_rs = 0.0
    out_mu_i = []
    out_logv_i = []
    out_mu_rs = []
    out_logv_rs = []
    out_true_i = []
    out_true_rs = []
    for Xd_b,Xs_b,Y_b,insitu_b in pbar:
        # move data to device
        Xd_b = Xd_b.to(device=device, dtype=torch.float32)
        Xs_b = Xs_b.to(device=device, dtype=torch.float32)
        Y_b = Y_b.to(device=device, dtype=torch.float32)
        insitu_b = insitu_b.to(device=device, dtype=torch.float32)
        Y_b = Y_b.view(-1)
        insitu_b = insitu_b.view(-1)
        if train_model:
            preds = model(Xd_b, Xs_b)
        else:
            with torch.no_grad():
                preds = model(Xd_b, Xs_b)
        mu_i_b = preds['mu_insitu']
        logv_i_b = preds['log_var_insitu']
        #logv_i_b = torch.zeros_like(mu_i_b)  # homoscedastic for insitu
        mu_rs_b = preds['mu_rs']
        logv_rs_b = preds['log_var_rs']
        #logv_rs_b = torch.zeros_like(mu_rs_b)  # homoscedastic for rs
        m_i = insitu_b > 0.5
        m_rs = ~m_i
        if loss_fn is not None:
            loss_i = loss_fn(mu_i_b, logv_i_b, Y_b, mask=m_i)
            loss_rs = loss_fn(mu_rs_b, logv_rs_b, Y_b, mask=m_rs)
            n_i = int(m_i.sum().item())
            n_rs = int(m_rs.sum().item())
            n_i_tot += n_i
            n_rs_tot += n_rs
            n_samples = n_i + n_rs
            n_samples_tot += n_samples
            denominator = n_i + lambda_rs * n_rs
            total_loss = (n_i * loss_i + lambda_rs * n_rs * loss_rs) / denominator
            if n_i > 0:
                running_loss_insitu += loss_i.item() * n_i
            if n_rs > 0:
                running_loss_rs += loss_rs.item() * n_rs
        if train_model:
            if global_step < warmup_steps:
                this_t = global_step / warmup_steps
                lr = warmup_start_lr * ((warmup_end_lr / warmup_start_lr) ** this_t)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            global_step += 1
        #print('Xd_b:')
        #print(Xd_b)
        #print('Xs_b:')
        #print(Xs_b)
        #print('Y_b:')
        #print(Y_b)
        #print('mu_i_b:')
        #print(mu_i_b)
        #print('logv_i_b:')
        #print(logv_i_b)
        #print('mu_rs_b:')
        #print(mu_rs_b)
        #print('logv_rs_b:')
        #print(logv_rs_b)
        #print('loss_i:')
        #print(loss_i)
        #print('loss_rs:')
        #print(loss_rs)
        #print('n_i:')
        #print(n_i)
        #print('n_rs:')
        #print(n_rs)
        #print('denominator:')
        #print(denominator)
        #print('total_loss:')
        #print(total_loss)
        #sys.exit()
        out_mu_i.append(mu_i_b.detach().cpu())
        out_logv_i.append(logv_i_b.detach().cpu())
        out_mu_rs.append(mu_rs_b.detach().cpu())
        out_logv_rs.append(logv_rs_b.detach().cpu())
        out_true_i.append(Y_b[m_i].detach().cpu())
        out_true_rs.append(Y_b[m_rs].detach().cpu())
    # calculate running loss
    if loss_fn is not None and n_samples > 0:
        running_loss = running_loss_insitu + running_loss_rs * lambda_rs
        running_loss /= (n_i_tot + lambda_rs * n_rs_tot)
        running_loss_insitu /= n_i_tot
        running_loss_rs /= n_rs_tot
    else:
        running_loss = None
        running_loss_insitu = None
        running_loss_rs = None
    if len(out_mu_i) > 0:
        mu_i = torch.cat(out_mu_i).squeeze().numpy()
        logv_i = torch.cat(out_logv_i).squeeze().numpy()
        mu_rs = torch.cat(out_mu_rs).squeeze().numpy()
        logv_rs = torch.cat(out_logv_rs).squeeze().numpy()
        true_i = torch.cat(out_true_i).squeeze().numpy()
        true_rs = torch.cat(out_true_rs).squeeze().numpy()
    else:
        mu_i = np.array([])
        logv_i = np.array([])
        mu_rs = np.array([])
        logv_rs = np.array([])
        true_i = np.array([])
        true_rs = np.array([])
    return(
        model,
        running_loss,
        running_loss_insitu,
        running_loss_rs,
        mu_i,
        logv_i,
        mu_rs,
        logv_rs,
        true_i,
        true_rs,
        global_step
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
    rs_factor,
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
    lfmc = data[2]
    source = data[3]
    info = data[4]
    (
        test_daily_data, test_static_data, test_lfmc, test_source, test_info,
        remaining_daily_data, remaining_static_data, remaining_lfmc, remaining_source, remaining_info
    ) = mask_location_data_fast(
        this_locs,
        daily_data,
        static_data,
        lfmc,
        source,
        info
    )
    # split out the validation data
    remaining_insitu_obs = remaining_info[remaining_info['source'] == 'nfmd'].shape[0]
    remaining_rs_obs = remaining_info[remaining_info['source'] == 'rs'].shape[0]
    num_val_obs_insitu = remaining_insitu_obs * val_split
    num_val_obs_rs = remaining_rs_obs * val_split
    val_locs = create_site_split(
        remaining_info,
        desired_insitu_sample_size=int(num_val_obs_insitu),
        desired_rs_sample_size=int(num_val_obs_rs),
    )
    val_locs = np.array(val_locs)
    # perform the same masking as was done for the test sites
    (
        val_daily_data, val_static_data, val_lfmc, val_source, val_info,
        train_daily_data, train_static_data, train_lfmc, train_source, train_info
    ) = mask_location_data_fast(
        val_locs,
        remaining_daily_data,
        remaining_static_data,
        remaining_lfmc,
        remaining_source,
        remaining_info
    )
    # Sanity check
    total_test = test_info.shape[0]
    insitu_test = test_info[test_info['source'] == 'nfmd'].shape[0]
    rs_test = test_info[test_info['source'] == 'rs'].shape[0]
    total_val = val_info.shape[0]
    insitu_val = val_info[val_info['source'] == 'nfmd'].shape[0]
    rs_val = val_info[val_info['source'] == 'rs'].shape[0]
    total_train = train_info.shape[0]
    insitu_train = train_info[train_info['source'] == 'nfmd'].shape[0]
    rs_train = train_info[train_info['source'] == 'rs'].shape[0]
    print(
        f"Test: {total_test} ({insitu_test} insitu, {rs_test} rs) | "
        f"Val: {total_val} ({insitu_val} insitu, {rs_val} rs) | "
        f"Train: {total_train} ({insitu_train} insitu, {rs_train} rs)"
    )
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
    y_mean = np.nanmean(train_lfmc)
    y_std = np.nanstd(train_lfmc)
    for v,var in enumerate(var_names['daily_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var
        ):
            continue
        train_daily_data[:,:,v] = (train_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
        val_daily_data[:,:,v] = (val_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
        test_daily_data[:,:,v] = (test_daily_data[:,:,v] - train_daily_mean[v]) / train_daily_std[v]
    for v,var in enumerate(var_names['static_vars']):
        if (
            '_sin' in var or
            '_cos' in var or
            'lag' in var
        ):
            continue
        train_static_data[:,:,v] = (train_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        val_static_data[:,:,v] = (val_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        test_static_data[:,:,v] = (test_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
    train_lfmc = (train_lfmc - y_mean) / y_std
    val_lfmc = (val_lfmc - y_mean) / y_std
    test_lfmc = (test_lfmc - y_mean) / y_std
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
        train_lfmc,
        train_source
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
        val_lfmc,
        val_source
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
        test_lfmc,
        test_source
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    # set up the loss functions
    #criterion = nn.MSELoss()
    criterion = GaussianNLLLoss(reduction="mean")
    # make sure that we have the warmup end lr
    warmup_end_lr = optimizer.param_groups[0]['lr']
    # set up the things that we need to track
    train_loss = []
    train_loss_insitu = []
    train_loss_rs = []
    val_loss = []
    val_loss_insitu = []
    val_loss_rs = []
    global_step = 0
    for epoch in range(1,max_epochs):
        print(f'Fold {this_fold_num}, Epoch {epoch}/{max_epochs}')
        model.train()
        (
            model,
            this_train_loss,
            this_train_loss_insitu,
            this_train_loss_rs,
            _,
            _,
            _,
            _,
            _,
            _,
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
            warmup_end_lr=warmup_end_lr,
            lambda_rs=rs_factor
        )
        train_loss.append(this_train_loss)
        train_loss_insitu.append(this_train_loss_insitu)
        train_loss_rs.append(this_train_loss_rs) 
        print(f'Training total loss: {this_train_loss:.4f}')
        print(f'Training insitu loss: {this_train_loss_insitu:.4f}')
        print(f'Training rs loss: {this_train_loss_rs:.4f}')
        scheduler.step()
        # run the validation
        model.eval()
        (
            model,
            this_val_loss,
            this_val_loss_insitu,
            this_val_loss_rs,
            mu_i_val,
            logv_i_val,
            mu_rs_val,
            logv_rs_val,
            true_i,
            true_rs,
            _
        ) = run_model(
            model,
            val_loader,
            device,
            criterion,
            train_model=False,
            lambda_rs=rs_factor
        )
        val_loss.append(this_val_loss)
        val_loss_insitu.append(this_val_loss_insitu)
        val_loss_rs.append(this_val_loss_rs)
        print(f'Validation total loss: {this_val_loss:.4f}')
        print(f'Validation insitu loss: {this_val_loss_insitu:.4f}')
        print(f'Validation rs loss: {this_val_loss_rs:.4f}')
        # denorm
        lfmc_i_val_only = mu_i_val[val_source.numpy() == 1 ] * y_std + y_mean
        lfmc_std_i_val_only = np.sqrt(np.exp(logv_i_val[val_source.numpy() == 1])) * y_std
        lfmc_i_val_true = true_i * y_std + y_mean
        ## get mixture
        #mu_mix_val, logv_mix_val = fuse_gaussians(
        #    mu_i_val,
        #    logv_i_val,
        #    mu_rs_val,
        #    logv_rs_val,
        #)
        # calculate metrics of interet
        val_mae = np.mean(np.abs(lfmc_i_val_only - lfmc_i_val_true))
        val_r2 = r2_score(lfmc_i_val_true, lfmc_i_val_only)
        val_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_i_val_only) + ((lfmc_i_val_true - lfmc_i_val_only) ** 2) / (lfmc_std_i_val_only ** 2)))
        val_rmse = np.sqrt(np.mean((lfmc_i_val_only - lfmc_i_val_true) ** 2))
        ## and for the mixed data
        #lfmc_mix_val = mu_mix_val[val_source.numpy() == 1 ] * y_std + y_mean
        #lfmc_std_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 1])) * y_std
        #val_mae_mix = np.mean(np.abs(lfmc_mix_val - lfmc_i_val_true))
        #val_r2_mix = r2_score(lfmc_i_val_true, lfmc_mix_val)
        #val_nll_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_mix_val) + ((lfmc_i_val_true - lfmc_mix_val) ** 2) / (lfmc_std_mix_val ** 2)))
        #val_rmse_mix = np.sqrt(np.mean((lfmc_mix_val - lfmc_i_val_true) ** 2))
        # and for rs data
        if len(true_rs) > 0:
            lfmc_rs_val_only = mu_rs_val[val_source.numpy() == 0] * y_std + y_mean
            lfmc_std_rs_val_only = np.sqrt(np.exp(logv_rs_val[val_source.numpy() == 0])) * y_std
            lfmc_rs_val_true = true_rs * y_std + y_mean
            val_mae_rs = np.mean(np.abs(lfmc_rs_val_only - lfmc_rs_val_true))
            val_r2_rs = r2_score(lfmc_rs_val_true, lfmc_rs_val_only)
            val_nll_rs = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_val_only) + ((lfmc_rs_val_true - lfmc_rs_val_only) ** 2) / (lfmc_std_rs_val_only ** 2)))
            val_rmse_rs = np.sqrt(np.mean((lfmc_rs_val_only - lfmc_rs_val_true) ** 2))
            ## also calculate the mixtures
            #lfmc_rs_mix_val = mu_mix_val[val_source.numpy() == 0] * y_std + y_mean
            #lfmc_std_rs_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 0])) * y_std
            #val_mae_rs_mix = np.mean(np.abs(lfmc_rs_mix_val - lfmc_rs_val_true))
            #val_r2_rs_mix = r2_score(lfmc_rs_val_true, lfmc_rs_mix_val)
            #val_nll_rs_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_mix_val) + ((lfmc_rs_val_true - lfmc_rs_mix_val) ** 2) / (lfmc_std_rs_mix_val ** 2)))
            #val_rmse_rs_mix = np.sqrt(np.mean((lfmc_rs_mix_val - lfmc_rs_val_true) ** 2))
        # average values for sanity check
        #avg_val_pred = np.mean(lfmc_i_val)
        #avg_val_true = np.mean(lfmc_i_val_true)
        #avg_val_std = np.mean(lfmc_std_i_val)
        print(
            f'Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, NLL: {val_nll:.4f}'
        )
        #print(
        #    f'Validation Mixture MAE: {val_mae_mix:.4f}, RMSE: {val_rmse_mix:.4f}, R2: {val_r2_mix:.4f}, NLL: {val_nll_mix:.4f}'
        #)
        if len(true_rs) > 0:
            print(
                f'Validation RS MAE: {val_mae_rs:.4f}, RMSE: {val_rmse_rs:.4f}, R2: {val_r2_rs:.4f}, NLL: {val_nll_rs:.4f}'
            )
            #print(
            #    f'Validation RS Mixture MAE: {val_mae_rs_mix:.4f}, RMSE: {val_rmse_rs_mix:.4f}, R2: {val_r2_rs_mix:.4f}, NLL: {val_nll_rs_mix:.4f}'
            #)
        #print(
        #    f'Validation Avg Pred: {avg_val_pred:.4f}, Validation Avg True: {avg_val_true:.4f}, Validation Avg Std: {avg_val_std:.4f}'
        #)
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
        val_loss_insitu,
        val_loss_rs,
        mu_i_val,
        logv_i_val,
        mu_rs_val,
        logv_rs_val,
        true_i_val,
        true_rs_val,
        _
    ) = run_model(
        model,
        val_loader,
        device,
        criterion,
        train_model=False,
        lambda_rs=rs_factor
    )
    lfmc_i_val_only = mu_i_val[val_source.numpy() == 1 ] * y_std + y_mean
    lfmc_std_i_val_only = np.sqrt(np.exp(logv_i_val[val_source.numpy() == 1])) * y_std
    lfmc_i_val_true = true_i * y_std + y_mean
    # calculate metrics of interet
    val_mae = np.mean(np.abs(lfmc_i_val_only - lfmc_i_val_true))
    val_r2 = r2_score(lfmc_i_val_true, lfmc_i_val_only)
    val_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_i_val_only) + ((lfmc_i_val_true - lfmc_i_val_only) ** 2) / (lfmc_std_i_val_only ** 2)))
    val_rmse = np.sqrt(np.mean((lfmc_i_val_only - lfmc_i_val_true) ** 2))
    # and for rs data
    if len(true_rs) > 0:
        lfmc_rs_val_only = mu_rs_val[val_source.numpy() == 0] * y_std + y_mean
        lfmc_std_rs_val_only = np.sqrt(np.exp(logv_rs_val[val_source.numpy() == 0])) * y_std
        lfmc_rs_val_true = true_rs * y_std + y_mean
        val_mae_rs = np.mean(np.abs(lfmc_rs_val_only - lfmc_rs_val_true))
        val_r2_rs = r2_score(lfmc_rs_val_true, lfmc_rs_val_only)
        val_nll_rs = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_val_only) + ((lfmc_rs_val_true - lfmc_rs_val_only) ** 2) / (lfmc_std_rs_val_only ** 2)))
        val_rmse_rs = np.sqrt(np.mean((lfmc_rs_val_only - lfmc_rs_val_true) ** 2))
    # average values for sanity check
    #avg_val_pred = np.mean(lfmc_i_val)
    #avg_val_true = np.mean(lfmc_i_val_true)
    #avg_val_std = np.mean(lfmc_std_i_val)
    print(
        f'Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, NLL: {val_nll:.4f}'
    )
    if len(true_rs) > 0:
        print(
            f'Validation RS MAE: {val_mae_rs:.4f}, RMSE: {val_rmse_rs:.4f}, R2: {val_r2_rs:.4f}, NLL: {val_nll_rs:.4f}'
        )
    # run the test
    (
        model,
        test_loss,
        test_loss_insitu,
        test_loss_rs,
        mu_i_test,
        logv_i_test,
        mu_rs_test,
        logv_rs_test,
        true_i_test,
        true_rs_test,
        _
    ) = run_model(
        model,
        test_loader,
        device,
        criterion,
        train_model=False,
        lambda_rs=rs_factor
    )
    # denorm
    if len(mu_i_test) == 0:
        test_loss = np.nan
        test_loss_insitu = np.nan
        test_loss_rs = np.nan
        lfmc_i_test_only = np.nan
        lfmc_std_i_test_only = np.nan
        lfmc_rs_test_only = np.nan
        lfmc_std_rs_test_only = np.nan
        lfmc_i_test_true = np.nan
        lfmc_rs_test_true = np.nan
        test_mae = np.nan
        test_r2 = np.nan
        test_nll = np.nan
        test_rmse = np.nan
    else:
        lfmc_i_test_only = mu_i_test[test_source.numpy() == 1 ] * y_std + y_mean
        lfmc_std_i_test_only = np.sqrt(np.exp(logv_i_test[test_source.numpy() == 1])) * y_std
        lfmc_i_test_true = true_i_test * y_std + y_mean
        # calculate metrics of interet
        test_mae = np.mean(np.abs(lfmc_i_test_only - lfmc_i_test_true))
        test_r2 = r2_score(lfmc_i_test_true, lfmc_i_test_only)
        test_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_i_test_only) + ((lfmc_i_test_true - lfmc_i_test_only) ** 2) / (lfmc_std_i_test_only ** 2)))
        test_rmse = np.sqrt(np.mean((lfmc_i_test_only - lfmc_i_test_true) ** 2))
        # and for rs data
        if len(true_rs) > 0:
            lfmc_rs_test_only = mu_rs_test[test_source.numpy() == 0] * y_std + y_mean
            lfmc_std_rs_test_only = np.sqrt(np.exp(logv_rs_test[test_source.numpy() == 0])) * y_std
            lfmc_rs_test_true = true_rs_test * y_std + y_mean
            test_mae_rs = np.mean(np.abs(lfmc_rs_test_only - lfmc_rs_test_true))
            test_r2_rs = r2_score(lfmc_rs_test_true, lfmc_rs_test_only)
            test_nll_rs = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_test_only) + ((lfmc_rs_test_true - lfmc_rs_test_only) ** 2) / (lfmc_std_rs_test_only ** 2)))
            test_rmse_rs = np.sqrt(np.mean((lfmc_rs_test_only - lfmc_rs_test_true) ** 2))
        # average testues for sanity check
        #avg_test_pred = np.mean(lfmc_i_test)
        #avg_test_true = np.mean(lfmc_i_test_true)
        #avg_test_std = np.mean(lfmc_std_i_test)
        print(
            f'testidation MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, NLL: {test_nll:.4f}'
        )
        if len(true_rs) > 0:
            print(
                f'testidation RS MAE: {test_mae_rs:.4f}, RMSE: {test_rmse_rs:.4f}, R2: {test_r2_rs:.4f}, NLL: {test_nll_rs:.4f}'
            )
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
            'loss_insitu':val_loss_insitu,
            'loss_rs':val_loss_rs,
            'lfmc_insitu_preds':lfmc_i_val_only,
            'lfmc_insitu_std':lfmc_std_i_val_only,
            'lfmc_rs_preds':lfmc_rs_val_only,
            'lfmc_rs_std':lfmc_std_rs_val_only,
            'lfmc_insitu_true':lfmc_i_val_true,
            'lfmc_rs_true':lfmc_rs_val_true
        },
        os.path.join(fold_save_dir,'val_outputs.pth')
    )
    torch.save(
        {
            'loss':test_loss,
            'loss_insitu':test_loss_insitu,
            'loss_rs':test_loss_rs,
            'lfmc_insitu_preds':lfmc_i_test_only,
            'lfmc_insitu_std':lfmc_std_i_test_only,
            'lfmc_rs_preds':lfmc_rs_test_only,
            'lfmc_rs_std':lfmc_std_rs_test_only,
            'lfmc_insitu_true':lfmc_i_test_true,
            'lfmc_rs_true':lfmc_rs_test_true
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
    input_data_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs'
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs'
    # training settings
    batch_size = 128
    max_epochs = 100
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available, using CPU. This will be slow!')
    warmup_steps = 2000
    base_lr = lr
    warmup_start_lr = 1e-6
    val_split = 0.15
    adam_weight_decay = 1e-2
    patience = 8
    # model hyperparameters (go back to the github and get what I deleted here)
    d_model = 96
    nhead = 2
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    rs_factor = 1.0 # weighting for RS loss
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
        f'_wd{adam_weight_decay}_rsf{rs_factor}_35k'
    )
    full_save_dir = os.path.join(save_dir, this_model_name)
    if os.path.exists(full_save_dir):
        shutil.rmtree(full_save_dir)
    os.makedirs(full_save_dir)
    # build the folds by location
    daily_data = datasets[0]
    info = datasets[4]
    n_folds = 10
    total_obs_insitu_obs = (
        info[info['source'] == 'nfmd'].shape[0]
    )
    total_obs_rs_obs = (
        info[info['source'] == 'rs'].shape[0]
    )
    desired_insitu_obs_per_fold = total_obs_insitu_obs / n_folds
    desired_rs_obs_per_fold = total_obs_rs_obs / n_folds
    fold_locs = {}
    used_sites = []
    for fold in range(n_folds):
        print(f'Getting locations for fold {fold+1}/{n_folds}')
        this_locs = create_site_split(
            info,
            desired_insitu_sample_size=int(desired_insitu_obs_per_fold),
            desired_rs_sample_size=int(desired_rs_obs_per_fold),
            used_sites=used_sites
        )
        used_sites.extend(this_locs)
        fold_locs[fold + 1] = this_locs
    # save this fold info
    with open(os.path.join(full_save_dir, 'fold_info.json'), 'w') as f:
        json.dump(fold_locs, f)
    # train this fold
    for fold, locs in enumerate(fold_locs.items()):
        print(f'Training fold {fold+1}/{n_folds} with {len(locs[1])} locations held out for testing')
        # build the model
        model = LFMCTransformerMultiTask(
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
            T_max=20,
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
            val_split,
            rs_factor
        )
    # one final version of the model trained on all the data
    print('Training final model on all data')
    model = LFMCTransformerMultiTask(
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
        T_max=20,
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
        val_split,
        rs_factor
    )

            

if __name__ == "__main__":
    main()