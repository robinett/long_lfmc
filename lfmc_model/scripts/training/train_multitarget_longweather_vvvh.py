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
from torch.utils.data import Sampler
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_model import LFMCTransformer
from transformer_model_multitask import LFMCTransformer as LFMCTransformerMultiTask
from transformer_multitask_longclimate import LFMCTransformer as LFMCTransformerMultiTaskLongClimate
import plotting

import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",
    category=UserWarning,
)

class StratifiedBatchSampler(Sampler):
    """
    Yields batches of indices such that each batch has
    approximately the same class distribution as the full
    label vector.
    """
    def __init__(self, labels, batch_size, shuffle=True, seed=None):
        """
        labels: 1D array-like, length N
            Stratifier / class labels (e.g. land-cover codes).
        batch_size: int
        shuffle: bool
            Whether to shuffle indices within each class.
        seed: int or None
            For reproducibility.
        """
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.classes, counts = np.unique(self.labels, return_counts=True)
        self.num_samples = len(self.labels)
        # Precompute per-class index lists
        self.class_indices_base = {
            c: np.where(self.labels == c)[0].tolist()
            for c in self.classes
        }
    def __iter__(self):
        # Copy and optionally shuffle within each class for this epoch
        class_indices = {
            c: idxs.copy() for c, idxs in self.class_indices_base.items()
        }
        if self.shuffle:
            for idxs in class_indices.values():
                self.rng.shuffle(idxs)
        while True:
            # Remaining counts per class
            rem_counts = np.array(
                [len(class_indices[c]) for c in self.classes]
            )
            total_rem = rem_counts.sum()
            if total_rem == 0:
                break
            # If we are near the end, we may have fewer than batch_size left
            current_batch_size = min(self.batch_size, total_rem)
            # Distribution based on remaining samples (keeps proportions stable)
            probs = rem_counts / rem_counts.sum()
            expected = probs * current_batch_size
            base = np.floor(expected).astype(int)
            remainder = current_batch_size - base.sum()
            # Give leftover slots to classes with largest fractional parts
            frac = expected - base
            order = np.argsort(-frac)
            for i in order[:remainder]:
                base[i] += 1
            batch = []
            for c, k in zip(self.classes, base):
                take = min(k, len(class_indices[c]))
                if take > 0:
                    batch.extend(class_indices[c][:take])
                    del class_indices[c][:take]
            if not batch:
                break
            yield batch
    def __len__(self):
        # Approx number of batches per epoch
        return int(np.ceil(self.num_samples / self.batch_size))

def pick_sites_stratified(
    df: pd.DataFrame,
    goal: int,
    strat_col: str = "stratifier",
    random_state: int = 42
):
    """
    Pick sites so that:
      * Total observations (sum of n) ~= goal
      * Land cover distribution (by sum of n) in the picked set
        is roughly equal to the full df distribution.

    Parameters
    ----------
    df : DataFrame
        Must have columns ['lat', 'lon', 'n', strat_col].
        Each row = one site.
    goal : int
        Desired total number of observations (sum of n).
    strat_col : str
        Column name with stratifier labels (e.g. NLCD code).
    random_state : int or None
        Seed for reproducible randomness.

    Returns
    -------
    list[tuple]
        List of (lat, lon) for selected sites.
        (You can also grab df.loc[selected_idx] if you
         want idx/stratifier/etc.)
    """
    if df.empty or goal <= 0:
        return []
    if strat_col not in df.columns:
        raise ValueError(f"'{strat_col}' column not found in df.")
    rng = np.random.default_rng(random_state)
    # --- 1. overall obs per stratum (by n) ----------------------
    obs_per_stratum = df.groupby(strat_col)["n"].sum()
    total_obs = int(obs_per_stratum.sum())
    # If you ask for more than available, cap at total.
    if goal > total_obs:
        goal = total_obs
    # --- 2. ideal quotas per stratum ----------------------------
    ideal = obs_per_stratum / total_obs * goal  # float
    quotas = np.floor(ideal).astype(int)
    # distribute leftover observations based on largest fractional part
    remainder = goal - int(quotas.sum())
    if remainder > 0:
        frac = ideal - quotas
        # strata with largest fractional parts get +1
        extra_order = frac.sort_values(ascending=False).index
        for s in extra_order[:remainder]:
            quotas[s] += 1
    # --- 3. pick minimal sites per stratum to hit each quota ----
    chosen_parts = []
    for s, quota in quotas.items():
        if quota <= 0:
            continue
        group = df[df[strat_col] == s].copy()
        if group.empty:
            continue
        # shuffle sites in this stratum
        group = group.sample(
            frac=1,
            random_state=rng.integers(0, 2**32 - 1)
        )
        csum = group["n"].to_numpy().cumsum()
        k = np.searchsorted(csum, quota, side="left")
        k = min(k, len(group) - 1)
        chosen_parts.append(group.iloc[:k+1])
    if not chosen_parts:
        return []
    chosen = pd.concat(chosen_parts, ignore_index=True)
    # Total obs will typically be close to `goal`, but may overshoot
    # slightly because we can only add whole sites.
    # Optional final shuffle for randomness in overall order
    chosen = chosen.sample(
        frac=1,
        random_state=rng.integers(0, 2**32 - 1)
    ).reset_index(drop=True)
    # Return (lat, lon) tuples like before
    take = chosen[["lat", "lon"]]
    return list(map(tuple, take.to_numpy()))

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
    short_data,
    long_data,           # torch.Tensor [N, ...]
    static_data,          # torch.Tensor [N, ...]
    lfmc_insitu,          # torch.Tensor [N, ...]
    source,               # torch.Tensor [N]
    info,                 # pd.DataFrame with 'latitude','longitude'
    stratifier,
    masking_radius_m=1000.0,
    match_tolerance_m=5.0,   # how close is "same location"
    round_decimals=6,        # snap to grid to avoid FP drift
):
    # Edge cases
    if len(info) == 0 or len(keep_locs) == 0:
        # nothing to keep or nothing to act on → everything is remaining
        empty = slice(0, 0)
        return (short_data[empty], long_data[empty], static_data[empty], lfmc_insitu[empty], source[empty], info.iloc[:0],
                short_data, long_data, static_data, lfmc_insitu, source, info, stratifier)

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
    kept_short_data   = short_data[m_kept]
    kept_long_data    = long_data[m_kept]
    kept_static_data  = static_data[m_kept]
    kept_lfmc_insitu  = lfmc_insitu[m_kept]
    kept_source       = source[m_kept]
    kept_info         = info.loc[mask_kept]
    kept_info         = kept_info.reset_index(drop=True)
    kept_stratifier   = stratifier[m_kept]

    remaining_short_data   = short_data[m_rem]
    remaining_long_data    = long_data[m_rem]
    remaining_static_data  = static_data[m_rem]
    remaining_lfmc_insitu  = lfmc_insitu[m_rem]
    remaining_source       = source[m_rem]
    remaining_info         = info.loc[mask_remaining]
    remaining_info         = remaining_info.reset_index(drop=True)
    remaining_stratifier   = stratifier[m_rem]

    return (
        kept_short_data, kept_long_data, kept_static_data, kept_lfmc_insitu, kept_source, kept_info, kept_stratifier,
        remaining_short_data, remaining_long_data, remaining_static_data, remaining_lfmc_insitu, remaining_source, remaining_info, remaining_stratifier
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
    X_short = torch.load(
        os.path.join(center_data_dir, 'X_short.pt'),
        weights_only=False
    )
    X_long = torch.load(
        os.path.join(center_data_dir, 'X_long.pt'),
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
    stratifier = np.load(
        os.path.join(center_data_dir, 'stratifier.npy')
    )
    # check for nan in any of these
    assert not torch.isnan(X_short).any(), "NaN found in X_short"
    assert not torch.isnan(X_long).any(), "NaN found in X_long"
    assert not torch.isnan(X_static).any(), "NaN found in X_static"
    assert not torch.isnan(Y_lfmc).any(), "NaN found in Y_lfmc"
    assert not torch.isnan(source).any(), "NaN found in source"
    assert not np.isnan(stratifier).any(), "NaN found in stratifier"
    # load the center info
    center_info = pd.read_csv(os.path.join(center_data_dir, 'info.csv'))
    all_center_data = [
        X_short, X_long, X_static, Y_lfmc, source, center_info, stratifier
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
    desired_vv_sample_size: int,
    desired_vh_sample_size: int,
    seed: int = 42,
    used_sites = None,
    round_decimals: int = 6,
    stratifier = None
):
    # split sources
    insitu = data_info[data_info['source'] == 'nfmd']
    vv     = data_info[data_info['source'] == 'VV']
    vh     = data_info[data_info['source'] == 'VH']
    # get the index and split by stratifier if provided
    if stratifier is not None:
        insitu_idx = data_info[data_info['source'] == 'nfmd'].index
        vv_idx     = data_info[data_info['source'] == 'VV'].index
        vh_idx     = data_info[data_info['source'] == 'VH'].index
        insitu_strat = stratifier[insitu_idx]
        vv_strat     = stratifier[vv_idx]
        vh_strat     = stratifier[vh_idx]
    # clean/standardize lat/lon once
    def clean(df):
        out = df[['date', 'latitude', 'longitude']].copy()
        out = out.rename(columns={'latitude': 'lat',
                                  'longitude': 'lon'})
        out['lat'] = pd.to_numeric(out['lat'], errors='coerce')
        out['lon'] = pd.to_numeric(out['lon'], errors='coerce')
        out = out.dropna(subset=['lat', 'lon'])
        # optional: snap to grid to avoid tiny fp diffs
        out['lat'] = out['lat'].round(round_decimals)
        out['lon'] = out['lon'].round(round_decimals)
        return out
    insitu = clean(insitu)
    vv     = clean(vv)
    vh     = clean(vh)
    if insitu.empty and vv.empty and vh.empty:
        return []
    # group counts per site (lat, lon)
    #insitu_counts = (insitu.groupby(['lat', 'lon'])
    #                 .size().rename('n').reset_index())
    #vv_counts = (vv.groupby(['lat', 'lon'])
    #             .size().rename('n').reset_index())
    #vh_counts = (vh.groupby(['lat', 'lon'])
    #             .size().rename('n').reset_index())
    # get the land cover type for each site if stratifier is provided
    insitu_counts = (
        insitu.groupby(["lat", "lon"])
              .agg(
                  n=("date", "size"),
                  idx=("date", lambda s: list(s.index))
              )
              .reset_index()
    )
    vv_counts = (
        vv.groupby(["lat", "lon"])
            .agg(
                n=("date", "size"),
                idx=("date", lambda s: list(s.index))
            )
            .reset_index()
    )
    vh_counts = (
        vh.groupby(["lat", "lon"])
            .agg(
                n=("date", "size"),
                idx=("date", lambda s: list(s.index))
            )
            .reset_index()
    )
    # get the land cover type for each grouping
    if stratifier is not None:
        insitu_lcs = []
        vv_lcs = []
        vh_lcs = []
        for idx,row in insitu_counts.iterrows():
            all_idx = row['idx']
            all_lcs = insitu_strat[all_idx]
            ex_idx = row['idx'][0]
            insitu_lcs.append(insitu_strat[ex_idx])
        for idx,row in vv_counts.iterrows():
            ex_idx = row['idx'][0]
            vv_lcs.append(vv_strat[ex_idx])
        for idx,row in vh_counts.iterrows():
            ex_idx = row['idx'][0]
            vh_lcs.append(vh_strat[ex_idx])
        insitu_counts['stratifier'] = insitu_lcs
        vv_counts['stratifier'] = vv_lcs
        vh_counts['stratifier'] = vh_lcs
    # fast exclude used_sites (as set of tuples)
    if used_sites:
        us = {(round(float(lat), round_decimals),
               round(float(lon), round_decimals))
              for (lat, lon) in used_sites}
        if not insitu_counts.empty:
            mi_i = pd.MultiIndex.from_frame(insitu_counts[['lat', 'lon']])
            mask_i = ~mi_i.isin(us)
            insitu_counts = insitu_counts.loc[mask_i]
        if not vv_counts.empty:
            mi_vv = pd.MultiIndex.from_frame(vv_counts[['lat', 'lon']])
            mask_vv = ~mi_vv.isin(us)
            vv_counts = vv_counts.loc[mask_vv]
        if not vh_counts.empty:
            mi_vh = pd.MultiIndex.from_frame(vh_counts[['lat', 'lon']])
            mask_vh = ~mi_vh.isin(us)
            vh_counts = vh_counts.loc[mask_vh]
    # shuffle sites reproducibly
    rng = np.random.default_rng(seed)
    if not insitu_counts.empty:
        insitu_counts = insitu_counts.iloc[
            rng.permutation(len(insitu_counts))
        ].reset_index(drop=True)
    if not vv_counts.empty:
        vv_counts = vv_counts.iloc[
            rng.permutation(len(vv_counts))
        ].reset_index(drop=True)
    if not vh_counts.empty:
        vh_counts = vh_counts.iloc[
            rng.permutation(len(vh_counts))
        ].reset_index(drop=True)
    # pick minimum number of sites needed to hit desired obs
    val_i  = pick_sites_stratified(insitu_counts, desired_insitu_sample_size)
    val_vv = pick_sites_stratified(vv_counts,     desired_vv_sample_size)
    val_vh = pick_sites_stratified(vh_counts,     desired_vh_sample_size)
    # combine (allow duplicates if the same site is selected for multiple sources)
    val_locs = val_i + val_vv + val_vh
    # If you want unique sites only:
    # val_locs = list(dict.fromkeys(val_locs))
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
    warmup_end_lr=None,
    lambda_vv=1.0,
    lambda_vh=1.0
):
    pbar = tqdm.tqdm(
        loader,
        desc='Batch'
    )
    # tracking paraphanalia
    n_samples_tot = 0.0
    n_i_tot = 0.0
    n_vv_tot = 0.0
    n_vh_tot = 0.0
    running_loss = 0.0
    running_loss_insitu = 0.0
    running_loss_vv = 0.0
    running_loss_vh = 0.0
    out_mu_i = []
    out_logv_i = []
    out_mu_vv = []
    out_logv_vv = []
    out_mu_vh = []
    out_logv_vh = []
    out_true_i = []
    out_true_vv = []
    out_true_vh = []
    for Xsh_b,Xl_b,Xst_b,Y_b,insitu_b in pbar:
        # move data to device
        Xsh_b = Xsh_b.to(device=device, dtype=torch.float32)
        Xl_b = Xl_b.to(device=device, dtype=torch.float32)
        Xst_b = Xst_b.to(device=device, dtype=torch.float32)
        Y_b = Y_b.to(device=device, dtype=torch.float32)
        insitu_b = insitu_b.to(device=device, dtype=torch.float32)
        Y_b = Y_b.view(-1)
        insitu_b = insitu_b.view(-1)
        if train_model:
            preds = model(Xsh_b, Xl_b, Xst_b)
        else:
            with torch.no_grad():
                preds = model(Xsh_b, Xl_b, Xst_b)
        mu_i_b = preds['mu_insitu']
        logv_i_b = preds['log_var_insitu']
        #logv_i_b = torch.zeros_like(mu_i_b)  # homoscedastic for insitu
        mu_vv_b = preds['mu_vv']
        logv_vv_b = preds['log_var_vv']
        mu_vh_b = preds['mu_vh']
        logv_vh_b = preds['log_var_vh']
        #logv_rs_b = torch.zeros_like(mu_rs_b)  # homoscedastic for rs
        m_i = insitu_b == 0
        m_vv = insitu_b == 1
        m_vh = insitu_b == 2
        if loss_fn is not None:
            loss_i = loss_fn(mu_i_b, logv_i_b, Y_b, mask=m_i)
            loss_vv = loss_fn(mu_vv_b, logv_vv_b, Y_b, mask=m_vv)
            loss_vh = loss_fn(mu_vh_b, logv_vh_b, Y_b, mask=m_vh)
            n_i = int(m_i.sum().item())
            n_vv = int(m_vv.sum().item())
            n_vh = int(m_vh.sum().item())
            n_i_tot += n_i
            n_vv_tot += n_vv
            n_vh_tot += n_vh
            n_samples = n_i + n_vv + n_vh
            n_samples_tot += n_samples
            denominator = n_i + lambda_vv * n_vv + lambda_vh * n_vh
            total_loss = (
                n_i * loss_i + 
                lambda_vv * n_vv * loss_vv + 
                lambda_vh * n_vh * loss_vh
            ) / denominator
            if n_i > 0:
                running_loss_insitu += loss_i.item() * n_i
            if n_vv > 0:
                running_loss_vv += loss_vv.item() * n_vv
            if n_vh > 0:
                running_loss_vh += loss_vh.item() * n_vh
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
        out_mu_i.append(mu_i_b.detach().cpu())
        out_logv_i.append(logv_i_b.detach().cpu())
        out_mu_vv.append(mu_vv_b.detach().cpu())
        out_logv_vv.append(logv_vv_b.detach().cpu())
        out_true_i.append(Y_b[m_i].detach().cpu())
        out_true_vv.append(Y_b[m_vv].detach().cpu())
    # calculate running loss
    if loss_fn is not None and n_samples > 0:
        running_loss = running_loss_insitu + running_loss_vv * lambda_vv + running_loss_vh * lambda_vh
        running_loss /= (n_i_tot + lambda_vv * n_vv_tot + lambda_vh * n_vh_tot)
        running_loss_insitu /= n_i_tot
        if n_vv_tot > 0:
            running_loss_vv /= n_vv_tot
        else:
            running_loss_vv = 0.0
        if n_vh_tot > 0:
            running_loss_vh /= n_vh_tot
        else:
            running_loss_vh = 0.0
    else:
        running_loss = None
        running_loss_insitu = None
        running_loss_vv = None
        running_loss_vh = None
    if len(out_mu_i) > 0:
        mu_i = torch.cat(out_mu_i).squeeze().numpy()
        logv_i = torch.cat(out_logv_i).squeeze().numpy()
        true_i = torch.cat(out_true_i).squeeze().numpy()
    else:
        mu_i = np.array([])
        logv_i = np.array([])
        true_i = np.array([])
    if len(out_mu_vv) > 0:
        mu_vv = torch.cat(out_mu_vv).squeeze().numpy()
        logv_vv = torch.cat(out_logv_vv).squeeze().numpy()
        true_vv = torch.cat(out_true_vv).squeeze().numpy()
    else:
        mu_vv = np.array([])
        logv_vv = np.array([])
        true_vv = np.array([])
    if len(out_mu_vh) > 0:
        mu_vh = torch.cat(out_mu_vh).squeeze().numpy()
        logv_vh = torch.cat(out_logv_h).squeeze().numpy()
        true_vh = torch.cat(out_true_vh).squeeze().numpy()
    else:
        mu_vh = np.array([])
        logv_vh = np.array([])
        true_vh = np.array([])
    return(
        model,
        running_loss,
        running_loss_insitu,
        running_loss_vv,
        running_loss_vh,
        mu_i,
        logv_i,
        mu_vv,
        logv_vv,
        mu_vh,
        logv_vh,
        true_i,
        true_vv,
        true_vh,
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
    short_data = data[0]
    long_data = data[1]
    static_data = data[2]
    lfmc = data[3]
    source = data[4]
    info = data[5]
    stratifier = data[6]
    (
        test_short_data, test_long_data, test_static_data, test_lfmc, test_source, test_info, test_stratifier,
        remaining_short_data, remaining_long_data, remaining_static_data, remaining_lfmc, remaining_source, remaining_info, remaining_stratifier
    ) = mask_location_data_fast(
        this_locs,
        short_data,
        long_data,
        static_data,
        lfmc,
        source,
        info,
        stratifier
    )
    # split out the validation data
    remaining_insitu_obs = remaining_info[remaining_info['source_legible'] == 'nfmd'].shape[0]
    remaining_vv_obs = remaining_info[remaining_info['source_legible'] == 'vv'].shape[0]
    remaining_vh_obs = remaining_info[remaining_info['source_legible'] == 'vh'].shape[0]
    
    num_val_obs_insitu = remaining_insitu_obs * val_split
    num_val_obs_vv = remaining_vv_obs * val_split
    num_val_obs_vh = remaining_vh_obs * val_split
    val_locs = create_site_split(
        remaining_info,
        desired_insitu_sample_size=int(num_val_obs_insitu),
        desired_vv_sample_size=int(num_val_obs_vv),
        desired_vh_sample_size=int(num_val_obs_vh),
        stratifier=remaining_stratifier
    )
    val_locs = np.array(val_locs)
    # perform the same masking as was done for the test sites
    (
        val_short_data, val_long_data, val_static_data, val_lfmc, val_source, val_info, val_stratifier,
        train_short_data, train_long_data, train_static_data, train_lfmc, train_source, train_info, train_stratifier
    ) = mask_location_data_fast(
        val_locs,
        remaining_short_data,
        remaining_long_data,
        remaining_static_data,
        remaining_lfmc,
        remaining_source,
        remaining_info,
        remaining_stratifier
    )
    # Sanity check
    total_test = test_info.shape[0]
    insitu_test = test_info[test_info['source'] == 'nfmd'].shape[0]
    vv_test = test_info[test_info['source'] == 'vv'].shape[0]
    vh_test = test_info[test_info['source'] == 'vh'].shape[0]
    total_val = val_info.shape[0]
    insitu_val = val_info[val_info['source'] == 'nfmd'].shape[0]
    vv_val = val_info[val_info['source'] == 'vv'].shape[0]
    vh_val = val_info[val_info['source'] == 'vh'].shape[0]
    total_train = train_info.shape[0]
    insitu_train = train_info[train_info['source'] == 'nfmd'].shape[0]
    vv_train = train_info[train_info['source'] == 'vv'].shape[0]
    vh_train = train_info[train_info['source'] == 'vh'].shape[0]
    print(
        f"Test: {total_test} ({insitu_test} insitu, {vv_test} vv, {vh_test} vh) | "
        f"Val: {total_val} ({insitu_val} insitu, {vv_val} vv, {vh_val} vh) | "
        f"Train: {total_train} ({insitu_train} insitu, {vv_train} vv, {vh_train} vh)"
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
    train_short_mean = np.nanmean(train_short_data, axis=(0,1))
    train_short_std = np.nanstd(train_short_data, axis=(0,1))
    train_long_std = np.nanstd(train_long_data, axis=(0,1))
    train_long_mean = np.nanmean(train_long_data, axis=(0,1))
    train_static_mean = np.nanmean(train_static_data, axis=(0,1))
    train_static_std = np.nanstd(train_static_data, axis=(0,1))
    y_mean = np.nanmean(train_lfmc)
    y_std = np.nanstd(train_lfmc)
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
        train_short_data[:,:,v] = (train_short_data[:,:,v] - train_short_mean[v]) / train_short_std[v]
        val_short_data[:,:,v] = (val_short_data[:,:,v] - train_short_mean[v]) / train_short_std[v]
        test_short_data[:,:,v] = (test_short_data[:,:,v] - train_short_mean[v]) / train_short_std[v]
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
        train_long_data[:,:,v] = (train_long_data[:,:,v] - train_long_mean[v]) / train_long_std[v]
        val_long_data[:,:,v] = (val_long_data[:,:,v] - train_long_mean[v]) / train_long_std[v]
        test_long_data[:,:,v] = (test_long_data[:,:,v] - train_long_mean[v]) / train_long_std[v]
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
        train_static_data[:,:,v] = (train_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        val_static_data[:,:,v] = (val_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
        test_static_data[:,:,v] = (test_static_data[:,:,v] - train_static_mean[v]) / train_static_std[v]
    train_lfmc = (train_lfmc - y_mean) / y_std
    val_lfmc = (val_lfmc - y_mean) / y_std
    test_lfmc = (test_lfmc - y_mean) / y_std
    # save the normalization parameters for later use
    norm_params = {
        'train_short_mean': train_short_mean.tolist(),
        'train_short_std': train_short_std.tolist(),
        'train_long_mean': train_long_mean.tolist(),
        'train_long_std': train_long_std.tolist(),
        'train_static_mean': train_static_mean.tolist(),
        'train_static_std': train_static_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist()
    }
    # save the normalization parameters to disk
    with open(os.path.join(fold_save_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f)
    # create the datasets and dataloaders
    # stratify by land cover type to stabilize training
    # there are nan's somewhere... we need to find out where
    if np.isnan(train_stratifier).any():
        raise ValueError("NaN found in train_stratifier")
    if torch.isnan(train_short_data).any():
        raise ValueError("NaN found in train_short_data")
    if torch.isnan(train_long_data).any():
        raise ValueError("NaN found in train_long_data")
    if torch.isnan(train_static_data).any():
        raise ValueError("NaN found in train_static_data")
    if torch.isnan(train_lfmc).any():
        raise ValueError("NaN found in train_lfmc")
    train_dataset = TensorDataset(
        train_short_data,
        train_long_data,
        train_static_data,
        train_lfmc,
        train_source
    )
    batch_sampler = StratifiedBatchSampler(
        labels=train_stratifier,
        batch_size=batch_size,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        pin_memory=True
    )
    val_dataset = TensorDataset(
        val_short_data,
        val_long_data,
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
        test_short_data,
        test_long_data,
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
    train_loss_vv = []
    train_loss_vh = []
    val_loss = []
    val_loss_insitu = []
    val_loss_vv = []
    val_loss_vh = []
    global_step = 0
    for epoch in range(1,max_epochs):
        print(f'Fold {this_fold_num}, Epoch {epoch}/{max_epochs}')
        model.train()
        (
            model,
            this_train_loss,
            this_train_loss_insitu,
            this_train_loss_vv,
            this_train_loss_vh,
            _,
            _,
            _,
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
            lambda_vv=rs_factor,
            lambda_vh=rs_factor
        )
        train_loss.append(this_train_loss)
        train_loss_insitu.append(this_train_loss_insitu)
        train_loss_vv.append(this_train_loss_vv)
        train_loss_vh.append(this_train_loss_vh)
        print(f'Training total loss: {this_train_loss:.4f}')
        print(f'Training insitu loss: {this_train_loss_insitu:.4f}')
        print(f'Training vv loss: {this_train_loss_vv:.4f}')
        print(f'Training vh loss: {this_train_loss_vh:.4f}')
        scheduler.step()
        # run the validation
        model.eval()
        (
            model,
            this_val_loss,
            this_val_loss_insitu,
            this_val_loss_vv,
            this_val_loss_vh,
            mu_i_val,
            logv_i_val,
            mu_vv_val,
            logv_vv_val,
            mu_vh_val,
            logv_vh_val,
            true_i,
            true_vv,
            true_vh,
            _
        ) = run_model(
            model,
            val_loader,
            device,
            criterion,
            train_model=False,
            lambda_vv=rs_factor,
            lambda_vh=rs_factor
        )
        val_loss.append(this_val_loss)
        val_loss_insitu.append(this_val_loss_insitu)
        val_loss_vv.append(this_val_loss_vv)
        val_loss_vh.append(this_val_loss_vh)
        print(f'Validation total loss: {this_val_loss:.4f}')
        print(f'Validation insitu loss: {this_val_loss_insitu:.4f}')
        print(f'Validation vv loss: {this_val_loss_vv:.4f}')
        print(f'Validation vh loss: {this_val_loss_vh:.4f}')
        # denorm
        lfmc_i_val_only = mu_i_val[val_source.numpy() == 0 ] * y_std + y_mean
        lfmc_std_i_val_only = np.sqrt(np.exp(logv_i_val[val_source.numpy() == 0])) * y_std
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
        # and for vv data
        if len(true_vv) > 0:
            lfmc_vv_val_only = mu_vv_val[val_source.numpy() == 1] * y_std + y_mean
            lfmc_std_vv_val_only = np.sqrt(np.exp(logv_vv_val[val_source.numpy() == 1])) * y_std
            lfmc_vv_val_true = true_vv * y_std + y_mean
            val_mae_vv = np.mean(np.abs(lfmc_vv_val_only - lfmc_vv_val_true))
            val_r2_vv = r2_score(lfmc_vv_val_true, lfmc_vv_val_only)
            val_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vv_val_only) + ((lfmc_vv_val_true - lfmc_vv_val_only) ** 2) / (lfmc_std_vv_val_only ** 2)))
            val_rmse_rs = np.sqrt(np.mean((lfmc_rs_val_only - lfmc_rs_val_true) ** 2))
            ## also calculate the mixtures
            #lfmc_rs_mix_val = mu_mix_val[val_source.numpy() == 0] * y_std + y_mean
            #lfmc_std_rs_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 0])) * y_std
            #val_mae_rs_mix = np.mean(np.abs(lfmc_rs_mix_val - lfmc_rs_val_true))
            #val_r2_rs_mix = r2_score(lfmc_rs_val_true, lfmc_rs_mix_val)
            #val_nll_rs_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_mix_val) + ((lfmc_rs_val_true - lfmc_rs_mix_val) ** 2) / (lfmc_std_rs_mix_val ** 2)))
            #val_rmse_rs_mix = np.sqrt(np.mean((lfmc_rs_mix_val - lfmc_rs_val_true) ** 2))
        if len(true_vh) > 0:
            lfmc_vh_val_only = mu_vh_val[val_source.numpy() == 2] * y_std + y_mean
            lfmc_std_vh_val_only = np.sqrt(np.exp(logv_vh_val[val_source.numpy() == 2])) * y_std
            lfmc_vh_val_true = true_vh * y_std + y_mean
            val_mae_vh = np.mean(np.abs(lfmc_vh_val_only - lfmc_vh_val_true))
            val_r2_vh = r2_score(lfmc_vh_val_true, lfmc_vh_val_only)
            val_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vh_val_only) + ((lfmc_vh_val_true - lfmc_vh_val_only) ** 2) / (lfmc_std_vh_val_only ** 2)))
            val_rmse_vh = np.sqrt(np.mean((lfmc_vh_val_only - lfmc_vh_val_true) ** 2))
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
        if len(true_vv) > 0:
            print(
                f'Validation VV MAE: {val_mae_vv:.4f}, RMSE: {val_rmse_vv:.4f}, R2: {val_r2_vv:.4f}, NLL: {val_nll_vv:.4f}'
            )
            #print(
            #    f'Validation VV Mixture MAE: {val_mae_vv_mix:.4f}, RMSE: {val_rmse_vv_mix:.4f}, R2: {val_r2_vv_mix:.4f}, NLL: {val_nll_vv_mix:.4f}'
            #)
        if len(true_vh) > 0:
            print(
                f'Validation VH MAE: {val_mae_vh:.4f}, RMSE: {val_rmse_vh:.4f}, R2: {val_r2_vh:.4f}, NLL: {val_nll_vh:.4f}'
            )
            #print(
            #    f'Validation VH Mixture MAE: {val_mae_vh_mix:.4f}, RMSE: {val_rmse_vh_mix:.4f}, R2: {val_r2_vh_mix:.4f}, NLL: {val_nll_vh_mix:.4f}'
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
        val_loss_vv,
        val_loss_vh,
        mu_i_val,
        logv_i_val,
        mu_vv_val,
        logv_vv_val,
        mu_vh_val,
        logv_vh_val,
        true_i_val,
        true_vv_val,
        true_vh_val,
        _
    ) = run_model(
        model,
        val_loader,
        device,
        criterion,
        train_model=False,
        lambda_vv=rs_factor,
        lambda_vh=rs_factor
    )
    lfmc_i_val_only = mu_i_val[val_source.numpy() == 0] * y_std + y_mean
    lfmc_std_i_val_only = np.sqrt(np.exp(logv_i_val[val_source.numpy() == 0])) * y_std
    lfmc_i_val_true = true_i * y_std + y_mean
    # calculate metrics of interet
    val_mae = np.mean(np.abs(lfmc_i_val_only - lfmc_i_val_true))
    val_r2 = r2_score(lfmc_i_val_true, lfmc_i_val_only)
    val_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_i_val_only) + ((lfmc_i_val_true - lfmc_i_val_only) ** 2) / (lfmc_std_i_val_only ** 2)))
    val_rmse = np.sqrt(np.mean((lfmc_i_val_only - lfmc_i_val_true) ** 2))
    # and for vv data
    if len(true_vv) > 0:
        lfmc_vv_val_only = mu_vv_val[val_source.numpy() == 1] * y_std + y_mean
        lfmc_std_vv_val_only = np.sqrt(np.exp(logv_vv_val[val_source.numpy() == 1])) * y_std
        lfmc_vv_val_true = true_vv * y_std + y_mean
        val_mae_vv = np.mean(np.abs(lfmc_vv_val_only - lfmc_vv_val_true))
        val_r2_vv = r2_score(lfmc_vv_val_true, lfmc_vv_val_only)
        val_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vv_val_only) + ((lfmc_vv_val_true - lfmc_vv_val_only) ** 2) / (lfmc_std_vv_val_only ** 2)))
        val_rmse_vv = np.sqrt(np.mean((lfmc_vv_val_only - lfmc_vv_val_true) ** 2))
    else:
        lfmc_vv_val_only = np.nan
        lfmc_std_vv_val_only = np.nan
        lfmc_vv_val_true = np.nan
        val_mae_vv = np.nan
        val_r2_vv = np.nan
        val_nll_vv = np.nan
        val_rmse_vv = np.nan
    if len(true_vh) > 0:
        lfmc_vh_val_only = mu_vh_val[val_source.numpy() == 2] * y_std + y_mean
        lfmc_std_vh_val_only = np.sqrt(np.exp(logv_vh_val[val_source.numpy() == 2])) * y_std
        lfmc_vh_val_true = true_vh * y_std + y_mean
        val_mae_vh = np.mean(np.abs(lfmc_vh_val_only - lfmc_vh_val_true))
        val_r2_vh = r2_score(lfmc_vh_val_true, lfmc_vh_val_only)
        val_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vh_val_only) + ((lfmc_vh_val_true - lfmc_vh_val_only) ** 2) / (lfmc_std_vh_val_only ** 2)))
        val_rmse_vh = np.sqrt(np.mean((lfmc_vh_val_only - lfmc_vh_val_true) ** 2))
    else:
        lfmc_vh_val_only = np.nan
        lfmc_std_vh_val_only = np.nan
        lfmc_vh_val_true = np.nan
        val_mae_vh = np.nan
        val_r2_vh = np.nan
        val_nll_vh = np.nan
        val_rmse_vh = np.nan
    # average values for sanity check
    #avg_val_pred = np.mean(lfmc_i_val)
    #avg_val_true = np.mean(lfmc_i_val_true)
    #avg_val_std = np.mean(lfmc_std_i_val)
    print(
        f'Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, NLL: {val_nll:.4f}'
    )
    if len(true_vv) > 0:
        print(
            f'Validation VV MAE: {val_mae_vv:.4f}, RMSE: {val_rmse_vv:.4f}, R2: {val_r2_vv:.4f}, NLL: {val_nll_vv:.4f}'
        )
    if len(true_vh) > 0:
        print(
            f'Validation VH MAE: {val_mae_vh:.4f}, RMSE: {val_rmse_vh:.4f}, R2: {val_r2_vh:.4f}, NLL: {val_nll_vh:.4f}'
        )
    # run the test
    (
        model,
        test_loss,
        test_loss_insitu,
        test_loss_vv,
        test_loss_vh,
        mu_i_test,
        logv_i_test,
        mu_vv_test,
        logv_vv_test,
        mu_vh_test,
        logv_vh_test,
        true_i_test,
        true_vv_test,
        true_vh_test,
        _
    ) = run_model(
        model,
        test_loader,
        device,
        criterion,
        train_model=False,
        lambda_vv=rs_factor,
        lambda_vh=rs_factor
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
        lfmc_i_test_only = mu_i_test[test_source.numpy() == 0] * y_std + y_mean
        lfmc_std_i_test_only = np.sqrt(np.exp(logv_i_test[test_source.numpy() == 0])) * y_std
        lfmc_i_test_true = true_i_test * y_std + y_mean
        # calculate metrics of interet
        test_mae = np.mean(np.abs(lfmc_i_test_only - lfmc_i_test_true))
        test_r2 = r2_score(lfmc_i_test_true, lfmc_i_test_only)
        test_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_i_test_only) + ((lfmc_i_test_true - lfmc_i_test_only) ** 2) / (lfmc_std_i_test_only ** 2)))
        test_rmse = np.sqrt(np.mean((lfmc_i_test_only - lfmc_i_test_true) ** 2))
        # and for vv data
        if len(true_vv) > 0:
            lfmc_vv_test_only = mu_vv_test[test_source.numpy() == 1] * y_std + y_mean
            lfmc_std_vv_test_only = np.sqrt(np.exp(logv_vv_test[test_source.numpy() == 1])) * y_std
            lfmc_vv_test_true = true_vv_test * y_std + y_mean
            test_mae_vv = np.mean(np.abs(lfmc_vv_test_only - lfmc_vv_test_true))
            test_r2_vv = r2_score(lfmc_vv_test_true, lfmc_vv_test_only)
            test_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vv_test_only) + ((lfmc_vv_test_true - lfmc_vv_test_only) ** 2) / (lfmc_std_vv_test_only ** 2)))
            test_rmse_vv = np.sqrt(np.mean((lfmc_vv_test_only - lfmc_vv_test_true) ** 2))
        else:
            lfmc_vv_test_only = np.nan
            lfmc_std_vv_test_only = np.nan
            lfmc_vv_test_true = np.nan
            test_mae_vv = np.nan
            test_r2_vv = np.nan
            test_nll_vv = np.nan
            test_rmse_vv = np.nan
        if len(true_vh) > 0:
            lfmc_vh_test_only = mu_vh_test[test_source.numpy() == 2] * y_std + y_mean
            lfmc_std_vh_test_only = np.sqrt(np.exp(logv_vh_test[test_source.numpy() == 2])) * y_std
            lfmc_vh_test_true = true_vh_test * y_std + y_mean
            test_mae_vh = np.mean(np.abs(lfmc_vh_test_only - lfmc_vh_test_true))
            test_r2_vh = r2_score(lfmc_vh_test_true, lfmc_vh_test_only)
            test_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_vh_test_only) + ((lfmc_vh_test_true - lfmc_vh_test_only) ** 2) / (lfmc_std_vh_test_only ** 2)))
            test_rmse_vh = np.sqrt(np.mean((lfmc_vh_test_only - lfmc_vh_test_true) ** 2))
        else:
            lfmc_vh_test_only = np.nan
            lfmc_std_vh_test_only = np.nan
            lfmc_vh_test_true = np.nan
            test_mae_vh = np.nan
            test_r2_vh = np.nan
            test_nll_vh = np.nan
            test_rmse_vh = np.nan
        # average testues for sanity check
        #avg_test_pred = np.mean(lfmc_i_test)
        #avg_test_true = np.mean(lfmc_i_test_true)
        #avg_test_std = np.mean(lfmc_std_i_test)
        print(
            f'testidation MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, NLL: {test_nll:.4f}'
        )
        if len(true_vv) > 0:
            print(
                f'testidation VV MAE: {test_mae_vv:.4f}, RMSE: {test_rmse_vv:.4f}, R2: {test_r2_vv:.4f}, NLL: {test_nll_vv:.4f}'
            )
        if len(true_vh) > 0:
            print(
                f'testidation VH MAE: {test_mae_vh:.4f}, RMSE: {test_rmse_vh:.4f}, R2: {test_r2_vh:.4f}, NLL: {test_nll_vh:.4f}'
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
            'loss_vv':val_loss_vv,
            'loss_vh':val_loss_vh,
            'lfmc_insitu_preds':lfmc_i_val_only,
            'lfmc_insitu_std':lfmc_std_i_val_only,
            'lfmc_vv_preds':lfmc_vv_val_only,
            'lfmc_vv_std':lfmc_std_vv_val_only,
            'lfmc_vh_preds':lfmc_vh_val_only,
            'lfmc_vh_std':lfmc_std_vh_val_only,
            'lfmc_insitu_true':lfmc_i_val_true,
            'lfmc_vv_true':lfmc_vv_val_true,
            'lfmc_vh_true':lfmc_vh_val_true
        },
        os.path.join(fold_save_dir,'val_outputs.pth')
    )
    torch.save(
        {
            'loss':test_loss,
            'loss_insitu':test_loss_insitu,
            'loss_vv':test_loss_vv,
            'loss_vh':test_loss_vh,
            'lfmc_insitu_preds':lfmc_i_test_only,
            'lfmc_insitu_std':lfmc_std_i_test_only,
            'lfmc_vv_preds':lfmc_vv_test_only,
            'lfmc_vv_std':lfmc_std_vv_test_only,
            'lfmc_vh_preds':lfmc_vh_test_only,
            'lfmc_vh_std':lfmc_std_vh_test_only,
            'lfmc_insitu_true':lfmc_i_test_true,
            'lfmc_vv_true':lfmc_vv_test_true,
            'lfmc_vh_true':lfmc_vh_test_true
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
    # load passed hyperparameter settings
    # configs
    # directories, etc.
    input_data_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs_base'
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/outputs'
    # training settings
    batch_size = 128
    max_epochs = 100
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available, using CPU. This will be slow!')
    warmup_steps = 400
    base_lr = lr
    warmup_start_lr = 1e-6
    val_split = 0.2
    adam_weight_decay = 1e-4
    patience = 8
    # model hyperparameters (go back to the github and get what I deleted here)
    d_model = 64
    nhead = 2
    num_layers = 2
    dim_feedforward = 128
    dropout = 0.2
    rs_factor = 1.0 # weighting for RS loss
    # long model hyperparameters
    long_d_model = 64
    long_nhead = 2
    long_num_layers = 2
    long_dim_feedforward = 128
    long_out_dim = 64
    # load the data
    datasets = load_data(input_data_dir)
    # early check that we don't have nans ANYWHERE
    for i, data in enumerate(datasets):
        if type(data) is np.ndarray:
            if np.isnan(data).any():
                raise ValueError(f'Data array {i} contains NaNs!')
        elif type(data) is pd.DataFrame:
            if data.isnull().values.any():
                raise ValueError(f'DataFrame {i} contains NaNs!')
        elif type(data) is torch.Tensor:
            if torch.isnan(data).any():
                raise ValueError(f'Tensor {i} contains NaNs!')
    var_names = json.load(
        open(os.path.join(input_data_dir, 'var_names.json'), 'r')
    )
    # get the input dims that we are working with to build the model
    short_input_dim = datasets[0].shape[-1]
    long_input_dim = datasets[1].shape[-1]
    static_input_dim = datasets[2].shape[-1]
    # build the folds by location
    short_data = datasets[0]
    long_data = datasets[1]
    static_data = datasets[2]
    info = datasets[5]
    stratifier = datasets[6]
    num_rs_obs = info[info['source'] == 'rs'].shape[0]
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}_rsf{rs_factor}_rsobs{num_rs_obs}'
        f'_dmlong{long_d_model}_nhlong{long_nhead}_nllong{long_num_layers}'
        f'_dflong{long_dim_feedforward}_outlong{long_out_dim}'
        f'_basic'
    )
    # set up the save directories
    full_save_dir = os.path.join(save_dir, this_model_name)
    if os.path.exists(full_save_dir):
        shutil.rmtree(full_save_dir)
    os.makedirs(full_save_dir)
    n_folds = 10
    total_obs_insitu_obs = (
        info[info['source_legible'] == 'nfmd'].shape[0]
    )
    total_obs_vv_obs = (
        info[info['source_legible'] == 'vh'].shape[0]
    )
    total_obs_vh_obs = (
        info[info['source_legible'] == 'vh'].shape[0]
    )
    desired_insitu_obs_per_fold = total_obs_insitu_obs / n_folds
    desired_vv_obs_per_fold = total_obs_vv_obs / n_folds
    desired_vh_obs_per_fold = total_obs_vh_obs / n_folds
    fold_locs = {}
    used_sites = []
    for fold in range(n_folds):
        print(f'Getting locations for fold {fold+1}/{n_folds}')
        this_locs = create_site_split(
            info,
            desired_insitu_sample_size=int(desired_insitu_obs_per_fold),
            desired_vv_sample_size=int(desired_vv_obs_per_fold),
            desired_vh_sample_size=int(desired_vh_obs_per_fold),
            used_sites=used_sites,
            stratifier=stratifier
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
        model = LFMCTransformerMultiTaskLongClimate(
            short_input_dim=short_input_dim,
            static_input_dim=static_input_dim,
            long_input_dim=long_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=2,
            long_d_model=long_d_model,
            long_nhead=long_nhead,
            long_num_layers=long_num_layers,
            long_dim_feedforward=long_dim_feedforward,
            long_out_dim=long_out_dim
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
            rs_factor,
        )
    # one final version of the model trained on all the data
    print('Training final model on all data')
    model = LFMCTransformerMultiTaskLongClimate(
        short_input_dim=short_input_dim,
        static_input_dim=static_input_dim,
        long_input_dim=long_input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=2,
        long_d_model=long_d_model,
        long_nhead=long_nhead,
        long_num_layers=long_num_layers,
        long_dim_feedforward=long_dim_feedforward,
        long_out_dim=long_out_dim
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
    #parser = argparse.ArgumentParser(
    #    description='Train LFMC Transformer'
    #)
    #parser.add_argument(
    
    
    main()