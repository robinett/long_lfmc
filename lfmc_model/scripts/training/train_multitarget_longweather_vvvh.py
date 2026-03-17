import os
import sys
import copy
import json
import shutil
import random
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
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_multitask import LFMCTransformer
from transformer_multitask_longclimate import LFMCTransformer as LFMCTransformerMultiTask
#from transformer_multitask_longclimate import LFMCTransformer as LFMCTransformerMultiTaskLongClimate
#from transformer_multitask_longclimate_uncertainty import LFMCTransformer as LFMCTransformerMultiTaskLongClimate
import plotting

import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",
    category=UserWarning,
)

def check_tensor(name, x):
    print(
        name,
        "nan:", torch.isnan(x).any().item(),
        "inf:", torch.isinf(x).any().item(),
        "min:", x.min().item(),
        "max:", x.max().item()
    )

def _safe_mae_rmse_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan, np.nan, np.nan
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    if y_true.size == 0:
        return np.nan, np.nan, np.nan
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if y_true.size < 2:
        r2 = np.nan
    elif np.allclose(y_true, y_true[0]):
        r2 = np.nan
    else:
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = np.nan
    return mae, rmse, r2

class GradNorm:
    """
    GradNorm (Chen et al., 2018) for *T* tasks.

    Call pattern (once *only when you want to* apply GradNorm):

        total_loss, L_grad = grad_norm.update(
            task_losses,       # list of scalar tensors  [L0, L1, …]
            task_weights,      # nn.Parameter, len = T
            model,             # your (possibly DDP-wrapped) nn.Module
            shared_param_names # optional list[str] names of shared params
        )

    You then:
        total_loss.backward(retain_graph=True)
        L_grad.backward()      # updates task_weights only
    """
    def __init__(self, num_tasks: int, alpha: float = 0.5, device="cuda"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha should be in [0, 1]")
        self.T = num_tasks
        self.alpha = alpha
        self.device = torch.device(device)
        self.L0 = None  # will store initial losses
    # ------------------------------------------------------------
    def _unwrap(self, model):
        """Return the underlying module if model is DDP-wrapped."""
        return model.module if hasattr(model, "module") else model
    def _shared_params(self, model, shared_param_names):
        net = self._unwrap(model)
        if shared_param_names is None:
            # heuristic: everything *not* containing "head" is shared
            return [p for n, p in net.named_parameters()
                    if "head" not in n and p.requires_grad]
        return [p for n, p in net.named_parameters()
                if n in shared_param_names]
    def _grad_norm(self, scalar, params):
        grads = torch.autograd.grad(
            scalar, params,
            retain_graph=True, create_graph=True, allow_unused=True
        )
        return torch.norm(
            torch.stack([g.norm() for g in grads if g is not None])
        )
    # ------------------------------------------------------------
    def update(
        self,
        task_losses,             # list[Tensor] length T
        task_weights,            # nn.Parameter length T
        model,                   # nn.Module (DDP or not)
        shared_param_names=None  # optional list[str]
    ):
        """
        Returns
        -------
        total_loss : Tensor  (weighted sum, use for main backward)
        L_grad     : Tensor  (GradNorm loss, backprop into weights only)
        """
        L_vec = torch.stack(task_losses)               # shape (T,)
        w_pos = F.relu(task_weights)                   # keep ≥ 0
        total_loss = (w_pos * L_vec).sum()
        # Record initial (un-weighted) losses once
        if self.L0 is None:
            self.L0 = L_vec.detach()
        # ---------- compute GradNorm quantities ----------
        shared_params = self._shared_params(model, shared_param_names)
        G = torch.stack([
            self._grad_norm(w_pos[i] * L_vec[i], shared_params)
            for i in range(self.T)
        ])                                             # (T,)
        G_bar = G.mean().detach()
        with torch.no_grad():
            r = (L_vec.detach() / (self.L0 + 1e-8))
            r = r / r.mean()                           # relative inverse rate
        target = G_bar * (r ** self.alpha)
        L_grad = F.l1_loss(G, target, reduction="sum")
        return total_loss, L_grad

def get_args():
    parser = argparse.ArgumentParser(
        description='Train LFMC Transformer'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        help='Directory containing input data',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Directory to save model outputs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training',
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate for training',
    )
    parser.add_argument(
        '--val_split',
        type=float,
        help='Fraction of data to use for validation',
    )
    parser.add_argument(
        '--adam_wd',
        type=float,
        help='Weight decay for Adam optimizer',
    )
    parser.add_argument(
        '--d_model',
        type=int,
        help='Dimensionality of the model',
    )
    parser.add_argument(
        '--nhead',
        type=int,
        help='Number of attention heads',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        help='Number of layers in the transformer',
    )
    parser.add_argument(
        '--dim_feedforward',
        type=int,
        help='Dimensionality of the feedforward layer',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate',
    )
    parser.add_argument(
        '--long_d_model',
        type=int,
        help='Dimensionality of the long-term model',
    )
    parser.add_argument(
        '--long_nhead',
        type=int,
        help='Number of attention heads in the long-term model',
    )
    parser.add_argument(
        '--long_num_layers',
        type=int,
        help='Number of layers in the long-term transformer',
    )
    parser.add_argument(
        '--long_dim_feedforward',
        type=int,
        help='Dimensionality of the feedforward layer in the long-term model',
    )
    parser.add_argument(
        '--long_out_dim',
        type=int,
        help='Dimensionality of the output layer in the long-term model',
    )
    parser.add_argument(
        '--num_tasks',
        type=int,
        help='Number of tasks for gradient normalization',
    )
    parser.add_argument(
        '--task_weight_type',
        type=str,
        choices=['default', 'manual', 'gradnorm'],
        help='Type of task weighting to use',
    )
    parser.add_argument(
        '--manual_task_weights',
        type=float,
        nargs='+',
        default=None,
        help='Manual task weights (must match num_tasks)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global random seed for model initialization and training randomness',
    )
    parser.add_argument(
        '--split_seed',
        type=int,
        default=42,
        help='Seed for fold/site split generation (keep fixed across ensemble members)',
    )
    parser.add_argument(
        '--batch_seed',
        type=int,
        default=None,
        help='Seed for stratified batch sampling (defaults to --seed when omitted)',
    )
    parser.add_argument(
        '--fold_info_in',
        type=str,
        default=None,
        help='Optional path to an existing fold_info.json to reuse exact folds',
    )
    parser.add_argument(
        '--run_tag',
        type=str,
        default=None,
        help='Optional suffix appended to the model output directory name (e.g., seed000)',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If set, delete an existing output directory before training',
    )
    args = parser.parse_args()
    return args


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
    orig_df: pd.DataFrame,
    goal: int,
    must_use_sites: list[tuple] = None,
    not_allowed_sites: list[tuple] = None,
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
    orig_df : DataFrame
        Original DataFrame (unfiltered) for getting the original
        stratifier distribution. Can be same as df if no filtering
        has been done.
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
    # get rid of sites we can't use if needed
    if not_allowed_sites is not None:
        df_clean = df[~df[['lat', 'lon']].apply(tuple, axis=1).isin(not_allowed_sites)]
    else:
        df_clean = df
    # --- 1. overall obs per stratum (by n) ----------------------
    obs_per_stratum = df_clean.groupby(strat_col)["n"].sum()
    total_obs = int(obs_per_stratum.sum())
    # If you ask for more than available, cap at total.
    #print(goal)
    #if goal > total_obs:
    #    goal = total_obs
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
    # use the sites that we are mandated to, if passed
    chosen_parts = []
    if must_use_sites is not None:
        must_use_df = df[
            df[['lat', 'lon']].apply(tuple, axis=1).isin(must_use_sites)
        ]
        # and get rid from normal df because we will just add here
        df_clean = df_clean[~df_clean[['lat', 'lon']].apply(tuple, axis=1).isin(must_use_sites)]
        # update the quotas that we need to add based on what we now already have
        for s, quota in quotas.items():
            this_quota_data = must_use_df[must_use_df[strat_col] == s]
            # continue if empty
            if this_quota_data.empty:
                continue
            quotas[s] -= int(this_quota_data['n'].sum())
            quotas[s] = max(0, quotas[s])  # Cap at 0
            chosen_parts.append(this_quota_data)
    # --- 3. pick minimal sites per stratum to hit each quota ----
    for s, quota in quotas.items():
        if quota <= 0:
            continue
        group = df_clean[df_clean[strat_col] == s].copy()
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
        this_chosen = group.iloc[:k+1]
        chosen_parts.append(this_chosen)
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
    y,          # torch.Tensor [N, ...]
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
        return (short_data[empty], long_data[empty], static_data[empty], y[empty], source[empty], info.iloc[:0], stratifier[empty],
                short_data, long_data, static_data, y, source, info, stratifier)

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
    kept_y            = y[m_kept]
    kept_source       = source[m_kept]
    kept_info         = info.loc[mask_kept]
    kept_info         = kept_info.reset_index(drop=True)
    kept_stratifier   = stratifier[m_kept]

    remaining_short_data   = short_data[m_rem]
    remaining_long_data    = long_data[m_rem]
    remaining_static_data  = static_data[m_rem]
    remaining_y            = y[m_rem]
    remaining_source       = source[m_rem]
    remaining_info         = info.loc[mask_remaining]
    remaining_info         = remaining_info.reset_index(drop=True)
    remaining_stratifier   = stratifier[m_rem]

    return (
        kept_short_data, kept_long_data, kept_static_data, kept_y, kept_source, kept_info, kept_stratifier,
        remaining_short_data, remaining_long_data, remaining_static_data, remaining_y, remaining_source, remaining_info, remaining_stratifier
    )

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        """
        MSE loss with masking and finite-value filtering.

        Args:
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred:   (...,) predicted mean
            target: (...,) ground truth
            mask:   optional boolean mask for valid entries
                    (same shape as pred/target) or broadcastable
        """
        # start with finite target mask
        valid = torch.isfinite(target)

        if mask is not None:
            # allow mask to be float/bool; convert to bool
            # (e.g., 1/0 or True/False)
            if mask.dtype != torch.bool:
                mask = mask != 0
            valid = valid & mask

        # filter
        pred = pred[valid]
        target = target[valid]

        if target.numel() == 0:
            # no valid targets: return scalar 0 with grad
            return pred.new_tensor(0.0, requires_grad=True)

        mse = (pred - target) ** 2

        if self.reduction == "mean":
            return mse.mean()
        elif self.reduction == "sum":
            return mse.sum()
        else:  # 'none'
            return mse

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
    Y = torch.load(
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
    assert not torch.isnan(Y).any(), "NaN found in Y"
    assert not torch.isnan(source).any(), "NaN found in source"
    assert not np.isnan(stratifier).any(), "NaN found in stratifier"
    # load the center info
    center_info = pd.read_csv(os.path.join(center_data_dir, 'info.csv'))
    all_center_data = [
        X_short, X_long, X_static, Y, source, center_info, stratifier
    ]
    return all_center_data


def _round_site_tuple(site, round_decimals: int = 10):
    return (
        round(float(site[0]), round_decimals),
        round(float(site[1]), round_decimals),
    )


def _dedupe_sites_in_order(sites, round_decimals: int = 10):
    ordered = []
    seen = set()
    for site in sites:
        if site is None:
            continue
        rounded = _round_site_tuple(site, round_decimals=round_decimals)
        if rounded in seen:
            continue
        seen.add(rounded)
        ordered.append(rounded)
    return ordered


def _extract_climate_zone_codes_from_static_tensor(static_data, static_var_names):
    climate_slots = []
    climate_codes = []
    for idx, feat in enumerate(static_var_names):
        if feat.startswith("climate_zone_"):
            climate_slots.append(idx)
            climate_codes.append(int(feat.split("_")[-1]))
    if not climate_slots:
        raise ValueError("No climate_zone_* features found in static_var_names")

    if isinstance(static_data, torch.Tensor):
        static_np = static_data.detach().cpu().numpy()
    else:
        static_np = np.asarray(static_data)
    if static_np.ndim == 3:
        static_np = static_np[:, 0, :]
    if static_np.ndim != 2:
        raise ValueError(f"Expected static data with 2 or 3 dims, got shape {static_np.shape}")

    climate_block = static_np[:, climate_slots]
    active_mask = climate_block > 0.5
    active_counts = active_mask.sum(axis=1)
    if np.any(active_counts == 0):
        raise ValueError(
            "Found rows with no active climate-zone one-hot channel in X_static"
        )
    if np.any(active_counts > 1):
        raise ValueError(
            "Found rows with multiple active climate-zone one-hot channels in X_static"
        )
    return np.asarray(climate_codes, dtype=np.int16)[np.argmax(active_mask, axis=1)]


def _build_site_lookup(values, data_info, round_decimals: int = 10, column_name: str = "value"):
    if len(values) != data_info.shape[0]:
        raise ValueError(
            f"{column_name} length mismatch: {len(values)} vs data_info rows {data_info.shape[0]}"
        )
    site_df = pd.DataFrame(
        {
            "lat": pd.to_numeric(data_info["latitude"], errors="coerce").round(round_decimals),
            "lon": pd.to_numeric(data_info["longitude"], errors="coerce").round(round_decimals),
            column_name: np.asarray(values),
        }
    ).dropna(subset=["lat", "lon"])
    nunique = site_df.groupby(["lat", "lon"])[column_name].nunique()
    bad = nunique[nunique > 1]
    if not bad.empty:
        example = bad.index[0]
        raise ValueError(
            f"Site {example} has multiple {column_name} values; cannot build site lookup cleanly"
        )
    site_lookup = (
        site_df.groupby(["lat", "lon"], as_index=False)[column_name]
        .first()
    )
    return {
        _round_site_tuple((row["lat"], row["lon"]), round_decimals=round_decimals): row[column_name]
        for _, row in site_lookup.iterrows()
    }


def _site_counts_by_climate_zone(site_climate_lookup):
    counts = {}
    for climate_code in site_climate_lookup.values():
        climate_code = int(climate_code)
        counts[climate_code] = counts.get(climate_code, 0) + 1
    return counts


def _enforce_climate_zone_train_support(
    selected_sites,
    candidate_sites,
    site_climate_lookup,
    total_sites_by_climate_zone,
    round_decimals: int = 10,
):
    selected_unique = _dedupe_sites_in_order(selected_sites, round_decimals=round_decimals)
    candidate_unique = _dedupe_sites_in_order(candidate_sites, round_decimals=round_decimals)
    target_n = len(selected_unique)
    candidate_order = selected_unique + [
        site for site in candidate_unique if site not in set(selected_unique)
    ]
    chosen = []
    chosen_set = set()
    chosen_counts_by_zone = {}
    for site in candidate_order:
        if len(chosen) >= target_n:
            break
        if site in chosen_set:
            continue
        if site not in site_climate_lookup:
            raise ValueError(f"Missing climate-zone lookup for site {site}")
        climate_code = int(site_climate_lookup[site])
        max_holdout_for_zone = int(total_sites_by_climate_zone[climate_code]) - 1
        if max_holdout_for_zone < 1:
            continue
        next_count = chosen_counts_by_zone.get(climate_code, 0) + 1
        if next_count > max_holdout_for_zone:
            continue
        chosen.append(site)
        chosen_set.add(site)
        chosen_counts_by_zone[climate_code] = next_count
    if len(chosen) < target_n:
        print(
            "WARNING: climate-zone feasibility reduced selected site count from "
            f"{target_n} to {len(chosen)}"
        )
    return chosen


def _assign_remaining_sites_to_test_folds(
    fold_locs,
    all_sites,
    site_climate_lookup,
    site_stratifier_lookup,
    round_decimals: int = 10,
):
    fold_locs = {
        int(fold): _dedupe_sites_in_order(locs, round_decimals=round_decimals)
        for fold, locs in fold_locs.items()
    }
    all_sites = _dedupe_sites_in_order(all_sites, round_decimals=round_decimals)
    total_sites_by_climate_zone = _site_counts_by_climate_zone(site_climate_lookup)
    fold_zone_counts = {
        int(fold): {} for fold in fold_locs
    }
    fold_strat_counts = {
        int(fold): {} for fold in fold_locs
    }
    assigned_sites = set()
    for fold, locs in fold_locs.items():
        for site in locs:
            assigned_sites.add(site)
            climate_code = int(site_climate_lookup[site])
            fold_zone_counts[fold][climate_code] = fold_zone_counts[fold].get(climate_code, 0) + 1
            stratifier_code = int(site_stratifier_lookup[site])
            fold_strat_counts[fold][stratifier_code] = fold_strat_counts[fold].get(stratifier_code, 0) + 1

    unassigned_sites = [site for site in all_sites if site not in assigned_sites]
    unassigned_sites = sorted(
        unassigned_sites,
        key=lambda site: (
            total_sites_by_climate_zone[int(site_climate_lookup[site])],
            int(site_stratifier_lookup[site]),
            float(site[0]),
            float(site[1]),
        ),
    )
    for site in unassigned_sites:
        climate_code = int(site_climate_lookup[site])
        feasible_folds = [
            fold
            for fold in sorted(fold_locs)
            if fold_zone_counts[fold].get(climate_code, 0) + 1
            <= total_sites_by_climate_zone[climate_code] - 1
        ]
        if not feasible_folds:
            raise ValueError(
                f"Could not place unassigned site {site} into any test fold without "
                f"eliminating climate zone {climate_code} from train"
            )
        stratifier_code = int(site_stratifier_lookup[site])
        best_fold = min(
            feasible_folds,
            key=lambda fold: (
                fold_strat_counts[fold].get(stratifier_code, 0),
                len(fold_locs[fold]),
                fold,
            ),
        )
        fold_locs[best_fold].append(site)
        fold_zone_counts[best_fold][climate_code] = (
            fold_zone_counts[best_fold].get(climate_code, 0) + 1
        )
        fold_strat_counts[best_fold][stratifier_code] = (
            fold_strat_counts[best_fold].get(stratifier_code, 0) + 1
        )
    return fold_locs


def _validate_sites_assigned_exactly_once(fold_locs, all_sites, round_decimals: int = 10):
    site_counts = {}
    for fold, locs in fold_locs.items():
        for site in _dedupe_sites_in_order(locs, round_decimals=round_decimals):
            site_counts[site] = site_counts.get(site, 0) + 1
    all_sites = _dedupe_sites_in_order(all_sites, round_decimals=round_decimals)
    missing = [site for site in all_sites if site_counts.get(site, 0) == 0]
    repeated = [(site, count) for site, count in site_counts.items() if count != 1]
    if missing or repeated:
        raise ValueError(
            "Each site must appear in test exactly once. "
            f"Missing={len(missing)}, repeated={len(repeated)}"
        )

class EarlyStopping:
    def __init__(
        self,
        patience=5,
        rmse_delta=0.1,
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
    source,
    desired_insitu_sample_size: int,
    desired_vv_sample_size: int,
    desired_vh_sample_size: int,
    seed: int = 42,
    used_sites = None,
    round_decimals: int = 10,
    stratifier = None,
    climate_zone_codes = None,
):
    source_np = np.asarray(source).reshape(-1)
    if source_np.shape[0] != data_info.shape[0]:
        raise ValueError(
            f"source and data_info length mismatch: {source_np.shape[0]} vs {data_info.shape[0]}"
        )
    site_climate_lookup = None
    total_sites_by_climate_zone = None
    if climate_zone_codes is not None:
        site_climate_lookup = _build_site_lookup(
            climate_zone_codes,
            data_info,
            round_decimals=round_decimals,
            column_name="climate_zone_code",
        )
        total_sites_by_climate_zone = _site_counts_by_climate_zone(site_climate_lookup)
    # split sources by numeric source code
    # 0=insitu, 1=vv/ratios, 2=vh
    insitu_rows = np.where(source_np == 0)[0]
    vv_rows = np.where(source_np == 1)[0]
    vh_rows = np.where(source_np == 2)[0]
    insitu = data_info.iloc[insitu_rows].copy()
    vv = data_info.iloc[vv_rows].copy()
    vh = data_info.iloc[vh_rows].copy()
    insitu["row_idx"] = insitu_rows
    vv["row_idx"] = vv_rows
    vh["row_idx"] = vh_rows
    strat_np = None if stratifier is None else np.asarray(stratifier)
    # clean/standardize lat/lon once
    def clean(df):
        out = df[['date', 'latitude', 'longitude', 'row_idx']].copy()
        out = out.rename(columns={'latitude': 'lat',
                                  'longitude': 'lon'})
        out['lat'] = pd.to_numeric(out['lat'], errors='coerce')
        out['lon'] = pd.to_numeric(out['lon'], errors='coerce')
        out = out.dropna(subset=['lat', 'lon'])
        # optional: snap to grid to avoid tiny fp diffs
        out['lat'] = out['lat'].round(round_decimals)
        out['lon'] = out['lon'].round(round_decimals)
        if climate_zone_codes is not None:
            out["climate_zone_code"] = np.asarray(climate_zone_codes)[out["row_idx"].to_numpy(dtype=int)]
        return out
    insitu = clean(insitu)
    vv     = clean(vv)
    vh     = clean(vh)
    insitu = insitu.reset_index(drop=True)
    vv     = vv.reset_index(drop=True)
    vh     = vh.reset_index(drop=True)
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
    if strat_np is not None:
        insitu_lcs = []
        vv_lcs = []
        vh_lcs = []
        for idx,row in insitu_counts.iterrows():
            ex_idx = row['idx'][0]
            row_idx = int(insitu.iloc[ex_idx]['row_idx'])
            insitu_lcs.append(strat_np[row_idx])
        for idx,row in vv_counts.iterrows():
            ex_idx = row['idx'][0]
            row_idx = int(vv.iloc[ex_idx]['row_idx'])
            vv_lcs.append(strat_np[row_idx])
        for idx,row in vh_counts.iterrows():
            ex_idx = row['idx'][0]
            row_idx = int(vh.iloc[ex_idx]['row_idx'])
            vh_lcs.append(strat_np[row_idx])
        insitu_counts['stratifier'] = insitu_lcs
        vv_counts['stratifier'] = vv_lcs
        vh_counts['stratifier'] = vh_lcs
    if climate_zone_codes is not None:
        insitu_counts["climate_zone_code"] = [
            int(site_climate_lookup[_round_site_tuple((row["lat"], row["lon"]), round_decimals=round_decimals)])
            for _, row in insitu_counts.iterrows()
        ]
        vv_counts["climate_zone_code"] = [
            int(site_climate_lookup[_round_site_tuple((row["lat"], row["lon"]), round_decimals=round_decimals)])
            for _, row in vv_counts.iterrows()
        ]
        vh_counts["climate_zone_code"] = [
            int(site_climate_lookup[_round_site_tuple((row["lat"], row["lon"]), round_decimals=round_decimals)])
            for _, row in vh_counts.iterrows()
        ]
    insitu_counts_orig = insitu_counts.copy()
    vv_counts_orig = vv_counts.copy()
    vh_counts_orig = vh_counts.copy()
    # fast exclude used_sites (as set of tuples)
    if used_sites:
        us = {(round(float(lat), round_decimals),
               round(float(lon), round_decimals))
              for (lat, lon) in used_sites}
        for site in us:
            insitu_counts = insitu_counts[
                (insitu_counts['lat'] != site[0]) | (insitu_counts['lon'] != site[1])
            ]
            vv_counts = vv_counts[
                (vv_counts['lat'] != site[0]) | (vv_counts['lon'] != site[1])
            ]
            vh_counts = vh_counts[
                (vh_counts['lat'] != site[0]) | (vh_counts['lon'] != site[1])
            ]
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
    insitu_sites = insitu_counts[['lat', 'lon']].apply(tuple, axis=1)
    val_i  = pick_sites_stratified(insitu_counts, insitu_counts_orig, desired_insitu_sample_size)
    if len(insitu_sites) == len(val_i):
        desired_vv_sample_size = 1_000_000
        desired_vh_sample_size = 1_000_000
    rs_goal = int(desired_vv_sample_size + desired_vh_sample_size)
    rs_counts = pd.concat(
        [
            vv_counts[['lat', 'lon', 'n', 'stratifier']],
            vh_counts[['lat', 'lon', 'n', 'stratifier']]
        ],
        ignore_index=True
    )
    rs_counts = rs_counts.groupby(['lat', 'lon', 'stratifier'], as_index=False)['n'].sum()
    rs_counts_orig = pd.concat(
        [
            vv_counts_orig[['lat', 'lon', 'n', 'stratifier']],
            vh_counts_orig[['lat', 'lon', 'n', 'stratifier']]
        ],
        ignore_index=True
    )
    rs_counts_orig = rs_counts_orig.groupby(['lat', 'lon', 'stratifier'], as_index=False)['n'].sum()
    val_rs = pick_sites_stratified(
        rs_counts,
        rs_counts_orig,
        rs_goal,
        must_use_sites=val_i,
        not_allowed_sites=insitu_sites
    )
    val_vv = val_rs
    val_vh = val_rs
    perc_i_sites = len(val_i) / len(insitu_counts_orig) * 100 if len(insitu_counts_orig) > 0 else 0
    perc_vv_sites = len(val_vv) / len(vv_counts_orig) * 100 if len(vv_counts_orig) > 0 else 0
    perc_vh_sites = len(val_vh) / len(vh_counts_orig) * 100 if len(vh_counts_orig) > 0 else 0
    print(f'Selected {perc_i_sites:.2f}% of insitu sites, {perc_vv_sites:.2f}% of VV sites, {perc_vh_sites:.2f}% of VH sites')
    num_sel_i = 0
    for site in val_i:
        this_data = insitu[
            (insitu['lat'] == site[0]) & (insitu['lon'] == site[1])
        ]
        num_sel_i += this_data.shape[0]
    perc_data_i = num_sel_i / len(insitu) * 100 if len(insitu) > 0 else 0
    num_sel_vv = 0
    for site in val_vv:
        this_data = vv[
            (vv['lat'] == site[0]) & (vv['lon'] == site[1])
        ]
        num_sel_vv += this_data.shape[0]
    perc_data_vv = num_sel_vv / len(vv) * 100 if len(vv) > 0 else 0
    num_sel_vh = 0
    for site in val_vh:
        this_data = vh[
            (vh['lat'] == site[0]) & (vh['lon'] == site[1])
        ]
        num_sel_vh += this_data.shape[0]
    perc_data_vh = num_sel_vh / len(vh) * 100 if len(vh) > 0 else 0
    print(f'Selected {perc_data_i:.2f}% of insitu data, {perc_data_vv:.2f}% of VV data, {perc_data_vh:.2f}% of VH data')
    # combine (allow duplicates if the same site is selected for multiple sources)
    candidate_sites = []
    if not insitu_counts.empty:
        candidate_sites.extend(
            insitu_counts[["lat", "lon"]].apply(tuple, axis=1).tolist()
        )
    if not rs_counts.empty:
        candidate_sites.extend(
            rs_counts[["lat", "lon"]].apply(tuple, axis=1).tolist()
        )
    val_locs = _dedupe_sites_in_order(val_i + val_rs, round_decimals=round_decimals)
    if site_climate_lookup is not None:
        val_locs = _enforce_climate_zone_train_support(
            selected_sites=val_locs,
            candidate_sites=candidate_sites,
            site_climate_lookup=site_climate_lookup,
            total_sites_by_climate_zone=total_sites_by_climate_zone,
            round_decimals=round_decimals,
        )
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
    num_tasks=None,
    second_task_source_code=None,
    task_weight_type=None,
    grad_norm=None,
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
            if global_step < warmup_steps:
                this_t = global_step / warmup_steps
                lr = warmup_start_lr * ((warmup_end_lr / warmup_start_lr) ** this_t)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        if (
            not train_model or
            insitu_b.sum().item() == 0
            or task_weight_type != 'gradnorm'
        ):
            use_gradnorm = False
            if optimizer is not None:
                optimizer.param_groups[2]['lr'] = 0.0
        #elif global_step < warmup_steps:
        #    use_gradnorm = (global_step % 20 == 0)
        else:
            use_gradnorm = (global_step % 50 == 0) 
            if use_gradnorm:
                optimizer.param_groups[2]['lr'] = 1e-2
            else:
                optimizer.param_groups[2]['lr'] = 0.0
        if train_model:
            optimizer.zero_grad(set_to_none=True)
            if use_gradnorm:
                with torch.backends.cudnn.flags(enabled=False), \
                    sdpa_kernel(SDPBackend.MATH):
                    preds = model(Xsh_b, Xl_b, Xst_b)
            else:
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
        # task weights from gradnorm
        #task_weights = model.task_weights
        if loss_fn is not None:
            loss_i = loss_fn(mu_i_b, logv_i_b, Y_b, mask=m_i)
            loss_vv = loss_fn(mu_vv_b, logv_vv_b, Y_b, mask=m_vv)
            loss_vh = loss_fn(mu_vh_b, logv_vh_b, Y_b, mask=m_vh)
            #loss_i = loss_fn(mu_i_b,Y_b,mask=m_i)
            #loss_vv = loss_fn(mu_vv_b,Y_b,mask=m_vv)
            #loss_vh = loss_fn(mu_vh_b,Y_b,mask=m_vh)
            n_i = int(m_i.sum().item())
            n_vv = int(m_vv.sum().item())
            n_vh = int(m_vh.sum().item())
            loss_i = loss_i * n_i
            loss_vv = loss_vv * n_vv
            loss_vh = loss_vh * n_vh
            n_i_tot += n_i
            n_vv_tot += n_vv
            n_vh_tot += n_vh
            n_samples = n_i + n_vv + n_vh
            n_samples_tot += n_samples
            if num_tasks == 1:
                task_losses = [loss_i]
            elif num_tasks == 2:
                if second_task_source_code not in (1, 2):
                    raise ValueError(
                        f'num_tasks=2 requires second_task_source_code in {{1,2}}, got {second_task_source_code}'
                    )
                loss_second = loss_vv if second_task_source_code == 1 else loss_vh
                task_losses = [loss_i, loss_second]
            elif num_tasks == 3:
                task_losses = [loss_i, loss_vv, loss_vh]
            if train_model and use_gradnorm:
                total_loss, L_grad = grad_norm.update(
                    task_losses, model.task_weights, model
                )
            else:
                if num_tasks == 1:
                    total_loss = model.task_weights[0] * loss_i
                elif num_tasks == 2:
                    if second_task_source_code not in (1, 2):
                        raise ValueError(
                            f'num_tasks=2 requires second_task_source_code in {{1,2}}, got {second_task_source_code}'
                        )
                    loss_second = loss_vv if second_task_source_code == 1 else loss_vh
                    total_loss = (
                        model.task_weights[0] * loss_i +
                        model.task_weights[1] * loss_second
                    )
                elif num_tasks == 3:
                    total_loss = (
                        model.task_weights[0] * loss_i +
                        model.task_weights[1] * loss_vv +
                        model.task_weights[2] * loss_vh
                    )
        if train_model:
            if use_gradnorm:
                total_loss.backward(retain_graph=True)
                L_grad.backward()
            else:
                total_loss.backward()
                if model.task_weights.grad is not None:
                    model.task_weights.grad.zero_()
            optimizer.step()
            with torch.no_grad():
                if use_gradnorm:
                    model.task_weights.clamp_(min=1e-6)
                    model.task_weights *= grad_norm.T / model.task_weights.sum()
            global_step += 1
        out_mu_i.append(mu_i_b.detach().cpu())
        out_logv_i.append(logv_i_b.detach().cpu())
        out_mu_vv.append(mu_vv_b.detach().cpu())
        out_logv_vv.append(logv_vv_b.detach().cpu())
        out_mu_vh.append(mu_vh_b.detach().cpu())
        out_logv_vh.append(logv_vh_b.detach().cpu())
        out_true_i.append(Y_b[m_i].detach().cpu())
        out_true_vv.append(Y_b[m_vv].detach().cpu())
        out_true_vh.append(Y_b[m_vh].detach().cpu())
        if loss_fn is not None:
            running_loss += total_loss.item() * n_samples
            running_loss_insitu += loss_i.item()
            if n_vv > 0:
                running_loss_vv += loss_vv.item()
            if n_vh > 0:
                running_loss_vh += loss_vh.item()
    if loss_fn is not None and n_samples_tot > 0.0:
        if n_i_tot > 0:
            running_loss /= n_samples_tot
            running_loss_insitu /= n_i_tot
        if n_vv_tot > 0:
            running_loss_vv /= n_vv_tot
        if n_vh_tot > 0:
            running_loss_vh /= n_vh_tot
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
        logv_vh = torch.cat(out_logv_vh).squeeze().numpy()
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
    num_tasks,
    task_weight_type,
    grad_norm,
    split_seed=None,
    batch_seed=None,
    plot_distributions=False
):
    this_fold_num = fold_test_locs[0]
    this_locs = np.array(fold_test_locs[1])
    this_split_seed = None if split_seed is None else int(split_seed) + int(this_fold_num)
    this_batch_seed = None if batch_seed is None else int(batch_seed) + int(this_fold_num)
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
    y = data[3]
    source = data[4]
    info = data[5]
    stratifier = data[6]
    (
        test_short_data, test_long_data, test_static_data, test_y, test_source, test_info, test_stratifier,
        remaining_short_data, remaining_long_data, remaining_static_data, remaining_y, remaining_source, remaining_info, remaining_stratifier
    ) = mask_location_data_fast(
        this_locs,
        short_data,
        long_data,
        static_data,
        y,
        source,
        info,
        stratifier
    )
    # split out the validation data
    remaining_insitu_obs = int((remaining_source == 0).sum().item())
    remaining_vv_obs = int((remaining_source == 1).sum().item())
    remaining_vh_obs = int((remaining_source == 2).sum().item())
    num_val_obs_insitu = remaining_insitu_obs * val_split
    num_val_obs_vv = remaining_vv_obs * val_split
    num_val_obs_vh = remaining_vh_obs * val_split
    remaining_climate_zone_codes = _extract_climate_zone_codes_from_static_tensor(
        remaining_static_data,
        var_names["static_vars"],
    )
    val_locs = create_site_split(
        remaining_info,
        remaining_source,
        desired_insitu_sample_size=int(num_val_obs_insitu),
        desired_vv_sample_size=int(num_val_obs_vv),
        desired_vh_sample_size=int(num_val_obs_vh),
        seed=this_split_seed,
        stratifier=remaining_stratifier,
        climate_zone_codes=remaining_climate_zone_codes,
    )
    val_locs = np.array(val_locs)
    # perform the same masking as was done for the test sites
    (
        val_short_data, val_long_data, val_static_data, val_y, val_source, val_info, val_stratifier,
        train_short_data, train_long_data, train_static_data, train_y, train_source, train_info, train_stratifier
    ) = mask_location_data_fast(
        val_locs,
        remaining_short_data,
        remaining_long_data,
        remaining_static_data,
        remaining_y,
        remaining_source,
        remaining_info,
        remaining_stratifier
    )
    # Sanity check
    total_test = test_info.shape[0]
    insitu_test = int((test_source == 0).sum().item())
    vv_test = int((test_source == 1).sum().item())
    vh_test = int((test_source == 2).sum().item())
    total_val = val_info.shape[0]
    insitu_val = int((val_source == 0).sum().item())
    vv_val = int((val_source == 1).sum().item())
    vh_val = int((val_source == 2).sum().item())
    total_train = train_info.shape[0]
    insitu_train = int((train_source == 0).sum().item())
    vv_train = int((train_source == 1).sum().item())
    vh_train = int((train_source == 2).sum().item())
    second_task_source_code = None
    if num_tasks == 2:
        train_source_codes = np.unique(np.asarray(train_source)).astype(int)
        non_insitu_codes = sorted(
            int(code) for code in train_source_codes if int(code) != 0
        )
        if len(non_insitu_codes) != 1:
            raise ValueError(
                'num_tasks=2 requires exactly one non-insitu source code in train_source '
                f'for this fold, got {non_insitu_codes}'
            )
        second_task_source_code = non_insitu_codes[0]
        if second_task_source_code not in (1, 2):
            raise ValueError(
                'num_tasks=2 only supports non-insitu source codes 1 (vv) or 2 (vh), '
                f'got {second_task_source_code}'
            )
        second_task_name = 'vv' if second_task_source_code == 1 else 'vh'
        print(
            f'num_tasks=2: using second task source code={second_task_source_code} '
            f'({second_task_name})'
        )
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
    lfmc_mean = np.nanmean(train_y[train_source == 0])
    lfmc_std = np.nanstd(train_y[train_source == 0])
    #print(vv_train)
    #print(train_y[train_source == 1])
    #print(np.unique(train_source))
    #sys.exit()
    if vv_train > 0:
        vv_mean = np.nanmean(train_y[train_source == 1])
        vv_std = np.nanstd(train_y[train_source == 1])
    else:
        vv_mean = np.array(np.nan)
        vv_std = np.array(np.nan)
    if vh_train > 0:
        vh_mean = np.nanmean(train_y[train_source == 2])
        vh_std = np.nanstd(train_y[train_source == 2])
    else:
        vh_mean = np.array(np.nan)
        vh_std = np.array(np.nan)
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
    train_y[train_source == 0] = (train_y[train_source == 0] - lfmc_mean) / lfmc_std
    val_y[val_source == 0] = (val_y[val_source == 0] - lfmc_mean) / lfmc_std
    test_y[test_source == 0] = (test_y[test_source == 0] - lfmc_mean) / lfmc_std
    if vv_train > 0:
        train_y[train_source == 1] = (train_y[train_source == 1] - vv_mean) / vv_std
        val_y[val_source == 1] = (val_y[val_source == 1] - vv_mean) / vv_std
        test_y[test_source == 1] = (test_y[test_source == 1] - vv_mean) / vv_std
    if vh_train > 0:
        train_y[train_source == 2] = (train_y[train_source == 2] - vh_mean) / vh_std
        val_y[val_source == 2] = (val_y[val_source == 2] - vh_mean) / vh_std
        test_y[test_source == 2] = (test_y[test_source == 2] - vh_mean) / vh_std
    # save the normalization parameters for later use
    norm_params = {
        'train_short_mean': train_short_mean.tolist(),
        'train_short_std': train_short_std.tolist(),
        'train_long_mean': train_long_mean.tolist(),
        'train_long_std': train_long_std.tolist(),
        'train_static_mean': train_static_mean.tolist(),
        'train_static_std': train_static_std.tolist(),
        'lfmc_mean': lfmc_mean.tolist(),
        'lfmc_std': lfmc_std.tolist(),
        'vv_mean': vv_mean.tolist(),
        'vv_std': vv_std.tolist(),
        'vh_mean': vh_mean.tolist(),
        'vh_std': vh_std.tolist(),
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
    if torch.isnan(train_y).any():
        raise ValueError("NaN found in train_y")
    train_dataset = TensorDataset(
        train_short_data,
        train_long_data,
        train_static_data,
        train_y,
        train_source
    )
    # make a combined stratifier label that is unique per land cover type and source
    train_stratifier_np = np.asarray(train_stratifier)
    train_source_np = np.asarray(train_source)
    train_joint_stratifier = np.stack([train_stratifier_np, train_source_np], axis=1)
    _,train_joint_stratifier = np.unique(
        train_joint_stratifier,
        axis=0,
        return_inverse=True
    )
    batch_sampler = StratifiedBatchSampler(
        labels=train_joint_stratifier,
        batch_size=batch_size,
        shuffle=True,
        seed=this_batch_seed,
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
        val_y,
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
        test_y,
        test_source
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    # set up the loss functions
    #criterion = nn.MSELoss(reduction="mean")
    criterion = GaussianNLLLoss(reduction="mean")
    #criterion = MaskedMSELoss(reduction="mean")
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
    test_loss = []
    test_loss_insitu = []
    test_loss_vv = []
    test_loss_vh = []
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
            loss_fn=criterion,
            train_model=True,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            global_step=global_step,
            warmup_start_lr=warmup_start_lr,
            warmup_end_lr=warmup_end_lr,
            num_tasks=num_tasks,
            second_task_source_code=second_task_source_code,
            task_weight_type=task_weight_type,
            grad_norm=grad_norm,
        )
        print(f'learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        train_loss.append(this_train_loss)
        train_loss_insitu.append(this_train_loss_insitu)
        train_loss_vv.append(this_train_loss_vv)
        train_loss_vh.append(this_train_loss_vh)
        print(f'Training total loss: {this_train_loss:.4f}')
        print(f'Training insitu loss: {this_train_loss_insitu:.4f}')
        print(f'Training vv loss: {this_train_loss_vv:.4f}')
        print(f'Training vh loss: {this_train_loss_vh:.4f}')
        if global_step > warmup_steps:
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
            loss_fn=criterion,
            train_model=False,
            num_tasks=num_tasks,
            second_task_source_code=second_task_source_code,
            task_weight_type=task_weight_type,
        )
        val_loss.append(this_val_loss)
        val_loss_insitu.append(this_val_loss_insitu)
        val_loss_vv.append(this_val_loss_vv)
        val_loss_vh.append(this_val_loss_vh)
        print(f'Validation total loss: {this_val_loss:.4f}')
        print(f'Validation insitu loss: {this_val_loss_insitu:.4f}')
        print(f'Validation vv loss: {this_val_loss_vv:.4f}')
        print(f'Validation vh loss: {this_val_loss_vh:.4f}')
        # run test each epoch so progress plots include test curves
        (
            _,
            this_test_loss_epoch,
            this_test_loss_insitu_epoch,
            this_test_loss_vv_epoch,
            this_test_loss_vh_epoch,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _
        ) = run_model(
            model,
            test_loader,
            device,
            loss_fn=criterion,
            train_model=False,
            num_tasks=num_tasks,
            second_task_source_code=second_task_source_code,
            task_weight_type=task_weight_type,
        )
        test_loss.append(this_test_loss_epoch)
        test_loss_insitu.append(this_test_loss_insitu_epoch)
        test_loss_vv.append(this_test_loss_vv_epoch)
        test_loss_vh.append(this_test_loss_vh_epoch)
        # denorm
        lfmc_val_only = mu_i_val[val_source == 0] * lfmc_std + lfmc_mean
        lfmc_std_val_only = np.sqrt(np.exp(logv_i_val[val_source == 0])) * lfmc_std
        lfmc_val_true = true_i * lfmc_std + lfmc_mean
        ## get mixture
        #mu_mix_val, logv_mix_val = fuse_gaussians(
        #    mu_i_val,
        #    logv_i_val,
        #    mu_rs_val,
        #    logv_rs_val,
        #)
        # calculate metrics of interet
        val_mae, val_rmse, val_r2 = _safe_mae_rmse_r2(lfmc_val_true, lfmc_val_only)
        val_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_val_only) + ((lfmc_val_true - lfmc_val_only) ** 2) / (lfmc_std_val_only ** 2)))
        val_avg_lfmc = np.mean(lfmc_val_only)
        val_avg_std = np.mean(lfmc_std_val_only)
        ## and for the mixed data
        #lfmc_mix_val = mu_mix_val[val_source.numpy() == 1 ] * y_std + y_mean
        #lfmc_std_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 1])) * y_std
        #val_mae_mix = np.mean(np.abs(lfmc_mix_val - lfmc_i_val_true))
        #val_r2_mix = r2_score(lfmc_i_val_true, lfmc_mix_val)
        #val_nll_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_mix_val) + ((lfmc_i_val_true - lfmc_mix_val) ** 2) / (lfmc_std_mix_val ** 2)))
        #val_rmse_mix = np.sqrt(np.mean((lfmc_mix_val - lfmc_i_val_true) ** 2))
        # and for vv data
        if len(true_vv) > 0:
            vv_val_only = mu_vv_val[val_source == 1] * vv_std + vv_mean
            vv_std_val_only = np.sqrt(np.exp(logv_vv_val[val_source == 1])) * vv_std
            vv_val_true = true_vv * vv_std + vv_mean
            val_mae_vv, val_rmse_vv, val_r2_vv = _safe_mae_rmse_r2(vv_val_true, vv_val_only)
            val_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vv_std_val_only) + ((vv_val_true - vv_val_only) ** 2) / (vv_std_val_only ** 2)))
            val_avg_vv = np.mean(vv_val_only)
            val_avg_std_vv = np.mean(vv_std_val_only)
            ## also calculate the mixtures
            #lfmc_rs_mix_val = mu_mix_val[val_source.numpy() == 0] * y_std + y_mean
            #lfmc_std_rs_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 0])) * y_std
            #val_mae_rs_mix = np.mean(np.abs(lfmc_rs_mix_val - lfmc_rs_val_true))
            #val_r2_rs_mix = r2_score(lfmc_rs_val_true, lfmc_rs_mix_val)
            #val_nll_rs_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_mix_val) + ((lfmc_rs_val_true - lfmc_rs_mix_val) ** 2) / (lfmc_std_rs_mix_val ** 2)))
            #val_rmse_rs_mix = np.sqrt(np.mean((lfmc_rs_mix_val - lfmc_rs_val_true) ** 2))
        if len(true_vh) > 0:
            vh_val_only = mu_vh_val[val_source == 2] * vh_std + vh_mean
            vh_std_val_only = np.sqrt(np.exp(logv_vh_val[val_source == 2])) * vh_std
            vh_val_true = true_vh * vh_std + vh_mean
            val_mae_vh, val_rmse_vh, val_r2_vh = _safe_mae_rmse_r2(vh_val_true, vh_val_only)
            val_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vh_std_val_only) + ((vh_val_true - vh_val_only) ** 2) / (vh_std_val_only ** 2)))
            val_avg_vh = np.mean(vh_val_only)
            val_avg_std_vh = np.mean(vh_std_val_only)
        # average values for sanity check
        #avg_val_pred = np.mean(lfmc_i_val)
        #avg_val_true = np.mean(lfmc_i_val_true)
        #avg_val_std = np.mean(lfmc_std_i_val)
        print(f'task weights: {model.task_weights}')
        print(
            f'Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, NLL: {val_nll:.4f}, avg: {val_avg_lfmc:.4f}, avg_std: {val_avg_std:.4f}'
        )
        #print(
        #    f'Validation Mixture MAE: {val_mae_mix:.4f}, RMSE: {val_rmse_mix:.4f}, R2: {val_r2_mix:.4f}, NLL: {val_nll_mix:.4f}'
        #)
        if len(true_vv) > 0:
            print(
                f'Validation VV MAE: {val_mae_vv:.4f}, RMSE: {val_rmse_vv:.4f}, R2: {val_r2_vv:.4f}, NLL: {val_nll_vv:.4f}, avg: {val_avg_vv:.4f}, avg_std: {val_avg_std_vv:.4f}'
            )
            #print(
            #    f'Validation VV Mixture MAE: {val_mae_vv_mix:.4f}, RMSE: {val_rmse_vv_mix:.4f}, R2: {val_r2_vv_mix:.4f}, NLL: {val_nll_vv_mix:.4f}'
            #)
        if len(true_vh) > 0:
            print(
                f'Validation VH MAE: {val_mae_vh:.4f}, RMSE: {val_rmse_vh:.4f}, R2: {val_r2_vh:.4f}, NLL: {val_nll_vh:.4f}, avg: {val_avg_vh:.4f}, avg_std: {val_avg_std_vh:.4f}'
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
        _,
        _,
        _,
        _,
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
        loss_fn=criterion,
        train_model=False,
        num_tasks=num_tasks,
        second_task_source_code=second_task_source_code,
        task_weight_type=task_weight_type,
    )
    # denorm
    lfmc_val_only = mu_i_val[val_source == 0] * lfmc_std + lfmc_mean
    lfmc_std_val_only = np.sqrt(np.exp(logv_i_val[val_source == 0])) * lfmc_std
    lfmc_val_true = true_i * lfmc_std + lfmc_mean
    ## get mixture
    #mu_mix_val, logv_mix_val = fuse_gaussians(
    #    mu_i_val,
    #    logv_i_val,
    #    mu_rs_val,
    #    logv_rs_val,
    #)
    # calculate metrics of interet
    val_mae, val_rmse, val_r2 = _safe_mae_rmse_r2(lfmc_val_true, lfmc_val_only)
    val_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_val_only) + ((lfmc_val_true - lfmc_val_only) ** 2) / (lfmc_std_val_only ** 2)))
    ## and for the mixed data
    #lfmc_mix_val = mu_mix_val[val_source.numpy() == 1 ] * y_std + y_mean
    #lfmc_std_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 1])) * y_std
    #val_mae_mix = np.mean(np.abs(lfmc_mix_val - lfmc_i_val_true))
    #val_r2_mix = r2_score(lfmc_i_val_true, lfmc_mix_val)
    #val_nll_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_mix_val) + ((lfmc_i_val_true - lfmc_mix_val) ** 2) / (lfmc_std_mix_val ** 2)))
    #val_rmse_mix = np.sqrt(np.mean((lfmc_mix_val - lfmc_i_val_true) ** 2))
    # and for vv data
    if len(true_vv) > 0:
        vv_val_only = mu_vv_val[val_source == 1] * vv_std + vv_mean
        vv_std_val_only = np.sqrt(np.exp(logv_vv_val[val_source == 1])) * vv_std
        vv_val_true = true_vv * vv_std + vv_mean
        val_mae_vv, val_rmse_vv, val_r2_vv = _safe_mae_rmse_r2(vv_val_true, vv_val_only)
        val_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vv_std_val_only) + ((vv_val_true - vv_val_only) ** 2) / (vv_std_val_only ** 2)))
        ## also calculate the mixtures
        #lfmc_rs_mix_val = mu_mix_val[val_source.numpy() == 0] * y_std + y_mean
        #lfmc_std_rs_mix_val = np.sqrt(np.exp(logv_mix_val[val_source.numpy() == 0])) * y_std
        #val_mae_rs_mix = np.mean(np.abs(lfmc_rs_mix_val - lfmc_rs_val_true))
        #val_r2_rs_mix = r2_score(lfmc_rs_val_true, lfmc_rs_mix_val)
        #val_nll_rs_mix = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_rs_mix_val) + ((lfmc_rs_val_true - lfmc_rs_mix_val) ** 2) / (lfmc_std_rs_mix_val ** 2)))
        #val_rmse_rs_mix = np.sqrt(np.mean((lfmc_rs_mix_val - lfmc_rs_val_true) ** 2))
    else:
        vv_val_only = np.array([])
        vv_std_val_only = np.array([])
        vv_val_true = np.array([])
        val_mae_vv = np.nan
        val_r2_vv = np.nan
        val_nll_vv = np.nan
        val_rmse_vv = np.nan
    if len(true_vh) > 0:
        vh_val_only = mu_vh_val[val_source == 2] * vh_std + vh_mean
        vh_std_val_only = np.sqrt(np.exp(logv_vh_val[val_source == 2])) * vh_std
        vh_val_true = true_vh * vh_std + vh_mean
        val_mae_vh, val_rmse_vh, val_r2_vh = _safe_mae_rmse_r2(vh_val_true, vh_val_only)
        val_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vh_std_val_only) + ((vh_val_true - vh_val_only) ** 2) / (vh_std_val_only ** 2)))
    else:
        vh_val_only = np.array([])
        vh_std_val_only = np.array([])
        vh_val_true = np.array([])
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
    # run the test
    (
        model,
        this_test_loss,
        this_test_loss_insitu,
        this_test_loss_vv,
        this_test_loss_vh,
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
        loss_fn=criterion,
        train_model=False,
        num_tasks=num_tasks,
        second_task_source_code=second_task_source_code,
        task_weight_type=task_weight_type,
    )
    # denorm
    if len(mu_i_test) == 0:
        this_test_loss = np.nan
        this_test_loss_insitu = np.nan
        this_test_loss_vv = np.nan
        this_test_loss_vh = np.nan
        lfmc_test_only = np.nan
        lfmc_std_test_only = np.nan
        vv_test_only = np.nan
        vv_std_test_only = np.nan
        vh_test_only = np.nan
        vh_std_test_only = np.nan
        lfmc_test_true = np.nan
        vv_test_true = np.nan
        vh_test_true = np.nan
        test_mae = np.nan
        test_r2 = np.nan
        test_nll = np.nan
        test_rmse = np.nan
    else:
        plotting.plot_training_progression(
            train_loss,
            val_loss,
            test_loss,
            best_epoch,
            'all',
            fold_save_dir
        )
        if len(true_i_test) > 0:
            lfmc_test_only = mu_i_test[test_source.numpy() == 0] * lfmc_std + lfmc_mean
            lfmc_std_test_only = np.sqrt(np.exp(logv_i_test[test_source.numpy() == 0])) * lfmc_std
            lfmc_test_true = true_i_test * lfmc_std + lfmc_mean
            # calculate metrics of interet
            test_mae, test_rmse, test_r2 = _safe_mae_rmse_r2(lfmc_test_true, lfmc_test_only)
            test_nll = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(lfmc_std_test_only) + ((lfmc_test_true - lfmc_test_only) ** 2) / (lfmc_std_test_only ** 2)))
            plotting.plot_training_progression(
                train_loss_insitu,
                val_loss_insitu,
                test_loss_insitu,
                best_epoch,
                'insitu',
                fold_save_dir
            )
        else:
            lfmc_test_only = np.nan
            lfmc_std_test_only = np.nan
            lfmc_test_true = np.nan
            test_mae = np.nan
            test_r2 = np.nan
            test_nll = np.nan
            test_rmse = np.nan
        # and for vv data
        if len(true_vv) > 0:
            vv_test_only = mu_vv_test[test_source.numpy() == 1] * vv_std + vv_mean
            vv_std_test_only = np.sqrt(np.exp(logv_vv_test[test_source.numpy() == 1])) * vv_std
            vv_test_true = true_vv_test * vv_std + vv_mean
            test_mae_vv, test_rmse_vv, test_r2_vv = _safe_mae_rmse_r2(vv_test_true, vv_test_only)
            test_nll_vv = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vv_std_test_only) + ((vv_test_true - vv_test_only) ** 2) / (vv_std_test_only ** 2)))
            plotting.plot_training_progression(
                train_loss_vv,
                val_loss_vv,
                test_loss_vv,
                best_epoch,
                'vv',
                fold_save_dir
            )
        else:
            vv_test_only = np.nan
            vv_std_test_only = np.nan
            vv_test_true = np.nan
            test_mae_vv = np.nan
            test_r2_vv = np.nan
            test_nll_vv = np.nan
            test_rmse_vv = np.nan
        if len(true_vh) > 0:
            vh_test_only = mu_vh_test[test_source.numpy() == 2] * vh_std + vh_mean
            vh_std_test_only = np.sqrt(np.exp(logv_vh_test[test_source.numpy() == 2])) * vh_std
            vh_test_true = true_vh_test * vh_std + vh_mean
            test_mae_vh, test_rmse_vh, test_r2_vh = _safe_mae_rmse_r2(vh_test_true, vh_test_only)
            test_nll_vh = np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(vh_std_test_only) + ((vh_test_true - vh_test_only) ** 2) / (vh_std_test_only ** 2)))
            plotting.plot_training_progression(
                train_loss_vh,
                val_loss_vh,
                test_loss_vh,
                best_epoch,
                'vh',
                fold_save_dir
            )
        else:
            vh_test_only = np.nan
            vh_std_test_only = np.nan
            vh_test_true = np.nan
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
            'lfmc_preds':lfmc_val_only,
            'lfmc_std':lfmc_std_val_only,
            'vv_preds':vv_val_only,
            'vv_std':vv_std_val_only,
            'vh_preds':vh_val_only,
            'vh_std':vh_std_val_only,
            'lfmc_true':lfmc_val_true,
            'vv_true':vv_val_true,
            'vh_true':vh_val_true
        },
        os.path.join(fold_save_dir,'val_outputs.pth')
    )
    torch.save(
        {
            'loss':this_test_loss,
            'loss_insitu':this_test_loss_insitu,
            'loss_vv':this_test_loss_vv,
            'loss_vh':this_test_loss_vh,
            'lfmc_preds':lfmc_test_only,
            'lfmc_std':lfmc_std_test_only,
            'vv_preds':vv_test_only,
            'vv_std':vv_std_test_only,
            'vh_preds':vh_test_only,
            'vh_std':vh_std_test_only,
            'lfmc_true':lfmc_test_true,
            'vv_true':vv_test_true,
            'vh_true':vh_test_true
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


def _fold_locs_to_jsonable(fold_locs):
    return {
        str(int(fold)): [[float(lat), float(lon)] for (lat, lon) in locs]
        for fold, locs in fold_locs.items()
    }


def _load_fold_locs_json(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'fold_locs' in raw:
        raw = raw['fold_locs']
    fold_locs = {}
    for fold_key, locs in raw.items():
        fold_num = int(fold_key)
        fold_locs[fold_num] = [tuple(loc) for loc in locs]
    return dict(sorted(fold_locs.items(), key=lambda kv: int(kv[0])))


def main():
    # load passed hyperparameter settings
    args = get_args()
    seed = int(args.seed)
    split_seed = None if args.split_seed is None else int(args.split_seed)
    batch_seed = int(args.batch_seed) if args.batch_seed is not None else seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Using seeds: seed={seed}, split_seed={split_seed}, batch_seed={batch_seed}')
    # configs
    # directories, etc.
    input_data_dir = args.input_data_dir
    save_dir = args.save_dir
    # training settings
    n_folds = 6
    batch_size = args.batch_size
    max_epochs = 100
    lr = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available, using CPU. This will be slow!')
    #warmup_steps = 1200
    base_lr = lr
    warmup_start_lr = 1e-6
    val_split = args.val_split
    adam_weight_decay = args.adam_wd
    patience = 8
    gradnorm_alpha = 1.0
    num_tasks = args.num_tasks
    # model hyperparameters (go back to the github and get what I deleted here)
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    # long model hyperparameters
    long_d_model = args.long_d_model
    long_nhead = args.long_nhead
    long_num_layers = args.long_num_layers
    long_dim_feedforward = args.long_dim_feedforward
    long_out_dim = args.long_out_dim
    # load the data
    datasets = load_data(input_data_dir)
    # early check that we don't have nans ANYWHERE
    for i, data in enumerate(datasets):
        if type(data) is np.ndarray:
            if np.isnan(data).any():
                raise ValueError(f'Data array {i} contains NaNs!')
        elif type(data) is pd.DataFrame:
            if data.isnull().values.any():
                print(data[data.isnull().any(axis=1)])
                print(f'WARNING: DataFrame {i} contains NaNs!')
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
    source = datasets[4]
    info = datasets[5]
    stratifier = datasets[6]
    climate_zone_codes = _extract_climate_zone_codes_from_static_tensor(
        static_data,
        var_names["static_vars"],
    )
    all_sites = _dedupe_sites_in_order(
        info[["latitude", "longitude"]].to_numpy(),
        round_decimals=10,
    )
    site_climate_lookup = _build_site_lookup(
        climate_zone_codes,
        info,
        round_decimals=10,
        column_name="climate_zone_code",
    )
    site_stratifier_lookup = _build_site_lookup(
        stratifier,
        info,
        round_decimals=10,
        column_name="stratifier",
    )
    num_insitu_obs = int((source == 0).sum().item())
    num_vv_obs = int((source == 1).sum().item())
    num_vh_obs = int((source == 2).sum().item())
    # make warmup the first 3 epochs
    batches_per_epoch = (num_insitu_obs + num_vv_obs + num_vh_obs) / batch_size * 0.7
    warmup_steps = int(3 * batches_per_epoch)
    print(f'Number of insitu observations: {num_insitu_obs}')
    print(f'Number of VV observations: {num_vv_obs}')
    print(f'Number of VH observations: {num_vh_obs}')
    first_task_weight_tag = ''
    if (
        args.task_weight_type == 'manual'
        and args.manual_task_weights is not None
        and len(args.manual_task_weights) > 0
    ):
        first_task_weight_tag = f'_tw0{args.manual_task_weights[0]}'
    this_model_name = (
        f'transformer_dm{d_model}_nh{nhead}_nl{num_layers}_df{dim_feedforward}'
        f'_do{dropout}_bs{batch_size}_lr{lr}_warmup{warmup_steps}'
        f'_wd{adam_weight_decay}_iobs{num_insitu_obs}_vvobs{num_vv_obs}_vhobs{num_vh_obs}'
        f'_dmlong{long_d_model}_nhlong{long_nhead}_nllong{long_num_layers}'
        f'_dflong{long_dim_feedforward}_outlong{long_out_dim}'
        f'_firstweight{first_task_weight_tag}'
    )
    if args.run_tag is not None and len(args.run_tag) > 0:
        this_model_name = f'{this_model_name}_{args.run_tag}'
    # set up the save directories
    full_save_dir = os.path.join(save_dir, this_model_name)
    if os.path.exists(full_save_dir):
        if args.overwrite:
            shutil.rmtree(full_save_dir)
        else:
            raise FileExistsError(
                f'Output directory already exists: {full_save_dir}. '
                'Pass --overwrite to replace it, or use --run_tag to create a unique directory.'
            )
    os.makedirs(full_save_dir)
    if args.fold_info_in:
        print(f'Loading fold definitions from {args.fold_info_in}')
        fold_locs = _load_fold_locs_json(args.fold_info_in)
        n_folds = len(fold_locs)
        print(f'Using {n_folds} precomputed folds for training')
    else:
        desired_insitu_obs_per_fold = num_insitu_obs / n_folds
        desired_vv_obs_per_fold = num_vv_obs / n_folds
        desired_vh_obs_per_fold = num_vh_obs / n_folds
        fold_locs = {}
        used_sites = []
        for fold in range(n_folds):
            print(f'Getting locations for fold {fold+1}/{n_folds}')
            this_locs = create_site_split(
                info,
                source,
                desired_insitu_sample_size=int(desired_insitu_obs_per_fold),
                desired_vv_sample_size=int(desired_vv_obs_per_fold),
                desired_vh_sample_size=int(desired_vh_obs_per_fold),
                seed=split_seed,
                used_sites=used_sites,
                stratifier=stratifier,
                climate_zone_codes=climate_zone_codes,
            )
            used_sites.extend(this_locs)
            fold_locs[fold + 1] = this_locs
        fold_locs = _assign_remaining_sites_to_test_folds(
            fold_locs=fold_locs,
            all_sites=all_sites,
            site_climate_lookup=site_climate_lookup,
            site_stratifier_lookup=site_stratifier_lookup,
            round_decimals=10,
        )
        # if there is any fold with zero locations, raise an error
        # we are allowed to get rid of the last fold if it has no locations
        remove_last = False
        for fold in fold_locs:
            if len(fold_locs[fold]) == 0 and fold != n_folds:
                raise ValueError(f'Fold {fold} has no locations')
            elif len(fold_locs[fold]) == 0 and fold == n_folds:
                print(f'Fold {fold} has no locations, removing')
                remove_last = True
        if remove_last:
            del fold_locs[fold]
        n_folds = len(fold_locs)
        print(f'Using {n_folds} folds for training')
    _validate_sites_assigned_exactly_once(
        fold_locs,
        all_sites=all_sites,
        round_decimals=10,
    )
    # save fold info actually used for this run
    with open(os.path.join(full_save_dir, 'fold_info.json'), 'w') as f:
        json.dump(_fold_locs_to_jsonable(fold_locs), f)
    # train this fold
    for fold, locs in enumerate(fold_locs.items()):
        print(f'Training fold {fold+1}/{n_folds} with {len(locs[1])} locations held out for testing')
        # build the model
        model = LFMCTransformerMultiTask(
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
            long_out_dim=long_out_dim,
            num_task_weights=num_tasks
        ).to(device)
        # build the optimizer
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            elif 'task_weights' in name:
                continue
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
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': model.task_weights, 'weight_decay': 0.0}
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
        # initialize our task weights
        task_weight_type = args.task_weight_type
        if task_weight_type == 'manual':
            for i in range(num_tasks):
                model.task_weights.data[i] = args.manual_task_weights[i]
        if task_weight_type == 'gradnorm':
            grad_norm = GradNorm(
                num_tasks=args.num_tasks,
                alpha=gradnorm_alpha,
                device=device
            )
        else:
            grad_norm = None
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
            num_tasks,
            task_weight_type,
            grad_norm,
            split_seed=split_seed,
            batch_seed=batch_seed,
        )
    # one final version of the model trained on all the data
    print('Training final model on all data')
    model = LFMCTransformerMultiTask(
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
        long_out_dim=long_out_dim,
        num_task_weights=num_tasks
    ).to(device)
    # build the optimizer
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif 'task_weights' in name:
            continue
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
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': model.task_weights, 'weight_decay': 0.0}
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
    # initialize our task weights
    task_weight_type = args.task_weight_type
    if task_weight_type == 'manual':
        for i in range(num_tasks):
            model.task_weights.data[i] = args.manual_task_weights[i]
    if task_weight_type == 'gradnorm':
        grad_norm = GradNorm(
            num_tasks=args.num_tasks,
            alpha=gradnorm_alpha,
            device=device
        )
    else:
        grad_norm = None
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
        num_tasks,
        task_weight_type,
        grad_norm,
        split_seed=split_seed,
        batch_seed=batch_seed,
    )

            

if __name__ == "__main__":
    main()
