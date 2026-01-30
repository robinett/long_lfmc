import torch
import pandas as pd
import pickle
import re
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict, Counter
import glob
import datetime
import copy
from datetime import timedelta
import os
import ast
import fnmatch
from typing import Optional, Tuple, List, Dict
import json
import xarray as xr
import rioxarray as rxr
import pyproj

# append up one dir to path
sys.path.append(sys.path[0] + '/../..')

from utils import plotting  # Assuming utils is a module with plotting function

def _detect_reason_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    base = {
        c: re.split(r"_lag_\d+\D*$", c)[0]  # strip one lag tail
        for c in df.columns
    }
    retrieved_cols = [c for c, b in base.items() if b.startswith("retrieved")]
    filled_cols    = [c for c, b in base.items() if b.endswith("_filled")]
    return retrieved_cols, filled_cols

def drop_nans_with_reasons(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, pd.DataFrame]]:
    """
    Drop rows based on NaNs with the following rules:

    - Let `target_cols` be the columns containing target values.
    - "Non-target" columns are all other columns.

    A row is dropped if:
      1) Any non-target column has a NaN, OR
      2) All target columns are NaN (i.e., no target value).

    Only rows where all non-target columns are non-NaN and
    at least one target column is non-NaN are kept.

    Returned:
      kept:   DataFrame after dropping rows
      counts: dict of counts by reason
      dropped: dict of DataFrames by reason
    """
    retrieved_cols, filled_cols = _detect_reason_columns(df)

    # Normalize target columns to those that exist in df
    if target_cols:
        _tcols = [c for c in target_cols if c in df.columns]
    else:
        _tcols = []

    # Non-target columns are everything else
    non_target_cols = [c for c in df.columns if c not in _tcols]

    # NaNs in non-target columns
    if non_target_cols:
        nan_in_non_target = df[non_target_cols].isna().any(axis=1)
    else:
        # If no non-targets, this condition never triggers
        nan_in_non_target = pd.Series(False, index=df.index, dtype=bool)

    # Rows with no target value (all target cols NaN)
    if _tcols:
        no_target_value = df[_tcols].isna().all(axis=1)
    else:
        # If no target columns are specified, we never drop
        # based on "no target value"
        no_target_value = pd.Series(False, index=df.index, dtype=bool)

    # Drop if: NaN in any non-target OR no target value
    nan_mask = nan_in_non_target | no_target_value
    nan_rows = df.loc[nan_mask]

    # Reason masks within rows we’re dropping
    if retrieved_cols:
        retrieved_nan = nan_rows[retrieved_cols].isna().any(axis=1)
    else:
        retrieved_nan = pd.Series(False, index=nan_rows.index, dtype=bool)

    if filled_cols:
        filled_nan = nan_rows[filled_cols].isna().any(axis=1)
    else:
        filled_nan = pd.Series(False, index=nan_rows.index, dtype=bool)

    both_nan  = retrieved_nan & filled_nan
    only_ret  = retrieved_nan & ~filled_nan
    only_fill = filled_nan & ~retrieved_nan

    # "other" = NaNs outside retrieved/filled columns
    reason_cols = sorted(set(retrieved_cols) | set(filled_cols))
    if reason_cols:
        other_nan = (
            ~retrieved_nan
            & ~filled_nan
            & nan_rows.drop(columns=reason_cols).isna().any(axis=1)
        )
    else:
        # No retrieved/filled columns at all → everything is "other"
        other_nan = pd.Series(True, index=nan_rows.index, dtype=bool)

    total           = int(nan_mask.sum())
    only_ret_count  = int(only_ret.sum())
    only_fill_count = int(only_fill.sum())
    both_count      = int(both_nan.sum())
    other_count     = int(other_nan.sum())
    retrieved_count = only_ret_count + both_count
    filled_count    = only_fill_count + both_count

    # invariants with the explicit "other" bucket
    assert total == only_ret_count + only_fill_count + both_count + other_count
    assert retrieved_count == only_ret_count + both_count
    assert filled_count    == only_fill_count + both_count

    kept = df.loc[~nan_mask]

    counts = {
        "total_nan_rows": total,
        "retrieved":      retrieved_count,
        "filled":         filled_count,
        "both":           both_count,
        "only_retrieved": only_ret_count,
        "only_filled":    only_fill_count,
        "other":          other_count,
    }

    dropped = {
        "only_retrieved": nan_rows.loc[only_ret].copy(),
        "only_filled":    nan_rows.loc[only_fill].copy(),
        "both":           nan_rows.loc[both_nan].copy(),
        "other":          nan_rows.loc[other_nan].copy(),
        "retrieved_any":  nan_rows.loc[only_ret | both_nan].copy(),
        "filled_any":     nan_rows.loc[only_fill | both_nan].copy(),
        "any_nan":        nan_rows.copy(),
    }

    return kept, counts, dropped


#def drop_nans_with_reasons(
#    df: pd.DataFrame,
#    target_cols: Optional[List[str]] = None,
#) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, pd.DataFrame]]:
#    """
#    Drops any row with ≥1 NaN in non-target cols, and if <1 target col does not have a value.
#    """
#    retrieved_cols, filled_cols = _detect_reason_columns(df)
#
#    # Base "has any NaN" rule
#    nan_mask_base = df.isna().any(axis=1)
#
#    # Keep rows if any target col is non-NaN
#    if target_cols:
#        _tcols = [c for c in target_cols if c in df.columns]
#        if _tcols:
#            has_target_value = df[_tcols].notna().any(axis=1)
#        else:
#            has_target_value = pd.Series(False, index=df.index, dtype=bool)
#    else:
#        has_target_value = pd.Series(False, index=df.index, dtype=bool)
#
#    # Only drop if: (has any NaN) AND (no target col has value)
#    nan_mask = nan_mask_base & ~has_target_value
#    nan_rows = df.loc[nan_mask]
#
#    # masks within rows we’re dropping
#    if retrieved_cols:
#        retrieved_nan = nan_rows[retrieved_cols].isna().any(axis=1)
#    else:
#        retrieved_nan = pd.Series(False, index=nan_rows.index, dtype=bool)
#
#    if filled_cols:
#        filled_nan = nan_rows[filled_cols].isna().any(axis=1)
#    else:
#        filled_nan = pd.Series(False, index=nan_rows.index, dtype=bool)
#
#    both_nan  = retrieved_nan & filled_nan
#    only_ret  = retrieved_nan & ~filled_nan
#    only_fill = filled_nan & ~retrieved_nan
#
#    # "other" = NaNs outside retrieved/filled columns
#    reason_cols = sorted(set(retrieved_cols) | set(filled_cols))
#    if reason_cols:
#        other_nan = (~retrieved_nan & ~filled_nan &
#                     nan_rows.drop(columns=reason_cols).isna().any(axis=1))
#    else:
#        # no retrieved/filled columns at all → everything is "other"
#        other_nan = pd.Series(True, index=nan_rows.index, dtype=bool)
#
#    total            = int(nan_mask.sum())
#    only_ret_count   = int(only_ret.sum())
#    only_fill_count  = int(only_fill.sum())
#    both_count       = int(both_nan.sum())
#    other_count      = int(other_nan.sum())
#    retrieved_count  = only_ret_count + both_count
#    filled_count     = only_fill_count + both_count
#
#    # invariants with the explicit "other" bucket
#    assert total == only_ret_count + only_fill_count + both_count + other_count
#    assert retrieved_count == only_ret_count + both_count
#    assert filled_count    == only_fill_count + both_count
#
#    kept = df.loc[~nan_mask]
#
#    counts = {
#        "total_nan_rows": total,
#        "retrieved": retrieved_count,
#        "filled": filled_count,
#        "both": both_count,
#        "only_retrieved": only_ret_count,
#        "only_filled": only_fill_count,
#        "other": other_count,
#    }
#
#    dropped = {
#        "only_retrieved": nan_rows.loc[only_ret].copy(),
#        "only_filled":    nan_rows.loc[only_fill].copy(),
#        "both":           nan_rows.loc[both_nan].copy(),
#        "other":          nan_rows.loc[other_nan].copy(),
#        "retrieved_any":  nan_rows.loc[only_ret | both_nan].copy(),
#        "filled_any":     nan_rows.loc[only_fill | both_nan].copy(),
#        "any_nan":        nan_rows.copy(),
#    }
#
#    return kept, counts, dropped

#def drop_nans_with_reasons(
#    df: pd.DataFrame
#) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, pd.DataFrame]]:
#    """
#    Drops any row with ≥1 NaN. Returns:
#      kept_df: rows with no NaNs anywhere
#      counts:  totals per drop reason (same keys as before + a few helpers)
#      dropped: dataframes for each reason bucket
#
#    Buckets (mutually exclusive, sum to total_nan_rows):
#      - only_retrieved
#      - only_filled
#      - both
#      - other   (NaNs outside retrieved/filled cols)
#
#    Convenience (overlapping) views also included:
#      - retrieved_any = only_retrieved ∪ both
#      - filled_any    = only_filled ∪ both
#      - any_nan       = all dropped rows
#    """
#    retrieved_cols, filled_cols = _detect_reason_columns(df)
#
#    nan_mask = df.isna().any(axis=1)
#    nan_rows = df.loc[nan_mask]
#
#    # masks within rows we’re dropping
#    if retrieved_cols:
#        retrieved_nan = nan_rows[retrieved_cols].isna().any(axis=1)
#    else:
#        retrieved_nan = pd.Series(False, index=nan_rows.index, dtype=bool)
#
#    if filled_cols:
#        filled_nan = nan_rows[filled_cols].isna().any(axis=1)
#    else:
#        filled_nan = pd.Series(False, index=nan_rows.index, dtype=bool)
#
#    both_nan  = retrieved_nan & filled_nan
#    only_ret  = retrieved_nan & ~filled_nan
#    only_fill = filled_nan & ~retrieved_nan
#
#    # "other" = NaNs outside retrieved/filled columns
#    reason_cols = sorted(set(retrieved_cols) | set(filled_cols))
#    if reason_cols:
#        other_nan = (~retrieved_nan & ~filled_nan &
#                     nan_rows.drop(columns=reason_cols).isna().any(axis=1))
#    else:
#        # no retrieved/filled columns at all → everything is "other"
#        other_nan = pd.Series(True, index=nan_rows.index, dtype=bool)
#
#    total            = int(nan_mask.sum())
#    only_ret_count   = int(only_ret.sum())
#    only_fill_count  = int(only_fill.sum())
#    both_count       = int(both_nan.sum())
#    other_count      = int(other_nan.sum())
#    retrieved_count  = only_ret_count + both_count
#    filled_count     = only_fill_count + both_count
#
#    # invariants with the explicit "other" bucket
#    assert total == only_ret_count + only_fill_count + both_count + other_count
#    assert retrieved_count == only_ret_count + both_count
#    assert filled_count    == only_fill_count + both_count
#
#    kept = df.loc[~nan_mask]
#
#    counts = {
#        "total_nan_rows": total,
#        "retrieved": retrieved_count,
#        "filled": filled_count,
#        "both": both_count,
#        "only_retrieved": only_ret_count,
#        "only_filled": only_fill_count,
#        "other": other_count,
#    }
#
#    # Build the dropped DataFrame views
#    dropped = {
#        "only_retrieved": nan_rows.loc[only_ret].copy(),
#        "only_filled":    nan_rows.loc[only_fill].copy(),
#        "both":           nan_rows.loc[both_nan].copy(),
#        "other":          nan_rows.loc[other_nan].copy(),
#        # convenience (overlapping) views:
#        "retrieved_any":  nan_rows.loc[only_ret | both_nan].copy(),
#        "filled_any":     nan_rows.loc[only_fill | both_nan].copy(),
#        "any_nan":        nan_rows.copy(),
#    }
#
#    return kept, counts, dropped

def init_nan_totals() -> Dict[str, int]:
    return {
        "files_processed": 0,
        "total_nan_rows": 0,
        "retrieved": 0,
        "filled": 0,
        "both": 0,
        "only_retrieved": 0,
        "only_filled": 0,
        "other": 0,
    }

def accumulate_nan_counts(
    totals: Dict[str, int],
    counts: Dict[str, int]
) -> Dict[str, int]:
    keys = [
        "total_nan_rows",
        "retrieved",
        "filled",
        "both",
        "only_retrieved",
        "only_filled",
        "other",
    ]
    for k in keys:
        totals[k] = totals.get(k, 0) + int(counts.get(k, 0))
    totals["files_processed"] = totals.get("files_processed", 0) + 1
    return totals

def _unwrap_scalarish(x):
    """Unwrap list/array or strip brackets around strings like "[0.123]".
    Returns original value (possibly still string) if not clearly numeric.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0] if len(x) else np.nan
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if s.lower() in {"", "nan", "none", "null"}:
            return np.nan
        return s
    return x

def _looks_numeric(val) -> bool:
    """True if value is numeric or a numeric string."""
    if pd.isna(val):
        return True
    if isinstance(val, (int, float, np.number)):
        return True
    if isinstance(val, str):
        pattern = re.compile(
            r"""^\s*
                (?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
                \s*$
            """,
            re.VERBOSE,
        )
        return bool(pattern.match(val))
    return False

def load_and_clean_csv(
    path: str,
    date_col: str = "date",
    numeric_threshold: float = 0.90,
    force_numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Read CSV, normalize scalar-ish cells, and convert only columns that are
    predominantly numeric (>= numeric_threshold) to float. Preserves
    categorical/text columns.
    """
    df = pd.read_csv(
        path,
        parse_dates=[date_col],
        dtype={
            "category": "string",
            "sub_category": "string",
            "sample_status": "string",
            "site_name": "string",
            "fuel_type": "string",
            "method": "string",
        }
    )

    # Normalize scalar-ish cells
    for col in df.columns:
        if col == date_col:
            continue
        df[col] = df[col].map(_unwrap_scalarish)

    # Detect numeric-like columns
    numeric_like_cols = []
    for col in df.columns:
        if col == date_col:
            continue
        s = df[col]
        sample = s.sample(min(len(s), 5000), random_state=0) if len(s) > 5000 else s
        frac_numeric = np.mean([_looks_numeric(v) for v in sample])
        if frac_numeric >= numeric_threshold:
            numeric_like_cols.append(col)

    if force_numeric_cols:
        numeric_like_cols = sorted(set(numeric_like_cols).union(force_numeric_cols))

    # Convert selected columns to numeric
    for col in numeric_like_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def filter_lag_columns(
    df,
    static_vars,
    short_lag_days,
    short_vars,
    long_lag_days,
    long_vars,
    target_vars,
    info_vars,
    stratifier
):
    keep_cols = static_vars + target_vars + info_vars + [stratifier]
    for sv,s_var in enumerate(short_vars):
        for sd,s_day in enumerate(short_lag_days):
            keep_cols.append(f'{s_var}_lag_{s_day}d')
    for lv,l_var in enumerate(long_vars):
        for ld,l_day in enumerate(long_lag_days):
            keep_cols.append(f'{l_var}_lag_{l_day}d')
    keep_cols = list(dict.fromkeys(keep_cols)) # make unique
    cols_to_keep = [col for col in df.columns if col in keep_cols]
    df = df[cols_to_keep]
    return df

def _coerce_numeric_block(df_block: pd.DataFrame) -> np.ndarray:
    """Coerce a DataFrame block to float32 numpy (handles object dtypes)."""
    def _to_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return x[0] if len(x) else np.nan
        if isinstance(x, str):
            s = x.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1].strip()
            if s.lower() in {'nan', 'none', ''}:
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan
        return x
    clean = df_block.map(_to_scalar)
    clean = clean.apply(pd.to_numeric, errors="coerce")
    return clean.to_numpy(dtype=np.float32)

def df_to_tensor(
    df: pd.DataFrame,
    base_vars,
    lag_suffix=None,
    static_broadcast_to=None
) -> torch.Tensor:
    """
    Convert a DataFrame to a tensor of shape (N, seq_len, V).
    - If lag columns for the first var exist (e.g., var_lag_0d), stacks by lag.
    - Else treats columns as static (no lag) and returns seq_len=1,
      or broadcasts to seq_len=static_broadcast_to if provided.
    """
    # Normalize base names in case caller passed something like "srad_lag_0d"
    bases = [re.sub(r'_lag_\d+\D*$', '', v) for v in base_vars]
    first = bases[0]

    # Try lagged path: detect suffix if not provided
    if lag_suffix is None:
        auto_pat = re.compile(rf'^{re.escape(first)}_lag_\d+(\D+)$')
        suffixes = {m.group(1) for col in df.columns if (m := auto_pat.match(col))}
        # Note: absence of lag suffix means either static OR lag with empty suffix.
        # If exactly one suffix found, use it; else treat as "no lag columns".
        use_lag = len(suffixes) == 1
        if use_lag:
            lag_suffix = suffixes.pop()
        else:
            lag_suffix = None
    else:
        use_lag = True  # caller explicitly provided one

    if use_lag and lag_suffix is not None:
        # Discover lag indices from the first base
        pat0 = re.compile(rf'^{re.escape(first)}_lag_(\d+){re.escape(lag_suffix)}$')
        lags = sorted(int(m.group(1)) for col in df.columns if (m := pat0.match(col)))
        if not lags:
            use_lag = False  # fall back to static if no actual lag columns found

    if use_lag and lag_suffix is not None:
        arrays = []
        expected = set(lags)
        for var_in, base in zip(base_vars, bases):
            pat = re.compile(rf'^{re.escape(base)}_lag_(\d+){re.escape(lag_suffix)}$')
            found = {int(m.group(1)): col for col in df.columns if (m := pat.match(col))}
            if set(found) != expected:
                missing = sorted(expected - set(found))
                extra   = sorted(set(found) - expected)
                raise ValueError(
                    f"'{var_in}' (base '{base}') has lags {sorted(found)} but expected {lags}. "
                    f"Missing: {missing}, Extra: {extra}"
                )
            cols = [found[i] for i in lags]                # (seq_len)
            block = _coerce_numeric_block(df[cols])        # (N, seq_len)
            arrays.append(block[..., None])                # (N, seq_len, 1)
        out = np.concatenate(arrays, axis=-1)             # (N, seq_len, V)
        return torch.from_numpy(out).float()

    # Static path: require exact columns by base name
    missing_static = [c for c in bases if c not in df.columns]
    if missing_static:
        raise ValueError(f"Static columns missing: {missing_static}")

    block = _coerce_numeric_block(df[bases])              # (N, V)
    block = block[:, None, :]                             # (N, 1, V)

    if static_broadcast_to is not None and static_broadcast_to > 1:
        block = np.repeat(block, repeats=static_broadcast_to, axis=1)  # (N, T, V)

    return torch.from_numpy(block).float()

def build_inputs(
    csvs,
    first_label_date,
    last_label_date,
    static_features,
    short_features,
    long_features,
    info_features,
    target_cols,
    stratifier,
    var_names,
    short_lag_days,
    long_lag_days,
    save_dir,
    model_name='transformer',
    acceptable_lfmc_range=(30,500), # Example range for LFMC, adjust
    num_rs_samples=0.0, # will include num_nfmd_samples * factor random samples from RS data
    include_lag=True,
    make_plots=False,
    vh_locations='all'
):
    totals = init_nan_totals()
    krishna_transforms = {}
    for c,csv in enumerate(csvs):
        print(f'Processing CSV {c+1}/{len(csvs)}: {csv}')
        #df = pd.read_csv(csv, parse_dates=['date'])
        df = load_and_clean_csv(csv)
        print(df)
        #print(np.unique(df['source']))
        #sys.exit()
        # keep anything that we need for both long and short lag days
        combined_lag_days = list(set(short_lag_days).union(set(long_lag_days)))
        df = filter_lag_columns(
            df,
            static_features,
            short_lag_days,
            short_features,
            long_lag_days,
            long_features,
            target_cols,
            info_features,
            stratifier
        )
        # fill any missing columns
        required_cols = (
            static_features +
            info_features +
            target_cols +
            [stratifier] +
            [f'{var}_lag_{day}d' for var in short_features for day in short_lag_days] +
            [f'{var}_lag_{day}d' for var in long_features for day in long_lag_days]
        )
        # make sure that this is a unique list
        required_cols = list(dict.fromkeys(required_cols))
        missing_cols = [col for col in required_cols if col not in df.columns]
        for col in missing_cols:
            print(f'Warning: required column {col} not found in {csv}')
            # fill with zeros
            df = df.copy()
            df[col] = 0.0
        df = df[required_cols]
        # get rid of any columns outside of the acceptable lfmc range
        df = df[
            (df['lfmc'].isna()) |
            (
                (df['lfmc'] >= acceptable_lfmc_range[0]) &
                (df['lfmc'] <= acceptable_lfmc_range[1])
            )
        ]
        df, nan_counts, dropped_dfs = drop_nans_with_reasons(df,target_cols)
        totals = accumulate_nan_counts(totals,nan_counts)
        # make the plots diagnosing why rows were dropped, if desired
        if make_plots:
            dropped_for_retrieval = dropped_dfs['retrieved_any']
            # accumulate location and date
            if c == 0:
                dropped_dates = dropped_for_retrieval['date'].dt.date.tolist()
                dropped_lats = dropped_for_retrieval['latitude'].tolist()
                dropped_lons = dropped_for_retrieval['longitude'].tolist()
            else:
                dropped_dates.extend(dropped_for_retrieval['date'].dt.date.tolist())
                dropped_lats.extend(dropped_for_retrieval['latitude'].tolist())
                dropped_lons.extend(dropped_for_retrieval['longitude'].tolist())
        # add in the remote sensing samples according to the specified factor
        # get the number of insitu samples
        in_situ_df = df[df['source'] == 'nfmd']
        adding_df = in_situ_df.copy()
        num_insitu_samples = in_situ_df.shape[0]
        source_labels = np.zeros(len(in_situ_df), dtype=int)
        source_legible = ['nfmd' for i in range(len(in_situ_df))]
        # sample from the rs dataframe
        if 'VV' in target_cols:
            vv_df = df[df['VV'].notna()]
            if num_rs_samples > len(vv_df):
                print(
                    f"Warning: requested {num_rs_samples} samples from VV"
                    f" data, but only {len(vv_df)} available."
                )
                print("Using all available samples.")
                this_num_vv_samples = len(vv_df)
            else:
                this_num_vv_samples = int(num_rs_samples)
            vv_samples = vv_df.sample(n=this_num_vv_samples,random_state=42)
            adding_df = pd.concat([adding_df,vv_samples], ignore_index=True)
            source_labels = np.concatenate(
                [source_labels, np.ones(len(vv_samples),dtype=int)]
            )
            [source_legible.append('vv') for i in range(len(vv_samples))]
        if 'vh_backscatter' in target_cols:
            vh_df = df[df['vh_backscatter'].notna()]
            # further sub-select based on vh_locations
            if vh_locations == 'at_sites':
                vh_df = vh_df[vh_df['source'] == 'vh_at_sites']
            elif vh_locations == 'at_random':
                vh_df = vh_df[vh_df['source'] == 'vh_at_random']
            elif vh_locations == 'all':
                pass
            else:
                raise ValueError(f"Unknown VH location: {vh_locations}")
            if num_rs_samples > len(vh_df):
                print(
                    f"Warning: requested {num_rs_samples} samples from VH"
                    f" data, but only {len(vh_df)} available."
                )
                print("Using all available samples.")
                this_num_vh_samples = len(vh_df)
            else:
                this_num_vh_samples = int(num_rs_samples)
            vh_samples = vh_df.sample(n=this_num_vh_samples,random_state=42)
            adding_df = pd.concat([adding_df,vh_samples], ignore_index=True)
            source_labels = np.concatenate(
                [source_labels, (np.ones(len(vh_samples),dtype=int) + 1)]
            )
            [source_legible.append('vh_backscatter') for i in range(len(vh_samples))]
        df = adding_df
        df['source_legible'] = source_legible
        # separate out our different dataframes
        print('Separating variables')
        all_vars = df.columns.tolist()
        all_short_vars = []
        for s_var in short_features:
            s_var_fmt = f'{s_var}_lag_*d'
            this_short_vars = fnmatch.filter(
                all_vars,
                s_var_fmt
            )
            all_short_vars.extend(this_short_vars)
        all_long_vars = []
        for l_var in long_features:
            l_var_fmt = f'{l_var}_lag_*d'
            this_long_vars = fnmatch.filter(
                all_vars,
                l_var_fmt
            )
            all_long_vars.extend(this_long_vars)
        all_static_vars = static_features
        all_info_vars = info_features
        all_target_vars = target_cols
        # sort
        short_df = df[all_short_vars]
        long_df = df[all_long_vars]
        static_df = df[all_static_vars]
        info_df = df[all_info_vars]
        target_df = df[all_target_vars]
        stratifier_df = df[stratifier]
        # need to consolidate target df into a dataframe that is a single row
        all_target_vals = np.array([])
        for var in target_cols:
            # Select non-NaN values for this column
            valid_rows = target_df[target_df[var].notna()]
            all_target_vals = np.concatenate(
                (all_target_vals, valid_rows[var].to_numpy())
            )
        # Concatenate all non-null subsets
        target_df = target_df.copy()
        target_df['vals'] = all_target_vals
        if include_lag:
            short_features.append('lfrac')
            long_features.append('lfrac')
            for sd in short_lag_days:
                this_lag_frac = sd / max(short_lag_days)
                short_df = short_df.copy()
                short_df[f'lfrac_lag_{sd}d'] = this_lag_frac
            for ld in long_lag_days:
                this_lag_frac = ld / max(long_lag_days)
                long_df = long_df.copy()
                long_df[f'lfrac_lag_{ld}d'] = this_lag_frac
        # convert to tensors
        print('Converting to tensors')
        # if this is the first csv, initialize tensors
        if c == 0:
            X_short = df_to_tensor(
                short_df,
                short_features,
                'd'
            )
            X_long = df_to_tensor(
                long_df,
                long_features,
                'd'
            )
            X_static = df_to_tensor(
                static_df,
                static_features,
                None
            )
            Y = df_to_tensor(
                target_df,
                ['vals'],
                None
            )
            source = torch.from_numpy(source_labels)
            all_info_df = copy.deepcopy(info_df)
            all_stratifier = stratifier_df.to_numpy()
        else:
            X_short = torch.cat(
                (X_short, df_to_tensor(
                    short_df,
                    short_features,
                    'd'
                )),
                dim=0
            )
            X_long = torch.cat(
                (X_long, df_to_tensor(
                    long_df,
                    long_features,
                    'd'
                )),
                dim=0
            )
            X_static = torch.cat(
                (X_static, df_to_tensor(
                    static_df,
                    static_features,
                    None
                )),
                dim=0
            )
            Y = torch.cat(
                (Y, df_to_tensor(
                    target_df,
                    ['vals'],
                    None
                )),
                dim=0
            )
            source = torch.cat(
                (source, torch.from_numpy(source_labels)),
                dim=0
            )
            all_info_df = pd.concat(
                [all_info_df, info_df],
                ignore_index=True
            )
            all_stratifier = np.concatenate(
                [all_stratifier, stratifier_df.to_numpy()],
                axis=0
            )
        # check if any of the tensors have nans
        if torch.isnan(X_short).any():
            raise ValueError('NaNs found in X_short tensor after conversion')
        if torch.isnan(X_long).any():
            raise ValueError('NaNs found in X_long tensor after conversion')
        if torch.isnan(X_static).any():
            raise ValueError('NaNs found in X_static tensor after conversion')
        if torch.isnan(Y).any():
            raise ValueError('NaNs found in Y tensor after conversion')
        if torch.isnan(source).any():
            raise ValueError('NaNs found in source tensor after conversion')
        if np.isnan(all_stratifier).any():
            raise ValueError('NaNs found in stratifier array after conversion')
        if include_lag:
            # remove lfrac from dynamic features so that it is not duplicated
            short_features.remove('lfrac')
            long_features.remove('lfrac')
        print('current all info df:')
        print(all_info_df)
    # get rid of any static features that are all zeros
    non_zero_static_cols = (X_static.abs().sum(dim=(0, 1)) != 0).squeeze()
    all_zero_static_cols = (X_static.abs().sum(dim=(0, 1)) == 0).squeeze()
    X_static = X_static[:, :, non_zero_static_cols]
    static_features_new = []
    for i,col in enumerate(static_features):
        if all_zero_static_cols[i]:
            print(f'Warning: static feature {col} is all zeros. Dropping.')
        else:
            static_features_new.append(col)
    static_features = static_features_new
    var_names['static_vars'] = static_features
    # check why we are dropping data for Krishna's retrievals, if requested
    if make_plots:
        # for testing
        dropped_dates = dropped_dates
        dropped_lats = dropped_lats
        dropped_lons = dropped_lons
        # first we need to load up globcover
        cover_da = rxr.open_rasterio(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/globcover/'
            'GLOBCOVER_L4_200901_200912_V2.3.tif'
        )
        cover_keys_df = pd.read_excel(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/globcover/'
            'Globcover2009_Legend.xls'
        )
        value_to_label = dict(zip(cover_keys_df['Value'], cover_keys_df['Label']))
        # get the value for the first lat/lon just so that we can see what the data looks liek
        if 'band' in cover_da.dims and len(cover_da['band']) == 1:
            cover_da = cover_da.squeeze('band', drop=True)
        points = np.arange(len(dropped_lons))
        vals_da = cover_da.interp(
            x=xr.DataArray(dropped_lons, dims="points"),
            y=xr.DataArray(dropped_lats, dims="points"),
            method="nearest"
        )
        all_cover_vals = vals_da.values
        meaningful_labels = [value_to_label[v] for v in all_cover_vals]
        # count the different labels
        label_counts = Counter(meaningful_labels)
        labels,counts = zip(*label_counts.items())
        # bar plot of the labels
        plotting.bar_plot(
            labels,
            counts,
            'Land Cover Types',
            'Frequency',
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/plots/dropped_retrieved_landcover.png'
        )
        # plot the distribution of dates by month
        months = [d.month for d in dropped_dates]
        month_counts = Counter(months)
        values = [month_counts.get(i, 0) for i in range(1, 13)]
        month_labels = [
            'Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec'
        ]
        plotting.bar_plot(
            month_labels,
            values,
            'Month',
            'Frequency',
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/plots/dropped_retrieved_months.png'
        )
        # plot a map of all the dropped points
        # get the unique list of points and their counts
        point_tuples = list(zip(dropped_lats,dropped_lons))
        point_counts = Counter(point_tuples)
        unique_lats,unique_lons = zip(*point_counts.keys())
        counts = list(point_counts.values())
        plotting.map_points(
            unique_lons,
            unique_lats,
            counts,
            '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/plots/dropped_retrieved_map.png'
        )
    print(f'Total rows removed due to NaNs: {totals["total_nan_rows"]}')
    print(f'  of which due to filled MODIS: {totals["filled"]}')
    print(f'  of which due to retrieved static: {totals["retrieved"]}')
    print(f'  of which due to both: {totals["both"]}')
    print(f'  of which are solely retrieved static: {totals["only_retrieved"]}')
    print(f'  of which are solely filled MODIS: {totals["only_filled"]}')
    print(f'  of which are other: {totals["other"]}')
    frac_insitu = (source == 0).sum().item() / len(source)
    frac_vv = (source == 1).sum().item() / len(source)
    frac_vh = (source == 2).sum().item() / len(source)
    print(f'In final dataset, fraction insitu: {frac_insitu:.3f}, fraction vv: {frac_vv:.3f}, fraction vh: {frac_vh:.3f}')
    print(f'X_short shape: {X_short.shape}')
    print(f'X_long shape: {X_long.shape}')
    print(f'X_static shape: {X_static.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'info_df shape: {all_info_df.shape}')
    # save to disk
    print('Saving tensors to disk')
    torch.save(X_short,os.path.join(save_dir,'X_short.pt'))
    torch.save(X_long,os.path.join(save_dir,'X_long.pt'))
    torch.save(X_static,os.path.join(save_dir,'X_static.pt'))
    torch.save(Y,os.path.join(save_dir,'Y.pt'))
    torch.save(source,os.path.join(save_dir,'source.pt'))
    all_info_df.to_csv(os.path.join(save_dir,'info.csv'),index=False)
    np.save(
        os.path.join(save_dir,'stratifier.npy'),
        all_stratifier
    )
    with open(os.path.join(save_dir,'var_names.json'),'w') as f:
        json.dump(var_names,f)

if __name__ == "__main__":
    # set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # fill in follwing necessary information for producing the correct dataset
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs/news1_multitask_cleaned'
    os.makedirs(save_dir, exist_ok=True)
    csv_names = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/compiled/'
        'y_InsitucleanedVh_X_ModisfilledDaymetStaticClimatezoneSarstatsseasonalLandcoverfracLandcoverchange_Z_Nlcdclass_180d/'
        'compiled_data_*.csv'
    )
    csv_names_all = sorted(glob.glob(csv_names))
    first_label_date = pd.to_datetime('2003-01-01')
    last_label_date = pd.to_datetime('2023-12-31')
    csv_names = []
    for c,csv_n in enumerate(csv_names_all):
        fname = csv_n.split('/')[-1]
        fname = fname.split('.')[0]
        start_date = pd.to_datetime(fname.split('_')[2])
        end_date = pd.to_datetime(fname.split('_')[3])
        if start_date.year >= first_label_date.year and end_date.year <= last_label_date.year:
            csv_names.append(csv_n)
    static_features = [
        'slope',
        'elevation',
        #'canopy_height',
        'clay',
        'sand',
        'latitude',
        'longitude',
        'climate_zone_1',
        'climate_zone_2','climate_zone_3',
        'climate_zone_4','climate_zone_5',
        'climate_zone_6','climate_zone_7',
        'climate_zone_8','climate_zone_9',
        'climate_zone_10','climate_zone_11',
        'climate_zone_12','climate_zone_13',
        'climate_zone_14','climate_zone_15',
        'climate_zone_16','climate_zone_17',
        'climate_zone_18','climate_zone_19',
        'climate_zone_20','climate_zone_21',
        'climate_zone_22','climate_zone_23',
        'climate_zone_24','climate_zone_25',
        'climate_zone_26','climate_zone_27',
        'climate_zone_28','climate_zone_29',
        'barren','crops','deciduous_forest',
        'developed','evergreen_forest',
        'grass','mixed_forest','other',
        'shrub','water','wetlands',
        #'sar_vh_mean',
        #'sar_vh_seasonal_amp',
        #'sar_vh_annual_fraction'
        #'sar_vh_mean',
        #'sar_vh_std',
        #'sar_vh_min',
        #'sar_vh_max',
        #'vh_skewness',
        #'vh_kurtosis',
        #'vh_autocorr1',
        #'vh_autocorr2',
        #'sar_vh_jan_mean',
        #'sar_vh_feb_mean',
        #'sar_vh_mar_mean',
        #'sar_vh_apr_mean',
        #'sar_vh_may_mean',
        #'sar_vh_jun_mean',
        #'sar_vh_jul_mean',
        #'sar_vh_aug_mean',
        #'sar_vh_sep_mean',
        #'sar_vh_oct_mean',
        #'sar_vh_nov_mean',
        #'sar_vh_dec_mean',
        #'sar_vv_mean','sar_vh_mean',#'sar_vv_minus_vh_mean',
        #'sar_vv_std','sar_vh_std',#'sar_vv_minus_vh_std',
        #'sar_vv_min','sar_vh_min',#'sar_vv_minus_vh_min',
        #'sar_vv_max','sar_vh_max',#'sar_vv_minus_vh_max',
        #'sar_vv_jan_mean','sar_vh_jan_mean',#'sar_vv_minus_vh_jan_mean',
        #'sar_vv_feb_mean','sar_vh_feb_mean',#'sar_vv_minus_vh_feb_mean',
        #'sar_vv_mar_mean','sar_vh_mar_mean',#'sar_vv_minus_vh_mar_mean',
        #'sar_vv_apr_mean','sar_vh_apr_mean',#'sar_vv_minus_vh_apr_mean',
        #'sar_vv_may_mean','sar_vh_may_mean',#'sar_vv_minus_vh_may_mean',
        #'sar_vv_jun_mean','sar_vh_jun_mean',#'sar_vv_minus_vh_jun_mean',
        #'sar_vv_jul_mean','sar_vh_jul_mean',#'sar_vv_minus_vh_jul_mean',
        #'sar_vv_aug_mean','sar_vh_aug_mean',#'sar_vv_minus_vh_aug_mean',
        #'sar_vv_sep_mean','sar_vh_sep_mean',#'sar_vv_minus_vh_sep_mean',
        #'sar_vv_oct_mean','sar_vh_oct_mean',#'sar_vv_minus_vh_oct_mean',
        #'sar_vv_nov_mean','sar_vh_nov_mean',#'sar_vv_minus_vh_nov_mean',
        #'sar_vv_dec_mean','sar_vh_dec_mean',#'sar_vv_minus_vh_dec_mean',
        #'vv_skewness','vh_skewness',#'vv_minus_vh_skewness',
        #'vv_kurtosis','vh_kurtosis',#'vv_minus_vh_kurtosis',
        #'vv_autocorr1','vh_autocorr1',#'vv_minus_vh_autocorr1',
        #'vv_autocorr2','vh_autocorr2',#'vv_minus_vh_autocorr2',
        #'land_cover_change_flag'
    ]
    short_features = [
        'Nadir_Reflectance_Band1_filled',
        'Nadir_Reflectance_Band2_filled',
        'Nadir_Reflectance_Band3_filled',
        'Nadir_Reflectance_Band4_filled',
        'Nadir_Reflectance_Band5_filled',
        'Nadir_Reflectance_Band6_filled',
        'Nadir_Reflectance_Band7_filled',
    ]
    long_features = [
        'srad','prcp','swe','tmax','vp',
    ]
    stratifier = 'nlcd'
    include_lag = True
    #target_cols = ['lfmc']
    target_cols = ['lfmc','vh_backscatter']
    #target_cols = ['lfmc','VV','VH']
    #num_rs_samples = 0
    # just keep all of them
    num_rs_samples = 100_000_000
    # which vh samples to keep
    # options are:
    #   all: doesn't matter where from
    #   at_sites: only at sites where we already have lfmc measurements
    #   at_random: only at random loctions, not where we have lfmc measurements
    vh_locations = 'all'
    info_features = [
        'date',
        'latitude',
        'longitude',
        'source',
        'source_legible'
    ]
    # save the variable names for later use
    var_names = {
        'short_vars': short_features + (['lfrac'] if include_lag else []),
        'long_vars': long_features + (['lfrac'] if include_lag else []),
        'static_vars': static_features,
        'info_vars': info_features,
        'lfmc_vars': target_cols
    }
    short_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ]
    long_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,
        81,82,83,84,85,86,87,88,89,90,
        91,92,93,94,95,96,97,98,99,100,
        101,102,103,104,105,106,107,108,109,110,
        111,112,113,114,115,116,117,118,119,120,
        121,122,123,124,125,126,127,128,129,130,
        131,132,133,134,135,136,137,138,139,140,
        141,142,143,144,145,146,147,148,149,150,
        151,152,153,154,155,156,157,158,159,160,
        161,162,163,164,165,166,167,168,169,170,
        171,172,173,174,175,176,177,178,179,180,
    ]
    acceptable_lfmc_range = (30, 500)  # Example range for LFMC, adjust as needed
    build_inputs(
        csvs=csv_names,
        first_label_date=first_label_date,
        last_label_date=last_label_date,
        static_features=static_features,
        long_features=long_features,
        short_features=short_features,
        info_features=info_features,
        target_cols=target_cols,
        stratifier=stratifier,
        var_names=var_names,
        short_lag_days=short_lag_days,
        long_lag_days=long_lag_days,
        save_dir=save_dir,
        model_name='transformer',
        acceptable_lfmc_range=acceptable_lfmc_range,
        num_rs_samples=num_rs_samples,
        include_lag=include_lag,
        make_plots=False,
        vh_locations=vh_locations
    )
