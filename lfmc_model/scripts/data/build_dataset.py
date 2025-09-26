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

def drop_nans_with_reasons(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    retrieved_cols, filled_cols = _detect_reason_columns(df)

    nan_mask = df.isna().any(axis=1)
    nan_rows = df.loc[nan_mask]

    # masks within rows we’re dropping
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
        other_nan = ~retrieved_nan & ~filled_nan & nan_rows.drop(columns=reason_cols).isna().any(axis=1)
    else:
        # no retrieved/filled columns at all → everything is "other"
        other_nan = pd.Series(True, index=nan_rows.index, dtype=bool)

    total            = int(nan_mask.sum())
    only_ret_count   = int(only_ret.sum())
    only_fill_count  = int(only_fill.sum())
    both_count       = int(both_nan.sum())
    other_count      = int(other_nan.sum())
    retrieved_count  = only_ret_count + both_count
    filled_count     = only_fill_count + both_count

    # invariants with the explicit "other" bucket
    assert total == only_ret_count + only_fill_count + both_count + other_count
    assert retrieved_count == only_ret_count + both_count
    assert filled_count    == only_fill_count + both_count

    kept = df.loc[~nan_mask]

    counts = {
        "total_nan_rows": total,
        "retrieved": retrieved_count,
        "filled": filled_count,
        "both": both_count,
        "only_retrieved": only_ret_count,
        "only_filled": only_fill_count,
        "other": other_count,
    }
    return kept, counts

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
    df = pd.read_csv(path, parse_dates=[date_col])

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
    day_lags
):
    day_suffixes = [f'_lag_{day}d' for day in day_lags]
    matched_suffixes = set()
    def keep_column(col: str):
        match = re.search(r'lag_\d+[dh]',col)
        if match:
            for suffix in day_suffixes:
                if col.endswith(suffix):
                    matched_suffixes.add(suffix)
                    return True
            return False
        return True # non-lag columns are kept
    cols_to_keep = [col for col in df.columns if keep_column(col)]    
    unmatched_suffixes = set(day_suffixes) - matched_suffixes
    if unmatched_suffixes:
        raise ValueError(f"Unmatched lag suffixes: {unmatched_suffixes}")
    return df[cols_to_keep]    

#def df_to_tensor(
#    df: pd.DataFrame,
#    base_vars,
#    lag_suffix
#) -> torch.Tensor:
#    """
#    (same as before) returns a tensor of shape (N, seq_len, len(base_vars))
#    """
#    first = base_vars[0]
#    if lag_suffix is None:
#        auto_pat = re.compile(rf'^{re.escape(first)}_lag_\d+(\D+)$')
#        suffixes = {m.group(1) for col in df.columns if (m := auto_pat.match(col))}
#        if not suffixes:
#            raise ValueError(f"No lag columns for {first} to detect suffix")
#        if len(suffixes) > 1:
#            raise ValueError(f"Multiple suffixes {suffixes} for {first}")
#        lag_suffix = suffixes.pop()
#    # find all lag indices
#    pat0 = re.compile(rf'^{re.escape(first)}_lag_(\d+){re.escape(lag_suffix)}$')
#    lags = sorted(int(m.group(1)) for col in df.columns if (m := pat0.match(col)))
#    if not lags:
#        raise ValueError(f"No columns matching {first}_lag_<N>{lag_suffix}")
#    arrays = []
#    for var in base_vars:
#        pat = re.compile(rf'^{re.escape(var)}_lag_(\d+){re.escape(lag_suffix)}$')
#        found = {int(m.group(1)): col for col in df.columns if (m := pat.match(col))}
#        if set(found) != set(lags):
#            raise ValueError(f"{var} has lags {sorted(found)} but expected {lags}")
#        cols = [found[i] for i in lags]
#        arrays.append(df[cols].values[..., None])
#    print(arrays)
#    return torch.from_numpy(np.concatenate(arrays, axis=-1)).float()

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
    static_broadcast_to: int | None = None,
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
    dynamic_features,
    info_features,
    target_col,
    lag_days,
    save_dir,
    model_name='transformer',
    acceptable_lfmc_range=(30,500), # Example range for LFMC, adjust
):
    totals = init_nan_totals()
    for c,csv in enumerate(csvs):
        print(f'Processing CSV {c+1}/{len(csvs)}: {csv}')
        #df = pd.read_csv(csv, parse_dates=['date'])
        df = load_and_clean_csv(csv)
        df = filter_lag_columns(
            df,
            lag_days
        )
        # keep only the columns corresponding to any of our passed features
        required_cols = (
            static_features +
            info_features +
            target_col +
            [f'{var}_lag_{day}d' for var in dynamic_features for day in lag_days]
        )
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in {csv}: {missing_cols}')
        df = df[required_cols + ['date']]
        # get rid of any columns outside of the acceptable lfmc range
        df = df[(df['lfmc'] >= acceptable_lfmc_range[0]) & (df['lfmc'] <= acceptable_lfmc_range[1])]
        # get rid of any rows with nans
        df, nan_counts = drop_nans_with_reasons(df)
        totals = accumulate_nan_counts(totals,nan_counts)
        #base_feature_name = {
        #    col: col.split('_lag_')[0] if '_lag_' in col else col
        #    for col in df.columns
        #}
        #retrieved_cols = [
        #    col for col, base in base_feature_name.items()
        #    if base.startswith('retrieved')
        #]
        #filled_cols = [
        #    col for col, base in base_feature_name.items()
        #    if base.endswith('_filled')
        #]
        #nan_row_mask = df.isna().any(axis=1)
        #total_nan_rows = int(nan_row_mask.sum())
        #if total_nan_rows:
        #    nan_rows = df.loc[nan_row_mask]
        #    retrieved_nan_rows = (
        #        nan_rows[retrieved_cols].isna().any(axis=1)
        #        if retrieved_cols else pd.Series(False, index=nan_rows.index)
        #    )
        #    filled_nan_rows = (
        #        nan_rows[filled_cols].isna().any(axis=1)
        #        if filled_cols else pd.Series(False, index=nan_rows.index)
        #    )
        #    retrieved_count = int(retrieved_nan_rows.sum())
        #    filled_count = int(filled_nan_rows.sum())
        #    both_count = int((retrieved_nan_rows & filled_nan_rows).sum())
        #    total_removed += total_nan_rows
        #    total_removed_modis += filled_count
        #    total_removed_retrieved += retrieved_count
        #    total_removed_both += both_count
        #    print(
        #        'Dropping {total} rows due to NaNs '
        #        '(retrieved static: {retrieved}, filled MODIS: {filled}, both: {both})'.format(
        #            total=total_nan_rows,
        #            retrieved=retrieved_count,
        #            filled=filled_count,
        #            both=both_count
        #        )
        #    )
        #    df = df.loc[~nan_row_mask]
        # separate out our different dataframes
        print('Separating variables')
        all_vars = df.columns.tolist()
        all_daily_vars = []
        for d_var in dynamic_features:
            d_var_fmt = f'{d_var}_lag_*d'
            this_daily_vars = fnmatch.filter(
                all_vars,
                d_var_fmt
            )
            all_daily_vars.extend(this_daily_vars)
        all_static_vars = static_features
        all_info_vars = info_features
        all_target_vars = target_col
        # sort
        daily_df = df[all_daily_vars]
        static_df = df[all_static_vars]
        info_df = df[all_info_vars]
        target_df = df[all_target_vars]
        # convert to tensors
        print('Converting to tensors')
        # if this is the first csv, initialize tensors
        if c == 0:
            X_daily = df_to_tensor(
                daily_df,
                dynamic_features,
                'd'
            )
            X_static = df_to_tensor(
                static_df,
                static_features,
                None
            )
            Y = df_to_tensor(
                target_df,
                target_col,
                None
            )
            all_info_df = copy.deepcopy(info_df)
        else:
            X_daily = torch.cat(
                (X_daily, df_to_tensor(
                    daily_df,
                    dynamic_features,
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
                    target_col,
                    None
                )),
                dim=0
            )
            all_info_df = pd.concat(
                [all_info_df, info_df],
                ignore_index=True
            )
    print(f'Total rows removed due to NaNs: {totals["total_nan_rows"]}')
    print(f'  of which due to filled MODIS: {totals["filled"]}')
    print(f'  of which due to retrieved static: {totals["retrieved"]}')
    print(f'  of which due to both: {totals["both"]}')
    print(f'  of which are solely retrieved static: {totals["only_retrieved"]}')
    print(f'  of which are solely filled MODIS: {totals["only_filled"]}')
    print(f'  of which are other: {totals["other"]}')
    print(f'X_daily shape: {X_daily.shape}')
    print(f'X_static shape: {X_static.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'info_df shape: {all_info_df.shape}')
    # save to disk
    print('Saving tensors to disk')
    torch.save(X_daily,os.path.join(save_dir,'X_daily.pt'))
    torch.save(X_static,os.path.join(save_dir,'X_static.pt'))
    torch.save(Y,os.path.join(save_dir,'Y.pt'))
    all_info_df.to_csv(os.path.join(save_dir,'info.csv'),index=False)


if __name__ == "__main__":
    # set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # fill in follwing necessary information for producing the correct dataset
    save_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/inputs'
    csv_names = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/compiled/'
        'y_Insitu_X_ModisfilledDaymetStaticKrishnastatsWeatherstats_30days/'
        'compiled_data_*.csv'
    )
    csv_names = sorted(glob.glob(csv_names))
    first_label_date = datetime.date(2003, 1, 1)
    last_label_date = datetime.date(2023, 12, 31)
    static_features = [
        'slope','elevation','canopy_height','forest_cover',
        'clay','sand','latitude','longitude',
        #'retrieved_lfmc_mean',
        #'retrieved_lfmc_std','retrieved_lfmc_min','retrieved_lfmc_max',
        #'retrieved_lfmc_djf_mean','retrieved_lfmc_mam_mean',
        #'retrieved_lfmc_jja_mean','retrieved_lfmc_son_mean'
    ]
    dynamic_features = [
        'srad','prcp','swe','tmax','tmin','vp',
        'Nadir_Reflectance_Band1_filled',
        'Nadir_Reflectance_Band2_filled',
        'Nadir_Reflectance_Band3_filled',
        'Nadir_Reflectance_Band4_filled',
        'Nadir_Reflectance_Band5_filled',
        'Nadir_Reflectance_Band6_filled',
        'Nadir_Reflectance_Band7_filled'
        #'days_since_rain','max_precip_14_days',
        #'rolling_precip_14_days','max_temp_14_days',
        #'rolling_temp_14_days','max_vp_14_days',
        #'rolling_vp_14_days'
    ]
    target_col = ['lfmc']
    info_features = [
        'day_lat_lon','source'
    ]
    # save the variable names for later use
    var_names = {
        'daily_vars': dynamic_features,
        'static_vars': static_features,
        'info_vars': info_features,
        'lfmc_vars': target_col
    }
    with open(os.path.join(save_dir,'var_names.json'),'w') as f:
        json.dump(var_names,f)
    lag_days = [
        0,1,2,3,4,7,10,15,20,25,30
    ]
    acceptable_lfmc_range = (30, 500)  # Example range for LFMC, adjust as needed
    build_inputs(
        csvs=csv_names,
        first_label_date=first_label_date,
        last_label_date=last_label_date,
        static_features=static_features,
        dynamic_features=dynamic_features,
        info_features=info_features,
        target_col=target_col,
        lag_days=lag_days,
        save_dir=save_dir,
        model_name='transformer',
        acceptable_lfmc_range=acceptable_lfmc_range
    )
