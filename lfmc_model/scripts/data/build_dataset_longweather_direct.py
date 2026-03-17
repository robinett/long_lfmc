import argparse
import builtins
import datetime
import glob
import json
import os
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import torch

from longweather_direct_pipeline import (
    build_direct_tensors_from_sample_index,
    load_sample_index,
    open_source_datasets,
    save_direct_build_result,
)


def _print_with_timestamp(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)


print = _print_with_timestamp


def _expand_inputs(inputs):
    paths = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(matches)
        else:
            paths.append(item)
    return paths


def _normalize_key_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "date":
        dt = pd.to_datetime(df[col], errors="coerce", utc=True)
        return dt.dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("__NA__")
    if col in {"latitude", "longitude", "target_value"}:
        num = pd.to_numeric(df[col], errors="coerce")
        return num.round(8).map(lambda x: "__NA__" if pd.isna(x) else f"{x:.8f}")
    return df[col].astype(str).replace({"nan": "__NA__", "None": "__NA__"})


def _source_code_from_table(df: pd.DataFrame) -> np.ndarray:
    source_map = {
        "nfmd": 0,
        "vv": 1,
        "vv_minus_vh": 1,
        "vv_over_vh": 1,
        "vh": 2,
    }
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    if "source_code" in df.columns:
        out = pd.to_numeric(df["source_code"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isin(out, [0.0, 1.0, 2.0])
    if not valid.all():
        for fallback_col in ["source_legible", "source"]:
            if fallback_col in df.columns:
                mapped = (
                    df[fallback_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(source_map)
                    .to_numpy(dtype=np.float64)
                )
                fill_mask = (~valid) & np.isfinite(mapped)
                out[fill_mask] = mapped[fill_mask]
                valid = np.isin(out, [0.0, 1.0, 2.0])
                if valid.all():
                    break
    bad_n = int((~np.isin(out, [0.0, 1.0, 2.0])).sum())
    if bad_n > 0:
        raise ValueError(
            f"Could not infer valid source codes (0/1/2) for {bad_n:,} row(s) in sample index."
        )
    return out.astype(np.int64)


def _rebuild_source_tensor_only(sample_df: pd.DataFrame, info_df: pd.DataFrame) -> torch.Tensor:
    key_candidates = [
        "sample_id",
        "site_id",
        "date",
        "latitude",
        "longitude",
        "target_name",
        "target_value",
    ]
    key_cols = [c for c in key_candidates if c in sample_df.columns and c in info_df.columns]
    if not key_cols:
        raise ValueError("No shared key columns between sample index and info for source-only rebuild.")

    sample_work = sample_df.copy()
    info_work = info_df.copy()
    for c in key_cols:
        sample_work[f"_k_{c}"] = _normalize_key_col(sample_work, c)
        info_work[f"_k_{c}"] = _normalize_key_col(info_work, c)

    key_norm_cols = [f"_k_{c}" for c in key_cols]
    sample_work["_join_key"] = list(zip(*[sample_work[c] for c in key_norm_cols]))
    info_work["_join_key"] = list(zip(*[info_work[c] for c in key_norm_cols]))
    sample_work["_occ"] = sample_work.groupby("_join_key").cumcount()
    info_work["_occ"] = info_work.groupby("_join_key").cumcount()

    sample_codes = _source_code_from_table(sample_work)
    lookup: Dict[tuple, int] = {}
    for k, o, code in zip(sample_work["_join_key"], sample_work["_occ"], sample_codes):
        lookup[(k, int(o))] = int(code)

    out_codes = np.full(len(info_work), np.nan, dtype=np.float64)
    for i, (k, o) in enumerate(zip(info_work["_join_key"], info_work["_occ"])):
        code = lookup.get((k, int(o)))
        if code is not None:
            out_codes[i] = float(code)

    unmatched = int(np.isnan(out_codes).sum())
    if unmatched > 0:
        print(
            f"[source_only] Warning: {unmatched:,} row(s) unmatched on key join; "
            "falling back to info source labels."
        )
        info_fallback = _source_code_from_table(info_work)
        miss_mask = np.isnan(out_codes)
        out_codes[miss_mask] = info_fallback[miss_mask].astype(np.float64)

    unresolved = int(np.isnan(out_codes).sum())
    if unresolved > 0:
        raise ValueError(f"Failed to assign source code for {unresolved:,} output row(s).")

    out_int = out_codes.astype(np.int64)
    uniq, cnt = np.unique(out_int, return_counts=True)
    print(
        "[source_only] source code counts: "
        + ", ".join([f"{int(u)} -> {int(c):,}" for u, c in zip(uniq.tolist(), cnt.tolist())])
    )
    return torch.from_numpy(out_int)


def main():
    default_sample_index_path = (
        "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/"
        "sample_index_longweather_2000_2024_lfmc.parquet"
    )
    default_save_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/testing"
    parser = argparse.ArgumentParser(
        description=(
            "Build direct longweather tensors from a single sample-index file."
        )
    )
    parser.add_argument(
        "sample_index_path",
        type=str,
        nargs="?",
        default=None,
        help="Path or glob to sample-index parquet/csv (optional).",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output save directory (optional).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove save_dir before writing outputs.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "Only rewrite source.pt using sample index + existing info.* in save_dir. "
            "Skips all feature extraction/tensor rebuild."
        ),
    )
    args = parser.parse_args()
    if args.sample_index_path is None:
        print(
            "Warning: no sample_index_path provided. Defaulting to "
            f"{default_sample_index_path}"
        )
        sample_index_path = default_sample_index_path
    else:
        sample_index_path = str(args.sample_index_path)
    if args.save_dir is None:
        print(
            "Warning: no save_dir provided. Defaulting to "
            f"{default_save_dir}"
        )
        save_dir = default_save_dir
    else:
        save_dir = str(args.save_dir)

    # All configuration is explicitly set here.
    scratch_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc"

    sample_index_inputs = [sample_index_path]

    #start_date = None
    #end_date = None
    start_date = "2000-01-01"
    end_date = "2024-12-31"

    dataset_paths = {
        "daymet": os.path.join(scratch_dir, "daymet", "daymet_all_vars.zarr"),
        "modis": os.path.join(
            scratch_dir,
            "modis",
            "modis_regrid_interpolated",
            "modis_interp_5d.zarr",
        ),
        "static": os.path.join(
            scratch_dir,
            "static",
            "static_features_500m_epsg5070_float32.nc",
        ),
        "climate_zone": os.path.join(
            scratch_dir,
            "climate_zones",
            "climate_zone_per_pixel_fullgrid.nc4",
        ),
        "landcover_frac": os.path.join(
            scratch_dir,
            "nlcd",
            "nlcd_target_grid_2000_2024.zarr",
        ),
        "nlcd": os.path.join(
            scratch_dir,
            "nlcd",
            "nlcd_2000_2024.zarr",
        ),
    }

    grid_source = "modis"
    stratifier = "nlcd"
    include_lag_feature = True
    assume_prepared_index = True
    temporal_batch_mode = "needed_days_in_month_blocks"
    temporal_month_block_size = 6
    temporal_num_workers = 16
    temporal_max_inflight = 32

    short_features = [
        "Nadir_Reflectance_Band1_interp",
        "Nadir_Reflectance_Band2_interp",
        "Nadir_Reflectance_Band3_interp",
        "Nadir_Reflectance_Band4_interp",
        "Nadir_Reflectance_Band5_interp",
        "Nadir_Reflectance_Band6_interp",
        "Nadir_Reflectance_Band7_interp",
    ]

    long_features = [
        "srad",
        "prcp",
        "swe",
        "tmax",
        "vp",
    ]

    static_features = [
        "slope",
        "elevation",
        "clay",
        "sand",
        "canopy_height",
        #"latitude",
        #"longitude",
        "climate_zone_1",
        "climate_zone_2",
        "climate_zone_3",
        "climate_zone_4",
        "climate_zone_5",
        "climate_zone_6",
        "climate_zone_7",
        "climate_zone_8",
        "climate_zone_9",
        "climate_zone_10",
        "climate_zone_11",
        "climate_zone_12",
        "climate_zone_13",
        "climate_zone_14",
        "climate_zone_15",
        "climate_zone_16",
        "climate_zone_17",
        "climate_zone_18",
        "climate_zone_19",
        "climate_zone_20",
        "climate_zone_21",
        "climate_zone_22",
        "climate_zone_23",
        "climate_zone_24",
        "climate_zone_25",
        "climate_zone_26",
        "climate_zone_27",
        "climate_zone_28",
        "climate_zone_29",
        "barren",
        "crops",
        "deciduous_forest",
        "developed",
        "evergreen_forest",
        "grass",
        "mixed_forest",
        "other",
        "shrub",
        "water",
        "wetlands",
    ]

    short_lag_days = list(range(7))
    long_lag_days = list(range(365))

    var_locs = {
        "modis": [
            "Nadir_Reflectance_Band1_interp",
            "Nadir_Reflectance_Band2_interp",
            "Nadir_Reflectance_Band3_interp",
            "Nadir_Reflectance_Band4_interp",
            "Nadir_Reflectance_Band5_interp",
            "Nadir_Reflectance_Band6_interp",
            "Nadir_Reflectance_Band7_interp",
        ],
        "daymet": [
            "srad",
            "prcp",
            "swe",
            "tmax",
            "vp",
        ],
        "static": [
            "slope",
            "elevation",
            "canopy_height",
            "clay",
            "sand",
        ],
        "climate_zone": [
            "climate_zone_1",
            "climate_zone_2",
            "climate_zone_3",
            "climate_zone_4",
            "climate_zone_5",
            "climate_zone_6",
            "climate_zone_7",
            "climate_zone_8",
            "climate_zone_9",
            "climate_zone_10",
            "climate_zone_11",
            "climate_zone_12",
            "climate_zone_13",
            "climate_zone_14",
            "climate_zone_15",
            "climate_zone_16",
            "climate_zone_17",
            "climate_zone_18",
            "climate_zone_19",
            "climate_zone_20",
            "climate_zone_21",
            "climate_zone_22",
            "climate_zone_23",
            "climate_zone_24",
            "climate_zone_25",
            "climate_zone_26",
            "climate_zone_27",
            "climate_zone_28",
            "climate_zone_29",
        ],
        "landcover_frac": [
            "barren",
            "crops",
            "deciduous_forest",
            "developed",
            "evergreen_forest",
            "grass",
            "mixed_forest",
            "other",
            "shrub",
            "water",
            "wetlands",
        ],
    }

    build_cfg = {
        "sample_index_path": sample_index_path,
        "save_dir": save_dir,
        "overwrite": bool(args.overwrite),
        "source_only": bool(args.source_only),
        "dataset_paths": dataset_paths,
        "grid_source": grid_source,
        "var_locs": var_locs,
        "short_features": short_features,
        "long_features": long_features,
        "static_features": static_features,
        "short_lag_days": short_lag_days,
        "long_lag_days": long_lag_days,
        "include_lag_feature": include_lag_feature,
        "stratifier": stratifier,
        "assume_prepared_index": assume_prepared_index,
        "temporal_batch_mode": temporal_batch_mode,
        "temporal_month_block_size": temporal_month_block_size,
        "temporal_num_workers": temporal_num_workers,
        "temporal_max_inflight": temporal_max_inflight,
    }

    print("[build_dataset] Starting direct dataset build")
    print(
        "[build_dataset] Config summary: "
        f"sample_index_path={sample_index_path}, "
        f"save_dir={save_dir}, overwrite={bool(args.overwrite)}, "
        f"grid_source={grid_source}, stratifier={stratifier}, "
        f"short_features={len(short_features)}, long_features={len(long_features)}, "
        f"static_features={len(static_features)}, "
        f"short_lags={len(short_lag_days)}, long_lags={len(long_lag_days)}, "
        f"temporal_batch_mode={temporal_batch_mode}, "
        f"temporal_month_block_size={temporal_month_block_size}, "
        f"temporal_num_workers={temporal_num_workers}, "
        f"temporal_max_inflight={temporal_max_inflight}"
    )

    sample_paths = _expand_inputs(sample_index_inputs)
    print(f"[build_dataset] Sample index inputs ({len(sample_paths)}):")
    for p in sample_paths:
        print(f"  - {p}")

    print("[build_dataset] Loading sample index...")
    sample_df = load_sample_index(sample_paths)
    print(f"[build_dataset] Loaded sample index rows: {len(sample_df):,}")

    if args.source_only:
        print("[build_dataset] source-only mode enabled; skipping tensor rebuild.")
        info_parquet = os.path.join(save_dir, "info.parquet")
        info_csv = os.path.join(save_dir, "info.csv")
        if os.path.exists(info_parquet):
            info_df = pd.read_parquet(info_parquet)
            print(f"[build_dataset] Loaded existing info.parquet rows: {len(info_df):,}")
        elif os.path.exists(info_csv):
            info_df = pd.read_csv(info_csv)
            print(f"[build_dataset] Loaded existing info.csv rows: {len(info_df):,}")
        else:
            raise FileNotFoundError(
                f"source-only requested but no info.parquet/info.csv found in {save_dir}"
            )
        source_tensor = _rebuild_source_tensor_only(sample_df, info_df)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(source_tensor, os.path.join(save_dir, "source.pt"))
        print(f"[build_dataset] Wrote source.pt to {save_dir}")
        with open(os.path.join(save_dir, "build_config.json"), "w") as f:
            json.dump(build_cfg, f, indent=2, sort_keys=True)
        print(f"[build_dataset] Updated build config at {os.path.join(save_dir, 'build_config.json')}")
        return

    print("[build_dataset] Opening source datasets...")
    dss = open_source_datasets(dataset_paths=dataset_paths)
    print(f"[build_dataset] Loaded source datasets: {sorted(list(dss.keys()))}")

    print("[build_dataset] Building tensors from sample index...")
    result = build_direct_tensors_from_sample_index(
        sample_df=sample_df,
        dss=dss,
        short_features=short_features,
        long_features=long_features,
        static_features=static_features,
        short_lag_days=short_lag_days,
        long_lag_days=long_lag_days,
        stratifier=stratifier,
        include_lag_feature=include_lag_feature,
        start_date=start_date,
        end_date=end_date,
        var_locs=var_locs,
        grid_source=grid_source,
        assume_prepared_index=assume_prepared_index,
        temporal_batch_mode=temporal_batch_mode,
        temporal_month_block_size=temporal_month_block_size,
        temporal_num_workers=temporal_num_workers,
        temporal_max_inflight=temporal_max_inflight,
    )

    if args.overwrite and os.path.isdir(save_dir):
        print(f"[build_dataset] --overwrite set, removing existing directory: {save_dir}")
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    print(f"[build_dataset] Saving outputs to: {save_dir}")
    save_direct_build_result(result, save_dir)
    if result.build_metadata is not None:
        build_cfg["build_metadata"] = result.build_metadata
    with open(os.path.join(save_dir, "build_config.json"), "w") as f:
        json.dump(build_cfg, f, indent=2, sort_keys=True)

    print(f"Saved tensors to {save_dir}")
    print(f"X_short: {tuple(result.X_short.shape)}")
    print(f"X_long: {tuple(result.X_long.shape)}")
    print(f"X_static: {tuple(result.X_static.shape)}")
    print(f"Y: {tuple(result.Y.shape)}")
    print(f"Rows kept after NaN filtering: {len(result.info):,}")
    print(f"Wrote build config to {os.path.join(save_dir, 'build_config.json')}")


if __name__ == "__main__":
    main()
