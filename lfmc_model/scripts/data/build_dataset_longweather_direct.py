import argparse
import builtins
import datetime
import glob
import json
import os

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


def main():
    default_sample_index_path = (
        "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/"
        "sample_index_longweather_2000_2024_lfmc.parquet"
    )
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
    args = parser.parse_args()
    if args.sample_index_path is None:
        print(
            "Warning: no sample_index_path provided. Defaulting to "
            f"{default_sample_index_path}"
        )
        sample_index_path = default_sample_index_path
    else:
        sample_index_path = str(args.sample_index_path)

    # All configuration is explicitly set here.
    scratch_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc"
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"

    sample_index_inputs = [sample_index_path]

    save_dir = os.path.join(
        scratch_dir,
        "lfmc_model",
        "inputs",
        "lfmc",
    )

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
            oak_dir,
            "static",
            "static_features_500m_epsg5070_float32.nc",
        ),
        "climate_zone": os.path.join(
            oak_dir,
            "climate_zones",
            "climate_zone_per_pixel_westUS.nc4",
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
        "latitude",
        "longitude",
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
    long_lag_days = list(range(181))

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

    os.makedirs(save_dir, exist_ok=True)
    print(f"[build_dataset] Saving outputs to: {save_dir}")
    save_direct_build_result(result, save_dir)
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
