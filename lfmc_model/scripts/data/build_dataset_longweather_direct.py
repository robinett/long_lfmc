import argparse
import glob
import os

from longweather_direct_pipeline import (
    build_direct_tensors_from_sample_index,
    default_long_features,
    default_long_lag_days,
    default_short_features,
    default_short_lag_days,
    default_static_features,
    load_sample_index,
    open_source_datasets,
    save_direct_build_result,
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Build training tensors directly from zarr/nc + sample index "
            "(Parquet/CSV) without a lag-expanded compile CSV."
        )
    )
    parser.add_argument(
        "--sample-index",
        action="append",
        required=True,
        help="Sample index file path or glob (.parquet or .csv). Repeatable.",
    )
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=["lfmc", "vh_backscatter"],
        help="Target columns to use; each row should populate exactly one.",
    )
    parser.add_argument("--stratifier", type=str, default="nlcd")
    parser.add_argument("--num-rs-samples", type=int, default=100_000_000)
    parser.add_argument(
        "--vh-locations",
        type=str,
        default="all",
        choices=["all", "at_sites", "at_random"],
    )
    parser.add_argument(
        "--lfmc-min",
        type=float,
        default=30.0,
        help="Minimum acceptable LFMC for rows where target is lfmc.",
    )
    parser.add_argument(
        "--lfmc-max",
        type=float,
        default=500.0,
        help="Maximum acceptable LFMC for rows where target is lfmc.",
    )
    parser.add_argument(
        "--no-lfrac",
        action="store_true",
        help="Disable lfrac lag-position feature.",
    )
    args = parser.parse_args()

    sample_paths = _expand_inputs(args.sample_index)
    sample_df = load_sample_index(sample_paths)
    dss = open_source_datasets()

    result = build_direct_tensors_from_sample_index(
        sample_df=sample_df,
        dss=dss,
        short_features=default_short_features(),
        long_features=default_long_features(),
        static_features=default_static_features(),
        target_cols=args.target_cols,
        short_lag_days=default_short_lag_days(),
        long_lag_days=default_long_lag_days(),
        stratifier=args.stratifier,
        include_lag_feature=not args.no_lfrac,
        acceptable_lfmc_range=(args.lfmc_min, args.lfmc_max),
        num_rs_samples=args.num_rs_samples,
        vh_locations=args.vh_locations,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_direct_build_result(result, args.save_dir)

    print(f"Saved tensors to {args.save_dir}")
    print(f"X_short: {tuple(result.X_short.shape)}")
    print(f"X_long: {tuple(result.X_long.shape)}")
    print(f"X_static: {tuple(result.X_static.shape)}")
    print(f"Y: {tuple(result.Y.shape)}")
    print(f"Rows kept after NaN filtering: {len(result.info):,}")


if __name__ == "__main__":
    main()

