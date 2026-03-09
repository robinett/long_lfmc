#!/usr/bin/env python3

import argparse
import json
import os

from longweather_direct_pipeline import build_training_sample_index_from_label_sources


def _parse_key_value_list(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected non-empty KEY=VALUE, got: {item}")
        out[key] = value
    return out


def _parse_int_mapping(items):
    raw = _parse_key_value_list(items)
    return {k: int(v) for k, v in raw.items()}


def _parse_float_mapping(items):
    raw = _parse_key_value_list(items)
    return {k: float(v) for k, v in raw.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a longweather training sample index from LFMC and SAR label CSVs."
    )
    parser.add_argument("--start-date", type=str, default="2000-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument(
        "--label-source",
        action="append",
        default=[],
        help="Label source mapping in KEY=PATH format. Repeat as needed.",
    )
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=None,
        help="Target columns to keep in the prepared sample index.",
    )
    parser.add_argument("--acceptable-lfmc-min", type=float, default=30.0)
    parser.add_argument("--acceptable-lfmc-max", type=float, default=500.0)
    parser.add_argument("--vh-locations", type=str, default="all")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--target-sample-n",
        action="append",
        default=[],
        help="Optional per-target cap in TARGET=N format.",
    )
    parser.add_argument(
        "--target-sample-fraction",
        action="append",
        default=[],
        help="Optional per-target sampling fraction in TARGET=FRAC format.",
    )
    return parser.parse_args()


def default_config():
    scratch_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc"
    out_path = os.path.join(
        scratch_dir,
        "lfmc_model",
        "indexes",
        "sample_index_longweather_2000_2024_lfmc_vv_over_vh.parquet",
    )
    label_sources = {
        "nfmd": os.path.join(
            scratch_dir,
            "nfmd",
            "nfmd_processed.csv",
        ),
        "vv_over_vh_at_sites": os.path.join(
            scratch_dir,
            "sar",
            "vv_over_vh_samples_at_sites_matching.csv",
        ),
        "vv_over_vh_at_random": os.path.join(
            scratch_dir,
            "sar",
            "vv_over_vh_samples_random_matching.csv",
        ),
    }
    return {
        "out_path": out_path,
        "label_sources": label_sources,
        "target_cols": ["lfmc", "vv_over_vh"],
        "acceptable_lfmc_range": (30.0, 500.0),
        "vh_locations": "all",
        "random_seed": 42,
        "target_sample_n": {
            "lfmc": -1,
            "vv_over_vh": -1,
        },
        "target_sample_fraction": {},
    }


def main():
    args = parse_args()
    cfg = default_config()

    label_sources = (
        _parse_key_value_list(args.label_source)
        if args.label_source
        else cfg["label_sources"]
    )
    target_cols = list(args.target_cols) if args.target_cols is not None else cfg["target_cols"]
    out_path = args.out_path if args.out_path is not None else cfg["out_path"]
    acceptable_lfmc_range = (float(args.acceptable_lfmc_min), float(args.acceptable_lfmc_max))
    target_sample_n = (
        _parse_int_mapping(args.target_sample_n)
        if args.target_sample_n
        else cfg["target_sample_n"]
    )
    target_sample_fraction = (
        _parse_float_mapping(args.target_sample_fraction)
        if args.target_sample_fraction
        else cfg["target_sample_fraction"]
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = build_training_sample_index_from_label_sources(
        label_sources=label_sources,
        start_date=args.start_date,
        end_date=args.end_date,
        out_path=out_path,
        target_cols=target_cols,
        acceptable_lfmc_range=acceptable_lfmc_range,
        target_sample_n=target_sample_n,
        target_sample_fraction=target_sample_fraction,
        vh_locations=args.vh_locations,
        random_seed=int(args.random_seed),
        sort_by=["date", "latitude", "longitude"],
    )

    print(f"Wrote {len(df):,} rows to {out_path}")
    print("df:")
    print(df)
    if "target_name" in df.columns:
        print(f"Target counts: {df['target_name'].value_counts(dropna=False).to_dict()}")
    if "source_legible" in df.columns:
        print(f"Source counts: {df['source_legible'].value_counts(dropna=False).to_dict()}")
    print(
        json.dumps(
            {
                "target_cols": target_cols,
                "acceptable_lfmc_range": list(acceptable_lfmc_range),
                "label_sources": label_sources,
                "vh_locations": args.vh_locations,
                "random_seed": int(args.random_seed),
                "target_sample_n": target_sample_n,
                "target_sample_fraction": target_sample_fraction,
                "out_path": out_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
