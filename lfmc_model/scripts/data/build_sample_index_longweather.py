import argparse
import os

from longweather_direct_pipeline import (
    build_sample_index_from_label_sources,
    default_label_sources,
)


def _parse_label_sources(values):
    if not values:
        return default_label_sources()
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                "Each --label-source must look like name=/path/to/file.csv"
            )
        name, path = item.split("=", 1)
        out[name] = path
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact sample index (Parquet/CSV) from label CSVs. "
            "This replaces the wide lag-expanded compile CSV."
        )
    )
    parser.add_argument("--start-date", required=True, type=str)
    parser.add_argument("--end-date", required=True, type=str)
    parser.add_argument(
        "--out-path",
        required=True,
        type=str,
        help="Output table path (.parquet recommended, .csv supported).",
    )
    parser.add_argument(
        "--label-source",
        action="append",
        default=[],
        help=(
            "Label source spec in the form name=/path/to/file.csv. "
            "Repeat for multiple sources. If omitted, defaults to nfmd."
        ),
    )
    args = parser.parse_args()

    label_sources = _parse_label_sources(args.label_source)
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = build_sample_index_from_label_sources(
        label_sources=label_sources,
        start_date=args.start_date,
        end_date=args.end_date,
        out_path=args.out_path,
        sort_by=["date", "latitude", "longitude"],
    )
    print(f"Wrote {len(df):,} rows to {args.out_path}")
    print(f"Sources: {sorted(df['source'].astype(str).unique().tolist())}")


if __name__ == "__main__":
    main()
