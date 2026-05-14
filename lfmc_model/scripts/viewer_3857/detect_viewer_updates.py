#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from viewer_pipeline_common import (
    chunk_dates,
    dataset_status_dir,
    date_status_path,
    ensure_state_dirs,
    load_config,
    log,
    open_source_dataset,
    plan_path,
    source_quality_by_date,
    tile_status_dir,
    viewer_dataset_path,
    viewer_quality_by_date,
    write_json,
)


def detect(args):
    cfg = load_config(args.config)
    ensure_state_dirs(cfg)
    ds = open_source_dataset(cfg)
    try:
        source_quality = source_quality_by_date(ds, cfg)
        source_dates = list(source_quality.keys())
        quality_variable = str(cfg["dataset"]["quality_variable"])
        viewer_quality = viewer_quality_by_date(viewer_dataset_path(cfg), quality_variable)

        reasons = {}
        if args.mode == "init" or not viewer_quality:
            selected_dates = source_dates
            reasons = {date: "init" if args.mode == "init" else "missing_viewer_dataset" for date in selected_dates}
        else:
            selected_dates = []
            for date in source_dates:
                if date not in viewer_quality:
                    selected_dates.append(date)
                    reasons[date] = "new_date"
                elif int(source_quality[date]) != int(viewer_quality[date]):
                    selected_dates.append(date)
                    reasons[date] = "changed_quality_flag"
                elif not date_status_path(dataset_status_dir(cfg), date).exists():
                    selected_dates.append(date)
                    reasons[date] = "missing_dataset_status"
                elif not date_status_path(tile_status_dir(cfg), date).exists():
                    selected_dates.append(date)
                    reasons[date] = "missing_tile_status"

        block_days = args.block_days
        if block_days is None:
            key = "init_block_days" if args.mode == "init" else "append_block_days"
            block_days = int(cfg.get("batching", {}).get(key, 8))
        blocks = chunk_dates(selected_dates, block_days)
        payload = {
            "status": "completed",
            "mode": args.mode,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dataset_path": str(cfg["dataset"]["scientific_dataset_path"]),
            "viewer_dataset_path": str(viewer_dataset_path(cfg)),
            "quality_variable": quality_variable,
            "block_days": int(block_days),
            "date_count": len(selected_dates),
            "block_count": len(blocks),
            "dates": selected_dates,
            "reasons": reasons,
            "blocks": blocks,
        }
        out = args.plan_path or plan_path(cfg)
        write_json(out, payload)
        log(f"Wrote update plan {out} with {len(selected_dates)} dates in {len(blocks)} blocks")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Detect viewer dates needing rebuild from actual Zarr data.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "viewer_pipeline_config.yaml")
    parser.add_argument("--mode", choices=["init", "append"], default="append")
    parser.add_argument("--block-days", type=int, default=None)
    parser.add_argument("--plan-path", type=Path, default=None)
    return parser.parse_args()


def main():
    detect(parse_args())


if __name__ == "__main__":
    main()
