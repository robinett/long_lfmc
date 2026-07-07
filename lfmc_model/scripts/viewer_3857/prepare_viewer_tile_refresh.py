#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import xarray as xr
import yaml

from viewer_pipeline_common import chunk_dates, datestr_values, load_config, log, viewer_dataset_path, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a full-date viewer tile refresh plan.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--viewer-dataset-path", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evergreen-alpha", type=int, required=True)
    parser.add_argument("--block-days", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.base_config)
    cfg["output"]["viewer_dataset_path"] = str(args.viewer_dataset_path)
    cfg["output"]["asset_root"] = str(args.asset_root)
    cfg["output"]["state_dir"] = str(args.state_dir)
    cfg.setdefault("rendering", {})["evergreen_forest_alpha"] = int(args.evergreen_alpha)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"Wrote tile refresh config {args.output_config}")

    ds = xr.open_zarr(viewer_dataset_path(cfg), consolidated=False)
    try:
        dates = datestr_values(ds["time"].values)
        blocks = chunk_dates(dates, int(args.block_days))
        payload = {
            "status": "completed",
            "mode": "tile_refresh",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "viewer_dataset_path": str(viewer_dataset_path(cfg)),
            "asset_root": str(args.asset_root),
            "evergreen_forest_alpha": int(args.evergreen_alpha),
            "block_days": int(args.block_days),
            "date_count": len(dates),
            "block_count": len(blocks),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "dates": dates,
            "blocks": blocks,
        }
        write_json(args.plan_path, payload)
        log(
            "Wrote tile refresh plan "
            f"{args.plan_path} with {len(dates)} dates in {len(blocks)} blocks"
        )
        summary_path = args.plan_path.with_suffix(".summary.json")
        summary = {key: payload[key] for key in payload if key not in {"dates", "blocks"}}
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        log(f"Wrote tile refresh summary {summary_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
