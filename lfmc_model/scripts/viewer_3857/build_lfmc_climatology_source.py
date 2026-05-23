#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "viewer_pipeline_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path) -> Dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def parse_date(value) -> dt.date:
    return dt.date.fromisoformat(str(value))


def climatology_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    out = dict(cfg.get("climatology", {}))
    if not out.get("enabled", False):
        raise ValueError("climatology.enabled must be true to build source climatology")
    if int(out.get("calendar_day_count", 365)) != 365:
        raise ValueError("This builder currently supports calendar_day_count=365 only")
    if str(out.get("feb29_policy", "")) != "map_to_feb28":
        raise ValueError("This builder currently requires feb29_policy=map_to_feb28")
    return out


def source_dataset_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["dataset"]["scientific_dataset_path"]))


def output_path(cfg: Dict[str, object]) -> Path:
    return Path(str(climatology_cfg(cfg)["source_output_path"]))


def state_dir(cfg: Dict[str, object]) -> Path:
    return Path(str(climatology_cfg(cfg)["source_state_dir"]))


def plan_path(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "source_climatology_plan.json"


def block_status_dir(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "blocks"


def block_status_path(cfg: Dict[str, object], block_index: int) -> Path:
    return block_status_dir(cfg) / f"block_{block_index:05d}.json"


def calendar_index_365(date_value: dt.date) -> int:
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    template_date = dt.date(2001, date_value.month, date_value.day)
    return int(template_date.timetuple().tm_yday - 1)


def calendar_metadata() -> Dict[str, np.ndarray]:
    dates = [dt.date(2001, 1, 1) + dt.timedelta(days=idx) for idx in range(365)]
    return {
        "climatology_day": np.arange(1, 366, dtype=np.int16),
        "climatology_month": np.asarray([value.month for value in dates], dtype=np.int8),
        "climatology_day_of_month": np.asarray([value.day for value in dates], dtype=np.int8),
    }


def window_indices(center_idx: int, start_offset: int, end_offset: int) -> np.ndarray:
    return np.asarray([(center_idx + offset) % 365 for offset in range(start_offset, end_offset + 1)], dtype=np.int16)


def create_array(root, name: str, *, shape=None, dtype=None, chunks=None, data=None, dims=None, attrs=None, fill_value=None):
    array_attrs = dict(attrs or {})
    array_attrs["_ARRAY_DIMENSIONS"] = list(dims or [])
    create_kwargs = {
        "name": name,
        "chunks": chunks,
        "attributes": array_attrs,
        "overwrite": True,
    }
    if data is not None:
        create_kwargs["data"] = data
    else:
        create_kwargs["shape"] = shape
        create_kwargs["dtype"] = dtype
        create_kwargs["fill_value"] = fill_value
    return root.create_array(**create_kwargs)


def selected_time_metadata(ds: xr.Dataset, cfg: Dict[str, object]) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    start_date = parse_date(clim_cfg["baseline_start_date"])
    end_date = parse_date(clim_cfg["baseline_end_date"])
    all_dates = [parse_date(datestr(value)) for value in ds["time"].values]
    positions = [idx for idx, value in enumerate(all_dates) if start_date <= value <= end_date]
    if not positions:
        raise ValueError(f"No source times found between {start_date} and {end_date}")
    selected_dates = [all_dates[idx] for idx in positions]
    day_indices = np.asarray([calendar_index_365(value) for value in selected_dates], dtype=np.int16)
    return {
        "baseline_start_date": start_date.isoformat(),
        "baseline_end_date": end_date.isoformat(),
        "positions": positions,
        "dates": [value.isoformat() for value in selected_dates],
        "day_indices": day_indices,
    }


def build_blocks(ds: xr.Dataset, cfg: Dict[str, object]) -> List[Dict[str, int]]:
    clim_cfg = climatology_cfg(cfg)
    work_block = int(clim_cfg.get("source_work_block_size", 256))
    y_size = int(ds.sizes["y"])
    x_size = int(ds.sizes["x"])
    blocks = []
    for y_start in range(0, y_size, work_block):
        y_end = min(y_start + work_block, y_size)
        for x_start in range(0, x_size, work_block):
            x_end = min(x_start + work_block, x_size)
            blocks.append(
                {
                    "block_index": len(blocks),
                    "y_start": int(y_start),
                    "y_end": int(y_end),
                    "x_start": int(x_start),
                    "x_end": int(x_end),
                }
            )
    return blocks


def initialize(args) -> None:
    cfg = load_config(args.config)
    clim_cfg = climatology_cfg(cfg)
    out = output_path(cfg)
    state = state_dir(cfg)
    state.mkdir(parents=True, exist_ok=True)
    block_status_dir(cfg).mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if out.exists():
            log(f"Removing existing source climatology zarr {out}")
            shutil.rmtree(out)
        if block_status_dir(cfg).exists():
            log(f"Removing existing source climatology block state {block_status_dir(cfg)}")
            shutil.rmtree(block_status_dir(cfg))
            block_status_dir(cfg).mkdir(parents=True, exist_ok=True)

    log(f"Opening source dataset {source_dataset_path(cfg)}")
    ds = xr.open_zarr(source_dataset_path(cfg), consolidated=False)
    try:
        source_variable = str(clim_cfg["source_variable"])
        if source_variable not in ds:
            raise ValueError(f"Source variable {source_variable!r} not found in {source_dataset_path(cfg)}")
        time_meta = selected_time_metadata(ds, cfg)
        blocks = build_blocks(ds, cfg)
        spatial_chunk = int(clim_cfg.get("source_spatial_chunk_size", 128))
        day_chunk = int(clim_cfg.get("source_climatology_day_chunk_size", 365))
        window_start, window_end = [int(value) for value in clim_cfg["window_offsets"]]
        metadata = calendar_metadata()

        if not out.exists():
            log(f"Creating source climatology zarr {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            root = zarr.open_group(
                str(out),
                mode="w",
                zarr_format=2,
                attributes={
                    "source_dataset_path": str(source_dataset_path(cfg)),
                    "source_variable": source_variable,
                    "baseline_start_date": time_meta["baseline_start_date"],
                    "baseline_end_date": time_meta["baseline_end_date"],
                    "calendar_day_count": 365,
                    "feb29_policy": str(clim_cfg["feb29_policy"]),
                    "window_offsets": [window_start, window_end],
                    "window_size": int(window_end - window_start + 1),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            create_array(
                root,
                "x",
                data=np.asarray(ds["x"].values, dtype=np.float64),
                chunks=(min(spatial_chunk * 4, int(ds.sizes["x"])),),
                dims=["x"],
                attrs=dict(ds["x"].attrs),
            )
            create_array(
                root,
                "y",
                data=np.asarray(ds["y"].values, dtype=np.float64),
                chunks=(min(spatial_chunk * 4, int(ds.sizes["y"])),),
                dims=["y"],
                attrs=dict(ds["y"].attrs),
            )
            for name, values in metadata.items():
                create_array(root, name, data=values, chunks=(365,), dims=["climatology_day"])
            attrs = dict(ds[source_variable].attrs)
            attrs.update(
                {
                    "long_name": "365-day rolling calendar climatology of LFMC ensemble mean",
                    "source_variable": source_variable,
                    "calendar_day_count": 365,
                    "feb29_policy": str(clim_cfg["feb29_policy"]),
                    "window_offsets": json.dumps([window_start, window_end]),
                    "window_size": int(window_end - window_start + 1),
                }
            )
            create_array(
                root,
                "lfmc_climatology_mean",
                shape=(365, int(ds.sizes["y"]), int(ds.sizes["x"])),
                dtype=np.float32,
                chunks=(day_chunk, spatial_chunk, spatial_chunk),
                dims=["climatology_day", "y", "x"],
                attrs=attrs,
                fill_value=np.nan,
            )
        else:
            log(f"Using existing source climatology zarr {out}")

        payload = {
            "status": "initialized",
            "source_dataset_path": str(source_dataset_path(cfg)),
            "source_variable": source_variable,
            "source_output_path": str(out),
            "baseline_start_date": time_meta["baseline_start_date"],
            "baseline_end_date": time_meta["baseline_end_date"],
            "baseline_time_count": len(time_meta["positions"]),
            "calendar_day_count": 365,
            "feb29_policy": str(clim_cfg["feb29_policy"]),
            "window_offsets": [window_start, window_end],
            "window_size": int(window_end - window_start + 1),
            "source_work_block_size": int(clim_cfg.get("source_work_block_size", 256)),
            "source_time_block_size": int(clim_cfg.get("source_time_block_size", 128)),
            "source_spatial_chunk_size": spatial_chunk,
            "source_climatology_day_chunk_size": day_chunk,
            "block_count": len(blocks),
            "blocks": blocks,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(plan_path(cfg), payload)
        log(f"Wrote source climatology plan {plan_path(cfg)} with {len(blocks)} blocks")
    finally:
        ds.close()


def compute_block(ds: xr.Dataset, cfg: Dict[str, object], block: Dict[str, int]) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    source_variable = str(clim_cfg["source_variable"])
    time_meta = selected_time_metadata(ds, cfg)
    positions = np.asarray(time_meta["positions"], dtype=np.int64)
    day_indices = np.asarray(time_meta["day_indices"], dtype=np.int16)
    time_block = int(clim_cfg.get("source_time_block_size", 128))
    window_start, window_end = [int(value) for value in clim_cfg["window_offsets"]]

    y_slice = slice(int(block["y_start"]), int(block["y_end"]))
    x_slice = slice(int(block["x_start"]), int(block["x_end"]))
    height = int(block["y_end"]) - int(block["y_start"])
    width = int(block["x_end"]) - int(block["x_start"])
    sum_by_day = np.zeros((365, height, width), dtype=np.float64)
    count_by_day = np.zeros((365, height, width), dtype=np.uint16)

    for start in range(0, positions.size, time_block):
        end = min(start + time_block, positions.size)
        selected_positions = positions[start:end]
        if np.all(np.diff(selected_positions) == 1):
            time_selector = slice(int(selected_positions[0]), int(selected_positions[-1]) + 1)
        else:
            time_selector = selected_positions
        values = np.asarray(
            ds[source_variable].isel(time=time_selector, y=y_slice, x=x_slice).values,
            dtype=np.float32,
        )
        local_day_indices = day_indices[start:end]
        for day_idx in np.unique(local_day_indices):
            local_values = values[local_day_indices == day_idx, :, :]
            finite = np.isfinite(local_values)
            sum_by_day[day_idx, :, :] += np.where(finite, local_values, 0.0).sum(axis=0)
            count_by_day[day_idx, :, :] += finite.sum(axis=0).astype(np.uint16)
        log(
            "Accumulated block "
            f"{int(block['block_index'])} times {start}-{end - 1} of {positions.size - 1}"
        )

    climatology = np.full((365, height, width), np.nan, dtype=np.float32)
    min_count = None
    max_count = None
    for day_idx in range(365):
        window = window_indices(day_idx, window_start, window_end)
        window_sum = sum_by_day[window, :, :].sum(axis=0)
        window_count = count_by_day[window, :, :].sum(axis=0)
        valid = window_count > 0
        np.divide(
            window_sum,
            window_count,
            out=climatology[day_idx, :, :],
            where=valid,
            casting="unsafe",
        )
        if np.any(valid):
            valid_counts = window_count[valid]
            local_min = int(valid_counts.min())
            local_max = int(valid_counts.max())
            min_count = local_min if min_count is None else min(min_count, local_min)
            max_count = local_max if max_count is None else max(max_count, local_max)

    return {
        "climatology": climatology,
        "finite_fraction": float(np.isfinite(climatology).mean()),
        "min_count": None if min_count is None else int(min_count),
        "max_count": None if max_count is None else int(max_count),
    }


def run_worker(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    block_index = args.block_index
    if block_index is None and args.use_slurm_array:
        block_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if block_index is None:
        raise ValueError("Provide --block-index or --use-slurm-array")
    blocks = list(plan["blocks"])
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"Block index {block_index} outside block count {len(blocks)}")
    status_path = block_status_path(cfg, block_index)
    if status_path.exists() and not args.force:
        log(f"Using existing completed block status {status_path}")
        return

    block = blocks[block_index]
    log(
        "Starting source climatology block "
        f"{block_index}/{len(blocks) - 1} "
        f"y={block['y_start']}:{block['y_end']} x={block['x_start']}:{block['x_end']}"
    )
    ds = xr.open_zarr(source_dataset_path(cfg), consolidated=False)
    try:
        result = compute_block(ds, cfg, block)
    finally:
        ds.close()

    root = zarr.open_group(str(output_path(cfg)), mode="a")
    clim_arr = root["lfmc_climatology_mean"]
    clim_arr[
        :,
        int(block["y_start"]): int(block["y_end"]),
        int(block["x_start"]): int(block["x_end"]),
    ] = result["climatology"]

    write_json(
        status_path,
        {
            "status": "completed",
            "block": block,
            "finite_fraction": result["finite_fraction"],
            "min_window_count": result["min_count"],
            "max_window_count": result["max_count"],
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    log(f"Completed source climatology block {block_index}; wrote {status_path}")


def report_status(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    completed = sum(1 for block in plan["blocks"] if block_status_path(cfg, int(block["block_index"])).exists())
    payload = {
        "status": "running" if completed < int(plan["block_count"]) else "completed",
        "block_count": int(plan["block_count"]),
        "completed_count": int(completed),
        "remaining_count": int(plan["block_count"]) - int(completed),
        "source_output_path": str(output_path(cfg)),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(payload, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build 365-day source-grid LFMC climatology for the viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["init", "worker", "status"], required=True)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "init":
        initialize(args)
    elif args.mode == "worker":
        run_worker(args)
    elif args.mode == "status":
        report_status(args)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
