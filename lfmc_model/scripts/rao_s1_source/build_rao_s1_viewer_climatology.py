#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "rao_s1_source_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def climatology_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    out = dict(cfg.get("climatology", {}))
    if not out.get("enabled", False):
        raise ValueError("climatology.enabled must be true")
    if int(out.get("calendar_day_count", 365)) != 365:
        raise ValueError("Only a 365-day climatology calendar is supported")
    if str(out.get("feb29_policy", "")) != "map_to_feb28":
        raise ValueError("Only feb29_policy=map_to_feb28 is supported")
    return out


def viewer_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["paths"]["viewer_zarr_path"]))


def state_dir(cfg: Dict[str, object]) -> Path:
    return Path(str(climatology_cfg(cfg)["state_dir"]))


def plan_path(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "rao_s1_viewer_climatology_plan.json"


def block_status_dir(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "blocks"


def block_status_path(cfg: Dict[str, object], block_index: int) -> Path:
    return block_status_dir(cfg) / f"block_{block_index:05d}.json"


def date_strings(root) -> List[str]:
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def parse_date(value) -> dt.date:
    return dt.date.fromisoformat(str(value))


def calendar_index_365(date_value: dt.date) -> int:
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    return int(dt.date(2001, date_value.month, date_value.day).timetuple().tm_yday - 1)


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
    kwargs = {
        "name": name,
        "chunks": chunks,
        "attributes": array_attrs,
        "overwrite": True,
    }
    if data is None:
        kwargs.update({"shape": shape, "dtype": dtype, "fill_value": fill_value})
    else:
        kwargs["data"] = data
    return root.create_array(**kwargs)


def drop_consolidated_metadata(store_path: Path) -> None:
    metadata_path = store_path / ".zmetadata"
    if metadata_path.exists():
        log(f"Removing stale consolidated metadata before mutation: {metadata_path}")
        try:
            metadata_path.unlink()
        except FileNotFoundError:
            pass


def build_blocks(height: int, width: int, block_size: int) -> List[Dict[str, int]]:
    blocks = []
    for y_start in range(0, height, block_size):
        y_end = min(y_start + block_size, height)
        for x_start in range(0, width, block_size):
            x_end = min(x_start + block_size, width)
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


def selected_time_metadata(root, cfg: Dict[str, object]) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    start_date = parse_date(clim_cfg["baseline_start_date"])
    end_date = parse_date(clim_cfg["baseline_end_date"])
    all_dates = [parse_date(value) for value in date_strings(root)]
    positions = [idx for idx, value in enumerate(all_dates) if start_date <= value <= end_date]
    if not positions:
        raise ValueError(f"No viewer times found between {start_date} and {end_date}")
    selected_dates = [all_dates[idx] for idx in positions]
    day_indices = np.asarray([calendar_index_365(value) for value in selected_dates], dtype=np.int16)
    return {
        "baseline_start_date": start_date.isoformat(),
        "baseline_end_date": end_date.isoformat(),
        "positions": positions,
        "dates": [value.isoformat() for value in selected_dates],
        "day_indices": day_indices,
    }


def require_array(root, name: str, expected_shape: tuple, expected_chunks: tuple) -> None:
    if name not in root:
        raise ValueError(f"Expected array {name!r} to exist after initialization")
    arr = root[name]
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Array {name!r} shape {arr.shape} != expected {expected_shape}")
    if tuple(arr.chunks) != tuple(expected_chunks):
        raise ValueError(f"Array {name!r} chunks {arr.chunks} != expected {expected_chunks}")


def initialize(args) -> None:
    cfg = load_config(args.config)
    clim_cfg = climatology_cfg(cfg)
    viewer_store = viewer_path(cfg)
    state = state_dir(cfg)
    state.mkdir(parents=True, exist_ok=True)
    block_status_dir(cfg).mkdir(parents=True, exist_ok=True)

    if args.overwrite_status and block_status_dir(cfg).exists():
        log(f"Removing existing climatology block status dir {block_status_dir(cfg)}")
        shutil.rmtree(block_status_dir(cfg))
        block_status_dir(cfg).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log(f"Would initialize Rao S1 viewer climatology for {viewer_store}")
        return

    drop_consolidated_metadata(viewer_store)
    root = zarr.open_group(str(viewer_store), mode="a")
    source_variable = str(clim_cfg["source_variable"])
    if source_variable not in root:
        raise ValueError(f"Source variable {source_variable!r} not found in {viewer_store}")
    height = int(root["y"].shape[0])
    width = int(root["x"].shape[0])
    shape = (365, height, width)
    time_meta = selected_time_metadata(root, cfg)
    tile_variable = str(clim_cfg["tile_variable"])
    point_variable = str(clim_cfg["point_variable"])
    tile_chunks = tuple(int(value) for value in clim_cfg["tile_chunks"])
    point_chunks = tuple(int(value) for value in clim_cfg["point_chunks"])
    metadata = calendar_metadata()

    for name, values in metadata.items():
        if args.overwrite_arrays and name in root:
            del root[name]
        if name not in root:
            create_array(root, name, data=values, chunks=(365,), dims=["climatology_day"])

    attrs = dict(root[source_variable].attrs.asdict())
    attrs.update(
        {
            "long_name": "Frozen 365-day calendar climatology of Sentinel-1-informed LFMC",
            "source_variable": source_variable,
            "baseline_start_date": time_meta["baseline_start_date"],
            "baseline_end_date": time_meta["baseline_end_date"],
            "calendar_day_count": 365,
            "feb29_policy": str(clim_cfg["feb29_policy"]),
            "window_offsets": json.dumps([int(value) for value in clim_cfg["window_offsets"]]),
            "window_size": int(clim_cfg["window_offsets"][1] - clim_cfg["window_offsets"][0] + 1),
        }
    )
    for variable_name, chunks, role in (
        (tile_variable, tile_chunks, "tile"),
        (point_variable, point_chunks, "point_api"),
    ):
        if args.overwrite_arrays and variable_name in root:
            log(f"Deleting existing climatology array {variable_name}")
            del root[variable_name]
        if variable_name not in root:
            variable_attrs = dict(attrs)
            variable_attrs["viewer_climatology_role"] = role
            log(f"Creating {variable_name} shape={shape} chunks={chunks}")
            create_array(
                root,
                variable_name,
                shape=shape,
                dtype=np.float32,
                chunks=chunks,
                dims=["climatology_day", "y", "x"],
                attrs=variable_attrs,
                fill_value=np.nan,
            )
        else:
            log(f"Using existing climatology array {variable_name}")

    root.attrs.update(
        {
            "lfmc_climatology_tile_variable": tile_variable,
            "lfmc_climatology_point_variable": point_variable,
            "lfmc_climatology_baseline_start_date": time_meta["baseline_start_date"],
            "lfmc_climatology_baseline_end_date": time_meta["baseline_end_date"],
            "lfmc_climatology_initialized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    root = zarr.open_group(str(viewer_store), mode="a")
    require_array(root, tile_variable, shape, tile_chunks)
    require_array(root, point_variable, shape, point_chunks)
    blocks = build_blocks(height, width, int(clim_cfg.get("work_block_size", 256)))
    payload = {
        "status": "initialized",
        "viewer_zarr_path": str(viewer_store),
        "source_variable": source_variable,
        "tile_variable": tile_variable,
        "point_variable": point_variable,
        "tile_chunks": list(tile_chunks),
        "point_chunks": list(point_chunks),
        "baseline_start_date": time_meta["baseline_start_date"],
        "baseline_end_date": time_meta["baseline_end_date"],
        "baseline_time_count": len(time_meta["positions"]),
        "baseline_dates": time_meta["dates"],
        "calendar_day_count": 365,
        "feb29_policy": str(clim_cfg["feb29_policy"]),
        "window_offsets": [int(value) for value in clim_cfg["window_offsets"]],
        "height": height,
        "width": width,
        "block_count": len(blocks),
        "blocks": blocks,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(plan_path(cfg), payload)
    log(f"Wrote Rao S1 viewer climatology plan {plan_path(cfg)} with {len(blocks)} blocks")


def compute_block(root, cfg: Dict[str, object], block: Dict[str, int], plan: Dict[str, object]) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    source_variable = str(plan["source_variable"])
    positions = np.asarray([idx for idx, _date in enumerate(date_strings(root)) if _date in set(plan["baseline_dates"])], dtype=np.int64)
    baseline_dates = [parse_date(value) for value in plan["baseline_dates"]]
    day_indices = np.asarray([calendar_index_365(value) for value in baseline_dates], dtype=np.int16)
    time_block = int(clim_cfg.get("time_block_size", 64))
    window_start, window_end = [int(value) for value in plan["window_offsets"]]
    y_slice = slice(int(block["y_start"]), int(block["y_end"]))
    x_slice = slice(int(block["x_start"]), int(block["x_end"]))
    height = int(block["y_end"]) - int(block["y_start"])
    width = int(block["x_end"]) - int(block["x_start"])
    sum_by_day = np.zeros((365, height, width), dtype=np.float64)
    count_by_day = np.zeros((365, height, width), dtype=np.uint16)

    for start in range(0, positions.size, time_block):
        end = min(start + time_block, positions.size)
        selected_positions = positions[start:end]
        values = np.asarray(
            root[source_variable].get_orthogonal_selection((selected_positions, y_slice, x_slice)),
            dtype=np.float32,
        )
        local_day_indices = day_indices[start:end]
        for day_idx in np.unique(local_day_indices):
            local_values = values[local_day_indices == day_idx, :, :]
            finite = np.isfinite(local_values)
            sum_by_day[day_idx, :, :] += np.where(finite, local_values, 0.0).sum(axis=0)
            count_by_day[day_idx, :, :] += finite.sum(axis=0).astype(np.uint16)
        log(f"Accumulated block {block['block_index']} times {start}-{end - 1} of {positions.size - 1}")

    climatology = np.full((365, height, width), np.nan, dtype=np.float32)
    min_count = None
    max_count = None
    for day_idx in range(365):
        window = window_indices(day_idx, window_start, window_end)
        window_sum = sum_by_day[window, :, :].sum(axis=0)
        window_count = count_by_day[window, :, :].sum(axis=0)
        valid = window_count > 0
        np.divide(window_sum, window_count, out=climatology[day_idx, :, :], where=valid, casting="unsafe")
        if np.any(valid):
            valid_counts = window_count[valid]
            local_min = int(valid_counts.min())
            local_max = int(valid_counts.max())
            min_count = local_min if min_count is None else min(min_count, local_min)
            max_count = local_max if max_count is None else max(max_count, local_max)

    return {
        "climatology": climatology,
        "finite_fraction": float(np.isfinite(climatology).mean()),
        "min_count": min_count,
        "max_count": max_count,
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
        f"Starting Rao S1 climatology block {block_index}/{len(blocks) - 1} "
        f"y={block['y_start']}:{block['y_end']} x={block['x_start']}:{block['x_end']}"
    )
    root = zarr.open_group(str(viewer_path(cfg)), mode="a")
    result = compute_block(root, cfg, block, plan)
    y_slice = slice(int(block["y_start"]), int(block["y_end"]))
    x_slice = slice(int(block["x_start"]), int(block["x_end"]))
    root[str(plan["tile_variable"])][:, y_slice, x_slice] = result["climatology"]
    root[str(plan["point_variable"])][:, y_slice, x_slice] = result["climatology"]
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
    log(f"Completed Rao S1 climatology block {block_index}; wrote {status_path}")


def finalize(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    completed = sum(1 for block in plan["blocks"] if block_status_path(cfg, int(block["block_index"])).exists())
    if completed != int(plan["block_count"]):
        raise RuntimeError(f"Only {completed}/{plan['block_count']} climatology blocks are complete")
    root = zarr.open_group(str(viewer_path(cfg)), mode="a")
    require_array(root, str(plan["tile_variable"]), (365, int(plan["height"]), int(plan["width"])), tuple(plan["tile_chunks"]))
    require_array(root, str(plan["point_variable"]), (365, int(plan["height"]), int(plan["width"])), tuple(plan["point_chunks"]))
    root.attrs.update({"lfmc_climatology_finalized_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"Consolidating viewer zarr metadata for {viewer_path(cfg)}")
    zarr.consolidate_metadata(str(viewer_path(cfg)))
    write_json(
        state_dir(cfg) / "rao_s1_viewer_climatology_completed.json",
        {
            "status": "completed",
            "block_count": int(plan["block_count"]),
            "viewer_zarr_path": str(viewer_path(cfg)),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    log("Rao S1 viewer climatology finalize complete")


def report_status(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    completed = sum(1 for block in plan["blocks"] if block_status_path(cfg, int(block["block_index"])).exists())
    payload = {
        "status": "running" if completed < int(plan["block_count"]) else "completed",
        "block_count": int(plan["block_count"]),
        "completed_count": int(completed),
        "remaining_count": int(plan["block_count"]) - int(completed),
        "viewer_zarr_path": str(viewer_path(cfg)),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(payload, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build frozen Sentinel-1 LFMC viewer climatology arrays.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["init", "worker", "finalize", "status"], required=True)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--overwrite-arrays", action="store_true")
    parser.add_argument("--overwrite-status", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "init":
        initialize(args)
    elif args.mode == "worker":
        run_worker(args)
    elif args.mode == "finalize":
        finalize(args)
    elif args.mode == "status":
        report_status(args)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
