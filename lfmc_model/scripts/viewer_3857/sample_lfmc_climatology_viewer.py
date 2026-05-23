#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "viewer_pipeline_config.yaml"
SOURCE_CLIM_VARIABLE = "lfmc_climatology_mean"


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


def climatology_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    out = dict(cfg.get("climatology", {}))
    if not out.get("enabled", False):
        raise ValueError("climatology.enabled must be true")
    return out


def viewer_dataset_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["viewer_dataset_path"]))


def source_climatology_path(cfg: Dict[str, object]) -> Path:
    return Path(str(climatology_cfg(cfg)["source_output_path"]))


def lookup_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["lookup_path"]))


def state_dir(cfg: Dict[str, object]) -> Path:
    return Path(str(climatology_cfg(cfg)["viewer_state_dir"]))


def plan_path(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "viewer_climatology_plan.json"


def block_status_dir(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "blocks"


def block_status_path(cfg: Dict[str, object], block_index: int) -> Path:
    return block_status_dir(cfg) / f"block_{block_index:05d}.json"


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


def drop_consolidated_metadata(store_path: Path) -> None:
    metadata_path = Path(store_path) / ".zmetadata"
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


def require_array(root, name: str, expected_shape: tuple, expected_chunks: tuple) -> None:
    if name not in root:
        raise ValueError(f"Expected array {name!r} to exist after initialization")
    arr = root[name]
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Array {name!r} shape {arr.shape} != expected {expected_shape}")
    if tuple(arr.chunks) != tuple(expected_chunks):
        raise ValueError(f"Array {name!r} chunks {arr.chunks} != expected {expected_chunks}")


def maybe_create_output_arrays(cfg: Dict[str, object], overwrite: bool) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    source_root = zarr.open_group(str(source_climatology_path(cfg)), mode="r")
    drop_consolidated_metadata(viewer_dataset_path(cfg))
    viewer_root = zarr.open_group(str(viewer_dataset_path(cfg)), mode="a")

    tile_variable = str(clim_cfg["viewer_tile_variable"])
    point_variable = str(clim_cfg["viewer_point_variable"])
    tile_chunks = tuple(int(value) for value in clim_cfg["viewer_tile_chunks"])
    point_chunks = tuple(int(value) for value in clim_cfg["viewer_point_chunks"])
    source_arr = source_root[SOURCE_CLIM_VARIABLE]
    height = int(viewer_root["y"].shape[0])
    width = int(viewer_root["x"].shape[0])
    shape = (365, height, width)

    for name in ("climatology_day", "climatology_month", "climatology_day_of_month"):
        if overwrite and name in viewer_root:
            del viewer_root[name]
        if name not in viewer_root:
            create_array(
                viewer_root,
                name,
                data=np.asarray(source_root[name][:]),
                chunks=(365,),
                dims=["climatology_day"],
                attrs=dict(source_root[name].attrs.asdict()),
            )

    attrs = dict(source_arr.attrs.asdict())
    attrs.update(
        {
            "source_climatology_path": str(source_climatology_path(cfg)),
            "viewer_sampled_from": SOURCE_CLIM_VARIABLE,
        }
    )
    for variable_name, chunks, role in (
        (tile_variable, tile_chunks, "tile"),
        (point_variable, point_chunks, "point_api"),
    ):
        if overwrite and variable_name in viewer_root:
            del viewer_root[variable_name]
        if variable_name not in viewer_root:
            variable_attrs = dict(attrs)
            variable_attrs["viewer_climatology_role"] = role
            log(f"Creating viewer climatology array {variable_name} shape={shape} chunks={chunks}")
            create_array(
                viewer_root,
                variable_name,
                shape=shape,
                dtype=np.float32,
                chunks=chunks,
                dims=["climatology_day", "y", "x"],
                attrs=variable_attrs,
                fill_value=np.nan,
            )
        else:
            log(f"Using existing viewer climatology array {variable_name}")

    viewer_root.attrs.update(
        {
            "lfmc_climatology_source_path": str(source_climatology_path(cfg)),
            "lfmc_climatology_tile_variable": tile_variable,
            "lfmc_climatology_point_variable": point_variable,
            "lfmc_climatology_updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Reopen after structural changes before verification or further writes.
    viewer_root = zarr.open_group(str(viewer_dataset_path(cfg)), mode="a")
    require_array(viewer_root, tile_variable, shape, tile_chunks)
    require_array(viewer_root, point_variable, shape, point_chunks)
    return {"height": height, "width": width}


def initialize(args) -> None:
    cfg = load_config(args.config)
    clim_cfg = climatology_cfg(cfg)
    state_dir(cfg).mkdir(parents=True, exist_ok=True)
    block_status_dir(cfg).mkdir(parents=True, exist_ok=True)
    if args.overwrite_status and block_status_dir(cfg).exists():
        log(f"Removing existing viewer climatology status dir {block_status_dir(cfg)}")
        shutil.rmtree(block_status_dir(cfg))
        block_status_dir(cfg).mkdir(parents=True, exist_ok=True)

    shape_info = maybe_create_output_arrays(cfg, overwrite=args.overwrite_arrays)
    block_size = int(clim_cfg.get("viewer_work_block_size", 256))
    blocks = build_blocks(shape_info["height"], shape_info["width"], block_size)
    payload = {
        "status": "initialized",
        "viewer_dataset_path": str(viewer_dataset_path(cfg)),
        "source_climatology_path": str(source_climatology_path(cfg)),
        "lookup_path": str(lookup_path(cfg)),
        "tile_variable": str(clim_cfg["viewer_tile_variable"]),
        "point_variable": str(clim_cfg["viewer_point_variable"]),
        "tile_chunks": [int(value) for value in clim_cfg["viewer_tile_chunks"]],
        "point_chunks": [int(value) for value in clim_cfg["viewer_point_chunks"]],
        "height": int(shape_info["height"]),
        "width": int(shape_info["width"]),
        "block_size": block_size,
        "block_count": len(blocks),
        "blocks": blocks,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(plan_path(cfg), payload)
    log(f"Wrote viewer climatology plan {plan_path(cfg)} with {len(blocks)} blocks")


def sample_block(cfg: Dict[str, object], block: Dict[str, int]) -> Dict[str, object]:
    clim_cfg = climatology_cfg(cfg)
    source_root = zarr.open_group(str(source_climatology_path(cfg)), mode="r")
    lookup_root = zarr.open_group(str(lookup_path(cfg)), mode="r")
    source_arr = source_root[SOURCE_CLIM_VARIABLE]

    y_slice = slice(int(block["y_start"]), int(block["y_end"]))
    x_slice = slice(int(block["x_start"]), int(block["x_end"]))
    source_rows = np.asarray(lookup_root["source_row"][y_slice, x_slice], dtype=np.int32)
    source_cols = np.asarray(lookup_root["source_col"][y_slice, x_slice], dtype=np.int32)
    row_min = int(source_rows.min())
    row_max = int(source_rows.max())
    col_min = int(source_cols.min())
    col_max = int(source_cols.max())
    source_block = np.asarray(
        source_arr[:, row_min:row_max + 1, col_min:col_max + 1],
        dtype=np.float32,
    )
    local_rows = source_rows - row_min
    local_cols = source_cols - col_min
    sampled = source_block[:, local_rows, local_cols].astype(np.float32, copy=False)

    drop_consolidated_metadata(viewer_dataset_path(cfg))
    viewer_root = zarr.open_group(str(viewer_dataset_path(cfg)), mode="a")
    tile_variable = str(clim_cfg["viewer_tile_variable"])
    point_variable = str(clim_cfg["viewer_point_variable"])
    viewer_root[tile_variable][:, y_slice, x_slice] = sampled
    viewer_root[point_variable][:, y_slice, x_slice] = sampled

    return {
        "finite_fraction": float(np.isfinite(sampled).mean()),
        "source_row_min": row_min,
        "source_row_max": row_max,
        "source_col_min": col_min,
        "source_col_max": col_max,
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
        "Starting viewer climatology block "
        f"{block_index}/{len(blocks) - 1} "
        f"y={block['y_start']}:{block['y_end']} x={block['x_start']}:{block['x_end']}"
    )
    result = sample_block(cfg, block)
    write_json(
        status_path,
        {
            "status": "completed",
            "block": block,
            **result,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    log(f"Completed viewer climatology block {block_index}; wrote {status_path}")


def finalize(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    completed = sum(1 for block in plan["blocks"] if block_status_path(cfg, int(block["block_index"])).exists())
    if completed != int(plan["block_count"]):
        raise RuntimeError(f"Only {completed}/{plan['block_count']} viewer climatology blocks are complete")

    drop_consolidated_metadata(viewer_dataset_path(cfg))
    root = zarr.open_group(str(viewer_dataset_path(cfg)), mode="a")
    clim_cfg = climatology_cfg(cfg)
    require_array(
        root,
        str(clim_cfg["viewer_tile_variable"]),
        (365, int(plan["height"]), int(plan["width"])),
        tuple(int(value) for value in clim_cfg["viewer_tile_chunks"]),
    )
    require_array(
        root,
        str(clim_cfg["viewer_point_variable"]),
        (365, int(plan["height"]), int(plan["width"])),
        tuple(int(value) for value in clim_cfg["viewer_point_chunks"]),
    )
    root.attrs.update({"lfmc_climatology_finalized_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"Consolidating viewer zarr metadata for {viewer_dataset_path(cfg)}")
    zarr.consolidate_metadata(str(viewer_dataset_path(cfg)))
    write_json(
        state_dir(cfg) / "viewer_climatology_completed.json",
        {
            "status": "completed",
            "block_count": int(plan["block_count"]),
            "viewer_dataset_path": str(viewer_dataset_path(cfg)),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    log("Viewer climatology finalize complete")


def report_status(args) -> None:
    cfg = load_config(args.config)
    plan = read_json(plan_path(cfg))
    completed = sum(1 for block in plan["blocks"] if block_status_path(cfg, int(block["block_index"])).exists())
    payload = {
        "status": "running" if completed < int(plan["block_count"]) else "completed",
        "block_count": int(plan["block_count"]),
        "completed_count": int(completed),
        "remaining_count": int(plan["block_count"]) - int(completed),
        "viewer_dataset_path": str(viewer_dataset_path(cfg)),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(payload, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample source-grid LFMC climatology into the EPSG:3857 viewer zarr.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["init", "worker", "finalize", "status"], required=True)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--overwrite-arrays", action="store_true")
    parser.add_argument("--overwrite-status", action="store_true")
    parser.add_argument("--force", action="store_true")
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
