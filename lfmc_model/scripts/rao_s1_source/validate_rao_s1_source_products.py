#!/usr/bin/env python3

import argparse
import json
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


def date_strings(root) -> List[str]:
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def require_array(root, name: str, expected_shape=None, expected_chunks=None) -> None:
    if name not in root:
        raise ValueError(f"Missing required array {name!r}")
    arr = root[name]
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Array {name!r} shape {arr.shape} != expected {expected_shape}")
    if expected_chunks is not None and tuple(arr.chunks) != tuple(expected_chunks):
        raise ValueError(f"Array {name!r} chunks {arr.chunks} != expected {expected_chunks}")


def validate_anomaly_layer(cfg: Dict[str, object]) -> None:
    layers = cfg.get("layers", {})
    if "lfmc" not in layers:
        raise ValueError("Config is missing required lfmc layer")
    anomaly = dict(layers.get("anomaly", {}))
    if not anomaly:
        raise ValueError("Config is missing required anomaly layer")
    if str(anomaly.get("derived", "")).strip() != "lfmc_anomaly":
        raise ValueError("Anomaly layer must be configured as derived=lfmc_anomaly")
    value_min = float(anomaly["min"])
    value_max = float(anomaly["max"])
    if value_min >= 0 or value_max <= 0 or abs(value_min) != abs(value_max):
        raise ValueError(f"Anomaly layer min/max must be centered on zero; got {value_min}, {value_max}")
    if str(anomaly.get("source_variable", "")) != str(cfg["dataset"]["variable_name"]):
        raise ValueError("Anomaly layer source_variable must match dataset.variable_name")
    if str(anomaly.get("climatology_variable", "")) != str(cfg["climatology"]["tile_variable"]):
        raise ValueError("Anomaly layer climatology_variable must match climatology.tile_variable")


def validate_scientific_zarr(cfg: Dict[str, object], target_date: str) -> None:
    path = Path(str(cfg["paths"]["scientific_zarr_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Scientific zarr does not exist: {path}")
    root = zarr.open_group(str(path), mode="r")
    variable_name = str(cfg["dataset"]["variable_name"])
    require_array(root, variable_name)
    require_array(root, "quality_flag")
    dates = date_strings(root)
    if target_date not in dates:
        raise ValueError(f"Target date {target_date} is missing from scientific zarr {path}")
    time_count = len(dates)
    if root[variable_name].shape[0] != time_count:
        raise ValueError(f"Scientific {variable_name} time length does not match time coord")
    expected_time_chunk = int(cfg["chunks"]["scientific_time"])
    spatial_chunk = int(cfg["chunks"]["spatial"])
    expected_chunks = (expected_time_chunk, spatial_chunk, spatial_chunk)
    if tuple(root[variable_name].chunks) != expected_chunks:
        raise ValueError(f"Scientific {variable_name} chunks {root[variable_name].chunks} != {expected_chunks}")
    log(f"Scientific zarr validated: {path} target={target_date} time_count={time_count}")


def validate_viewer_zarr(cfg: Dict[str, object], target_date: str) -> None:
    path = Path(str(cfg["paths"]["viewer_zarr_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Viewer zarr does not exist: {path}")
    root = zarr.open_group(str(path), mode="r")
    variable_name = str(cfg["dataset"]["variable_name"])
    require_array(root, variable_name)
    require_array(root, "quality_flag")
    dates = date_strings(root)
    if target_date not in dates:
        raise ValueError(f"Target date {target_date} is missing from viewer zarr {path}")
    time_count = len(dates)
    height = int(root["y"].shape[0])
    width = int(root["x"].shape[0])
    expected_lfmc_chunks = (int(cfg["chunks"]["viewer_time"]), int(cfg["chunks"]["spatial"]), int(cfg["chunks"]["spatial"]))
    if tuple(root[variable_name].chunks) != expected_lfmc_chunks:
        raise ValueError(f"Viewer {variable_name} chunks {root[variable_name].chunks} != {expected_lfmc_chunks}")
    if root[variable_name].shape != (time_count, height, width):
        raise ValueError(f"Viewer {variable_name} shape {root[variable_name].shape} does not match time/y/x")

    clim_cfg = cfg["climatology"]
    tile_variable = str(clim_cfg["tile_variable"])
    point_variable = str(clim_cfg["point_variable"])
    expected_shape = (365, height, width)
    require_array(root, tile_variable, expected_shape=expected_shape, expected_chunks=tuple(clim_cfg["tile_chunks"]))
    require_array(root, point_variable, expected_shape=expected_shape, expected_chunks=tuple(clim_cfg["point_chunks"]))
    for attr_name, expected_value in (
        ("lfmc_climatology_baseline_start_date", str(clim_cfg["baseline_start_date"])),
        ("lfmc_climatology_baseline_end_date", str(clim_cfg["baseline_end_date"])),
    ):
        actual_value = root.attrs.get(attr_name)
        if actual_value != expected_value:
            raise ValueError(f"Viewer zarr attr {attr_name}={actual_value!r} != {expected_value!r}")
    if not root.attrs.get("lfmc_climatology_finalized_at"):
        raise ValueError("Viewer zarr is missing lfmc_climatology_finalized_at")
    log(f"Viewer zarr validated: {path} target={target_date} time_count={time_count}")


def validate_assets(cfg: Dict[str, object], target_date: str) -> None:
    asset_root = Path(str(cfg["paths"]["asset_root"]))
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Viewer asset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dates = list(manifest.get("dates", []))
    if target_date not in dates:
        raise ValueError(f"Target date {target_date} is missing from viewer asset manifest")
    layers = manifest.get("layers", {})
    for layer_key in ("lfmc", "anomaly"):
        if layer_key not in layers:
            raise ValueError(f"Viewer asset manifest is missing layer {layer_key!r}")
        template = str(layers[layer_key].get("tile_root_template", ""))
        if not template:
            raise ValueError(f"Viewer asset manifest layer {layer_key!r} is missing tile_root_template")
        sample_relpath = template.format(date=target_date, z=int(cfg["viewer"]["max_zoom"]), x=0, y=0)
        sample_path = asset_root / sample_relpath
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample {layer_key} tile for {target_date}: {sample_path}")
    if dates[-1] != target_date:
        log(f"Target date {target_date} is present in manifest; latest manifest date is {dates[-1]}")
    log(f"Viewer assets validated: {asset_root} target={target_date}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate local Rao S1 Source products before upload.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--check-assets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_anomaly_layer(cfg)
    validate_scientific_zarr(cfg, args.target_date)
    validate_viewer_zarr(cfg, args.target_date)
    if args.check_assets:
        validate_assets(cfg, args.target_date)
    log("Rao S1 local product validation complete")


if __name__ == "__main__":
    main()
