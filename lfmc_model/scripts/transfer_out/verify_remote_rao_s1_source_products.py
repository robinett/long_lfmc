#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "source_coop_transfer_configs.yaml"
SOURCE_HTTP_ROOT = "https://data.source.coop"
DEFAULT_RAO_S1_PRODUCT_PREFIX = "rseg/sentinel1-lfmc/"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def strip_slashes(value: str) -> str:
    return str(value).strip().strip("/")


def source_url(product_prefix: str, relpath: str) -> str:
    return f"{SOURCE_HTTP_ROOT}/{strip_slashes(product_prefix)}/{strip_slashes(relpath)}"


def load_remote_json(url: str):
    request = Request(url, headers={"User-Agent": "rao-s1-source-transfer/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def expected_dates_from_local_manifest(cfg) -> list:
    assets_cfg = cfg["datasets"]["rao_s1_viewer_3857_assets"]
    manifest_path = Path(str(assets_cfg["source_path"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dates = manifest.get("dates", [])
    if not dates:
        raise ValueError(f"Local Rao S1 viewer manifest has no dates: {manifest_path}")
    return list(dates)


def check_head(url: str, label: str) -> None:
    request = Request(url, method="HEAD", headers={"User-Agent": "rao-s1-source-transfer/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status >= 400:
            raise ValueError(f"{label} HEAD failed with status {response.status}")


def check_zarr(label: str, store_url: str, expected_chunks, expected_time_count: int) -> None:
    metadata_url = f"{store_url}/.zmetadata"
    log(f"Opening remote {label} zarr metadata {metadata_url}")
    metadata = load_remote_json(metadata_url).get("metadata", {})
    if ".zgroup" not in metadata:
        raise ValueError(f"{label} is missing .zgroup metadata")
    if "lfmc/.zarray" not in metadata:
        raise ValueError(f"{label} is missing lfmc array metadata")
    if "time/.zarray" not in metadata:
        raise ValueError(f"{label} is missing time array metadata")

    lfmc_meta = metadata["lfmc/.zarray"]
    time_meta = metadata["time/.zarray"]
    shape = tuple(lfmc_meta.get("shape", ()))
    chunks = tuple(lfmc_meta.get("chunks", ()))
    time_shape = tuple(time_meta.get("shape", ()))
    if not shape or shape[0] != expected_time_count:
        raise ValueError(f"{label} lfmc time length check failed: shape={shape}")
    if time_shape != (expected_time_count,):
        raise ValueError(f"{label} time coordinate shape check failed: shape={time_shape}")
    if chunks != tuple(expected_chunks):
        raise ValueError(f"{label} chunks {chunks} did not match {expected_chunks}")

    last_time_chunk = f"{store_url}/time/{max(0, (expected_time_count - 1) // time_meta.get('chunks', [expected_time_count])[0])}"
    check_head(last_time_chunk, f"{label} final time chunk")
    log(f"Verified {label}: shape={shape} chunks={chunks}")


def check_assets(
    product_prefix: str,
    destination_relpath: str,
    expected_last_date: str,
    expected_time_count: int,
) -> None:
    manifest_relpath = f"{strip_slashes(destination_relpath)}/manifest.json"
    manifest_url = source_url(product_prefix, manifest_relpath)
    log(f"Opening remote viewer manifest {manifest_url}")
    manifest = load_remote_json(manifest_url)
    dates = manifest.get("dates", [])
    if len(dates) != expected_time_count or dates[-1] != expected_last_date:
        raise ValueError(f"Manifest date check failed: count={len(dates)} last={dates[-1] if dates else None}")
    layers = manifest.get("layers", {})
    if sorted(layers) != ["lfmc"]:
        raise ValueError(f"Expected only lfmc layer, found {sorted(layers)}")
    template = str(layers["lfmc"].get("tile_root_template", ""))
    sample_tile_relpath = (
        f"{strip_slashes(destination_relpath)}/"
        + template.format(date=expected_last_date, z=3, x=0, y=0)
    )
    sample_tile_url = source_url(product_prefix, sample_tile_relpath)
    check_head(sample_tile_url, "sample tile")
    log(f"Verified assets manifest and sample tile {sample_tile_url}")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify remote Rao S1 Source products.")
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product_prefix", default=None)
    parser.add_argument("--expected_last_date", default=None)
    parser.add_argument("--expected_time_count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path)
    source_cfg = cfg["source_coop"]
    datasets = cfg["datasets"]
    product_prefix = args.product_prefix or DEFAULT_RAO_S1_PRODUCT_PREFIX or source_cfg["product_prefix"]
    expected_dates = expected_dates_from_local_manifest(cfg)
    expected_last_date = args.expected_last_date or expected_dates[-1]
    expected_time_count = args.expected_time_count or len(expected_dates)

    check_zarr(
        "scientific",
        source_url(product_prefix, datasets["rao_s1_scientific_lfmc_maps"]["destination_relpath"]),
        expected_chunks=(128, 256, 256),
        expected_time_count=expected_time_count,
    )
    check_zarr(
        "viewer",
        source_url(product_prefix, datasets["rao_s1_viewer_3857_lfmc_maps"]["destination_relpath"]),
        expected_chunks=(32, 256, 256),
        expected_time_count=expected_time_count,
    )
    check_assets(
        product_prefix,
        datasets["rao_s1_viewer_3857_assets"]["destination_relpath"],
        expected_last_date,
        expected_time_count,
    )
    log("Remote Rao S1 Source products verified")


if __name__ == "__main__":
    main()
