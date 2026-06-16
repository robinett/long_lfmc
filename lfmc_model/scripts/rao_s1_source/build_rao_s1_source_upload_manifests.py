#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def date_index(root, target_date: str) -> int:
    dates = np.asarray(root["time"][:]).astype("datetime64[D]")
    matches = np.where(dates == np.datetime64(target_date))[0]
    if matches.size != 1:
        raise ValueError(f"Expected exactly one {target_date} time coordinate, found {matches.size}")
    return int(matches[0])


def existing_relpaths(root_path: Path, relpaths: Iterable[str]) -> List[str]:
    found = []
    for relpath in relpaths:
        if (root_path / relpath).is_file():
            found.append(relpath)
    return found


def zarr_metadata_relpaths(root_path: Path, array_names: Sequence[str]) -> List[str]:
    relpaths = []
    top_level_candidates = [
        ".zgroup",
        ".zattrs",
        ".zmetadata",
        "zarr.json",
        "zarr.json.bak",
    ]
    relpaths.extend(existing_relpaths(root_path, top_level_candidates))
    for array_name in array_names:
        relpaths.extend(
            existing_relpaths(
                root_path,
                [
                    f"{array_name}/.zarray",
                    f"{array_name}/.zattrs",
                    f"{array_name}/zarr.json",
                ],
            )
        )
    return relpaths


def zarr_array_chunk_relpaths(root_path: Path, array, array_name: str, time_chunk_index: int) -> List[str]:
    if len(array.shape) == 1:
        relpath = f"{array_name}/{time_chunk_index}"
        if (root_path / relpath).is_file():
            return [relpath]
        return []
    if len(array.shape) != 3:
        raise ValueError(f"Expected {array_name!r} to be 1D or 3D, got shape {array.shape}")
    array_path = root_path / array_name
    prefix = f"{time_chunk_index}."
    return [
        path.relative_to(root_path).as_posix()
        for path in sorted(array_path.glob(f"{prefix}*"))
        if path.is_file()
    ]


def manifest_stats(root_path: Path, relpaths: Sequence[str]) -> Tuple[int, int]:
    total_bytes = 0
    for relpath in relpaths:
        path = root_path / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Manifest entry does not exist: {path}")
        total_bytes += path.stat().st_size
    return len(relpaths), total_bytes


def write_manifest(path: Path, relpaths: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(dict.fromkeys(relpaths))
    path.write_text("\n".join(ordered) + "\n", encoding="utf-8")


def build_zarr_manifest(
    label: str,
    zarr_path: Path,
    target_date: str,
    variable_name: str,
    output_path: Path,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r")
    target_index = date_index(root, target_date)
    variable = root[variable_name]
    quality = root["quality_flag"]
    time_coord = root["time"]
    time_chunk_index = target_index // int(variable.chunks[0])
    relpaths = []
    relpaths.extend(zarr_array_chunk_relpaths(zarr_path, variable, variable_name, time_chunk_index))
    relpaths.extend(zarr_array_chunk_relpaths(zarr_path, quality, "quality_flag", target_index // int(quality.chunks[0])))
    relpaths.extend(zarr_array_chunk_relpaths(zarr_path, time_coord, "time", target_index // int(time_coord.chunks[0])))
    relpaths.extend(zarr_metadata_relpaths(zarr_path, [variable_name, "quality_flag", "time"]))
    write_manifest(output_path, relpaths)
    count, total_bytes = manifest_stats(zarr_path, output_path.read_text(encoding="utf-8").splitlines())
    log(
        f"{label} manifest: target_index={target_index} time_chunk={time_chunk_index} "
        f"files={count} bytes={total_bytes} path={output_path}"
    )


def iter_files_under(root_path: Path, rel_root: str) -> Iterable[str]:
    base = root_path / rel_root
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if path.is_file():
            yield path.relative_to(root_path).as_posix()


def build_assets_manifest(asset_root: Path, target_date: str, output_path: Path) -> None:
    relpaths = []
    relpaths.extend(existing_relpaths(asset_root, ["manifest.json"]))
    for path in asset_root.glob("*.json"):
        relpath = path.relative_to(asset_root).as_posix()
        if relpath not in relpaths:
            relpaths.append(relpath)
    for layer_name in ("lfmc", "anomaly"):
        relpaths.extend(iter_files_under(asset_root, f"tiles/{layer_name}/{target_date}"))
    if not any(relpath.startswith(f"tiles/lfmc/{target_date}/") for relpath in relpaths):
        raise FileNotFoundError(f"No LFMC tiles found for {target_date} under {asset_root}")
    if not any(relpath.startswith(f"tiles/anomaly/{target_date}/") for relpath in relpaths):
        raise FileNotFoundError(f"No anomaly tiles found for {target_date} under {asset_root}")
    write_manifest(output_path, relpaths)
    count, total_bytes = manifest_stats(asset_root, output_path.read_text(encoding="utf-8").splitlines())
    log(f"assets manifest: files={count} bytes={total_bytes} path={output_path}")


def write_summary(path: Path, manifest_paths: Dict[str, Path]) -> None:
    payload = {name: str(manifest_path) for name, manifest_path in manifest_paths.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build targeted upload manifests for Rao S1 Source artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    variable_name = str(cfg["dataset"]["variable_name"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        "scientific": output_dir / "rao_s1_scientific_lfmc_maps.txt",
        "viewer": output_dir / "rao_s1_viewer_3857_lfmc_maps.txt",
        "assets": output_dir / "rao_s1_viewer_3857_assets.txt",
    }
    build_zarr_manifest(
        "scientific",
        Path(str(cfg["paths"]["scientific_zarr_path"])),
        args.target_date,
        variable_name,
        manifest_paths["scientific"],
    )
    build_zarr_manifest(
        "viewer",
        Path(str(cfg["paths"]["viewer_zarr_path"])),
        args.target_date,
        variable_name,
        manifest_paths["viewer"],
    )
    build_assets_manifest(
        Path(str(cfg["paths"]["asset_root"])),
        args.target_date,
        manifest_paths["assets"],
    )
    write_summary(output_dir / "manifest_paths.json", manifest_paths)
    log(f"Wrote manifest summary: {output_dir / 'manifest_paths.json'}")


if __name__ == "__main__":
    main()
