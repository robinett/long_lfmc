#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
INFERENCE_SCRIPT_DIR = REPO_ROOT / "lfmc_model/scripts/inference"
if str(INFERENCE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_SCRIPT_DIR))

from low_latency_rollback import restore_rollback_dir, timestamped_message


DEFAULT_MANIFEST = REPO_ROOT / "logs/low_latency_forward_setup/low_latency_forward_setup_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast-reset the low-latency forward test sandbox by rolling zarrs back to "
            "their captured pre-run state while preserving reusable source caches."
        )
    )
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--rollback_dir",
        type=Path,
        default=None,
        help="Rollback directory to restore. Defaults to the newest directory under production metadata/rollback.",
    )
    parser.add_argument(
        "--drop_source_caches",
        action="store_true",
        help="Also remove raw/mosaic/regridded source caches. Default keeps them for faster reruns.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing low-latency forward setup manifest: {path}")
    return json.loads(path.read_text())


def latest_rollback_dir(metadata_dir: Path) -> Path:
    root = metadata_dir / "rollback"
    if not root.exists():
        raise FileNotFoundError(f"Rollback root does not exist: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No rollback directories found under {root}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def remove_path(path: Path, dry_run: bool = False) -> None:
    if not path.exists():
        return
    print(timestamped_message(f"Removing reset artifact: {path}"))
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean_metadata(metadata_dir: Path, dry_run: bool = False) -> None:
    remove_path(metadata_dir / "status_reports", dry_run=dry_run)
    remove_path(metadata_dir / "locks", dry_run=dry_run)
    for update_record in sorted(metadata_dir.glob("production_update_*.json")):
        remove_path(update_record, dry_run=dry_run)


def cleanup_test_artifacts(paths: dict, drop_source_caches: bool, dry_run: bool = False) -> None:
    for key in ["modis_staging_root", "append_coord_dir", "map_run_root"]:
        if key in paths:
            remove_path(Path(paths[key]), dry_run=dry_run)
    clean_metadata(Path(paths["production_metadata_dir"]), dry_run=dry_run)

    if not drop_source_caches:
        print(timestamped_message("Preserving source caches: MODIS raw/mosaic/regrid and low-latency climate files"))
        return

    cache_keys = [
        "modis_raw_root",
        "modis_mosaic_root",
        "modis_regrid_root",
        "low_latency_regrid_root",
    ]
    for key in cache_keys:
        if key in paths:
            remove_path(Path(paths[key]), dry_run=dry_run)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest_path)
    paths = manifest["paths"]
    metadata_dir = Path(paths["production_metadata_dir"])
    rollback_dir = args.rollback_dir or latest_rollback_dir(metadata_dir)

    print(timestamped_message(f"Using setup manifest: {args.manifest_path}"))
    print(timestamped_message(f"Using rollback directory: {rollback_dir}"))
    restore_rollback_dir(rollback_dir, dry_run=args.dry_run)
    cleanup_test_artifacts(paths, drop_source_caches=args.drop_source_caches, dry_run=args.dry_run)
    print(timestamped_message("Fast low-latency forward reset complete"))


if __name__ == "__main__":
    main()
