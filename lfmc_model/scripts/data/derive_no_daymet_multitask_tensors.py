#!/usr/bin/env python3

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import torch


SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc")
SOURCE_TAG = (
    "multisource_fusion_k3_dw64_dm128_ds32_dc64_sh32_lfp64_sarp32_wd64_lr5e-4"
    "_tw5_vpd_anoms_nozone_clim20"
)
DEST_TAG = (
    "multisource_fusion_no_daymet_k3_dw64_dm128_ds32_dc64_sh32_lfp64_sarp32"
    "_wd64_lr5e-4_tw5_vpd_anoms_nozone_clim20"
)
LINK_FILES = [
    "X_short.pt",
    "X_static.pt",
    "Y.pt",
    "source.pt",
    "stratifier.npy",
    "info.csv",
    "info.parquet",
]


def default_source_root() -> Path:
    return SCRATCH_ROOT / "lfmc_model" / "inputs" / "ensemble" / ("lfmc_vh_vv_365_" + SOURCE_TAG)


def default_dest_root() -> Path:
    return SCRATCH_ROOT / "lfmc_model" / "inputs" / "ensemble" / ("lfmc_vh_vv_365_" + DEST_TAG)


def default_source_fold_info() -> Path:
    return (
        SCRATCH_ROOT
        / "lfmc_model"
        / "outputs"
        / "shared_training"
        / ("canonical_fold_info_" + SOURCE_TAG + ".json")
    )


def default_dest_fold_info() -> Path:
    return (
        SCRATCH_ROOT
        / "lfmc_model"
        / "outputs"
        / "shared_training"
        / ("canonical_fold_info_" + DEST_TAG + ".json")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive multitask tensors with the Daymet/long encoder removed."
    )
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--dest-root", type=Path, default=default_dest_root())
    parser.add_argument("--source-fold-info", type=Path, default=default_source_fold_info())
    parser.add_argument("--dest-fold-info", type=Path, default=default_dest_fold_info())
    parser.add_argument("--start-member", type=int, default=1000)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(source_path: Path, dest_path: Path):
    if not source_path.exists():
        return False
    if dest_path.exists():
        dest_path.unlink()
    try:
        os.link(source_path, dest_path)
    except OSError:
        shutil.copy2(source_path, dest_path)
    return True


def write_json(path: Path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    os.replace(tmp_path, path)


def update_var_names(source_path: Path, dest_path: Path):
    with source_path.open("r") as file_obj:
        var_names = json.load(file_obj)
    out = copy.deepcopy(var_names)
    out["long_vars"] = []
    write_json(dest_path, out)


def update_build_config(source_path: Path, dest_path: Path, source_member_dir: Path):
    with source_path.open("r") as file_obj:
        build_config = json.load(file_obj)
    out = copy.deepcopy(build_config)
    out["long_features"] = []
    out["long_lag_days"] = []
    if "var_locs" in out and isinstance(out["var_locs"], dict):
        out["var_locs"]["daymet"] = []
    out["derived_no_daymet"] = {
        "source_member_dir": str(source_member_dir),
        "removed_tensor": "X_long.pt",
        "removed_long_vars": build_config.get("long_features", []),
        "note": "X_long is intentionally empty so the multisource fusion model omits the Daymet encoder.",
    }
    write_json(dest_path, out)


def derive_member(source_member_dir: Path, dest_member_dir: Path, overwrite: bool):
    if not source_member_dir.is_dir():
        raise FileNotFoundError(f"Missing source member tensor directory: {source_member_dir}")
    if dest_member_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists; pass --overwrite to replace: {dest_member_dir}")
        shutil.rmtree(dest_member_dir)
    dest_member_dir.mkdir(parents=True, exist_ok=True)

    x_short_path = source_member_dir / "X_short.pt"
    if not x_short_path.exists():
        raise FileNotFoundError(f"Missing X_short.pt in {source_member_dir}")
    x_short = torch.load(x_short_path, map_location="cpu", weights_only=False)
    n_rows = int(x_short.shape[0])
    empty_long = torch.empty((n_rows, 0, 0), dtype=x_short.dtype)
    torch.save(empty_long, dest_member_dir / "X_long.pt")

    for filename in LINK_FILES:
        link_or_copy(source_member_dir / filename, dest_member_dir / filename)

    update_var_names(source_member_dir / "var_names.json", dest_member_dir / "var_names.json")
    update_build_config(
        source_member_dir / "build_config.json",
        dest_member_dir / "build_config.json",
        source_member_dir,
    )
    print(f"derived {dest_member_dir} rows={n_rows:,} X_long={tuple(empty_long.shape)}", flush=True)


def copy_fold_info(source_fold_info: Path, dest_fold_info: Path, overwrite: bool):
    if not source_fold_info.exists():
        raise FileNotFoundError(f"Missing source fold info: {source_fold_info}")
    if dest_fold_info.exists() and not overwrite:
        print(f"fold info already exists: {dest_fold_info}", flush=True)
        return
    dest_fold_info.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_fold_info, dest_fold_info)
    print(f"copied fold info to {dest_fold_info}", flush=True)


def main():
    args = parse_args()
    args.dest_root.mkdir(parents=True, exist_ok=True)
    print(f"source_root={args.source_root}", flush=True)
    print(f"dest_root={args.dest_root}", flush=True)
    for member_id in range(args.start_member, args.start_member + args.ensemble_size):
        member_name = f"lfmc_vh_vv_ds{member_id:04d}"
        derive_member(
            source_member_dir=args.source_root / member_name,
            dest_member_dir=args.dest_root / member_name,
            overwrite=bool(args.overwrite),
        )
    copy_fold_info(
        source_fold_info=args.source_fold_info,
        dest_fold_info=args.dest_fold_info,
        overwrite=bool(args.overwrite),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
