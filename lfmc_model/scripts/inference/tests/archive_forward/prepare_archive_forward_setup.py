#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
LOG_DIR = REPO_ROOT / "logs/archive_forward_setup_20260417_111322"
SCRATCH_DIR = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/tests/archive_forward_setup_20260417_111322"
)
SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc")
OAK_ROOT = Path("/oak/stanford/groups/konings/trobinet/long_lfmc/final_lfmc")

CANONICAL_SOURCE_REGISTRY = REPO_ROOT / "lfmc_model/scripts/inference/source_registry.yaml"
CANONICAL_MAP_CONFIG = REPO_ROOT / "lfmc_model/scripts/inference/map_configs_final_update.yaml"
CANONICAL_MULTIYEAR_MAP_CONFIG = REPO_ROOT / "lfmc_model/scripts/inference/map_configs_multisource_fusion_clim20_multiyear.yaml"

SRC_SCIENTIFIC = SCRATCH_ROOT / "lfmc_model/inference/final_products/lfmc_vh_vv_365_multisource_fusion_clim20_2001_2024.zarr"
SRC_NLCD_ANNUAL = SCRATCH_ROOT / "nlcd/nlcd_target_grid_2000_2024.zarr"
SRC_DAYMET_CLIM20 = SCRATCH_ROOT / "daymet/daymet_vars_and_anoms_clim20.zarr"
SRC_DAYMET_CLIM = SCRATCH_ROOT / "daymet/daymet_vars_and_anoms_clim20_climatology.zarr"

TEST_YEAR = 2024
PREP_END_DATE = "2023-12-31"

SCRATCH_PRODUCTION_ZARR = SCRATCH_DIR / "production/scientific_2001_2024_scrambled_test.zarr"
SCRATCH_PRODUCTION_METADATA = SCRATCH_DIR / "production/metadata"
SCRATCH_NLCD_RAW_DIR = SCRATCH_DIR / "nlcd/raw_tifs"
SCRATCH_NLCD_RAW_ZARR = SCRATCH_DIR / "nlcd/nlcd_2024_update_raw_test.zarr"
SCRATCH_NLCD_ANNUAL_ZARR = SCRATCH_DIR / "nlcd/nlcd_target_grid_2000_2023_test.zarr"
SCRATCH_DAYMET_RAW_2024_ZARR = SCRATCH_DIR / "daymet/daymet_all_vars_2024_test.zarr"
SCRATCH_DAYMET_CLIM20_ZARR = SCRATCH_DIR / "daymet/daymet_vars_and_anoms_clim20_2000_2023_test.zarr"
SCRATCH_DAYMET_CLIM_ZARR = SCRATCH_DIR / "daymet/daymet_vars_and_anoms_clim20_climatology_test.zarr"
SCRATCH_DAYMET_EARTHACCESS_ROOT = SCRATCH_DIR / "daymet/daymet_earthaccess"
SCRATCH_DAYMET_DAILY_ROOT = SCRATCH_DIR / "daymet/daymet_daily"
SCRATCH_DAYMET_REGRID_ROOT = SCRATCH_DIR / "daymet/daymet_regrid"
SCRATCH_MAP_RUN_ROOT = SCRATCH_DIR / "map_runs/final_2024"

TEST_SOURCE_REGISTRY = LOG_DIR / "source_registry_test.yaml"
TEST_MAP_CONFIG = LOG_DIR / "map_configs_final_test.yaml"
TEST_MANIFEST = LOG_DIR / "archive_forward_setup_manifest.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def scratch_to_oak(scratch_path: Path) -> Path:
    return OAK_ROOT / scratch_path.relative_to(SCRATCH_ROOT)


def rsync_path(src_path: Path, dst_path: Path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if src_path.is_dir():
        dst_path.mkdir(parents=True, exist_ok=True)
        run_cmd(["rsync", "-a", f"{src_path}/", f"{dst_path}/"])
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["rsync", "-a", src_path, dst_path])


def run_cmd(cmd):
    print("Running command:")
    print("  " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def require_oak_copy(scratch_path: Path, label: str) -> Path:
    oak_path = scratch_to_oak(scratch_path)
    if oak_path.exists():
        print(f"Using OAK {label}: {oak_path}")
        return oak_path
    raise FileNotFoundError(f"Missing required OAK baseline for {label}: {oak_path}")


def open_zarr_nonconsolidated(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def strip_zarr_encoding(ds: xr.Dataset) -> xr.Dataset:
    cleaned = ds.copy(deep=False)
    cleaned.encoding = {}
    for name in cleaned.variables:
        cleaned[name].encoding = {}
    for name in cleaned.coords:
        cleaned[name].encoding = {}
    return cleaned


def subset_zarr_time(src_path: Path, dst_path: Path, selector_dim: str, end_value, extra_indexers=None) -> None:
    print(f"Opening source zarr: {src_path}")
    ds = open_zarr_nonconsolidated(src_path)
    indexers = {selector_dim: slice(None, end_value)}
    if extra_indexers is not None:
        indexers.update(extra_indexers)
    subset = strip_zarr_encoding(ds.sel(**indexers))
    print(f"Subset sizes for {dst_path.name}: {dict(subset.sizes)}")
    ensure_parent(dst_path)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    print(f"Writing subset zarr in explicit format v2: {dst_path}")
    subset.to_zarr(
        dst_path,
        mode="w",
        consolidated=False,
        safe_chunks=False,
        zarr_format=2,
    )
    zarr.consolidate_metadata(str(dst_path))
    ds.close()
    subset.close()
    print(f"Wrote subset zarr: {dst_path}")


def copy_tree(src_path: Path, dst_path: Path, label: str, reset_existing: bool = True) -> None:
    ensure_parent(dst_path)
    if reset_existing and dst_path.exists():
        shutil.rmtree(dst_path)
    print(f"Copying {label}: {src_path} -> {dst_path}")
    rsync_path(src_path, dst_path)


def read_scientific_time_and_quality(zarr_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    root = zarr.open_group(str(zarr_path), mode="r")
    time_arr = root["time"]
    time_vals = np.asarray(time_arr[:])
    time_units = str(time_arr.attrs.get("units", ""))
    if np.issubdtype(time_vals.dtype, np.datetime64):
        times = pd.to_datetime(time_vals)
    elif time_units.startswith("nanoseconds since 1970-01-01"):
        times = pd.to_datetime(np.asarray(time_vals, dtype=np.int64))
    else:
        ds = open_zarr_nonconsolidated(zarr_path)
        times = pd.to_datetime(ds["time"].values)
        ds.close()
    quality = np.asarray(root["quality_flag"][:], dtype=np.uint8)
    return times, quality


def scientific_year_is_scrambled(zarr_path: Path, year: int) -> bool:
    if not zarr_path.exists():
        return False
    try:
        times, quality_vals = read_scientific_time_and_quality(zarr_path)
        year_indices = np.where(times.year == int(year))[0]
        if len(year_indices) == 0:
            return False
        return bool(np.all(quality_vals[year_indices] == 1))
    except Exception as exc:
        print(f"Scientific zarr state check failed for {zarr_path}: {exc}")
        return False


def subset_matches_end_value(dst_path: Path, selector_dim: str, expected_end_value) -> bool:
    if not dst_path.exists():
        return False
    try:
        ds = open_zarr_nonconsolidated(dst_path)
        coord = ds[selector_dim]
        if int(coord.sizes.get(selector_dim, 0)) == 0:
            ds.close()
            return False
        last_value = coord.values[-1]
        ds.close()
        return pd.Timestamp(last_value) == pd.Timestamp(expected_end_value)
    except Exception as exc:
        print(f"Subset state check failed for {dst_path}: {exc}")
        return False


def copy_is_usable(dst_path: Path) -> bool:
    if not dst_path.exists():
        return False
    try:
        ds = open_zarr_nonconsolidated(dst_path)
        ds.close()
        return True
    except Exception as exc:
        print(f"Copy usability check failed for {dst_path}: {exc}")
        return False


def scramble_scientific_year(zarr_path: Path, year: int) -> None:
    print(f"Scrambling scientific zarr year {year} with low-latency quality flags: {zarr_path}")
    times, _ = read_scientific_time_and_quality(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    year_indices = np.where(times.year == int(year))[0]
    if len(year_indices) == 0:
        raise ValueError(f"No time slices found for year {year} in {zarr_path}")

    rng = np.random.default_rng(year)
    mean_arr = root["lfmc_ens_mean"]
    std_arr = root["lfmc_ens_std"]
    quality_arr = root["quality_flag"]
    y_size = int(mean_arr.shape[1])
    x_size = int(mean_arr.shape[2])
    for idx in year_indices:
        mean_arr[idx, :, :] = (20.0 + 260.0 * rng.random((y_size, x_size), dtype=np.float32)).astype(np.float32)
        std_arr[idx, :, :] = (1.0 + 40.0 * rng.random((y_size, x_size), dtype=np.float32)).astype(np.float32)
        quality_arr[idx] = np.uint8(1)
    zarr.consolidate_metadata(str(zarr_path))
    print(f"Scrambled {len(year_indices)} daily slices for {year}")


def write_yaml(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote YAML: {path}")


def write_manifest(payload: dict) -> None:
    ensure_parent(TEST_MANIFEST)
    TEST_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote manifest: {TEST_MANIFEST}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    oak_nlcd_annual = require_oak_copy(SRC_NLCD_ANNUAL, "NLCD annual target zarr")
    oak_daymet_clim20 = require_oak_copy(SRC_DAYMET_CLIM20, "Daymet clim20 zarr")
    oak_daymet_clim = require_oak_copy(SRC_DAYMET_CLIM, "Daymet clim20 climatology zarr")

    if scientific_year_is_scrambled(SCRATCH_PRODUCTION_ZARR, TEST_YEAR):
        print(f"Scratch scientific zarr already has scrambled low-latency {TEST_YEAR}; skipping restore/scramble")
    else:
        print("Restoring scratch scientific zarr through 2024 from canonical scratch baseline")
        copy_tree(SRC_SCIENTIFIC, SCRATCH_PRODUCTION_ZARR, "scientific zarr", reset_existing=False)
        scramble_scientific_year(SCRATCH_PRODUCTION_ZARR, TEST_YEAR)

    if subset_matches_end_value(SCRATCH_NLCD_ANNUAL_ZARR, "year", "2023-12-31"):
        print("Scratch NLCD annual zarr already stops at 2023; skipping rebuild")
    else:
        print("Preparing scratch NLCD annual zarr through 2023")
        subset_zarr_time(
            oak_nlcd_annual,
            SCRATCH_NLCD_ANNUAL_ZARR,
            selector_dim="year",
            end_value=np.datetime64("2023-12-31"),
        )

    if subset_matches_end_value(SCRATCH_DAYMET_CLIM20_ZARR, "time", PREP_END_DATE):
        print("Scratch Daymet clim20 zarr already stops at 2023-12-31; skipping rebuild")
    else:
        print("Preparing scratch Daymet clim20 zarr through 2023")
        subset_zarr_time(
            oak_daymet_clim20,
            SCRATCH_DAYMET_CLIM20_ZARR,
            selector_dim="time",
            end_value=PREP_END_DATE,
        )

    if copy_is_usable(SCRATCH_DAYMET_CLIM_ZARR):
        print("Scratch Daymet climatology zarr is already present and readable; skipping restore")
    else:
        print("Restoring scratch Daymet climatology from OAK baseline")
        copy_tree(oak_daymet_clim, SCRATCH_DAYMET_CLIM_ZARR, "Daymet clim20 climatology zarr", reset_existing=False)

    registry_cfg = yaml.safe_load(CANONICAL_SOURCE_REGISTRY.read_text())
    registry_cfg["storage"]["oak_root"] = str(OAK_ROOT)
    registry_cfg["sources"]["daymet"]["combined_path"] = str(SCRATCH_DAYMET_CLIM20_ZARR)
    registry_cfg["sources"]["daymet"]["climatology_path"] = str(SCRATCH_DAYMET_CLIM_ZARR)
    registry_cfg["sources"]["daymet"]["archive_path"] = str(SCRATCH_DAYMET_RAW_2024_ZARR)
    registry_cfg["sources"]["nlcd"]["raw_path"] = str(SCRATCH_NLCD_RAW_ZARR)
    registry_cfg["sources"]["nlcd"]["annual_path"] = str(SCRATCH_NLCD_ANNUAL_ZARR)
    registry_cfg["processing"]["daymet"]["earthaccess_root"] = str(SCRATCH_DAYMET_EARTHACCESS_ROOT)
    registry_cfg["processing"]["daymet"]["daily_root"] = str(SCRATCH_DAYMET_DAILY_ROOT)
    registry_cfg["processing"]["daymet"]["regrid_root"] = str(SCRATCH_DAYMET_REGRID_ROOT)
    registry_cfg["processing"]["nlcd"]["raw_dir"] = str(SCRATCH_NLCD_RAW_DIR)
    registry_cfg["sources"]["scientific_current"]["zarr_path"] = str(SCRATCH_PRODUCTION_ZARR)
    registry_cfg["sources"]["production"]["zarr_path"] = str(SCRATCH_PRODUCTION_ZARR)
    registry_cfg["sources"]["production"]["metadata_dir"] = str(SCRATCH_PRODUCTION_METADATA)
    write_yaml(TEST_SOURCE_REGISTRY, registry_cfg)

    map_cfg = yaml.safe_load(CANONICAL_MAP_CONFIG.read_text())
    multiyear_cfg = yaml.safe_load(CANONICAL_MULTIYEAR_MAP_CONFIG.read_text())
    multiyear_submission = multiyear_cfg.get("submission", {})
    map_submission = map_cfg.setdefault("submission", {})
    for key in ["owners_partition", "owners_gpu_time_limit", "owners_gpu_cpus_per_task", "owners_gpu_mem"]:
        if key in multiyear_submission:
            map_submission[key] = multiyear_submission[key]
    map_submission["owners_gpu_max_jobs"] = 100
    map_submission["owners_gpu_constraint"] = ""
    map_cfg["sources"]["registry_path"] = str(TEST_SOURCE_REGISTRY)
    map_cfg["data"]["requested_start_date"] = f"{TEST_YEAR}-01-01"
    map_cfg["data"]["requested_end_date"] = f"{TEST_YEAR}-12-31"
    map_cfg["paths"]["run_root"] = str(SCRATCH_MAP_RUN_ROOT)
    write_yaml(TEST_MAP_CONFIG, map_cfg)

    manifest = {
        "test_year": TEST_YEAR,
        "prep_end_date": PREP_END_DATE,
        "scratch_dir": str(SCRATCH_DIR),
        "scrambled_year": TEST_YEAR,
        "scrambled_quality_flag_value": 1,
        "paths": {
            "production_zarr": str(SCRATCH_PRODUCTION_ZARR),
            "production_metadata_dir": str(SCRATCH_PRODUCTION_METADATA),
            "nlcd_raw_dir": str(SCRATCH_NLCD_RAW_DIR),
            "nlcd_raw_zarr": str(SCRATCH_NLCD_RAW_ZARR),
            "nlcd_annual_zarr": str(SCRATCH_NLCD_ANNUAL_ZARR),
            "daymet_raw_zarr": str(SCRATCH_DAYMET_RAW_2024_ZARR),
            "daymet_clim20_zarr": str(SCRATCH_DAYMET_CLIM20_ZARR),
            "daymet_climatology_zarr": str(SCRATCH_DAYMET_CLIM_ZARR),
            "daymet_earthaccess_root": str(SCRATCH_DAYMET_EARTHACCESS_ROOT),
            "daymet_daily_root": str(SCRATCH_DAYMET_DAILY_ROOT),
            "daymet_regrid_root": str(SCRATCH_DAYMET_REGRID_ROOT),
            "map_run_root": str(SCRATCH_MAP_RUN_ROOT),
            "source_registry_test": str(TEST_SOURCE_REGISTRY),
            "map_config_test": str(TEST_MAP_CONFIG),
        },
        "oak_paths": {
            "nlcd_annual": str(oak_nlcd_annual),
            "daymet_clim20": str(oak_daymet_clim20),
            "daymet_climatology": str(oak_daymet_clim),
        },
        "source_paths": {
            "scientific": str(SRC_SCIENTIFIC),
        },
    }
    write_manifest(manifest)
    print("Scratch archive-forward setup staging complete")


if __name__ == "__main__":
    main()
