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
LOG_DIR = REPO_ROOT / "logs/low_latency_forward_setup"
SCRATCH_DIR = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/tests/low_latency_forward_setup"
)
SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc")
OAK_ROOT = Path("/oak/stanford/groups/konings/trobinet/long_lfmc/final_lfmc")

CANONICAL_SOURCE_REGISTRY = REPO_ROOT / "lfmc_model/scripts/inference/source_registry.yaml"
CANONICAL_MAP_CONFIG = REPO_ROOT / "lfmc_model/scripts/inference/map_configs_low_latency_update.yaml"
CANONICAL_MULTIYEAR_MAP_CONFIG = (
    REPO_ROOT / "lfmc_model/scripts/inference/map_configs_multisource_fusion_clim20_multiyear.yaml"
)

SRC_LFMC_TARGET = SCRATCH_ROOT / "lfmc_model/inference/final_products/lfmc_vh_vv_365_multisource_fusion_clim20_2001_2024.zarr"
SRC_MODIS_CANONICAL = SCRATCH_ROOT / "modis/modis_regrid_interpolated/modis_interp_5d.zarr"
SRC_NLCD_ANNUAL = SCRATCH_ROOT / "nlcd/nlcd_target_grid_2000_2024.zarr"
SRC_DAYMET_CLIM20 = SCRATCH_ROOT / "daymet/daymet_vars_and_anoms_clim20.zarr"
SRC_DAYMET_CLIM = SCRATCH_ROOT / "daymet/daymet_vars_and_anoms_clim20_climatology.zarr"

LFMC_BASELINE_START_DATE = os.environ.get("LFMC_BASELINE_START_DATE", "2023-01-01")
LFMC_BASELINE_END_DATE = os.environ.get(
    "LFMC_BASELINE_END_DATE",
    os.environ.get("PREP_END_DATE", "2023-12-31"),
)
SOURCE_BASELINE_START_DATE = os.environ.get("SOURCE_BASELINE_START_DATE", "2022-01-01")
SOURCE_BASELINE_END_DATE = os.environ.get("SOURCE_BASELINE_END_DATE", "2022-12-31")
NLCD_BASELINE_START_DATE = os.environ.get("NLCD_BASELINE_START_DATE", LFMC_BASELINE_START_DATE)
NLCD_BASELINE_END_DATE = os.environ.get("NLCD_BASELINE_END_DATE", LFMC_BASELINE_END_DATE)
TEST_START_DATE = os.environ.get("TEST_START_DATE", "2024-01-01")
TEST_END_DATE = os.environ.get("TEST_END_DATE", "2024-12-31")
SOURCE_PREWARM_START_DATE = os.environ.get("SOURCE_PREWARM_START_DATE", "2023-01-01")
SOURCE_PREWARM_END_DATE = os.environ.get("SOURCE_PREWARM_END_DATE", "2023-12-31")
TODAY_OVERRIDE = os.environ.get("TODAY_OVERRIDE", "2025-01-07")
LFMC_BASELINE_START_YEAR = pd.Timestamp(LFMC_BASELINE_START_DATE).year
LFMC_BASELINE_END_YEAR = pd.Timestamp(LFMC_BASELINE_END_DATE).year
SOURCE_BASELINE_START_YEAR = pd.Timestamp(SOURCE_BASELINE_START_DATE).year
SOURCE_BASELINE_END_YEAR = pd.Timestamp(SOURCE_BASELINE_END_DATE).year
NLCD_BASELINE_START_YEAR = pd.Timestamp(NLCD_BASELINE_START_DATE).year
NLCD_BASELINE_END_YEAR = pd.Timestamp(NLCD_BASELINE_END_DATE).year
TEST_START_YEAR = pd.Timestamp(TEST_START_DATE).year
TEST_END_YEAR = pd.Timestamp(TEST_END_DATE).year

SCRATCH_PRODUCTION_ZARR = (
    SCRATCH_DIR / f"production/lfmc_{LFMC_BASELINE_START_YEAR}_{LFMC_BASELINE_END_YEAR}_archive_test.zarr"
)
SCRATCH_PRODUCTION_METADATA = SCRATCH_DIR / "production/metadata"
SCRATCH_MODIS_CANONICAL_ZARR = (
    SCRATCH_DIR
    / f"modis/modis_interp_5d_{SOURCE_BASELINE_START_YEAR}_{SOURCE_BASELINE_END_YEAR}_baseline_test.zarr"
)
SCRATCH_MODIS_RAW_ROOT = SCRATCH_DIR / "modis/modis_earthaccess"
SCRATCH_MODIS_MOSAIC_ROOT = SCRATCH_DIR / "modis/modis_combined"
SCRATCH_MODIS_REGRID_ROOT = SCRATCH_DIR / "modis/modis_regrid"
SCRATCH_MODIS_STAGING_ROOT = SCRATCH_DIR / "modis/modis_interp_staging"
SCRATCH_MODIS_PLOTS_DIR = SCRATCH_DIR / "modis/plots"
SCRATCH_NLCD_ANNUAL_ZARR = (
    SCRATCH_DIR / f"nlcd/nlcd_target_grid_{NLCD_BASELINE_START_YEAR}_{NLCD_BASELINE_END_YEAR}_test.zarr"
)
SCRATCH_DAYMET_COMBINED_ZARR = (
    SCRATCH_DIR
    / f"weather/daymet_vars_and_anoms_clim20_{SOURCE_BASELINE_START_YEAR}_{SOURCE_BASELINE_END_YEAR}_baseline_test.zarr"
)
SCRATCH_DAYMET_CLIM_ZARR = SCRATCH_DIR / "weather/daymet_vars_and_anoms_clim20_climatology_test.zarr"
SCRATCH_LL_REGRID_ROOT = SCRATCH_DIR / "climate_low_latency/regridded_daily"
SCRATCH_LL_STANDARD_ZARR = SCRATCH_DIR / "climate_low_latency/prism_snodas_low_latency_all_vars_test.zarr"
SCRATCH_LL_APPEND_COORD_DIR = SCRATCH_DIR / "climate_low_latency/append_coord"
SCRATCH_PRISM_RAW_ROOT = SCRATCH_DIR / "climate_low_latency/prism_raw"
SCRATCH_PRISM_EXTRACTED_ROOT = SCRATCH_DIR / "climate_low_latency/prism_extracted"
SCRATCH_PRISM_TARGET_ROOT = SCRATCH_DIR / "climate_low_latency/prism_target_daily"
SCRATCH_PRISM_PLOTS_DIR = SCRATCH_DIR / "climate_low_latency/plots/prism"
SCRATCH_SNODAS_RAW_ROOT = SCRATCH_DIR / "climate_low_latency/snodas_raw"
SCRATCH_SNODAS_TARGET_ROOT = SCRATCH_DIR / "climate_low_latency/snodas_target_daily"
SCRATCH_SNODAS_PLOTS_DIR = SCRATCH_DIR / "climate_low_latency/plots/snodas"
SCRATCH_MAP_RUN_ROOT = SCRATCH_DIR / f"map_runs/low_latency_{TEST_START_YEAR}_{TEST_END_YEAR}"

TEST_SOURCE_REGISTRY = LOG_DIR / "source_registry_test.yaml"
TEST_MAP_CONFIG = LOG_DIR / "map_configs_low_latency_test.yaml"
TEST_MANIFEST = LOG_DIR / "low_latency_forward_setup_manifest.json"
RESET_PROCESSING_ROOTS = os.environ.get("RESET_PROCESSING_ROOTS", "0").lower() in {"1", "true", "yes"}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def scratch_to_oak(scratch_path: Path) -> Path:
    return OAK_ROOT / scratch_path.relative_to(SCRATCH_ROOT)


def run_cmd(cmd):
    print("Running command:")
    print("  " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def rsync_path(src_path: Path, dst_path: Path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if src_path.is_dir():
        dst_path.mkdir(parents=True, exist_ok=True)
        run_cmd(["rsync", "-a", f"{src_path}/", f"{dst_path}/"])
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["rsync", "-a", src_path, dst_path])


def choose_stage_source(scratch_path: Path, label: str) -> Path:
    oak_path = scratch_to_oak(scratch_path)
    if scratch_path.exists():
        print(f"Using canonical scratch {label}: {scratch_path}")
        return scratch_path
    if oak_path.exists():
        print(f"Using OAK {label}: {oak_path}")
        return oak_path
    raise FileNotFoundError(f"Missing required baseline for {label}: scratch={scratch_path} oak={oak_path}")


def require_scratch_source(path: Path, label: str) -> Path:
    if path.exists():
        print(f"Using canonical scratch {label}: {path}")
        return path
    raise FileNotFoundError(f"Missing required scratch baseline for {label}: {path}")


def open_zarr_nonconsolidated(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def open_zarr_consolidated(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def strip_zarr_encoding(ds: xr.Dataset) -> xr.Dataset:
    cleaned = ds.copy(deep=False)
    cleaned.encoding = {}
    for name in cleaned.variables:
        cleaned[name].encoding = {}
    for name in cleaned.coords:
        cleaned[name].encoding = {}
    return cleaned


def rechunk_subset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    chunk_map = {}
    for dim_name, dim_size in ds.sizes.items():
        dim_size = int(dim_size)
        if dim_size <= 0:
            continue
        if dim_name == "time":
            chunk_map[dim_name] = min(128, dim_size)
        elif dim_name in {"year", "landcover_year"}:
            chunk_map[dim_name] = min(1, dim_size)
    if len(chunk_map) == 0:
        return ds
    print(f"Rechunking staged subset before zarr write: {chunk_map}")
    return ds.chunk(chunk_map)


def subset_zarr_coord(
    src_path: Path,
    dst_path: Path,
    selector_dim: str,
    start_value,
    end_value,
    extra_indexers=None,
    use_consolidated_source: bool = False,
    validate_xy_coords: bool = False,
) -> None:
    print(f"Opening source zarr: {src_path}")
    ds = open_zarr_consolidated(src_path) if use_consolidated_source else open_zarr_nonconsolidated(src_path)
    indexers = {selector_dim: slice(start_value, end_value)}
    if extra_indexers is not None:
        indexers.update(extra_indexers)
    subset = rechunk_subset_for_zarr(strip_zarr_encoding(ds.sel(**indexers)))
    print(f"Subset sizes for {dst_path.name}: {dict(subset.sizes)}")
    ensure_parent(dst_path)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    subset.to_zarr(dst_path, mode="w", consolidated=False, safe_chunks=False)
    zarr.consolidate_metadata(str(dst_path))
    ds.close()
    subset.close()
    if validate_xy_coords and not zarr_xy_coords_are_valid(dst_path, use_consolidated_source=False):
        raise ValueError(f"Staged zarr failed x/y coordinate validation: {dst_path}")
    print(f"Wrote subset zarr: {dst_path}")


def coord_value_matches(value, expected_value, compare_as_year: bool = False) -> bool:
    if compare_as_year:
        return pd.Timestamp(value).year == pd.Timestamp(expected_value).year
    return pd.Timestamp(value) == pd.Timestamp(expected_value)


def coord_is_finite_unique(ds: xr.Dataset, coord_name: str, path: Path) -> bool:
    if coord_name not in ds.coords:
        print(f"Coordinate validation failed for {path}: missing coord {coord_name}")
        return False
    values = np.asarray(ds[coord_name].values)
    if values.ndim != 1:
        print(f"Coordinate validation failed for {path}: coord {coord_name} is not 1D")
        return False
    if values.size == 0:
        print(f"Coordinate validation failed for {path}: coord {coord_name} is empty")
        return False
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        missing_count = int((~np.isfinite(values)).sum())
        print(
            f"Coordinate validation failed for {path}: "
            f"coord {coord_name} has {missing_count} non-finite values"
        )
        return False
    unique_count = int(np.unique(values).size)
    if unique_count != values.size:
        print(
            f"Coordinate validation failed for {path}: "
            f"coord {coord_name} has {values.size - unique_count} duplicate values"
        )
        return False
    return True


def zarr_xy_coords_are_valid(dst_path: Path, use_consolidated_source: bool = False) -> bool:
    if not dst_path.exists():
        return False
    try:
        ds = open_zarr_consolidated(dst_path) if use_consolidated_source else open_zarr_nonconsolidated(dst_path)
        valid = coord_is_finite_unique(ds, "x", dst_path) and coord_is_finite_unique(ds, "y", dst_path)
        ds.close()
        return valid
    except Exception as exc:
        print(f"Coordinate validation failed for {dst_path}: {exc}")
        return False


def copy_tree(src_path: Path, dst_path: Path, label: str, reset_existing: bool = True) -> None:
    ensure_parent(dst_path)
    if reset_existing and dst_path.exists():
        shutil.rmtree(dst_path)
    print(f"Copying {label}: {src_path} -> {dst_path}")
    rsync_path(src_path, dst_path)


def subset_matches_bounds(
    dst_path: Path,
    selector_dim: str,
    expected_start_value,
    expected_end_value,
    use_consolidated_source: bool = False,
    compare_as_year: bool = False,
    require_valid_xy: bool = False,
) -> bool:
    if not dst_path.exists():
        return False
    try:
        ds = open_zarr_consolidated(dst_path) if use_consolidated_source else open_zarr_nonconsolidated(dst_path)
        coord = ds[selector_dim]
        if int(coord.sizes.get(selector_dim, 0)) == 0:
            ds.close()
            return False
        first_value = coord.values[0]
        last_value = coord.values[-1]
        ds.close()
        matches_bounds = coord_value_matches(first_value, expected_start_value, compare_as_year) and coord_value_matches(
            last_value,
            expected_end_value,
            compare_as_year,
        )
        if not matches_bounds:
            return False
        if require_valid_xy and not zarr_xy_coords_are_valid(dst_path, use_consolidated_source=use_consolidated_source):
            return False
        return True
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


def iter_low_latency_map_roots() -> list[Path]:
    map_runs_root = SCRATCH_DIR / "map_runs"
    if not map_runs_root.exists():
        return [SCRATCH_MAP_RUN_ROOT]
    roots = list(map_runs_root.glob("low_latency_*"))
    if SCRATCH_MAP_RUN_ROOT not in roots:
        roots.append(SCRATCH_MAP_RUN_ROOT)
    return sorted(roots)


def reset_processing_roots() -> None:
    paths = [
        SCRATCH_DIR / "production/lfmc_2001_2023_test.zarr",
        SCRATCH_DIR / "modis/modis_interp_5d_2001_2023_test.zarr",
        SCRATCH_DIR / "nlcd/nlcd_target_grid_2000_2023_test.zarr",
        SCRATCH_DIR / "weather/daymet_vars_and_anoms_clim20_2000_2023_test.zarr",
        SCRATCH_MODIS_RAW_ROOT,
        SCRATCH_MODIS_MOSAIC_ROOT,
        SCRATCH_MODIS_REGRID_ROOT,
        SCRATCH_MODIS_STAGING_ROOT,
        SCRATCH_MODIS_PLOTS_DIR,
        SCRATCH_LL_REGRID_ROOT,
        SCRATCH_LL_STANDARD_ZARR,
        SCRATCH_LL_APPEND_COORD_DIR,
        SCRATCH_PRISM_RAW_ROOT,
        SCRATCH_PRISM_EXTRACTED_ROOT,
        SCRATCH_PRISM_TARGET_ROOT,
        SCRATCH_PRISM_PLOTS_DIR,
        SCRATCH_SNODAS_RAW_ROOT,
        SCRATCH_SNODAS_TARGET_ROOT,
        SCRATCH_SNODAS_PLOTS_DIR,
        SCRATCH_PRODUCTION_METADATA,
    ]
    paths.extend(iter_low_latency_map_roots())
    for path in paths:
        if path.exists():
            print(f"Removing prior test processing path: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def prepare_processing_roots() -> None:
    paths = [
        SCRATCH_MODIS_RAW_ROOT,
        SCRATCH_MODIS_MOSAIC_ROOT,
        SCRATCH_MODIS_REGRID_ROOT,
        SCRATCH_MODIS_STAGING_ROOT,
        SCRATCH_MODIS_PLOTS_DIR,
        SCRATCH_LL_REGRID_ROOT,
        SCRATCH_LL_APPEND_COORD_DIR,
        SCRATCH_PRISM_RAW_ROOT,
        SCRATCH_PRISM_EXTRACTED_ROOT,
        SCRATCH_PRISM_TARGET_ROOT,
        SCRATCH_PRISM_PLOTS_DIR,
        SCRATCH_SNODAS_RAW_ROOT,
        SCRATCH_SNODAS_TARGET_ROOT,
        SCRATCH_SNODAS_PLOTS_DIR,
        SCRATCH_MAP_RUN_ROOT,
        SCRATCH_PRODUCTION_METADATA,
    ]
    for path in paths:
        if path.exists():
            print(f"Reusing existing test processing path: {path}")
            continue
        path.mkdir(parents=True, exist_ok=True)
        print(f"Creating missing test processing path: {path}")


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
    print("Low-latency forward setup configuration:")
    print(f"  lfmc_baseline_start_date={LFMC_BASELINE_START_DATE}")
    print(f"  lfmc_baseline_end_date={LFMC_BASELINE_END_DATE}")
    print(f"  source_baseline_start_date={SOURCE_BASELINE_START_DATE}")
    print(f"  source_baseline_end_date={SOURCE_BASELINE_END_DATE}")
    print(f"  nlcd_baseline_start_date={NLCD_BASELINE_START_DATE}")
    print(f"  nlcd_baseline_end_date={NLCD_BASELINE_END_DATE}")
    print(f"  test_start_date={TEST_START_DATE}")
    print(f"  test_end_date={TEST_END_DATE}")
    print(f"  source_prewarm_start_date={SOURCE_PREWARM_START_DATE}")
    print(f"  source_prewarm_end_date={SOURCE_PREWARM_END_DATE}")
    print(f"  today_override={TODAY_OVERRIDE}")
    print(f"  scratch_dir={SCRATCH_DIR}")

    if RESET_PROCESSING_ROOTS:
        print("RESET_PROCESSING_ROOTS=true; removing prior low-latency processing paths")
        reset_processing_roots()
    else:
        print("RESET_PROCESSING_ROOTS=false; preserving prior low-latency processing paths")
    prepare_processing_roots()

    lfmc_source = require_scratch_source(SRC_LFMC_TARGET, "LFMC scientific target zarr")
    modis_source = choose_stage_source(SRC_MODIS_CANONICAL, "canonical MODIS zarr")
    nlcd_source = choose_stage_source(SRC_NLCD_ANNUAL, "NLCD annual target zarr")
    daymet_combined_source = choose_stage_source(SRC_DAYMET_CLIM20, "Daymet combined clim20 zarr")
    daymet_clim_source = choose_stage_source(SRC_DAYMET_CLIM, "Daymet climatology zarr")

    if subset_matches_bounds(SCRATCH_PRODUCTION_ZARR, "time", LFMC_BASELINE_START_DATE, LFMC_BASELINE_END_DATE):
        print(
            f"Scratch LFMC target zarr already spans "
            f"{LFMC_BASELINE_START_DATE} -> {LFMC_BASELINE_END_DATE}; skipping rebuild"
        )
    else:
        print(f"Preparing scratch LFMC target zarr for {LFMC_BASELINE_START_DATE} -> {LFMC_BASELINE_END_DATE}")
        subset_zarr_coord(
            lfmc_source,
            SCRATCH_PRODUCTION_ZARR,
            selector_dim="time",
            start_value=LFMC_BASELINE_START_DATE,
            end_value=LFMC_BASELINE_END_DATE,
            extra_indexers={"landcover_year": slice(LFMC_BASELINE_START_YEAR, LFMC_BASELINE_END_YEAR)},
        )

    if subset_matches_bounds(
        SCRATCH_MODIS_CANONICAL_ZARR,
        "time",
        SOURCE_BASELINE_START_DATE,
        SOURCE_BASELINE_END_DATE,
    ):
        print(
            f"Scratch MODIS canonical zarr already spans "
            f"{SOURCE_BASELINE_START_DATE} -> {SOURCE_BASELINE_END_DATE}; skipping rebuild"
        )
    else:
        print(f"Preparing scratch MODIS canonical zarr for {SOURCE_BASELINE_START_DATE} -> {SOURCE_BASELINE_END_DATE}")
        subset_zarr_coord(
            modis_source,
            SCRATCH_MODIS_CANONICAL_ZARR,
            selector_dim="time",
            start_value=SOURCE_BASELINE_START_DATE,
            end_value=SOURCE_BASELINE_END_DATE,
        )

    if subset_matches_bounds(
        SCRATCH_NLCD_ANNUAL_ZARR,
        "year",
        NLCD_BASELINE_START_DATE,
        NLCD_BASELINE_END_DATE,
        use_consolidated_source=True,
        compare_as_year=True,
        require_valid_xy=True,
    ):
        print(
            f"Scratch NLCD annual zarr already spans "
            f"{NLCD_BASELINE_START_YEAR} -> {NLCD_BASELINE_END_YEAR}; skipping rebuild"
        )
    else:
        print(f"Preparing scratch NLCD annual zarr for {NLCD_BASELINE_START_YEAR} -> {NLCD_BASELINE_END_YEAR}")
        subset_zarr_coord(
            nlcd_source,
            SCRATCH_NLCD_ANNUAL_ZARR,
            selector_dim="year",
            start_value=f"{NLCD_BASELINE_START_YEAR}-01-01",
            end_value=f"{NLCD_BASELINE_END_YEAR}-12-31",
            use_consolidated_source=True,
            validate_xy_coords=True,
        )

    if subset_matches_bounds(
        SCRATCH_DAYMET_COMBINED_ZARR,
        "time",
        SOURCE_BASELINE_START_DATE,
        SOURCE_BASELINE_END_DATE,
    ):
        print(
            f"Scratch combined weather store already spans "
            f"{SOURCE_BASELINE_START_DATE} -> {SOURCE_BASELINE_END_DATE}; skipping rebuild"
        )
    else:
        print(f"Preparing scratch combined weather store for {SOURCE_BASELINE_START_DATE} -> {SOURCE_BASELINE_END_DATE}")
        subset_zarr_coord(
            daymet_combined_source,
            SCRATCH_DAYMET_COMBINED_ZARR,
            selector_dim="time",
            start_value=SOURCE_BASELINE_START_DATE,
            end_value=SOURCE_BASELINE_END_DATE,
        )

    if copy_is_usable(SCRATCH_DAYMET_CLIM_ZARR):
        print("Scratch climatology store is already present and readable; skipping restore")
    else:
        print("Restoring scratch climatology store")
        copy_tree(daymet_clim_source, SCRATCH_DAYMET_CLIM_ZARR, "Daymet climatology zarr", reset_existing=False)

    registry_cfg = yaml.safe_load(CANONICAL_SOURCE_REGISTRY.read_text())
    registry_cfg["storage"]["oak_root"] = str(OAK_ROOT)
    registry_cfg["sources"]["modis"]["path"] = str(SCRATCH_MODIS_CANONICAL_ZARR)
    registry_cfg["sources"]["daymet"]["combined_path"] = str(SCRATCH_DAYMET_COMBINED_ZARR)
    registry_cfg["sources"]["daymet"]["climatology_path"] = str(SCRATCH_DAYMET_CLIM_ZARR)
    registry_cfg["sources"]["climate_low_latency"]["path"] = str(SCRATCH_LL_STANDARD_ZARR)
    registry_cfg["sources"]["nlcd"]["annual_path"] = str(SCRATCH_NLCD_ANNUAL_ZARR)
    registry_cfg["sources"]["scientific_current"]["zarr_path"] = str(SCRATCH_PRODUCTION_ZARR)
    registry_cfg["sources"]["production"]["zarr_path"] = str(SCRATCH_PRODUCTION_ZARR)
    registry_cfg["sources"]["production"]["metadata_dir"] = str(SCRATCH_PRODUCTION_METADATA)
    registry_cfg["processing"]["modis"]["raw_root"] = str(SCRATCH_MODIS_RAW_ROOT)
    registry_cfg["processing"]["modis"]["mosaic_root"] = str(SCRATCH_MODIS_MOSAIC_ROOT)
    registry_cfg["processing"]["modis"]["regrid_root"] = str(SCRATCH_MODIS_REGRID_ROOT)
    registry_cfg["processing"]["modis"]["staging_root"] = str(SCRATCH_MODIS_STAGING_ROOT)
    registry_cfg["processing"]["modis"]["plots_dir"] = str(SCRATCH_MODIS_PLOTS_DIR)
    registry_cfg["processing"]["modis"]["quality_flag"] = 1
    registry_cfg["processing"]["prism"]["raw_root"] = str(SCRATCH_PRISM_RAW_ROOT)
    registry_cfg["processing"]["prism"]["extracted_root"] = str(SCRATCH_PRISM_EXTRACTED_ROOT)
    registry_cfg["processing"]["prism"]["target_daily_root"] = str(SCRATCH_PRISM_TARGET_ROOT)
    registry_cfg["processing"]["prism"]["plots_dir"] = str(SCRATCH_PRISM_PLOTS_DIR)
    registry_cfg["processing"]["snodas"]["raw_root"] = str(SCRATCH_SNODAS_RAW_ROOT)
    registry_cfg["processing"]["snodas"]["target_daily_root"] = str(SCRATCH_SNODAS_TARGET_ROOT)
    registry_cfg["processing"]["snodas"]["plots_dir"] = str(SCRATCH_SNODAS_PLOTS_DIR)
    registry_cfg["processing"]["climate_low_latency"]["regrid_root"] = str(SCRATCH_LL_REGRID_ROOT)
    registry_cfg["processing"]["climate_low_latency"]["zarr_path"] = str(SCRATCH_LL_STANDARD_ZARR)
    registry_cfg["processing"]["climate_low_latency"]["append_coord_dir"] = str(SCRATCH_LL_APPEND_COORD_DIR)
    write_yaml(TEST_SOURCE_REGISTRY, registry_cfg)

    map_cfg = yaml.safe_load(CANONICAL_MAP_CONFIG.read_text())
    multiyear_cfg = yaml.safe_load(CANONICAL_MULTIYEAR_MAP_CONFIG.read_text())
    multiyear_submission = multiyear_cfg.get("submission", {})
    map_submission = map_cfg.setdefault("submission", {})
    for key in [
        "owners_partition",
        "owners_gpu_time_limit",
        "owners_gpu_cpus_per_task",
        "owners_gpu_mem",
    ]:
        if key in multiyear_submission:
            map_submission[key] = multiyear_submission[key]
    map_submission["owners_gpu_max_jobs"] = 100
    map_submission["owners_gpu_constraint"] = ""
    map_submission["dynamic_gpu_work_queue"] = True
    map_submission["gpu_fine_tasks_per_job"] = 1
    map_submission["gpu_max_jobs"] = 0
    map_submission["max_prepared_ahead_of_completed_shards"] = 1000
    map_submission["prepare_failure_threshold"] = 3
    print("Configuring low-latency test for owners-only GPU workers (gpu_max_jobs=0, prepare_failure_threshold=3)")
    map_cfg["sources"]["registry_path"] = str(TEST_SOURCE_REGISTRY)
    map_cfg["data"]["requested_start_date"] = TEST_START_DATE
    map_cfg["data"]["requested_end_date"] = TEST_END_DATE
    map_cfg["paths"]["run_root"] = str(SCRATCH_MAP_RUN_ROOT)
    write_yaml(TEST_MAP_CONFIG, map_cfg)

    manifest = {
        "lfmc_baseline_start_date": LFMC_BASELINE_START_DATE,
        "lfmc_baseline_end_date": LFMC_BASELINE_END_DATE,
        "source_baseline_start_date": SOURCE_BASELINE_START_DATE,
        "source_baseline_end_date": SOURCE_BASELINE_END_DATE,
        "nlcd_baseline_start_date": NLCD_BASELINE_START_DATE,
        "nlcd_baseline_end_date": NLCD_BASELINE_END_DATE,
        "test_start_date": TEST_START_DATE,
        "test_end_date": TEST_END_DATE,
        "source_prewarm_start_date": SOURCE_PREWARM_START_DATE,
        "source_prewarm_end_date": SOURCE_PREWARM_END_DATE,
        "today_override": TODAY_OVERRIDE,
        "modis_quality_flag": registry_cfg["processing"]["modis"]["quality_flag"],
        "scratch_dir": str(SCRATCH_DIR),
        "paths": {
            "production_zarr": str(SCRATCH_PRODUCTION_ZARR),
            "production_metadata_dir": str(SCRATCH_PRODUCTION_METADATA),
            "modis_canonical_zarr": str(SCRATCH_MODIS_CANONICAL_ZARR),
            "modis_raw_root": str(SCRATCH_MODIS_RAW_ROOT),
            "modis_mosaic_root": str(SCRATCH_MODIS_MOSAIC_ROOT),
            "modis_regrid_root": str(SCRATCH_MODIS_REGRID_ROOT),
            "modis_staging_root": str(SCRATCH_MODIS_STAGING_ROOT),
            "nlcd_annual_zarr": str(SCRATCH_NLCD_ANNUAL_ZARR),
            "combined_weather_zarr": str(SCRATCH_DAYMET_COMBINED_ZARR),
            "climatology_zarr": str(SCRATCH_DAYMET_CLIM_ZARR),
            "low_latency_standard_zarr": str(SCRATCH_LL_STANDARD_ZARR),
            "low_latency_regrid_root": str(SCRATCH_LL_REGRID_ROOT),
            "append_coord_dir": str(SCRATCH_LL_APPEND_COORD_DIR),
            "source_registry_test": str(TEST_SOURCE_REGISTRY),
            "map_config_test": str(TEST_MAP_CONFIG),
        },
        "source_paths": {
            "lfmc_target": str(lfmc_source),
            "modis_canonical": str(modis_source),
            "nlcd_annual": str(nlcd_source),
            "daymet_combined": str(daymet_combined_source),
            "daymet_climatology": str(daymet_clim_source),
        },
    }
    write_manifest(manifest)
    print("Scratch low-latency forward setup staging complete")


if __name__ == "__main__":
    main()
