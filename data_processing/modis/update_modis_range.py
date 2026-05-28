#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import earthaccess
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from get_modis import (
    DEFAULT_GRID_PATH,
    TILES_V,
    collect_links,
    dict_to_key,
    existing_file_keys,
    expected_file_keys_for_month,
    load_missing_granules_manifest,
)
from update_modis_month import (
    DEFAULT_REGISTRY_PATH,
    build_staging_zarr,
    candidate_regrid_paths,
    canonical_time_bounds,
    ensure_daily_mosaics_for_dates,
    ensure_regridded_files_for_dates,
    ensure_raw_downloads,
    modis_bounding_box,
    mosaic_dates_needing_build,
    prune_source_artifacts,
    promote_staging_window,
    regrid_dates_needing_build,
    run_cmd,
    validate_regridded_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download, process, interpolate, and append a MODIS date range into the "
            "canonical zarr, with extra tail context for interpolation."
        )
    )
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--registry_path", type=Path, default=Path(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--grid_path", type=Path, default=None)
    parser.add_argument("--raw_root", type=Path, default=None)
    parser.add_argument("--mosaic_root", type=Path, default=None)
    parser.add_argument("--regrid_root", type=Path, default=None)
    parser.add_argument("--canonical_zarr", type=Path, default=None)
    parser.add_argument("--staging_root", type=Path, default=None)
    parser.add_argument("--plots_dir", type=Path, default=None)
    parser.add_argument("--tail_context_days", type=int, default=None)
    parser.add_argument("--max_interpolation_days", type=int, default=None)
    parser.add_argument("--buffer_days", type=int, default=None)
    parser.add_argument("--xy_chunk_size", type=int, default=None)
    parser.add_argument("--time_chunk_size", type=int, default=None)
    parser.add_argument("--quality_flag", type=int, default=None)
    parser.add_argument("--check_only", action="store_true")
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, "r") as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    proc = registry.get("processing", {}).get("modis", {})
    sources = registry.get("sources", {})
    args.grid_path = args.grid_path or Path(DEFAULT_GRID_PATH)
    args.raw_root = args.raw_root or Path(proc["raw_root"])
    args.mosaic_root = args.mosaic_root or Path(proc["mosaic_root"])
    args.regrid_root = args.regrid_root or Path(proc["regrid_root"])
    args.canonical_zarr = args.canonical_zarr or Path(sources["modis"]["path"])
    args.staging_root = args.staging_root or Path(proc["staging_root"])
    args.plots_dir = args.plots_dir or Path(proc["plots_dir"])
    args.tail_context_days = args.tail_context_days or int(proc["low_latency_tail_context_days"])
    args.max_interpolation_days = args.max_interpolation_days or int(proc["interpolation_max_days"])
    args.buffer_days = args.buffer_days or int(proc["interpolation_buffer_days"])
    args.xy_chunk_size = args.xy_chunk_size or int(proc["interpolation_xy_chunk_size"])
    args.time_chunk_size = args.time_chunk_size or int(proc["interpolation_time_chunk_size"])
    args.interpolation_num_workers = int(proc.get("interpolation_num_workers", 32))
    args.interpolation_worker_cpus = int(proc.get("interpolation_worker_cpus", 4))
    args.interpolation_worker_mem = str(proc.get("interpolation_worker_mem", "16G"))
    args.interpolation_worker_time = str(proc.get("interpolation_worker_time", "08:00:00"))
    args.interpolation_array_max_retries = int(proc.get("interpolation_array_max_retries", 3))
    args.regrid_chunk_buffer = int(proc.get("regrid_chunk_buffer", 100))
    args.regrid_chunk_size = int(proc.get("regrid_chunk_size", 2000))
    args.retain_staging_after_success = bool(proc.get("retain_staging_after_success", True))
    args.quality_flag = args.quality_flag if args.quality_flag is not None else int(proc["quality_flag"])
    return args


def parse_date_range(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    return start, end


def range_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start_date, end_date, freq="D")


def canonical_has_full_range(canonical_zarr: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    current_min, current_max = canonical_time_bounds(canonical_zarr)
    if current_min is None or current_max is None:
        return False
    return current_min <= start_date and current_max >= end_date


def regridded_range_complete(regrid_root: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    return all(
        any(path.exists() for path in candidate_regrid_paths(regrid_root, ts))
        for ts in range_dates(start_date, end_date)
    )


def raw_downloads_cover_range(raw_root: Path, start_date: pd.Timestamp, end_date: pd.Timestamp):
    missing = []
    missing_manifest = load_missing_granules_manifest(raw_root)
    allowed_missing = {
        dict_to_key(record)
        for records in missing_manifest.values()
        for record in records
    }
    allowed_hits = []
    month_starts = pd.date_range(
        start=start_date.replace(day=1),
        end=end_date.replace(day=1),
        freq="MS",
    )
    for month_start in month_starts:
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        this_start = max(month_start.normalize(), start_date)
        this_end = min(month_end, end_date)
        if this_end < this_start:
            continue
        month_days = pd.date_range(start=max(this_start, start_date), end=this_end, freq="D")
        year_dir = raw_root / month_end.strftime("%Y")
        existing = existing_file_keys(year_dir, "MCD43A4") | existing_file_keys(year_dir, "MCD43A2")
        for short_name in ("MCD43A4", "MCD43A2"):
            expected = expected_file_keys_for_month(month_days, short_name)
            for short, date_tag, tile_tag, version in sorted(set(expected) - existing):
                key = (short, date_tag, tile_tag, version)
                if key in allowed_missing:
                    allowed_hits.append(f"{short}.{date_tag}.{tile_tag}.{version}")
                    continue
                missing.append(f"{short}.{date_tag}.{tile_tag}.{version}")
    if allowed_hits:
        print(
            f"Allowing {len(allowed_hits)} MODIS logical gaps for NaN fallback; "
            f"sample={allowed_hits[:10]}"
        )
    return len(missing) == 0, missing


def remote_range_available(start_date: pd.Timestamp, end_date: pd.Timestamp, grid_path: Path):
    earthaccess.login()
    bbox = modis_bounding_box(grid_path)
    dates = range_dates(start_date, end_date)
    desired = len(dates) * len(TILES_V)
    results_data = earthaccess.search_data(
        short_name="MCD43A4",
        version="061",
        temporal=(start_date, end_date),
        bounding_box=bbox,
        downloadable=True,
    )
    data_links = collect_links(results_data, dates, "MCD43A4")
    results_quality = earthaccess.search_data(
        short_name="MCD43A2",
        version="061",
        temporal=(start_date, end_date),
        bounding_box=bbox,
        downloadable=True,
    )
    quality_links = collect_links(results_quality, dates, "MCD43A2")
    available = True
    data_missing = max(desired - len(data_links), 0)
    quality_missing = max(desired - len(quality_links), 0)
    reason = f"data_links={len(data_links)}/{desired};quality_links={len(quality_links)}/{desired}"
    if data_missing:
        reason = (
            f"{reason};data_missing={data_missing};"
            "data_partial_allowed=nan_fallback"
        )
    if quality_missing:
        reason = (
            f"{reason};quality_missing={quality_missing};"
            "quality_partial_allowed=nan_fallback"
        )
    return available, reason


def compute_refresh_start(canonical_zarr: Path, requested_start_date: pd.Timestamp, max_interpolation_days: int):
    current_min, current_max = canonical_time_bounds(canonical_zarr)
    if current_max is None:
        return requested_start_date
    refresh_candidate = current_max - pd.Timedelta(days=max_interpolation_days)
    refresh_start = min(requested_start_date, refresh_candidate)
    if current_min is not None:
        refresh_start = max(refresh_start, current_min)
    return pd.Timestamp(refresh_start).normalize()


def main():
    args = apply_registry_defaults(parse_args())
    requested_start, requested_end = parse_date_range(args.start_date, args.end_date)
    source_end = requested_end + pd.Timedelta(days=int(args.tail_context_days))
    refresh_start = compute_refresh_start(args.canonical_zarr, requested_start, args.max_interpolation_days)
    source_context_start = refresh_start - pd.Timedelta(days=int(args.buffer_days))
    source_context_end = source_end + pd.Timedelta(days=int(args.buffer_days))

    print(
        "Updating MODIS for range "
        f"{requested_start.date()} -> {requested_end.date()} "
        f"with tail context through {source_end.date()}"
    )
    print(f"  refresh_start={refresh_start.date()}")
    print(f"  source_context={source_context_start.date()} -> {source_context_end.date()}")
    print(f"  raw_root={args.raw_root}")
    print(f"  mosaic_root={args.mosaic_root}")
    print(f"  regrid_root={args.regrid_root}")
    print(f"  canonical_zarr={args.canonical_zarr}")

    if args.check_only:
        if canonical_has_full_range(args.canonical_zarr, requested_start, source_end):
            print(
                "MODIS availability check: available=True "
                "reason=canonical_zarr_already_contains_requested_range_with_tail_context"
            )
            raise SystemExit(0)
        available, reason = remote_range_available(source_context_start, source_context_end, args.grid_path)
        print(f"MODIS availability check: available={available} reason={reason}")
        raise SystemExit(0 if available else 1)

    if canonical_has_full_range(args.canonical_zarr, requested_start, source_end):
        print(
            "Canonical MODIS zarr already contains requested range with tail context; "
            "nothing to do"
        )
        return

    ensure_raw_downloads(source_context_start, source_context_end, args.raw_root, args.grid_path)
    raw_complete, missing_raw = raw_downloads_cover_range(args.raw_root, source_context_start, source_context_end)
    if not raw_complete:
        raise FileNotFoundError(
            f"Raw MODIS download did not produce the expected HDF files for "
            f"{source_context_start.date()} -> {source_context_end.date()}; "
            f"missing_count={len(missing_raw)} sample_missing={missing_raw[:10]}"
        )
    total_source_days = len(range_dates(source_context_start, source_context_end))
    regrid_missing_or_invalid_dates = regrid_dates_needing_build(
        args.regrid_root,
        source_context_start,
        source_context_end,
    )
    mosaic_build_dates = []
    for ts in regrid_missing_or_invalid_dates:
        if mosaic_dates_needing_build(args.mosaic_root, ts, ts):
            mosaic_build_dates.append(ts)
    print(
        "Regridded MODIS reuse check: "
        f"{total_source_days - len(regrid_missing_or_invalid_dates)}/{total_source_days} "
        "source-context dates already have valid regrids"
    )
    print(
        "Native MODIS mosaic dependency check: "
        f"{len(mosaic_build_dates)}/{len(regrid_missing_or_invalid_dates)} "
        "dates needing regrid also need a mosaic built first"
    )
    ensure_daily_mosaics_for_dates(
        mosaic_build_dates,
        args.raw_root,
        args.mosaic_root,
        args.quality_flag,
    )

    ensure_regridded_files_for_dates(
        regrid_missing_or_invalid_dates,
        args.grid_path,
        args.mosaic_root,
        args.regrid_root,
        args.regrid_chunk_size,
        args.regrid_chunk_buffer,
    )
    validate_regridded_grid(args.regrid_root, args.canonical_zarr, source_context_start, source_context_end)

    staging_zarr = build_staging_zarr(args, refresh_start, source_end)
    result = promote_staging_window(staging_zarr, args.canonical_zarr, refresh_start, source_end)
    print(f"MODIS range update complete: {result}")
    print(f"  staging_zarr={staging_zarr}")
    print(f"  canonical_zarr={args.canonical_zarr}")
    prune_source_artifacts(
        args.raw_root,
        args.mosaic_root,
        args.regrid_root,
        source_context_start,
        source_context_end,
    )
    if args.retain_staging_after_success:
        print(f"Retaining staging zarr after success: {staging_zarr}")


if __name__ == "__main__":
    main()
