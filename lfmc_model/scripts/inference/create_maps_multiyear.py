#!/usr/bin/env python3

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from map_config import default_map_config_path, get_cfg, load_map_config

here = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the ensemble map pipeline one year at a time while appending into a "
            "persistent multi-year output store."
        )
    )
    parser.add_argument("--config_path", type=str, default=default_map_config_path())
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _latest_run_dir(run_root: Path) -> Path:
    candidates = [
        path for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No run directories found under {run_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _require_year_range(cfg, args):
    start_year = args.start_year
    if start_year is None:
        start_year = get_cfg(cfg, "multiyear", "start_year", default=None)
    end_year = args.end_year
    if end_year is None:
        end_year = get_cfg(cfg, "multiyear", "end_year", default=None)
    if start_year is None or end_year is None:
        raise ValueError(
            "Multiyear runs require start/end years via --start_year/--end_year or "
            "multiyear.start_year/multiyear.end_year in the config"
        )
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError(f"start_year must be <= end_year; got {start_year} > {end_year}")
    return start_year, end_year


def _year_marker_path(status_dir: Path, year: int) -> Path:
    return status_dir / f"year_{year}_completed.json"


def _completed_years(status_dir: Path, start_year: int, end_year: int) -> list[int]:
    completed = []
    for year in range(start_year, end_year + 1):
        if _year_marker_path(status_dir, year).exists():
            completed.append(year)
    return completed


def _year_sequence(start_year: int, end_year: int) -> list[int]:
    return list(range(end_year, start_year - 1, -1))


def _cleanup_year_run_artifacts(year_run_root: Path, run_dir: Path):
    if run_dir.exists():
        print(f"[create_maps_multiyear] removing successful year run directory {run_dir}")
        shutil.rmtree(run_dir)
    if year_run_root.exists() and not any(year_run_root.iterdir()):
        print(f"[create_maps_multiyear] removing empty year container {year_run_root}")
        year_run_root.rmdir()


def _write_year_config(
    cfg: dict,
    year: int,
    start_year: int,
    end_year: int,
    base_run_root: Path,
    persistent_out_zarr_path: str,
    config_dir: Path,
) -> Path:
    cfg_year = copy.deepcopy(cfg)
    cfg_year.pop("_config_path", None)
    cfg_year.setdefault("data", {})
    cfg_year.setdefault("paths", {})
    cfg_year.setdefault("submission", {})

    cfg_year["data"]["requested_start_date"] = f"{year:04d}-01-01"
    cfg_year["data"]["requested_end_date"] = f"{year:04d}-12-31"
    cfg_year["data"]["output_store_start_date"] = f"{start_year:04d}-01-01"
    cfg_year["data"]["output_store_end_date"] = f"{end_year:04d}-12-31"

    cfg_year["paths"]["run_root"] = str(base_run_root / "years" / f"year_{year:04d}")
    cfg_year["paths"]["persistent_out_zarr_path"] = str(persistent_out_zarr_path)

    # Multiyear mode needs a blocking coordinator and cleanup after successful validation.
    cfg_year["submission"]["wait_for_validation_completion"] = True
    cfg_year["submission"]["cleanup_prepared_tensors_after_success"] = True

    config_path = config_dir / f"map_config_year_{year:04d}.yaml"
    with open(config_path, "w") as f:
        json.dump(cfg_year, f, indent=2, sort_keys=False)
    return config_path


def main():
    args = get_args()
    cfg = load_map_config(args.config_path)
    start_year, end_year = _require_year_range(cfg, args)

    base_run_root = Path(str(get_cfg(cfg, "paths", "run_root")))
    persistent_out_zarr_path = get_cfg(cfg, "paths", "persistent_out_zarr_path", default=None)
    if persistent_out_zarr_path in {None, "", "None"}:
        raise ValueError(
            "Multiyear runs require paths.persistent_out_zarr_path in the config"
        )

    status_dir = base_run_root / "multiyear_status"
    config_dir = base_run_root / "multiyear_configs"
    status_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    completed_years = _completed_years(status_dir, start_year, end_year)
    year_sequence = _year_sequence(start_year, end_year)
    cleanup_year_run_dir_after_success = bool(
        get_cfg(cfg, "multiyear", "cleanup_year_run_dir_after_success", default=True)
    )

    print(
        f"[create_maps_multiyear] years={start_year}-{end_year}; "
        f"persistent_out_zarr_path={persistent_out_zarr_path}"
    )
    print(
        f"[create_maps_multiyear] cleanup_year_run_dir_after_success="
        f"{cleanup_year_run_dir_after_success}"
    )
    if args.resume and completed_years:
        next_incomplete_year = next(
            (year for year in year_sequence if year not in completed_years),
            None,
        )
        print(
            f"[create_maps_multiyear] resume enabled; completed_years={completed_years}; "
            f"next_incomplete_year={next_incomplete_year}"
        )

    print(f"[create_maps_multiyear] run order={year_sequence}")

    total_years = len(year_sequence)
    for year_idx, year in enumerate(year_sequence, start=1):
        marker_path = _year_marker_path(status_dir, year)
        if args.resume and marker_path.exists():
            print(
                f"[create_maps_multiyear] skipping year {year}: completion marker exists at {marker_path}"
            )
            continue

        year_config_path = _write_year_config(
            cfg=cfg,
            year=year,
            start_year=start_year,
            end_year=end_year,
            base_run_root=base_run_root,
            persistent_out_zarr_path=str(persistent_out_zarr_path),
            config_dir=config_dir,
        )
        year_run_root = base_run_root / "years" / f"year_{year:04d}"
        year_run_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            os.path.join(here, "create_maps.py"),
            "--config_path",
            str(year_config_path),
        ]
        if args.manifest_only:
            cmd.append("--manifest_only")

        print(
            f"[create_maps_multiyear] starting year {year} with config {year_config_path}"
        )
        print(f"[create_maps_multiyear] command: {' '.join(cmd)}")
        year_started_at_epoch = time.time()
        year_started_at_iso = dt.datetime.now().isoformat(timespec="seconds")
        child_env = os.environ.copy()
        child_env["MULTIYEAR_CURRENT_YEAR"] = str(year)
        child_env["MULTIYEAR_START_YEAR"] = str(start_year)
        child_env["MULTIYEAR_END_YEAR"] = str(end_year)
        child_env["MULTIYEAR_TOTAL_YEARS"] = str(total_years)
        child_env["MULTIYEAR_YEAR_ORDINAL"] = str(year_idx)
        child_env["MULTIYEAR_STATUS_DIR"] = str(status_dir)
        child_env["MULTIYEAR_YEAR_STARTED_AT_EPOCH"] = f"{year_started_at_epoch:.3f}"
        subprocess.run(cmd, check=True, env=child_env)

        latest_run_dir = _latest_run_dir(year_run_root)
        marker_payload = {
            "year": year,
            "year_ordinal": year_idx,
            "total_years": total_years,
            "started_at": year_started_at_iso,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.time() - year_started_at_epoch,
            "run_dir": str(latest_run_dir),
            "year_config_path": str(year_config_path),
            "persistent_out_zarr_path": str(persistent_out_zarr_path),
            "manifest_only": bool(args.manifest_only),
            "run_dir_removed": False,
        }
        with open(marker_path, "w") as f:
            json.dump(marker_payload, f, indent=2, sort_keys=True)
        print(
            f"[create_maps_multiyear] completed year {year}; marker={marker_path}"
        )
        if cleanup_year_run_dir_after_success and not args.manifest_only:
            _cleanup_year_run_artifacts(year_run_root, latest_run_dir)
            marker_payload["run_dir_removed"] = True
            marker_payload["run_dir_removed_at"] = dt.datetime.now().isoformat(timespec="seconds")
            with open(marker_path, "w") as f:
                json.dump(marker_payload, f, indent=2, sort_keys=True)
            print(
                f"[create_maps_multiyear] cleaned year {year} scratch artifacts after success"
            )


if __name__ == "__main__":
    main()
