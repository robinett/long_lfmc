#!/usr/bin/env python3

import argparse
import copy
import datetime as dt
import json
import os
import pandas as pd
import subprocess
import sys
import time
from pathlib import Path

from map_config import default_map_config_path, get_cfg, load_map_config

here = os.path.dirname(os.path.abspath(__file__))
PROMOTE_SBATCH = os.path.join(here, "promote_maps_ensemble.sbatch")
VERIFY_SBATCH = os.path.join(here, "verify_maps_ensemble.sbatch")
CREATE_MAPS_SCRIPT = os.path.join(here, "create_maps.py")


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


def _maybe_latest_run_dir(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    candidates = [
        path for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    ]
    if len(candidates) == 0:
        return None
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


def _year_sequence(start_year: int, end_year: int) -> list[int]:
    return list(range(end_year, start_year - 1, -1))


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

    # Multiyear mode needs a blocking coordinator through validation.
    cfg_year["submission"]["wait_for_validation_completion"] = True

    config_path = config_dir / f"map_config_year_{year:04d}.yaml"
    with open(config_path, "w") as f:
        json.dump(cfg_year, f, indent=2, sort_keys=False)
    return config_path


def _load_run_context(run_dir: Path) -> tuple[dict, pd.DataFrame, Path]:
    run_config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.csv"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    manifest_df = pd.read_csv(manifest_path)
    return run_config, manifest_df, run_config_path


def _write_run_config(run_config_path: Path, run_config: dict) -> None:
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)


def _local_store_path(run_dir: Path, run_config: dict) -> Path:
    merged_dir = run_config.get("merged_dir")
    if merged_dir not in {None, "", "None"}:
        merged_dir_path = Path(str(merged_dir))
    else:
        merged_dir_path = run_dir / "merged"
    store_name = (
        run_config.get("config", {})
        .get("paths", {})
        .get("merged_store_name")
    )
    if store_name in {None, "", "None"}:
        store_name = Path(str(run_config.get("out_zarr_path", "lfmc_maps.zarr"))).name
    if store_name in {"", ".", ".."}:
        store_name = "lfmc_maps.zarr"
    return merged_dir_path / store_name


def _ensure_local_store_target(
    run_dir: Path,
    persistent_out_zarr_path: str,
) -> tuple[dict, pd.DataFrame]:
    run_config, manifest_df, run_config_path = _load_run_context(run_dir)
    expected_local_store = _local_store_path(run_dir, run_config)
    current_out = run_config.get("out_zarr_path")
    if current_out != str(expected_local_store):
        current_out_str = str(current_out) if current_out not in {None, ""} else ""
        merged_dir = expected_local_store.parent
        current_out_path = Path(current_out_str) if current_out_str != "" else None
        is_legacy_local_store = (
            current_out_path is not None
            and current_out_path.parent == merged_dir
        )
        if (
            current_out_str == str(persistent_out_zarr_path)
            or current_out_str == ""
            or is_legacy_local_store
        ):
            run_config["out_zarr_path"] = str(expected_local_store)
            _write_run_config(run_config_path, run_config)
            print(
                f"[create_maps_multiyear] rewired run {run_dir.name} to year-local merged store "
                f"{expected_local_store}"
            )
        else:
            raise ValueError(
                f"Run {run_dir} has unexpected out_zarr_path={current_out_str}; "
                f"expected {expected_local_store} or legacy persistent path {persistent_out_zarr_path}"
            )
    return run_config, manifest_df


def _shard_count(run_dir: Path) -> int:
    shard_dir = run_dir / "shards"
    if not shard_dir.exists():
        return 0
    return sum(1 for _ in shard_dir.glob("*.npz"))


def _prepared_reference_count(run_dir: Path) -> int:
    prepared_dir = run_dir / "prepared_tensors"
    if not prepared_dir.exists():
        return 0
    return sum(1 for _ in prepared_dir.glob("*_reference.pt"))


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _verification_artifact_path(run_dir: Path, kind: str) -> Path:
    return run_dir / f"{kind}_store_verification.json"


def _load_verification_artifact(run_dir: Path, kind: str) -> dict | None:
    artifact_path = _verification_artifact_path(run_dir, kind)
    payload = _load_json(artifact_path)
    if payload is None:
        return None
    if not isinstance(payload.get("verification"), dict):
        return None
    return payload


def _completed_state_from_marker(marker_path: Path) -> dict | None:
    marker = _load_json(marker_path)
    if marker is None:
        return None
    local_verify = marker.get("local_store_verification")
    production_verify = marker.get("production_store_verification")
    promotion_record_path = marker.get("promotion_record_path")
    if not isinstance(local_verify, dict) or not bool(local_verify.get("ok")):
        return None
    if not isinstance(production_verify, dict) or not bool(production_verify.get("ok")):
        return None
    if promotion_record_path in {None, "", "None"}:
        return None
    promotion_record = Path(str(promotion_record_path))
    if not promotion_record.exists():
        return None
    return {
        "verification_selection": marker.get("verification_selection"),
        "local_verify": local_verify,
        "production_verify": production_verify,
        "promotion_record_path": str(promotion_record),
    }


def _job_state(job_id: str) -> str:
    sacct_cmd = [
        "sacct",
        "-j",
        str(job_id),
        "--format=State",
        "-n",
        "-P",
    ]
    try:
        sacct_output = subprocess.check_output(sacct_cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        sacct_output = ""
    for line in sacct_output.splitlines():
        state = line.split("|", 1)[0].strip()
        if state != "":
            return state.split()[0]
    squeue_cmd = [
        "squeue",
        "-h",
        "-j",
        str(job_id),
        "-o",
        "%T",
    ]
    try:
        squeue_output = subprocess.check_output(squeue_cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        squeue_output = ""
    for line in squeue_output.splitlines():
        state = line.strip()
        if state != "":
            return state.split()[0]
    return "UNKNOWN"


def _job_failed(state: str) -> bool:
    return state in {
        "BOOT_FAIL",
        "CANCELLED",
        "CANCELLED+",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }


def _wait_for_job_completion(job_id: str, label: str, poll_seconds: int = 30) -> None:
    while True:
        state = _job_state(job_id)
        if _job_failed(state):
            raise RuntimeError(f"{label} job {job_id} failed with state {state}")
        if state == "COMPLETED":
            print(f"[create_maps_multiyear] {label} job {job_id} completed")
            return
        print(
            f"[create_maps_multiyear] {label} monitor: job={job_id} state={state}; "
            f"sleeping {poll_seconds}s"
        )
        time.sleep(float(poll_seconds))


def _year_state(
    year_run_root: Path,
    persistent_out_zarr_path: str,
) -> dict:
    marker_path = _year_marker_path(
        year_run_root.parent.parent / "multiyear_status",
        int(year_run_root.name.split("_")[-1]),
    )
    latest_run_dir = _maybe_latest_run_dir(year_run_root)
    state = {
        "run_dir": str(latest_run_dir) if latest_run_dir is not None else None,
        "marker_exists": marker_path.exists(),
        "state": "missing",
        "task_count": 0,
        "shard_count": 0,
        "prepared_count": 0,
        "verification_selection": None,
        "local_verify": None,
        "production_verify": None,
        "promotion_record_path": None,
        "local_verification_path": str(_verification_artifact_path(latest_run_dir, "local")) if latest_run_dir is not None else None,
        "production_verification_path": str(_verification_artifact_path(latest_run_dir, "production")) if latest_run_dir is not None else None,
    }
    if latest_run_dir is None:
        return state
    marker_state = _completed_state_from_marker(marker_path)
    if marker_state is not None:
        state.update(marker_state)
        state["state"] = "completed"
        return state
    run_config, manifest_df = _ensure_local_store_target(
        latest_run_dir,
        persistent_out_zarr_path=persistent_out_zarr_path,
    )
    task_count = int(len(manifest_df))
    shard_count = _shard_count(latest_run_dir)
    prepared_count = _prepared_reference_count(latest_run_dir)
    state["task_count"] = task_count
    state["shard_count"] = shard_count
    state["prepared_count"] = prepared_count
    local_store_path = Path(str(run_config["out_zarr_path"]))
    if task_count > 0 and shard_count == task_count:
        local_artifact = _load_verification_artifact(latest_run_dir, "local")
        if local_artifact is None:
            if local_store_path.exists():
                state["state"] = "needs_local_verification"
            else:
                state["state"] = "needs_merge"
            return state
        state["verification_selection"] = local_artifact.get("verification_selection")
        state["local_verify"] = local_artifact.get("verification")
        if not bool(state["local_verify"].get("ok")):
            state["state"] = "needs_merge"
            return state
        production_artifact = _load_verification_artifact(latest_run_dir, "production")
        if production_artifact is not None:
            state["production_verify"] = production_artifact.get("verification")
            state["promotion_record_path"] = production_artifact.get("promotion_record_path")
        promotion_record_path = state["promotion_record_path"]
        if (
            isinstance(state["production_verify"], dict)
            and bool(state["production_verify"].get("ok"))
            and promotion_record_path not in {None, "", "None"}
            and Path(str(promotion_record_path)).exists()
        ):
            state["state"] = "completed"
        else:
            state["state"] = "needs_promotion"
    elif shard_count > 0 or prepared_count > 0:
        state["state"] = "partial"
    return state


def _run_year_workflow(
    year_config_path: Path,
    manifest_only: bool,
    child_env: dict,
) -> None:
    cmd = [
        sys.executable,
        CREATE_MAPS_SCRIPT,
        "--config_path",
        str(year_config_path),
    ]
    if manifest_only:
        cmd.append("--manifest_only")
    print(f"[create_maps_multiyear] command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=child_env)


def _submit_sbatch(script_path: str, export_items: list[str], label: str) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        f"--export={','.join(export_items)}",
        script_path,
    ]
    print(f"[create_maps_multiyear] submitting {label} via: {' '.join(cmd)}")
    job_id = subprocess.check_output(cmd, text=True).strip().split(";", 1)[0].strip()
    if job_id == "":
        raise RuntimeError(f"{label} sbatch did not return a job id")
    print(f"[create_maps_multiyear] submitted {label} job {job_id}")
    return job_id


def _verify_local_store(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.csv"
    export_items = [
        "ALL",
        f"MANIFEST_PATH={manifest_path}",
        "VERIFY_KIND=local",
    ]
    job_id = _submit_sbatch(
        script_path=VERIFY_SBATCH,
        export_items=export_items,
        label="local verification",
    )
    _wait_for_job_completion(job_id, label="local verification")
    return job_id


def _promote_year(
    run_config: dict,
    run_dir: Path,
    persistent_out_zarr_path: str,
    product_tier: str,
) -> str:
    staging_zarr = str(run_config["out_zarr_path"])
    requested_start_date = str(run_config["requested_start_date"])
    requested_end_date = str(run_config["requested_end_date"])
    metadata_dir = str(Path(persistent_out_zarr_path).parent / "metadata")
    export_env = ",".join(
        [
            "ALL",
            f"MANIFEST_PATH={str(run_dir / 'manifest.csv')}",
            f"STAGING_ZARR={staging_zarr}",
            f"PRODUCTION_ZARR={str(persistent_out_zarr_path)}",
            f"METADATA_DIR={metadata_dir}",
            f"START_DATE={requested_start_date}",
            f"END_DATE={requested_end_date}",
            "PROMOTION_MODE=overwrite_time_range",
            f"PRODUCT_TIER={str(product_tier)}",
            "INITIALIZE_IF_MISSING=1",
        ]
    )
    job_id = _submit_sbatch(
        script_path=PROMOTE_SBATCH,
        export_items=export_env.split(","),
        label="promotion",
    )
    _wait_for_job_completion(job_id, label="promotion")
    return job_id


def _write_completion_marker(
    marker_path: Path,
    year: int,
    year_idx: int,
    total_years: int,
    year_started_at_iso: str,
    year_started_at_epoch: float,
    latest_run_dir: Path,
    year_config_path: Path,
    persistent_out_zarr_path: str,
    manifest_only: bool,
    verification_selection: dict | None,
    local_verify: dict,
    production_verify: dict,
    promotion_record_path: str | None,
) -> None:
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
        "manifest_only": bool(manifest_only),
        "verification_selection": verification_selection,
        "local_store_verification": local_verify,
        "production_store_verification": production_verify,
        "promotion_record_path": promotion_record_path,
    }
    with open(marker_path, "w") as f:
        json.dump(marker_payload, f, indent=2, sort_keys=True)


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
    year_sequence = _year_sequence(start_year, end_year)
    product_tier = str(get_cfg(cfg, "product", "tier", default="final"))

    print(
        f"[create_maps_multiyear] years={start_year}-{end_year}; "
        f"persistent_out_zarr_path={persistent_out_zarr_path}; "
        "verification_mode=batch_artifacts_only"
    )
    print(f"[create_maps_multiyear] run order={year_sequence}")

    total_years = len(year_sequence)
    for year_idx, year in enumerate(year_sequence, start=1):
        marker_path = _year_marker_path(status_dir, year)
        year_run_root = base_run_root / "years" / f"year_{year:04d}"
        year_config_path = _write_year_config(
            cfg=cfg,
            year=year,
            start_year=start_year,
            end_year=end_year,
            base_run_root=base_run_root,
            persistent_out_zarr_path=str(persistent_out_zarr_path),
            config_dir=config_dir,
        )
        year_run_root.mkdir(parents=True, exist_ok=True)
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
        year_state = _year_state(
            year_run_root=year_run_root,
            persistent_out_zarr_path=str(persistent_out_zarr_path),
        )
        if args.resume and year_state["state"] == "completed":
            print(
                f"[create_maps_multiyear] skipping year {year}: production verification already "
                f"passes for run {year_state['run_dir']}"
            )
            if not marker_path.exists():
                _write_completion_marker(
                    marker_path=marker_path,
                    year=year,
                    year_idx=year_idx,
                    total_years=total_years,
                    year_started_at_iso=year_started_at_iso,
                    year_started_at_epoch=year_started_at_epoch,
                    latest_run_dir=Path(str(year_state["run_dir"])),
                    year_config_path=year_config_path,
                    persistent_out_zarr_path=str(persistent_out_zarr_path),
                    manifest_only=bool(args.manifest_only),
                    verification_selection=year_state["verification_selection"],
                    local_verify=year_state["local_verify"],
                    production_verify=year_state["production_verify"],
                    promotion_record_path=year_state["promotion_record_path"],
                )
            continue

        if year_state["state"] in {"missing", "partial", "needs_merge"}:
            print(
                f"[create_maps_multiyear] starting/resuming year {year}; "
                f"state={year_state['state']}; marker_exists={year_state['marker_exists']}"
            )
            _run_year_workflow(
                year_config_path=year_config_path,
                manifest_only=bool(args.manifest_only),
                child_env=child_env,
            )
            if args.manifest_only:
                print(
                    f"[create_maps_multiyear] manifest_only completed for year {year}; "
                    "not writing a completion marker"
                )
                continue
            year_state = _year_state(
                year_run_root=year_run_root,
                persistent_out_zarr_path=str(persistent_out_zarr_path),
            )

        if year_state["state"] == "needs_local_verification":
            latest_run_dir = Path(str(year_state["run_dir"]))
            print(
                f"[create_maps_multiyear] verifying existing year-local store for {year}: "
                f"{year_state['local_verification_path']}"
            )
            _verify_local_store(latest_run_dir)
            year_state = _year_state(
                year_run_root=year_run_root,
                persistent_out_zarr_path=str(persistent_out_zarr_path),
            )

        if year_state["state"] == "needs_promotion":
            latest_run_dir = Path(str(year_state["run_dir"]))
            run_config, _ = _ensure_local_store_target(
                latest_run_dir,
                persistent_out_zarr_path=str(persistent_out_zarr_path),
            )
            print(
                f"[create_maps_multiyear] promoting verified year-local store for {year}: "
                f"{run_config['out_zarr_path']}"
            )
            _promote_year(
                run_config=run_config,
                run_dir=latest_run_dir,
                persistent_out_zarr_path=str(persistent_out_zarr_path),
                product_tier=product_tier,
            )
            year_state = _year_state(
                year_run_root=year_run_root,
                persistent_out_zarr_path=str(persistent_out_zarr_path),
            )

        if year_state["state"] != "completed":
            raise RuntimeError(
                f"Year {year} did not reach a verified completed state. "
                f"state={year_state['state']}; local_verify={year_state['local_verify']}; "
                f"production_verify={year_state['production_verify']}"
            )

        latest_run_dir = Path(str(year_state["run_dir"]))
        _write_completion_marker(
            marker_path=marker_path,
            year=year,
            year_idx=year_idx,
            total_years=total_years,
            year_started_at_iso=year_started_at_iso,
            year_started_at_epoch=year_started_at_epoch,
            latest_run_dir=latest_run_dir,
            year_config_path=year_config_path,
            persistent_out_zarr_path=str(persistent_out_zarr_path),
            manifest_only=bool(args.manifest_only),
            verification_selection=year_state["verification_selection"],
            local_verify=year_state["local_verify"],
            production_verify=year_state["production_verify"],
            promotion_record_path=year_state["promotion_record_path"],
        )
        print(f"[create_maps_multiyear] completed year {year}; marker={marker_path}")


if __name__ == "__main__":
    main()
