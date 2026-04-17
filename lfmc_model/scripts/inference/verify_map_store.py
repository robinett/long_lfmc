#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


STORE_COMPARISON_ATOL = 1.0e-5


def get_args():
    parser = argparse.ArgumentParser(
        description="Verify a merged map zarr store against nonempty shard samples."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--kind", choices=["local", "production"], required=True)
    parser.add_argument("--store_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--metadata_dir", type=str, default=None)
    parser.add_argument("--staging_zarr", type=str, default=None)
    parser.add_argument("--production_zarr", type=str, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    return parser.parse_args()


def _load_run_context(manifest_path: Path):
    run_dir = manifest_path.parent
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    manifest_df = pd.read_csv(manifest_path)
    return run_dir, run_config, manifest_df


def _output_path(run_dir: Path, kind: str, explicit: str | None) -> Path:
    if explicit not in {None, "", "None"}:
        return Path(str(explicit))
    return run_dir / f"{kind}_store_verification.json"


def _sample_n(run_config: dict, explicit_sample_n: int | None) -> int:
    if explicit_sample_n is not None:
        return max(1, int(explicit_sample_n))
    return int(
        run_config.get("config", {})
        .get("multiyear", {})
        .get("verification_sample_shards", 4)
    )


def _verification_store_path(run_config: dict, kind: str, explicit_store_path: str | None) -> str:
    if explicit_store_path not in {None, "", "None"}:
        return str(explicit_store_path)
    if kind == "local":
        return str(run_config["out_zarr_path"])
    raise ValueError("store_path must be provided for production verification")


def _matching_promotion_record(
    metadata_dir: Path,
    staging_zarr: str,
    production_zarr: str,
    start_date: str,
    end_date: str,
) -> str | None:
    if not metadata_dir.exists():
        return None
    matching_paths = []
    for record_path in sorted(metadata_dir.glob("production_update_*.json")):
        with open(record_path, "r") as f:
            record = json.load(f)
        if record.get("start_date") != start_date or record.get("end_date") != end_date:
            continue
        if record.get("production_zarr") != production_zarr:
            continue
        if record.get("staging_zarr") != staging_zarr:
            continue
        if record.get("status") not in {"completed", "completed_initialized_from_staging"}:
            continue
        matching_paths.append(str(record_path))
    if len(matching_paths) == 0:
        return None
    return matching_paths[-1]


def _nonempty_verification_rows(manifest_df: pd.DataFrame, sample_n: int) -> tuple[pd.DataFrame, dict]:
    ordered = manifest_df.sort_values(
        ["block_idx", "tile_iy", "tile_ix", "task_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    inspected_rows = []
    nonempty_rows = []
    for row_idx, row in enumerate(ordered.itertuples(index=False), start=1):
        shard_path = Path(str(row.shard_path))
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard while selecting verification rows: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as npz:
            shard_mean = np.asarray(npz["lfmc_ens_mean"], dtype=np.float32)
            finite_n = int(np.isfinite(shard_mean).sum())
        inspected_rows.append(str(shard_path))
        if finite_n > 0:
            row_payload = row._asdict()
            row_payload["verification_finite_n"] = finite_n
            nonempty_rows.append(row_payload)
        if row_idx == 1 or row_idx == len(ordered) or row_idx % 2000 == 0:
            print(
                f"[verify_map_store] scanned {row_idx}/{len(ordered)} shards; "
                f"nonempty={len(nonempty_rows)}"
            )
    selection_summary = {
        "inspected_shards": len(inspected_rows),
        "nonempty_shards_found": len(nonempty_rows),
    }
    if len(nonempty_rows) == 0:
        return pd.DataFrame(), selection_summary
    nonempty_df = pd.DataFrame(nonempty_rows)
    sample_n = max(1, min(int(sample_n), len(nonempty_df)))
    if sample_n == len(nonempty_df):
        sampled = nonempty_df.reset_index(drop=True)
    else:
        sample_idx = np.linspace(0, len(nonempty_df) - 1, sample_n, dtype=int)
        sample_idx = sorted(set(int(idx) for idx in sample_idx.tolist()))
        sampled = nonempty_df.iloc[sample_idx].reset_index(drop=True)
    selection_summary["selected_nonempty_shards"] = int(len(sampled))
    return sampled, selection_summary


def _verify_store_against_shards(
    store_path: Path,
    run_config: dict,
    sample_rows: pd.DataFrame,
) -> dict:
    result = {
        "ok": False,
        "store_path": str(store_path),
        "checked_shards": [],
        "reason": "",
    }
    if not store_path.exists():
        result["reason"] = f"missing_store:{store_path}"
        return result
    if len(sample_rows) == 0:
        result["reason"] = "no_nonempty_verification_shards"
        return result
    root = zarr.open_group(str(store_path), mode="r")
    required_vars = ["time", "lfmc_ens_mean", "lfmc_ens_std", "quality_flag"]
    missing_vars = [name for name in required_vars if name not in root]
    if missing_vars:
        result["reason"] = f"missing_arrays:{','.join(missing_vars)}"
        return result
    time_lookup = {
        int(time_value): idx
        for idx, time_value in enumerate(np.asarray(root["time"][:], dtype=np.int64).tolist())
    }
    expected_quality = int(run_config["quality_flag_value"])
    for row in sample_rows.itertuples(index=False):
        shard_path = Path(str(row.shard_path))
        with np.load(shard_path, allow_pickle=False) as npz:
            dates = np.asarray(npz["dates"], dtype="datetime64[ns]").astype("int64")
            if int(dates[0]) not in time_lookup or int(dates[-1]) not in time_lookup:
                result["reason"] = (
                    f"time_lookup_missing:{shard_path.name}:{int(dates[0])}:{int(dates[-1])}"
                )
                return result
            t0 = int(time_lookup[int(dates[0])])
            t1 = int(time_lookup[int(dates[-1])]) + 1
            y0 = int(npz["y0"])
            y1 = int(npz["y1"])
            x0 = int(npz["x0"])
            x1 = int(npz["x1"])
            shard_mean = np.asarray(npz["lfmc_ens_mean"], dtype=np.float32)
            shard_std = np.asarray(npz["lfmc_ens_std"], dtype=np.float32)
            store_mean = np.asarray(root["lfmc_ens_mean"][t0:t1, y0:y1, x0:x1], dtype=np.float32)
            store_std = np.asarray(root["lfmc_ens_std"][t0:t1, y0:y1, x0:x1], dtype=np.float32)
            store_quality = np.asarray(root["quality_flag"][t0:t1], dtype=np.uint8)
        if shard_mean.shape != store_mean.shape or shard_std.shape != store_std.shape:
            result["reason"] = f"shape_mismatch:{shard_path.name}"
            return result
        finite_mask = np.isfinite(shard_mean)
        if bool(finite_mask.any()):
            if not np.array_equal(np.isfinite(store_mean), finite_mask):
                result["reason"] = f"finite_mask_mismatch:{shard_path.name}"
                return result
            if not np.allclose(
                store_mean[finite_mask],
                shard_mean[finite_mask],
                rtol=0.0,
                atol=STORE_COMPARISON_ATOL,
            ):
                result["reason"] = f"mean_value_mismatch:{shard_path.name}"
                return result
            if not np.allclose(
                store_std[finite_mask],
                shard_std[finite_mask],
                rtol=0.0,
                atol=STORE_COMPARISON_ATOL,
            ):
                result["reason"] = f"std_value_mismatch:{shard_path.name}"
                return result
        if not np.all(store_quality == expected_quality):
            result["reason"] = f"quality_flag_mismatch:{shard_path.name}"
            return result
        result["checked_shards"].append(str(shard_path))
    result["ok"] = True
    result["reason"] = "verified"
    return result


def main():
    args = get_args()
    manifest_path = Path(args.manifest_path)
    run_dir, run_config, manifest_df = _load_run_context(manifest_path)
    sample_n = _sample_n(run_config, args.sample_n)
    store_path = Path(_verification_store_path(run_config, args.kind, args.store_path))
    output_path = _output_path(run_dir, args.kind, args.output_path)

    sample_rows, selection_summary = _nonempty_verification_rows(manifest_df, sample_n)
    verification = _verify_store_against_shards(
        store_path=store_path,
        run_config=run_config,
        sample_rows=sample_rows,
    )
    payload = {
        "kind": args.kind,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "store_path": str(store_path),
        "requested_start_date": str(run_config.get("requested_start_date")),
        "requested_end_date": str(run_config.get("requested_end_date")),
        "verification_selection": selection_summary,
        "verification": verification,
        "promotion_record_path": None,
    }
    if args.kind == "production":
        metadata_dir = args.metadata_dir if args.metadata_dir is not None else str(store_path.parent / "metadata")
        staging_zarr = args.staging_zarr if args.staging_zarr is not None else str(run_config["out_zarr_path"])
        production_zarr = args.production_zarr if args.production_zarr is not None else str(store_path)
        start_date = args.start_date if args.start_date is not None else str(run_config["requested_start_date"])
        end_date = args.end_date if args.end_date is not None else str(run_config["requested_end_date"])
        payload["promotion_record_path"] = _matching_promotion_record(
            metadata_dir=Path(metadata_dir),
            staging_zarr=staging_zarr,
            production_zarr=production_zarr,
            start_date=start_date,
            end_date=end_date,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[verify_map_store] wrote verification artifact to {output_path}")


if __name__ == "__main__":
    main()
