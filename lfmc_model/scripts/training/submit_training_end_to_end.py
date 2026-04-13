#!/usr/bin/env python3

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc")

REQUIRED_TENSOR_FILES = [
    "X_short.pt",
    "X_long.pt",
    "X_static.pt",
    "Y.pt",
    "source.pt",
    "stratifier.npy",
    "info.csv",
]

SERC_GPU_MAX_JOBS = 8
OWNERS_GPU_MAX_JOBS = 50
SERC_PARTITION = "serc"
OWNERS_PARTITION = "owners"
OWNERS_GPU_CONSTRAINT = (
    "GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|"
    "GPU_SKU:H200_SXM5|GPU_SKU:L40S|GPU_SKU:RTX_2080Ti|GPU_SKU:RTX_3090|GPU_SKU:V100S_PCIE"
)

POLL_SECONDS = 60
SUBMIT_SLEEP_SECONDS = 5
SLURM_POLL_SLEEP_SECONDS = 60
PENDING_START_TIMEOUT_SECONDS = 45 * 60
HEARTBEAT_STALE_SECONDS = 15 * 60
SLURM_RECHECK_MIN_SECONDS = 15 * 60
PREPROCESS_STALE_SECONDS = 45 * 60

BATCH_SIZE = "128"
LR = "5e-4"
VAL_SPLIT = "0.15"
ADAM_WD = "1e-4"
DROPOUT = "0.15"
WARMUP_EPOCHS = "2"
SCHEDULER_T_MAX = "40"

MODEL_FAMILY = "multisource_fusion"
DAYMET_ZARR_CLIM20 = str(SCRATCH_ROOT / "daymet" / "daymet_vars_and_anoms_clim20.zarr")
DEFAULT_SUBMISSION_TAG = (
    "multisource_fusion_k3_dw64_dm128_ds32_dc64_sh32_lfp64_sarp32_wd64_lr5e-4"
    "_tw5_vpd_anoms_nozone_clim20"
)
DEFAULT_EXISTING_TENSOR_TAG = "dm128_vpd_anoms_nozone_clim20"
NO_DAYMET_SUBMISSION_TAG = (
    "multisource_fusion_no_daymet_k3_dw64_dm128_ds32_dc64_sh32_lfp64_sarp32"
    "_wd64_lr5e-4_tw5_vpd_anoms_nozone_clim20"
)

D_MODEL = "128"
NHEAD = "4"
NUM_LAYERS = "3"
DIM_FEEDFORWARD = "256"

LONG_D_MODEL = "256"
LONG_NHEAD = "8"
LONG_NUM_LAYERS = "3"
LONG_DIM_FEEDFORWARD = "512"
LONG_OUT_DIM = "128"

WEATHER_KERNEL_SIZE = "3"
WEATHER_D_MODEL = "64"
WEATHER_MAX_DILATION = "64"
MODIS_D_MODEL = "128"
STATIC_D_MODEL = "32"
COMMON_D_MODEL = "64"
SHARED_LATENT_DIM = "32"
LFMC_PRIVATE_DIM = "64"
SAR_PRIVATE_DIM = "32"

MULTITASK_NUM_TASKS = "3"
MULTITASK_WEIGHTING_TYPE = "manual"
MULTITASK_TASK_WEIGHTS = ["5.0", "1.0", "1.0"]

SINGLE_NUM_TASKS = "1"
SINGLE_WEIGHTING_TYPE = "manual"
SINGLE_TASK_WEIGHTS = ["1.0"]

COMPLETE_STATE = "COMPLETE"
FATAL_STATE = "FAILED_FATAL"
RETRYABLE_STATE = "FAILED_RETRYABLE"
RUNNING_STATES = {"SUBMITTED", "STARTED"}


def now_ts():
    return time.time()


def utc_now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message):
    print("[%s] %s" % (utc_now_iso(), message), flush=True)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(tmp_path), str(path))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception:
        return None


def validate_input_dir(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return False
    for required_file in REQUIRED_TENSOR_FILES:
        if not (input_dir / required_file).exists():
            return False
    return True


def training_output_dir(save_root, run_tag):
    matches = sorted(Path(save_root).glob("*_" + run_tag))
    return matches[0] if matches else None


def training_outputs_complete(save_root, run_tag):
    run_dir = training_output_dir(save_root, run_tag)
    if run_dir is None:
        return False
    if not (run_dir / "fold_info.json").exists():
        return False
    for fold_idx in range(1, 7):
        fold_dir = run_dir / ("fold_%d" % fold_idx)
        if not (fold_dir / "test_info.csv").exists():
            return False
        if not (fold_dir / "test_outputs.pth").exists():
            return False
    return True


def acquire_lock(lock_path):
    lock_fh = Path(lock_path).open("a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        lock_fh.seek(0)
        existing = lock_fh.read().strip()
        raise RuntimeError("Coordinator appears to already be running: %s pid=%s" % (lock_path, existing))
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(str(os.getpid()) + "\n")
    lock_fh.flush()
    return lock_fh


def connect_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preprocess_steps (
            step_key TEXT PRIMARY KEY,
            step_type TEXT NOT NULL,
            job_name TEXT NOT NULL,
            output_path TEXT NOT NULL,
            state TEXT NOT NULL,
            job_id TEXT,
            submit_ts REAL,
            last_slurm_poll_ts REAL,
            last_error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_jobs (
            run_name TEXT PRIMARY KEY,
            run_tag TEXT NOT NULL,
            task_group TEXT NOT NULL,
            member_idx INTEGER NOT NULL,
            input_dir TEXT NOT NULL,
            save_root TEXT NOT NULL,
            pool TEXT,
            state TEXT NOT NULL,
            active_job_id TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            submit_ts REAL,
            start_ts REAL,
            end_ts REAL,
            heartbeat_ts REAL,
            last_slurm_poll_ts REAL,
            last_error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ts REAL NOT NULL,
            scope TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            job_id TEXT,
            message TEXT
        )
        """
    )
    conn.commit()
    return conn


def record_event(conn, scope, entity_key, event_type, job_id=None, message=None):
    conn.execute(
        """
        INSERT INTO events (event_ts, scope, entity_key, event_type, job_id, message)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (now_ts(), scope, entity_key, event_type, job_id, message),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-tensors", action="store_true")
    parser.add_argument("--use-existing-tensors", action="store_true")
    parser.add_argument("--multitask-only", action="store_true")
    parser.add_argument("--single-task-only", action="store_true")
    parser.add_argument(
        "--no-daymet-multitask",
        action="store_true",
        help=(
            "Train the derived multitask no-Daymet variant from existing tensors. "
            "Run derive_no_daymet_multitask_tensors.py first."
        ),
    )
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument(
        "--submission-tag",
        default=DEFAULT_SUBMISSION_TAG,
    )
    parser.add_argument("--existing-tensor-tag", default=DEFAULT_EXISTING_TENSOR_TAG)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.no_daymet_multitask:
        if args.build_tensors:
            raise RuntimeError("--no-daymet-multitask uses pre-derived tensors; do not pass --build-tensors.")
        if args.single_task_only:
            raise RuntimeError("--no-daymet-multitask is only defined for multitask training.")
        args.use_existing_tensors = True
        args.multitask_only = True
        if args.submission_tag == DEFAULT_SUBMISSION_TAG:
            args.submission_tag = NO_DAYMET_SUBMISSION_TAG
        if args.existing_tensor_tag == DEFAULT_EXISTING_TENSOR_TAG:
            args.existing_tensor_tag = NO_DAYMET_SUBMISSION_TAG

    if args.build_tensors and args.use_existing_tensors:
        raise RuntimeError("Choose only one of --build-tensors or --use-existing-tensors.")
    if args.use_existing_tensors:
        args.build_tensors = False
    else:
        args.build_tensors = True
    args.train_multitask = not args.single_task_only
    args.train_single = not args.multitask_only
    if not args.train_multitask and not args.train_single:
        raise RuntimeError("At least one of multitask or single-task training must be enabled.")
    return args


def build_paths(args):
    tensor_tag = args.submission_tag if args.build_tensors else args.existing_tensor_tag
    shared_fold_root = SCRATCH_ROOT / "lfmc_model" / "outputs" / "shared_training"
    state_root = shared_fold_root / ("_coord_" + args.submission_tag)
    return {
        "tensor_tag": tensor_tag,
        "multitask_sar_root": SCRATCH_ROOT / "sar" / "ensemble" / ("lfmc_vh_vv_365_" + tensor_tag),
        "multitask_sample_index_root": SCRATCH_ROOT / "lfmc_model" / "indexes" / "ensemble" / ("lfmc_vh_vv_365_" + tensor_tag),
        "multitask_input_root": SCRATCH_ROOT / "lfmc_model" / "inputs" / "ensemble" / ("lfmc_vh_vv_365_" + tensor_tag),
        "multitask_save_root": SCRATCH_ROOT / "lfmc_model" / "outputs" / ("lfmc_vh_vv_365_" + args.submission_tag),
        "single_sample_index_path": SCRATCH_ROOT / "lfmc_model" / "indexes" / ("sample_index_longweather_2000_2024_lfmc_" + tensor_tag + ".parquet"),
        "single_input_dir": SCRATCH_ROOT / "lfmc_model" / "inputs" / ("lfmc_365_" + tensor_tag),
        "single_save_root": SCRATCH_ROOT / "lfmc_model" / "outputs" / ("lfmc_365_" + args.submission_tag),
        "shared_fold_root": shared_fold_root,
        "fold_info_path": shared_fold_root / ("canonical_fold_info_" + tensor_tag + ".json"),
        "state_root": state_root,
        "db_path": state_root / "training_state.sqlite",
        "coordinator_lock_path": SCRATCH_ROOT / "lfmc_model" / "gpu_locks" / ("training_scheduler_" + args.submission_tag + ".lock"),
        "lock_dir": SCRATCH_ROOT / "lfmc_model" / "gpu_locks",
    }


def ensure_dirs(paths):
    for key in [
        "multitask_sar_root",
        "multitask_sample_index_root",
        "multitask_input_root",
        "multitask_save_root",
        "shared_fold_root",
        "state_root",
        "lock_dir",
    ]:
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    Path(paths["single_sample_index_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["single_save_root"]).mkdir(parents=True, exist_ok=True)


def marker_base(paths, run_name):
    return Path(paths["state_root"]) / "training_runs" / run_name


def marker_path(paths, run_name, marker_name):
    return marker_base(paths, run_name) / (marker_name + ".json")


def cleanup_training_markers(paths, run_name):
    for marker_name in ["started", "heartbeat", "completed", "failed"]:
        marker = marker_path(paths, run_name, marker_name)
        if marker.exists():
            marker.unlink()


def write_submitted_marker(paths, spec, pool, job_id):
    cleanup_training_markers(paths, spec["run_name"])
    write_json_atomic(
        marker_path(paths, spec["run_name"], "submitted"),
        {
            "timestamp_utc": utc_now_iso(),
            "timestamp_unix": now_ts(),
            "job_id": str(job_id),
            "pool": pool,
            "run_name": spec["run_name"],
            "run_tag": spec["run_tag"],
        },
    )


def build_training_specs(args, paths):
    multitask_specs = []
    single_specs = []
    for member_idx in range(args.ensemble_size):
        data_seed = 1000 + member_idx
        model_seed = 1000 + member_idx
        data_tag = "ds%04d" % data_seed
        multitask_specs.append(
            {
                "task_group": "multitask",
                "member_idx": member_idx,
                "run_tag": "multi_%s_ms%04d" % (data_tag, model_seed),
                "run_name": "multi_ens%02d_%s" % (member_idx, args.submission_tag),
                "input_dir": str(paths["multitask_input_root"] / ("lfmc_vh_vv_" + data_tag)),
                "save_root": str(paths["multitask_save_root"]),
                "seed": str(model_seed),
                "num_tasks": MULTITASK_NUM_TASKS,
                "weighting_type": MULTITASK_WEIGHTING_TYPE,
                "task_weights": MULTITASK_TASK_WEIGHTS,
            }
        )
        single_specs.append(
            {
                "task_group": "single",
                "member_idx": member_idx,
                "run_tag": "single_ms%04d" % model_seed,
                "run_name": "single_ens%02d_%s" % (member_idx, args.submission_tag),
                "input_dir": str(paths["single_input_dir"]),
                "save_root": str(paths["single_save_root"]),
                "seed": str(model_seed),
                "num_tasks": SINGLE_NUM_TASKS,
                "weighting_type": SINGLE_WEIGHTING_TYPE,
                "task_weights": SINGLE_TASK_WEIGHTS,
            }
        )
    return multitask_specs, single_specs


def upsert_training_specs(conn, specs):
    for spec in specs:
        updated = conn.execute(
            """
            UPDATE training_jobs
            SET run_tag = ?, task_group = ?, member_idx = ?, input_dir = ?, save_root = ?
            WHERE run_name = ?
            """,
            (
                spec["run_tag"],
                spec["task_group"],
                spec["member_idx"],
                spec["input_dir"],
                spec["save_root"],
                spec["run_name"],
            ),
        )
        if updated.rowcount == 0:
            conn.execute(
                """
                INSERT INTO training_jobs (run_name, run_tag, task_group, member_idx, input_dir, save_root, state)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec["run_name"],
                    spec["run_tag"],
                    spec["task_group"],
                    spec["member_idx"],
                    spec["input_dir"],
                    spec["save_root"],
                    "NEW",
                ),
            )
    conn.commit()


def build_preprocess_defs(args, paths):
    if not args.build_tensors:
        return []

    preprocess_defs = []
    multitask_preprocess_members = args.ensemble_size if args.train_multitask else 1
    for member_idx in range(multitask_preprocess_members):
        data_seed = 1000 + member_idx
        model_seed = 1000 + member_idx
        data_tag = "ds%04d" % data_seed
        run_name = "multi_ens%02d_%s" % (member_idx, args.submission_tag)
        member_sar_dir = paths["multitask_sar_root"] / data_tag
        member_sample_index = paths["multitask_sample_index_root"] / ("sample_index_longweather_2000_2024_lfmc_vh_vv_" + data_tag + ".parquet")
        member_input_dir = paths["multitask_input_root"] / ("lfmc_vh_vv_" + data_tag)
        preprocess_defs.append(
            {
                "step_key": "multitask_tensor_%02d" % member_idx,
                "step_type": "multitask_tensor",
                "job_name": "tensor_%s" % run_name,
                "member_idx": member_idx,
                "run_name": run_name,
                "data_seed": data_seed,
                "data_tag": data_tag,
                "model_seed": model_seed,
                "sar_dir": member_sar_dir,
                "sample_index": member_sample_index,
                "input_dir": member_input_dir,
                "output_path": member_input_dir,
            }
        )

    preprocess_defs.append(
        {
            "step_key": "fold_info",
            "step_type": "fold_info",
            "job_name": "fold_%s" % args.submission_tag,
            "output_path": paths["fold_info_path"],
            "input_dir": paths["multitask_input_root"] / "lfmc_vh_vv_ds1000",
        }
    )

    if args.train_single:
        preprocess_defs.append(
            {
                "step_key": "single_tensor",
                "step_type": "single_tensor",
                "job_name": "tensor_single_%s" % args.submission_tag,
                "output_path": paths["single_input_dir"],
                "sample_index": paths["single_sample_index_path"],
                "input_dir": paths["single_input_dir"],
            }
        )
    return preprocess_defs


def preprocess_output_complete(step):
    if step["step_type"] == "fold_info":
        return Path(step["output_path"]).exists()
    return validate_input_dir(step["output_path"])


def sync_preprocess_defs(conn, defs):
    for step in defs:
        updated = conn.execute(
            """
            UPDATE preprocess_steps
            SET step_type = ?, job_name = ?, output_path = ?
            WHERE step_key = ?
            """,
            (step["step_type"], step["job_name"], str(step["output_path"]), step["step_key"]),
        )
        if updated.rowcount == 0:
            conn.execute(
                """
                INSERT INTO preprocess_steps (step_key, step_type, job_name, output_path, state)
                VALUES (?, ?, ?, ?, ?)
                """,
                (step["step_key"], step["step_type"], step["job_name"], str(step["output_path"]), "NEW"),
            )
    conn.commit()


def refresh_preprocess_states(conn, defs):
    defs_by_key = {step["step_key"]: step for step in defs}
    for row in conn.execute("SELECT * FROM preprocess_steps").fetchall():
        step = defs_by_key.get(row["step_key"])
        if step is None:
            continue
        if preprocess_output_complete(step):
            if row["state"] != COMPLETE_STATE:
                conn.execute(
                    """
                    UPDATE preprocess_steps
                    SET state = ?, job_id = NULL, last_error = NULL
                    WHERE step_key = ?
                    """,
                    (COMPLETE_STATE, row["step_key"]),
                )
                record_event(conn, "preprocess", row["step_key"], "complete", job_id=row["job_id"])


def sleep_then_poll_slurm():
    time.sleep(SLURM_POLL_SLEEP_SECONDS)


def poll_slurm_jobs(job_names):
    sleep_then_poll_slurm()
    wanted = set(job_names)
    active = {}
    accounting = {}
    if not wanted:
        return active, accounting

    try:
        squeue_out = subprocess.check_output(
            ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%T|%P"],
            universal_newlines=True,
        )
    except Exception:
        squeue_out = ""

    for line in squeue_out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        job_id, job_name, state, partition = parts
        if job_name in wanted:
            active[job_name] = {"job_id": job_id, "state": state, "partition": partition}

    try:
        sacct_out = subprocess.check_output(
            [
                "sacct",
                "-u",
                os.environ["USER"],
                "--starttime",
                "today",
                "--format=JobIDRaw,JobName%80,State",
                "-n",
                "-P",
            ],
            universal_newlines=True,
        )
    except Exception:
        sacct_out = ""

    for line in sacct_out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id, job_name, state = parts[0], parts[1], parts[2]
        if job_name not in wanted:
            continue
        previous = accounting.get(job_name)
        if previous is None or int(job_id) > int(previous["job_id"]):
            accounting[job_name] = {"job_id": job_id, "state": state}

    return active, accounting


def maybe_reconcile_stale_preprocess(conn, defs):
    stale_rows = []
    current_ts = now_ts()
    for row in conn.execute("SELECT * FROM preprocess_steps WHERE state = 'SUBMITTED'").fetchall():
        if row["submit_ts"] is None:
            continue
        last_poll_ts = float(row["last_slurm_poll_ts"]) if row["last_slurm_poll_ts"] is not None else None
        if current_ts - float(row["submit_ts"]) >= PREPROCESS_STALE_SECONDS:
            if last_poll_ts is None or (current_ts - last_poll_ts) >= PREPROCESS_STALE_SECONDS:
                stale_rows.append(row)
    if not stale_rows:
        return

    job_names = [row["job_name"] for row in stale_rows]
    log("Sleeping %ss before rare Slurm reconciliation for stale preprocess jobs: %s" % (SLURM_POLL_SLEEP_SECONDS, ", ".join(job_names)))
    active, accounting = poll_slurm_jobs(job_names)
    current_ts = now_ts()
    for row in stale_rows:
        job_name = row["job_name"]
        if job_name in active:
            conn.execute(
                "UPDATE preprocess_steps SET last_slurm_poll_ts = ? WHERE step_key = ?",
                (current_ts, row["step_key"]),
            )
            continue
        latest = accounting.get(job_name)
        state_text = latest["state"] if latest else "UNKNOWN"
        conn.execute(
            """
            UPDATE preprocess_steps
            SET state = ?, last_slurm_poll_ts = ?, last_error = ?
            WHERE step_key = ?
            """,
            (FATAL_STATE, current_ts, "stale preprocess missing from Slurm state=%s" % state_text, row["step_key"]),
        )
        record_event(conn, "preprocess", row["step_key"], "failed_fatal", job_id=(latest or {}).get("job_id"), message=state_text)


def refresh_training_states(conn, specs, paths):
    spec_by_name = {spec["run_name"]: spec for spec in specs}
    for row in conn.execute("SELECT * FROM training_jobs").fetchall():
        spec = spec_by_name[row["run_name"]]
        if training_outputs_complete(spec["save_root"], spec["run_tag"]):
            if row["state"] != COMPLETE_STATE:
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET state = ?, active_job_id = NULL, end_ts = ?, last_error = NULL
                    WHERE run_name = ?
                    """,
                    (COMPLETE_STATE, now_ts(), row["run_name"]),
                )
                record_event(conn, "training", row["run_name"], "complete", job_id=row["active_job_id"], message="validated folds 1-6")
            for lock_path in Path(paths["lock_dir"]).glob("lock_*_%s.lock" % row["run_name"]):
                lock_path.unlink()
            continue

        submitted = read_json(marker_path(paths, row["run_name"], "submitted"))
        started = read_json(marker_path(paths, row["run_name"], "started"))
        heartbeat = read_json(marker_path(paths, row["run_name"], "heartbeat"))
        failed = read_json(marker_path(paths, row["run_name"], "failed"))

        if failed is not None:
            failed_job_id = str(failed.get("job_id", "")) or row["active_job_id"]
            exit_note = "worker_failed exit_code=%s" % failed.get("exit_code")
            if row["pool"] == "serc":
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET state = ?, active_job_id = NULL, end_ts = ?, last_error = ?
                    WHERE run_name = ?
                    """,
                    (FATAL_STATE, now_ts(), exit_note, row["run_name"]),
                )
                record_event(conn, "training", row["run_name"], "failed_fatal", job_id=failed_job_id, message=exit_note)
            else:
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET state = ?, active_job_id = NULL, end_ts = ?, last_error = ?
                    WHERE run_name = ?
                    """,
                    (RETRYABLE_STATE, now_ts(), exit_note, row["run_name"]),
                )
                record_event(conn, "training", row["run_name"], "failed_retryable", job_id=failed_job_id, message=exit_note)
            for lock_path in Path(paths["lock_dir"]).glob("lock_*_%s.lock" % row["run_name"]):
                lock_path.unlink()
            continue

        if started is not None:
            start_ts = float(started.get("timestamp_unix", row["start_ts"] or now_ts()))
            heartbeat_ts = start_ts
            if heartbeat is not None:
                heartbeat_ts = float(heartbeat.get("timestamp_unix", heartbeat_ts))
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, start_ts = COALESCE(start_ts, ?), heartbeat_ts = ?, active_job_id = COALESCE(active_job_id, ?)
                WHERE run_name = ?
                """,
                ("STARTED", start_ts, heartbeat_ts, str(started.get("job_id", "")), row["run_name"]),
            )
            continue

        if submitted is not None:
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, submit_ts = COALESCE(submit_ts, ?), active_job_id = COALESCE(active_job_id, ?)
                WHERE run_name = ?
                """,
                ("SUBMITTED", float(submitted.get("timestamp_unix", row["submit_ts"] or now_ts())), str(submitted.get("job_id", "")), row["run_name"]),
            )
            continue

        if row["state"] not in (COMPLETE_STATE, FATAL_STATE):
            conn.execute(
                "UPDATE training_jobs SET state = ?, active_job_id = NULL WHERE run_name = ?",
                ("NEW", row["run_name"]),
            )


def maybe_reconcile_stale_training(conn):
    stale_rows = []
    current_ts = now_ts()
    for row in conn.execute("SELECT * FROM training_jobs WHERE state IN ('SUBMITTED', 'STARTED')").fetchall():
        last_poll_ts = float(row["last_slurm_poll_ts"]) if row["last_slurm_poll_ts"] is not None else None
        if row["state"] == "SUBMITTED" and row["submit_ts"] is not None:
            if current_ts - float(row["submit_ts"]) >= PENDING_START_TIMEOUT_SECONDS:
                if last_poll_ts is None or (current_ts - last_poll_ts) >= PENDING_START_TIMEOUT_SECONDS:
                    stale_rows.append(row)
        elif row["state"] == "STARTED" and row["heartbeat_ts"] is not None:
            if current_ts - float(row["heartbeat_ts"]) >= HEARTBEAT_STALE_SECONDS:
                if last_poll_ts is None or (current_ts - last_poll_ts) >= HEARTBEAT_STALE_SECONDS:
                    stale_rows.append(row)
    if not stale_rows:
        return

    run_names = [row["run_name"] for row in stale_rows]
    log("Sleeping %ss before rare Slurm reconciliation for stale training jobs: %s" % (SLURM_POLL_SLEEP_SECONDS, ", ".join(run_names)))
    active, accounting = poll_slurm_jobs(run_names)
    current_ts = now_ts()
    for row in stale_rows:
        run_name = row["run_name"]
        if run_name in active:
            active_row = active[run_name]
            pool = "owners" if active_row["partition"] == OWNERS_PARTITION else "serc"
            state = "STARTED" if active_row["state"] == "RUNNING" else "SUBMITTED"
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, pool = ?, active_job_id = ?, last_slurm_poll_ts = ?
                WHERE run_name = ?
                """,
                (state, pool, active_row["job_id"], current_ts, run_name),
            )
            continue

        latest = accounting.get(run_name)
        state_text = latest["state"] if latest else "UNKNOWN"
        if row["pool"] == "serc":
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, active_job_id = NULL, end_ts = ?, last_slurm_poll_ts = ?, last_error = ?
                WHERE run_name = ?
                """,
                (FATAL_STATE, current_ts, current_ts, "stale training missing from Slurm state=%s" % state_text, run_name),
            )
            record_event(conn, "training", run_name, "failed_fatal", job_id=(latest or {}).get("job_id"), message=state_text)
        else:
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, active_job_id = NULL, end_ts = ?, last_slurm_poll_ts = ?, last_error = ?
                WHERE run_name = ?
                """,
                (RETRYABLE_STATE, current_ts, current_ts, "stale training missing from Slurm state=%s" % state_text, run_name),
            )
            record_event(conn, "training", run_name, "failed_retryable", job_id=(latest or {}).get("job_id"), message=state_text)
        for lock_path in Path(paths_global["lock_dir"]).glob("lock_*_%s.lock" % run_name):
            lock_path.unlink()


def submit_preprocess_step(conn, step, paths, args):
    if step["step_type"] == "multitask_tensor":
        step["sar_dir"].mkdir(parents=True, exist_ok=True)
        select_job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--job-name=sar_%s" % step["run_name"],
                str(REPO_ROOT / "data_processing" / "sar" / "sbatch_select_sar_sample.sh"),
                "--sample-at-sites",
                "--sample-at-random",
                "--random-seed",
                str(step["data_seed"]),
                "--vars-to-sample",
                "vv",
                "vh",
                "--output-dir",
                str(step["sar_dir"]),
                "--output-tag",
                step["data_tag"],
            ],
            universal_newlines=True,
        ).strip()
        index_job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency=afterok:%s" % select_job_id,
                "--job-name=idx_%s" % step["run_name"],
                str(REPO_ROOT / "lfmc_model" / "scripts" / "data" / "build_sample_index_longweather.sbatch"),
                "--out-path",
                str(step["sample_index"]),
                "--target-cols",
                "lfmc",
                "vv",
                "vh",
                "--random-seed",
                str(step["data_seed"]),
                "--label-source",
                "nfmd=%s" % (SCRATCH_ROOT / "nfmd" / "nfmd_processed.csv"),
                "--label-source",
                "vv_at_sites=%s" % (step["sar_dir"] / ("vv_samples_at_sites_matching_" + step["data_tag"] + ".csv")),
                "--label-source",
                "vv_at_random=%s" % (step["sar_dir"] / ("vv_samples_random_matching_" + step["data_tag"] + ".csv")),
                "--label-source",
                "vh_at_sites=%s" % (step["sar_dir"] / ("vh_samples_at_sites_matching_" + step["data_tag"] + ".csv")),
                "--label-source",
                "vh_at_random=%s" % (step["sar_dir"] / ("vh_samples_random_matching_" + step["data_tag"] + ".csv")),
                "--target-sample-n",
                "lfmc=-1",
                "--target-sample-n",
                "vv=-1",
                "--target-sample-n",
                "vh=-1",
            ],
            universal_newlines=True,
        ).strip()
        job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency=afterok:%s" % index_job_id,
                "--job-name=%s" % step["job_name"],
                str(REPO_ROOT / "lfmc_model" / "scripts" / "data" / "build_dataset_longweather_direct_single.sbatch"),
                str(step["sample_index"]),
                str(step["input_dir"]),
                "--daymet-zarr-path",
                DAYMET_ZARR_CLIM20,
                "--overwrite",
            ],
            universal_newlines=True,
        ).strip()
    elif step["step_type"] == "single_tensor":
        single_index_job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--job-name=idx_single_%s" % args.submission_tag,
                str(REPO_ROOT / "lfmc_model" / "scripts" / "data" / "build_sample_index_longweather.sbatch"),
                "--out-path",
                str(step["sample_index"]),
                "--target-cols",
                "lfmc",
                "--label-source",
                "nfmd=%s" % (SCRATCH_ROOT / "nfmd" / "nfmd_processed.csv"),
            ],
            universal_newlines=True,
        ).strip()
        job_id = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency=afterok:%s" % single_index_job_id,
                "--job-name=%s" % step["job_name"],
                str(REPO_ROOT / "lfmc_model" / "scripts" / "data" / "build_dataset_longweather_direct_single.sbatch"),
                str(step["sample_index"]),
                str(step["input_dir"]),
                "--daymet-zarr-path",
                DAYMET_ZARR_CLIM20,
                "--overwrite",
            ],
            universal_newlines=True,
        ).strip()
    else:
        first_tensor = conn.execute("SELECT job_id FROM preprocess_steps WHERE step_key = 'multitask_tensor_00'").fetchone()
        dependency = []
        if first_tensor is not None and first_tensor["job_id"]:
            dependency = ["--dependency=afterok:%s" % first_tensor["job_id"]]
        job_id = subprocess.check_output(
            ["sbatch", "--parsable"] + dependency + [
                "--job-name=%s" % step["job_name"],
                str(REPO_ROOT / "lfmc_model" / "scripts" / "training" / "generate_longweather_fold_info.sbatch"),
                "--input-data-dir",
                str(step["input_dir"]),
                "--out-path",
                str(paths["fold_info_path"]),
                "--split-seed",
                "42",
            ],
            universal_newlines=True,
        ).strip()

    conn.execute(
        """
        UPDATE preprocess_steps
        SET state = ?, job_id = ?, submit_ts = ?, last_error = NULL
        WHERE step_key = ?
        """,
        ("SUBMITTED", job_id, now_ts(), step["step_key"]),
    )
    record_event(conn, "preprocess", step["step_key"], "submitted", job_id=job_id)
    log("Submitted preprocess step %s as job %s." % (step["step_key"], job_id))


def choose_target_pool(conn):
    serc_count = conn.execute(
        "SELECT COUNT(*) AS n FROM training_jobs WHERE pool = 'serc' AND state IN ('SUBMITTED', 'STARTED')"
    ).fetchone()["n"]
    owners_count = conn.execute(
        "SELECT COUNT(*) AS n FROM training_jobs WHERE pool = 'owners' AND state IN ('SUBMITTED', 'STARTED')"
    ).fetchone()["n"]
    if int(serc_count) < SERC_GPU_MAX_JOBS:
        return "serc"
    if int(owners_count) < OWNERS_GPU_MAX_JOBS:
        return "owners"
    return None


def submit_training_job(conn, spec, pool, paths):
    state_dir = marker_base(paths, spec["run_name"])
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = Path(paths["lock_dir"]) / ("lock_%s_%s.lock" % (pool, spec["run_name"]))
    lock_file.write_text("RESERVED\n")
    export_arg = (
        "ALL,LOCK_FILE=%s,COORD_STATE_DIR=%s,COORD_RUN_NAME=%s,COORD_RUN_TAG=%s,COORD_POOL=%s"
        % (str(lock_file), str(state_dir), spec["run_name"], spec["run_tag"], pool)
    )
    if pool == "serc":
        gpu_args = ["--partition=%s" % SERC_PARTITION]
    else:
        gpu_args = ["--partition=%s" % OWNERS_PARTITION, "--constraint=%s" % OWNERS_GPU_CONSTRAINT]
    cmd = (
        ["sbatch"]
        + gpu_args
        + [
            "--parsable",
            "--export=%s" % export_arg,
            "--job-name=%s" % spec["run_name"],
            str(SCRIPT_DIR / "train_job_longweather.sbatch"),
            "--input_data_dir",
            spec["input_dir"],
            "--save_dir",
            spec["save_root"],
            "--batch_size",
            BATCH_SIZE,
            "--lr",
            LR,
            "--val_split",
            VAL_SPLIT,
            "--adam_wd",
            ADAM_WD,
            "--warmup_epochs",
            WARMUP_EPOCHS,
            "--scheduler_t_max",
            SCHEDULER_T_MAX,
            "--model_family",
            MODEL_FAMILY,
            "--d_model",
            D_MODEL,
            "--nhead",
            NHEAD,
            "--num_layers",
            NUM_LAYERS,
            "--dim_feedforward",
            DIM_FEEDFORWARD,
            "--dropout",
            DROPOUT,
            "--long_d_model",
            LONG_D_MODEL,
            "--long_nhead",
            LONG_NHEAD,
            "--long_num_layers",
            LONG_NUM_LAYERS,
            "--long_dim_feedforward",
            LONG_DIM_FEEDFORWARD,
            "--long_out_dim",
            LONG_OUT_DIM,
            "--weather_kernel_size",
            WEATHER_KERNEL_SIZE,
            "--weather_d_model",
            WEATHER_D_MODEL,
            "--weather_max_dilation",
            WEATHER_MAX_DILATION,
            "--modis_d_model",
            MODIS_D_MODEL,
            "--static_d_model",
            STATIC_D_MODEL,
            "--common_d_model",
            COMMON_D_MODEL,
            "--shared_latent_dim",
            SHARED_LATENT_DIM,
            "--lfmc_private_dim",
            LFMC_PRIVATE_DIM,
            "--sar_private_dim",
            SAR_PRIVATE_DIM,
            "--num_tasks",
            spec["num_tasks"],
            "--task_weight_type",
            spec["weighting_type"],
            "--manual_task_weights",
        ]
        + list(spec["task_weights"])
        + [
            "--seed",
            spec["seed"],
            "--batch_seed",
            spec["seed"],
            "--split_seed",
            "42",
            "--fold_info_in",
            str(paths["fold_info_path"]),
            "--run_tag",
            spec["run_tag"],
            "--overwrite",
        ]
    )
    try:
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
        job_id = output.split(";")[0].strip()
        write_submitted_marker(paths, spec, pool, job_id)
        lock_file.write_text(job_id + "\n")
        conn.execute(
            """
            UPDATE training_jobs
            SET state = ?, pool = ?, active_job_id = ?, submit_ts = ?, start_ts = NULL, end_ts = NULL,
                heartbeat_ts = NULL, last_error = NULL, attempt_count = attempt_count + 1
            WHERE run_name = ?
            """,
            ("SUBMITTED", pool, job_id, now_ts(), spec["run_name"]),
        )
        record_event(conn, "training", spec["run_name"], "submitted", job_id=job_id, message="pool=%s" % pool)
        log("Submitted training job %s as job %s on pool=%s." % (spec["run_name"], job_id, pool))
        return True
    except Exception as exc:
        if lock_file.exists():
            lock_file.unlink()
        conn.execute(
            "UPDATE training_jobs SET last_error = ? WHERE run_name = ?",
            (str(exc), spec["run_name"]),
        )
        record_event(conn, "training", spec["run_name"], "submit_failed", message=str(exc))
        conn.commit()
        log("Submit failed for %s: %s" % (spec["run_name"], exc))
        log("Waiting %ss before Slurm reconciliation for possible ambiguous submit." % SLURM_POLL_SLEEP_SECONDS)
        active, _ = poll_slurm_jobs([spec["run_name"]])
        if spec["run_name"] in active:
            recovered = active[spec["run_name"]]
            recovered_pool = "owners" if recovered["partition"] == OWNERS_PARTITION else "serc"
            recovered_job_id = recovered["job_id"]
            write_submitted_marker(paths, spec, recovered_pool, recovered_job_id)
            recovered_lock = Path(paths["lock_dir"]) / ("lock_%s_%s.lock" % (recovered_pool, spec["run_name"]))
            recovered_lock.write_text(recovered_job_id + "\n")
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, pool = ?, active_job_id = ?, submit_ts = ?, last_error = ?, attempt_count = attempt_count + 1
                WHERE run_name = ?
                """,
                ("SUBMITTED", recovered_pool, recovered_job_id, now_ts(), "Recovered from ambiguous submit: %s" % exc, spec["run_name"]),
            )
            record_event(conn, "training", spec["run_name"], "reattached_after_submit_failure", job_id=recovered_job_id, message="pool=%s" % recovered_pool)
            log("Reattached %s to existing job %s on pool=%s after ambiguous submit." % (spec["run_name"], recovered_job_id, recovered_pool))
            return True
        return False


def training_prerequisites_ready(spec, paths):
    return validate_input_dir(spec["input_dir"]) and Path(paths["fold_info_path"]).exists()


def completed_training_count(conn):
    return int(conn.execute("SELECT COUNT(*) AS n FROM training_jobs WHERE state = ?", (COMPLETE_STATE,)).fetchone()["n"])


def all_training_complete(conn):
    return int(conn.execute("SELECT COUNT(*) AS n FROM training_jobs WHERE state != ?", (COMPLETE_STATE,)).fetchone()["n"]) == 0


def ensure_submitted_markers_for_active_training(conn, specs, paths):
    spec_by_name = {spec["run_name"]: spec for spec in specs}
    for row in conn.execute("SELECT * FROM training_jobs WHERE state IN ('SUBMITTED', 'STARTED')").fetchall():
        marker = marker_path(paths, row["run_name"], "submitted")
        if marker.exists() or not row["active_job_id"]:
            continue
        spec = spec_by_name[row["run_name"]]
        write_submitted_marker(paths, spec, row["pool"] or "unknown", row["active_job_id"])


def validate_existing_inputs(args, paths, multitask_specs):
    if args.build_tensors:
        return
    if not Path(paths["fold_info_path"]).exists():
        raise RuntimeError("Expected fold file does not exist: %s" % paths["fold_info_path"])
    if args.train_multitask:
        for spec in multitask_specs:
            if not validate_input_dir(spec["input_dir"]):
                raise RuntimeError("Tensor directory invalid: %s" % spec["input_dir"])
    if args.train_single and not validate_input_dir(paths["single_input_dir"]):
        raise RuntimeError("Tensor directory invalid: %s" % paths["single_input_dir"])


def main():
    args = parse_args()
    global paths_global
    paths = build_paths(args)
    paths_global = paths
    ensure_dirs(paths)

    multitask_specs, single_specs = build_training_specs(args, paths)
    validate_existing_inputs(args, paths, multitask_specs)

    lock_fh = acquire_lock(paths["coordinator_lock_path"])
    conn = connect_db(paths["db_path"])

    specs = []
    if args.train_multitask:
        specs.extend(multitask_specs)
    if args.train_single:
        specs.extend(single_specs)
    upsert_training_specs(conn, specs)

    preprocess_defs = build_preprocess_defs(args, paths)
    sync_preprocess_defs(conn, preprocess_defs)
    ensure_submitted_markers_for_active_training(conn, specs, paths)

    log(
        "Starting training coordinator with build_tensors=%d train_multitask=%d train_single=%d "
        "serc_cap=%d owners_cap=%d."
        % (int(args.build_tensors), int(args.train_multitask), int(args.train_single), SERC_GPU_MAX_JOBS, OWNERS_GPU_MAX_JOBS)
    )

    try:
        while True:
            progress = False

            if args.build_tensors:
                refresh_preprocess_states(conn, preprocess_defs)
                maybe_reconcile_stale_preprocess(conn, preprocess_defs)
                refresh_preprocess_states(conn, preprocess_defs)

            refresh_training_states(conn, specs, paths)
            maybe_reconcile_stale_training(conn)
            refresh_training_states(conn, specs, paths)
            conn.commit()

            fatal_pre = conn.execute("SELECT * FROM preprocess_steps WHERE state = ?", (FATAL_STATE,)).fetchone()
            if fatal_pre is not None:
                raise RuntimeError("Fatal preprocessing failure for %s: %s" % (fatal_pre["step_key"], fatal_pre["last_error"]))
            fatal_train = conn.execute("SELECT * FROM training_jobs WHERE state = ?", (FATAL_STATE,)).fetchone()
            if fatal_train is not None:
                raise RuntimeError("Fatal training failure for %s: %s" % (fatal_train["run_name"], fatal_train["last_error"]))

            if args.build_tensors:
                for step in preprocess_defs:
                    row = conn.execute("SELECT * FROM preprocess_steps WHERE step_key = ?", (step["step_key"],)).fetchone()
                    if row["state"] in (COMPLETE_STATE, "SUBMITTED"):
                        continue
                    submit_preprocess_step(conn, step, paths, args)
                    conn.commit()
                    progress = True
                    time.sleep(SUBMIT_SLEEP_SECONDS)

            for spec in specs:
                row = conn.execute("SELECT * FROM training_jobs WHERE run_name = ?", (spec["run_name"],)).fetchone()
                if row["state"] in RUNNING_STATES or row["state"] == COMPLETE_STATE:
                    continue
                if not training_prerequisites_ready(spec, paths):
                    continue
                pool = choose_target_pool(conn)
                if pool is None:
                    break
                submit_training_job(conn, spec, pool, paths)
                conn.commit()
                progress = True
                time.sleep(SUBMIT_SLEEP_SECONDS)

            completed = completed_training_count(conn)
            active_serc = conn.execute(
                "SELECT COUNT(*) AS n FROM training_jobs WHERE pool = 'serc' AND state IN ('SUBMITTED', 'STARTED')"
            ).fetchone()["n"]
            active_owners = conn.execute(
                "SELECT COUNT(*) AS n FROM training_jobs WHERE pool = 'owners' AND state IN ('SUBMITTED', 'STARTED')"
            ).fetchone()["n"]
            log(
                "Coordinator status: completed=%d/%d active_serc=%d/%d active_owners=%d/%d."
                % (completed, len(specs), int(active_serc), SERC_GPU_MAX_JOBS, int(active_owners), OWNERS_GPU_MAX_JOBS)
            )

            if all_training_complete(conn):
                preprocess_done = True
                if args.build_tensors:
                    preprocess_done = all(
                        conn.execute("SELECT state FROM preprocess_steps WHERE step_key = ?", (step["step_key"],)).fetchone()["state"] == COMPLETE_STATE
                        for step in preprocess_defs
                    )
                if preprocess_done:
                    log("Completed all requested training jobs.")
                    break

            if args.once:
                break

            if not progress:
                log("No coordinator progress this round. Sleeping for %ss." % POLL_SECONDS)
                time.sleep(POLL_SECONDS)
    finally:
        conn.commit()
        conn.close()
        lock_fh.close()


if __name__ == "__main__":
    main()
