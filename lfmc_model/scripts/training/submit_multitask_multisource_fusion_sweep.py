#!/usr/bin/env python3

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc")
TENSOR_TAG = "dm128_vpd_anoms_nozone"
SUBMISSION_TAG = "multisource_fusion_sweep_vpd_anoms_nozone"

MULTITASK_INPUT_DIR = SCRATCH_ROOT / "lfmc_model" / "inputs" / "ensemble" / ("lfmc_vh_vv_365_" + TENSOR_TAG) / "lfmc_vh_vv_ds1000"
MULTITASK_SAVE_ROOT = SCRATCH_ROOT / "lfmc_model" / "outputs" / ("lfmc_vh_vv_365_" + SUBMISSION_TAG)
FOLD_INFO_PATH = SCRATCH_ROOT / "lfmc_model" / "outputs" / "shared_training" / ("canonical_fold_info_" + TENSOR_TAG + ".json")
LOCK_DIR = SCRATCH_ROOT / "lfmc_model" / "gpu_locks"
STATE_ROOT = MULTITASK_SAVE_ROOT / "_coordinator_state"
DB_PATH = STATE_ROOT / "sweep_state.sqlite"
COORDINATOR_LOCK_PATH = LOCK_DIR / ("sweep_scheduler_" + SUBMISSION_TAG + ".lock")

REQUIRED_TENSOR_FILES = ["X_short.pt", "X_long.pt", "X_static.pt", "Y.pt", "source.pt", "stratifier.npy", "info.csv"]

SERC_GPU_MAX_JOBS = 8
OWNERS_GPU_MAX_JOBS = 50
SERC_PARTITION = "serc"
OWNERS_PARTITION = "owners"
OWNERS_GPU_CONSTRAINT = "GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S|GPU_SKU:RTX_2080Ti|GPU_SKU:RTX_3090|GPU_SKU:V100S_PCIE"

POLL_SECONDS = 60
SUBMIT_SLEEP_SECONDS = 5
SLURM_POLL_SLEEP_SECONDS = 60
PENDING_START_TIMEOUT_SECONDS = 45 * 60
HEARTBEAT_INTERVAL_SECONDS = 5 * 60
HEARTBEAT_STALE_SECONDS = 15 * 60
SLURM_RECHECK_MIN_SECONDS = 15 * 60

BATCH_SIZE = 128
VAL_SPLIT = 0.15
ADAM_WD = "1e-4"
DROPOUT = "0.15"
WARMUP_EPOCHS = 2
SCHEDULER_T_MAX = 40

MODEL_FAMILY = "multisource_fusion"

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256

LONG_D_MODEL = 256
LONG_NHEAD = 8
LONG_NUM_LAYERS = 3
LONG_DIM_FEEDFORWARD = 512
LONG_OUT_DIM = 128

MULTITASK_NUM_TASKS = 3
MULTITASK_WEIGHTING_TYPE = "manual"

MODEL_SEED = 1000
SPLIT_SEED = 42

K_SIZES = [3, 5]
D_WEATHER_VALUES = [64, 128]
D_MODIS_VALUES = [64, 128]
D_STATIC_VALUES = [32, 64]
D_COMMON_VALUES = [64, 128]
WEATHER_MAX_DILATION_VALUES = [32, 64]
LR_VALUES = ["1e-4", "5e-4"]
FIRST_TASK_WEIGHT_VALUES = [1.0, 3.0, 5.0]

RUNNING_STATES = {"SUBMITTED", "STARTED"}
RETRYABLE_STATES = {"FAILED_RETRYABLE"}
COMPLETE_STATE = "COMPLETE"
FATAL_STATE = "FAILED_FATAL"


def now_ts():
    return time.time()


def utc_now_iso():
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def marker_path(run_name, marker_name):
    return STATE_ROOT / run_name / (marker_name + ".json")


def output_run_dirs_for_tag(run_tag):
    return sorted(MULTITASK_SAVE_ROOT.glob("*_" + run_tag))


def run_outputs_complete(run_tag):
    run_dirs = [d for d in output_run_dirs_for_tag(run_tag) if d.is_dir()]
    if not run_dirs:
        return False
    run_dir = run_dirs[0]
    if not (run_dir / "fold_info.json").exists():
        return False
    for fold_idx in range(1, 7):
        fold_dir = run_dir / ("fold_%d" % fold_idx)
        if not (fold_dir / "test_info.csv").exists():
            return False
        if not (fold_dir / "test_outputs.pth").exists():
            return False
    return True


def cleanup_stale_worker_markers(run_name):
    for marker_name in ["started", "heartbeat", "completed", "failed"]:
        marker = marker_path(run_name, marker_name)
        if marker.exists():
            marker.unlink()


def ensure_dirs():
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    MULTITASK_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)


def validate_inputs():
    if not MULTITASK_INPUT_DIR.exists():
        raise RuntimeError("Tensor directory missing: %s" % MULTITASK_INPUT_DIR)
    for required_file in REQUIRED_TENSOR_FILES:
        if not (MULTITASK_INPUT_DIR / required_file).exists():
            raise RuntimeError("Missing %s in tensor directory: %s" % (required_file, MULTITASK_INPUT_DIR))
    if not FOLD_INFO_PATH.exists():
        raise RuntimeError("Expected fold file does not exist: %s" % FOLD_INFO_PATH)


def acquire_coordinator_lock():
    lock_fh = COORDINATOR_LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        lock_fh.seek(0)
        existing = lock_fh.read().strip()
        raise RuntimeError("Sweep coordinator appears to already be running: %s pid=%s" % (COORDINATOR_LOCK_PATH, existing))
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(str(os.getpid()) + "\n")
    lock_fh.flush()
    return lock_fh


def connect_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS configs (
            config_idx INTEGER PRIMARY KEY,
            run_name TEXT UNIQUE NOT NULL,
            run_tag TEXT UNIQUE NOT NULL,
            params_json TEXT NOT NULL,
            state TEXT NOT NULL,
            pool TEXT,
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
            config_idx INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            job_id TEXT,
            message TEXT,
            FOREIGN KEY(config_idx) REFERENCES configs(config_idx)
        )
        """
    )
    conn.commit()
    return conn


def record_event(conn, config_idx, event_type, job_id=None, message=None):
    conn.execute(
        """
        INSERT INTO events (event_ts, config_idx, event_type, job_id, message)
        VALUES (?, ?, ?, ?, ?)
        """,
        (now_ts(), config_idx, event_type, job_id, message),
    )


def build_configs():
    configs = []
    config_idx = 0
    for k_size in K_SIZES:
        for d_weather in D_WEATHER_VALUES:
            for d_modis in D_MODIS_VALUES:
                for d_static in D_STATIC_VALUES:
                    for d_common in D_COMMON_VALUES:
                        for weather_max_dilation in WEATHER_MAX_DILATION_VALUES:
                            for lr in LR_VALUES:
                                for first_task_weight in FIRST_TASK_WEIGHT_VALUES:
                                    run_tag = "sweep_k%s_dw%s_dm%s_ds%s_dc%s_wdil%s_lr%s_tw%s" % (
                                        k_size,
                                        d_weather,
                                        d_modis,
                                        d_static,
                                        d_common,
                                        weather_max_dilation,
                                        lr,
                                        first_task_weight,
                                    )
                                    run_name = "sweep_%03d_%s" % (config_idx, SUBMISSION_TAG)
                                    params = {
                                        "K": k_size,
                                        "Dw": d_weather,
                                        "Dm": d_modis,
                                        "Ds": d_static,
                                        "Dc": d_common,
                                        "Wdil": weather_max_dilation,
                                        "LR": lr,
                                        "TW": first_task_weight,
                                    }
                                    configs.append(
                                        {
                                            "config_idx": config_idx,
                                            "run_name": run_name,
                                            "run_tag": run_tag,
                                            "params": params,
                                        }
                                    )
                                    config_idx += 1
    return configs


def upsert_configs(conn, configs):
    for cfg in configs:
        conn.execute(
            """
            INSERT INTO configs (config_idx, run_name, run_tag, params_json, state)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(config_idx) DO UPDATE SET
                run_name=excluded.run_name,
                run_tag=excluded.run_tag,
                params_json=excluded.params_json
            """,
            (
                cfg["config_idx"],
                cfg["run_name"],
                cfg["run_tag"],
                json.dumps(cfg["params"], sort_keys=True),
                "NEW",
            ),
        )
    conn.commit()


def get_all_configs(conn):
    return conn.execute("SELECT * FROM configs ORDER BY config_idx").fetchall()


def update_config_state(conn, config_idx, state=None, pool=None, active_job_id=None, submit_ts=None,
                        start_ts=None, end_ts=None, heartbeat_ts=None, last_slurm_poll_ts=None,
                        last_error=None, attempt_increment=False):
    current = conn.execute("SELECT * FROM configs WHERE config_idx = ?", (config_idx,)).fetchone()
    updates = {
        "state": state if state is not None else current["state"],
        "pool": pool if pool is not None else current["pool"],
        "active_job_id": active_job_id if active_job_id is not None else current["active_job_id"],
        "submit_ts": submit_ts if submit_ts is not None else current["submit_ts"],
        "start_ts": start_ts if start_ts is not None else current["start_ts"],
        "end_ts": end_ts if end_ts is not None else current["end_ts"],
        "heartbeat_ts": heartbeat_ts if heartbeat_ts is not None else current["heartbeat_ts"],
        "last_slurm_poll_ts": last_slurm_poll_ts if last_slurm_poll_ts is not None else current["last_slurm_poll_ts"],
        "last_error": last_error if last_error is not None else current["last_error"],
        "attempt_count": int(current["attempt_count"]) + (1 if attempt_increment else 0),
    }
    conn.execute(
        """
        UPDATE configs
        SET state = ?, pool = ?, active_job_id = ?, submit_ts = ?, start_ts = ?, end_ts = ?,
            heartbeat_ts = ?, last_slurm_poll_ts = ?, last_error = ?, attempt_count = ?
        WHERE config_idx = ?
        """,
        (
            updates["state"],
            updates["pool"],
            updates["active_job_id"],
            updates["submit_ts"],
            updates["start_ts"],
            updates["end_ts"],
            updates["heartbeat_ts"],
            updates["last_slurm_poll_ts"],
            updates["last_error"],
            updates["attempt_count"],
            config_idx,
        ),
    )


def count_active_by_pool(conn, pool):
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM configs WHERE pool = ? AND state IN (?, ?)",
        (pool, "SUBMITTED", "STARTED"),
    ).fetchone()
    return int(row["n"])


def choose_target_pool(conn):
    if count_active_by_pool(conn, "serc") < SERC_GPU_MAX_JOBS:
        return "serc"
    if OWNERS_GPU_MAX_JOBS > 0 and count_active_by_pool(conn, "owners") < OWNERS_GPU_MAX_JOBS:
        return "owners"
    return None


def sbatch_args_for_config(cfg, pool, state_dir, lock_file):
    params = cfg["params"]
    task_weights = [str(params["TW"]), "1.0", "1.0"]
    if pool == "serc":
        gpu_args = ["--partition=%s" % SERC_PARTITION]
    else:
        gpu_args = ["--partition=%s" % OWNERS_PARTITION, "--constraint=%s" % OWNERS_GPU_CONSTRAINT]
    return (
        ["sbatch"]
        + gpu_args
        + [
            "--parsable",
            "--export=ALL,LOCK_FILE=%s,COORD_STATE_DIR=%s,COORD_RUN_NAME=%s,COORD_RUN_TAG=%s,COORD_POOL=%s"
            % (str(lock_file), str(state_dir), cfg["run_name"], cfg["run_tag"], pool),
            "--job-name=%s" % cfg["run_name"],
            "train_job_longweather.sbatch",
            "--input_data_dir", str(MULTITASK_INPUT_DIR),
            "--save_dir", str(MULTITASK_SAVE_ROOT),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(params["LR"]),
            "--val_split", str(VAL_SPLIT),
            "--adam_wd", str(ADAM_WD),
            "--warmup_epochs", str(WARMUP_EPOCHS),
            "--scheduler_t_max", str(SCHEDULER_T_MAX),
            "--model_family", MODEL_FAMILY,
            "--d_model", str(D_MODEL),
            "--nhead", str(NHEAD),
            "--num_layers", str(NUM_LAYERS),
            "--dim_feedforward", str(DIM_FEEDFORWARD),
            "--dropout", str(DROPOUT),
            "--long_d_model", str(LONG_D_MODEL),
            "--long_nhead", str(LONG_NHEAD),
            "--long_num_layers", str(LONG_NUM_LAYERS),
            "--long_dim_feedforward", str(LONG_DIM_FEEDFORWARD),
            "--long_out_dim", str(LONG_OUT_DIM),
            "--weather_kernel_size", str(params["K"]),
            "--weather_d_model", str(params["Dw"]),
            "--weather_max_dilation", str(params["Wdil"]),
            "--modis_d_model", str(params["Dm"]),
            "--static_d_model", str(params["Ds"]),
            "--common_d_model", str(params["Dc"]),
            "--num_tasks", str(MULTITASK_NUM_TASKS),
            "--task_weight_type", MULTITASK_WEIGHTING_TYPE,
            "--manual_task_weights",
        ]
        + task_weights
        + [
            "--seed", str(MODEL_SEED),
            "--batch_seed", str(MODEL_SEED),
            "--split_seed", str(SPLIT_SEED),
            "--fold_info_in", str(FOLD_INFO_PATH),
            "--run_tag", cfg["run_tag"],
            "--skip_final_all_data_fold",
            "--overwrite",
        ]
    )


def parse_job_id_from_sbatch_output(output):
    text = output.strip()
    if not text:
        return None
    token = text.splitlines()[-1].split(";")[0].strip()
    return token if token.isdigit() else None


def write_submitted_marker(cfg, pool, job_id):
    cleanup_stale_worker_markers(cfg["run_name"])
    payload = {
        "timestamp_utc": utc_now_iso(),
        "job_id": str(job_id),
        "pool": pool,
        "run_name": cfg["run_name"],
        "run_tag": cfg["run_tag"],
        "attempt_time_unix": now_ts(),
    }
    write_json_atomic(marker_path(cfg["run_name"], "submitted"), payload)


def slurm_poll_snapshot():
    time.sleep(SLURM_POLL_SLEEP_SECONDS)
    active_by_name = {}
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
        job_id, run_name, state, partition = parts
        if "multisource_fusion_sweep" not in run_name:
            continue
        active_by_name.setdefault(run_name, []).append(
            {"job_id": job_id, "state": state, "partition": partition}
        )

    account_by_name = {}
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
        job_id, run_name, state = parts[0], parts[1], parts[2]
        if "multisource_fusion_sweep" not in run_name:
            continue
        account_by_name.setdefault(run_name, []).append(
            {"job_id": job_id, "state": state}
        )
    return active_by_name, account_by_name


def choose_keeper(rows):
    running_rows = [row for row in rows if row["state"] == "RUNNING"]
    if running_rows:
        return sorted(running_rows, key=lambda row: int(row["job_id"]))[-1]
    return sorted(rows, key=lambda row: int(row["job_id"]))[-1]


def dedupe_active_rows(run_name, rows):
    if len(rows) <= 1:
        return rows[0] if rows else None
    keeper = choose_keeper(rows)
    cancel_ids = [row["job_id"] for row in rows if row["job_id"] != keeper["job_id"]]
    if cancel_ids:
        log("Cancelling duplicate active jobs for %s: %s (keeping %s)." % (run_name, " ".join(cancel_ids), keeper["job_id"]))
        subprocess.call(["scancel"] + cancel_ids)
    return keeper


def submit_config(conn, cfg, pool):
    state_dir = STATE_ROOT / cfg["run_name"]
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / ("lock_%s_%s.lock" % (pool, cfg["run_name"]))
    write_json_atomic(lock_file, {"reserved": True, "run_name": cfg["run_name"], "pool": pool})
    cmd = sbatch_args_for_config(cfg, pool, state_dir, lock_file)
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        job_id = parse_job_id_from_sbatch_output(completed.stdout)
        if job_id is None:
            raise RuntimeError("sbatch returned unparsable output: %r" % completed.stdout)
        write_submitted_marker(cfg, pool, job_id)
        lock_file.write_text(str(job_id) + "\n")
        update_config_state(
            conn,
            cfg["config_idx"],
            state="SUBMITTED",
            pool=pool,
            active_job_id=job_id,
            submit_ts=now_ts(),
            start_ts=None,
            end_ts=None,
            heartbeat_ts=None,
            last_error=None,
            attempt_increment=True,
        )
        record_event(conn, cfg["config_idx"], "submitted", job_id=job_id, message="pool=%s" % pool)
        conn.commit()
        log("Submitted %s (%s) as job %s on pool=%s." % (cfg["run_name"], cfg["run_tag"], job_id, pool))
        return True
    except Exception as exc:
        if lock_file.exists():
            lock_file.unlink()
        update_config_state(conn, cfg["config_idx"], state="NEW", active_job_id=None, pool=None, last_error=str(exc))
        record_event(conn, cfg["config_idx"], "submit_failed", message=str(exc))
        conn.commit()
        log("Submit failed for %s (%s): %s" % (cfg["run_name"], cfg["run_tag"], exc))
        log("Waiting %ss before Slurm reconciliation for possible ambiguous submit." % SLURM_POLL_SLEEP_SECONDS)
        active_by_name, _ = slurm_poll_snapshot()
        rows = active_by_name.get(cfg["run_name"], [])
        if rows:
            keeper = dedupe_active_rows(cfg["run_name"], rows)
            pool = "owners" if keeper["partition"] == OWNERS_PARTITION else "serc"
            job_id = keeper["job_id"]
            write_submitted_marker(cfg, pool, job_id)
            lock_file = LOCK_DIR / ("lock_%s_%s.lock" % (pool, cfg["run_name"]))
            lock_file.write_text(str(job_id) + "\n")
            update_config_state(
                conn,
                cfg["config_idx"],
                state="SUBMITTED",
                pool=pool,
                active_job_id=job_id,
                submit_ts=now_ts(),
                last_error="Recovered from ambiguous submit: %s" % exc,
            )
            record_event(conn, cfg["config_idx"], "reattached_after_submit_failure", job_id=job_id, message="pool=%s" % pool)
            conn.commit()
            log("Reattached %s to existing job %s on pool=%s after ambiguous submit." % (cfg["run_name"], job_id, pool))
            return True
        log("No matching live job found for %s after submit failure." % cfg["run_name"])
        return False


def refresh_config_from_files(conn, cfg):
    state_dir = STATE_ROOT / cfg["run_name"]
    submitted = read_json(state_dir / "submitted.json")
    started = read_json(state_dir / "started.json")
    heartbeat = read_json(state_dir / "heartbeat.json")
    completed = read_json(state_dir / "completed.json")
    failed = read_json(state_dir / "failed.json")
    current = conn.execute("SELECT * FROM configs WHERE config_idx = ?", (cfg["config_idx"],)).fetchone()

    if run_outputs_complete(cfg["run_tag"]):
        if current["state"] != COMPLETE_STATE:
            update_config_state(
                conn,
                cfg["config_idx"],
                state=COMPLETE_STATE,
                active_job_id=None,
                end_ts=now_ts(),
                last_error=None,
            )
            record_event(conn, cfg["config_idx"], "complete", job_id=current["active_job_id"], message="validated folds 1-6")
        lock_glob = str(LOCK_DIR / ("lock_*_%s.lock" % cfg["run_name"]))
        for lock_path in Path(LOCK_DIR).glob("lock_*_%s.lock" % cfg["run_name"]):
            lock_path.unlink()
        return

    if completed is not None:
        update_config_state(conn, cfg["config_idx"], state="STARTED", end_ts=None)

    if failed is not None:
        pool = current["pool"]
        marker_job_id = str(failed.get("job_id", "")) or current["active_job_id"]
        last_error = "worker_failed exit_code=%s" % failed.get("exit_code")
        if pool == "serc":
            update_config_state(
                conn,
                cfg["config_idx"],
                state=FATAL_STATE,
                active_job_id=None,
                end_ts=now_ts(),
                last_error=last_error,
            )
            record_event(conn, cfg["config_idx"], "failed_fatal", job_id=marker_job_id, message=last_error)
        else:
            update_config_state(
                conn,
                cfg["config_idx"],
                state="FAILED_RETRYABLE",
                active_job_id=None,
                end_ts=now_ts(),
                last_error=last_error,
            )
            record_event(conn, cfg["config_idx"], "failed_retryable", job_id=marker_job_id, message=last_error)
        for lock_path in Path(LOCK_DIR).glob("lock_*_%s.lock" % cfg["run_name"]):
            lock_path.unlink()
        return

    if started is not None:
        start_ts = float(started.get("timestamp_unix", current["start_ts"] or now_ts()))
        heartbeat_ts = start_ts
        if heartbeat is not None:
            heartbeat_ts = float(heartbeat.get("timestamp_unix", heartbeat_ts))
        update_config_state(
            conn,
            cfg["config_idx"],
            state="STARTED",
            start_ts=start_ts,
            heartbeat_ts=heartbeat_ts,
        )
        return

    if submitted is not None:
        submit_ts = float(submitted.get("attempt_time_unix", current["submit_ts"] or now_ts()))
        marker_job_id = str(submitted.get("job_id", "")) or current["active_job_id"]
        update_config_state(
            conn,
            cfg["config_idx"],
            state="SUBMITTED",
            active_job_id=marker_job_id,
            submit_ts=submit_ts,
        )
        return

    if current["state"] not in (COMPLETE_STATE, FATAL_STATE):
        update_config_state(conn, cfg["config_idx"], state="NEW", active_job_id=None)


def stale_configs_requiring_slurm(conn):
    stale = []
    current_ts = now_ts()
    for cfg in get_all_configs(conn):
        if cfg["state"] == "SUBMITTED" and cfg["submit_ts"] is not None:
            if current_ts - float(cfg["submit_ts"]) >= PENDING_START_TIMEOUT_SECONDS:
                if cfg["last_slurm_poll_ts"] is None or (current_ts - float(cfg["last_slurm_poll_ts"])) >= SLURM_RECHECK_MIN_SECONDS:
                    stale.append(cfg)
        elif cfg["state"] == "STARTED" and cfg["heartbeat_ts"] is not None:
            if current_ts - float(cfg["heartbeat_ts"]) >= HEARTBEAT_STALE_SECONDS:
                if cfg["last_slurm_poll_ts"] is None or (current_ts - float(cfg["last_slurm_poll_ts"])) >= SLURM_RECHECK_MIN_SECONDS:
                    stale.append(cfg)
    return stale


def reconcile_stale_configs_with_slurm(conn, stale_rows):
    if not stale_rows:
        return
    names = [row["run_name"] for row in stale_rows]
    log("Sleeping %ss before rare Slurm reconciliation for stale configs: %s" % (SLURM_POLL_SLEEP_SECONDS, ", ".join(names)))
    active_by_name, account_by_name = slurm_poll_snapshot()
    poll_ts = now_ts()
    for row in stale_rows:
        run_name = row["run_name"]
        rows = active_by_name.get(run_name, [])
        if rows:
            keeper = dedupe_active_rows(run_name, rows)
            pool = "owners" if keeper["partition"] == OWNERS_PARTITION else "serc"
            update_kwargs = {
                "config_idx": row["config_idx"],
                "pool": pool,
                "active_job_id": keeper["job_id"],
                "last_slurm_poll_ts": poll_ts,
            }
            if keeper["state"] == "RUNNING":
                update_kwargs["state"] = "STARTED"
            else:
                update_kwargs["state"] = "SUBMITTED"
            update_config_state(conn, **update_kwargs)
            continue

        accounting_rows = account_by_name.get(run_name, [])
        latest_state = None
        latest_job_id = None
        if accounting_rows:
            latest = sorted(accounting_rows, key=lambda item: int(item["job_id"]))[-1]
            latest_job_id = latest["job_id"]
            latest_state = latest["state"]

        pool = row["pool"]
        if pool == "serc":
            update_config_state(
                conn,
                row["config_idx"],
                state=FATAL_STATE,
                active_job_id=None,
                end_ts=now_ts(),
                last_slurm_poll_ts=poll_ts,
                last_error="stale job missing from Slurm state=%s" % (latest_state or "UNKNOWN"),
            )
            record_event(conn, row["config_idx"], "failed_fatal", job_id=latest_job_id, message="Slurm reconcile state=%s" % (latest_state or "UNKNOWN"))
        else:
            update_config_state(
                conn,
                row["config_idx"],
                state="FAILED_RETRYABLE",
                active_job_id=None,
                end_ts=now_ts(),
                last_slurm_poll_ts=poll_ts,
                last_error="stale job missing from Slurm state=%s" % (latest_state or "UNKNOWN"),
            )
            record_event(conn, row["config_idx"], "failed_retryable", job_id=latest_job_id, message="Slurm reconcile state=%s" % (latest_state or "UNKNOWN"))
        for lock_path in Path(LOCK_DIR).glob("lock_*_%s.lock" % run_name):
            lock_path.unlink()


def all_complete(conn):
    row = conn.execute("SELECT COUNT(*) AS n FROM configs WHERE state != ?", (COMPLETE_STATE,)).fetchone()
    return int(row["n"]) == 0


def completed_count(conn):
    row = conn.execute("SELECT COUNT(*) AS n FROM configs WHERE state = ?", (COMPLETE_STATE,)).fetchone()
    return int(row["n"])


def next_submittable_configs(conn):
    rows = conn.execute(
        """
        SELECT * FROM configs
        WHERE state IN ('NEW', 'FAILED_RETRYABLE')
        ORDER BY config_idx
        """
    ).fetchall()
    return rows


def ensure_submitted_markers_for_active_configs(conn):
    for row in conn.execute("SELECT * FROM configs WHERE state IN ('SUBMITTED', 'STARTED')").fetchall():
        submitted_path = marker_path(row["run_name"], "submitted")
        if not submitted_path.exists() and row["active_job_id"]:
            write_submitted_marker(
                {"run_name": row["run_name"], "run_tag": row["run_tag"]},
                row["pool"] or "unknown",
                row["active_job_id"],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single reconciliation/submission pass and exit.")
    args = parser.parse_args()

    ensure_dirs()
    validate_inputs()
    lock_fh = acquire_coordinator_lock()
    conn = connect_db()
    configs = build_configs()
    config_by_idx = {cfg["config_idx"]: cfg for cfg in configs}
    upsert_configs(conn, configs)
    ensure_submitted_markers_for_active_configs(conn)
    log("Starting Python sweep coordinator for %d configs with serc_cap=%d owners_cap=%d." % (len(configs), SERC_GPU_MAX_JOBS, OWNERS_GPU_MAX_JOBS))

    try:
        while True:
            for cfg in configs:
                refresh_config_from_files(conn, cfg)
            conn.commit()

            stale_rows = stale_configs_requiring_slurm(conn)
            if stale_rows:
                reconcile_stale_configs_with_slurm(conn, stale_rows)
                conn.commit()
                for cfg in configs:
                    refresh_config_from_files(conn, cfg)
                conn.commit()

            fatal_rows = conn.execute("SELECT * FROM configs WHERE state = ?", (FATAL_STATE,)).fetchall()
            if fatal_rows:
                first = fatal_rows[0]
                raise RuntimeError("Fatal serc failure for %s: %s" % (first["run_name"], first["last_error"]))

            completed = completed_count(conn)
            log("Coordinator status: completed=%d/%d active_serc=%d/%d active_owners=%d/%d" % (
                completed,
                len(configs),
                count_active_by_pool(conn, "serc"),
                SERC_GPU_MAX_JOBS,
                count_active_by_pool(conn, "owners"),
                OWNERS_GPU_MAX_JOBS,
            ))

            progress = False
            for row in next_submittable_configs(conn):
                pool = choose_target_pool(conn)
                if pool is None:
                    break
                submit_config(conn, config_by_idx[row["config_idx"]], pool)
                conn.commit()
                progress = True
                time.sleep(SUBMIT_SLEEP_SECONDS)

            if all_complete(conn):
                log("Completed all %d sweep configs." % len(configs))
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
