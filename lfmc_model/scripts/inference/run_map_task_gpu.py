#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import time

import pandas as pd
import torch

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    build_static_superset_runtime,
    convert_tensor_payload_to_runtime,
    densify_tile_predictions,
    finalize_running_ensemble_predictions,
    initialize_running_ensemble_predictions,
    load_ensemble_runtimes,
    load_prepared_tensor_payload,
    load_runtime_forward_predictor,
    load_tile_payload,
    run_runtime_forward_loaded,
    save_tile_shard,
    timestamped_message,
    update_running_ensemble_predictions,
)


DEFAULT_GPU_TIME_MARGIN_SECONDS = 600
DEFAULT_GPU_IDLE_SLEEP_SECONDS = 15
DEFAULT_GPU_CLAIM_STALE_SECONDS = 1800
DEFAULT_GPU_NEXT_TASK_SAFETY_FACTOR = 1.15


class PreparedTensorNotReadyError(RuntimeError):
    pass


def _normalize_gpu_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _memory_bucket_batch_size(total_memory_gb: float) -> int:
    if total_memory_gb <= 12.5:
        return 2048
    if total_memory_gb <= 16.5:
        return 3072
    if total_memory_gb <= 32.5:
        return 7168
    if total_memory_gb <= 48.5:
        return 10240
    if total_memory_gb <= 80.5:
        return 20480
    return 32768


def _gpu_specific_batch_size(device_name: str, total_memory_gb: float) -> int | None:
    normalized_name = _normalize_gpu_name(device_name)
    if "RTX_2080TI" in normalized_name:
        return 2048
    if "TITAN_XP" in normalized_name:
        return 2048
    if "TITAN_V" in normalized_name:
        return 2048
    if "P100" in normalized_name:
        return 3072
    if "V100" in normalized_name:
        return 7168
    if "L40S" in normalized_name:
        return 12288
    if "A40" in normalized_name:
        return 8192
    if "A100" in normalized_name:
        if total_memory_gb >= 70.0:
            return 20480
        return 10240
    if "H100" in normalized_name:
        return 20480
    if "H200" in normalized_name:
        return 32768
    return None


def _resolve_forward_batch_size(run_config: dict, worker_label: str) -> int:
    configured_batch_size = int(run_config.get("forward_batch_size", 512))
    if not torch.cuda.is_available():
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} using configured batch_size="
            f"{configured_batch_size} because CUDA is not available"
        ))
        return configured_batch_size

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    total_memory_gb = (
        torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    )
    memory_bucket_batch_size = _memory_bucket_batch_size(total_memory_gb)
    gpu_specific_batch_size = _gpu_specific_batch_size(device_name, total_memory_gb)

    if gpu_specific_batch_size is None:
        resolved_batch_size = min(configured_batch_size, memory_bucket_batch_size)
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} using fallback batch_size="
            f"{resolved_batch_size} on gpu={device_name} memory={total_memory_gb:.1f}GB; "
            f"configured={configured_batch_size}; memory_bucket={memory_bucket_batch_size}"
        ))
        return resolved_batch_size

    resolved_batch_size = min(gpu_specific_batch_size, memory_bucket_batch_size)
    print(timestamped_message(
        f"[run_map_task_gpu] {worker_label} using adaptive batch_size={resolved_batch_size} "
        f"on gpu={device_name} memory={total_memory_gb:.1f}GB; "
        f"gpu_lookup={gpu_specific_batch_size}; memory_bucket={memory_bucket_batch_size}; "
        f"configured_default={configured_batch_size}"
    ))
    return resolved_batch_size


def get_args():
    parser = argparse.ArgumentParser(
        description="Run GPU ensemble forward passes from prepared tensor payloads."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--dynamic_work_queue", action="store_true")
    return parser.parse_args()


def _prepared_paths(prepared_dir: str, fine_task_id: int, member_idx: int | None = None) -> str:
    base = os.path.join(prepared_dir, f"task_{fine_task_id:06d}")
    if member_idx is None:
        return f"{base}_reference.pt"
    return f"{base}_member_{member_idx:02d}.pt"


def _format_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _parse_time_limit_to_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported time-limit format: {value}")
    hours, minutes, seconds = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def _claim_root(run_dir: str) -> str:
    return os.path.join(run_dir, "gpu_work_queue", "claims")


def _completion_log_path(run_dir: str) -> str:
    return os.path.join(run_dir, "gpu_work_queue", "completed_tasks.tsv")


def _serc_only_endgame_flag_path(run_dir: str) -> str:
    return os.path.join(run_dir, "gpu_work_queue", "serc_only_endgame.flag")


def _claim_dir(claims_root: str, fine_task_id: int) -> str:
    return os.path.join(claims_root, f"task_{int(fine_task_id):06d}")


def _claim_heartbeat_path(claim_dir: str) -> str:
    return os.path.join(claim_dir, "heartbeat.txt")


def _claim_metadata_path(claim_dir: str) -> str:
    return os.path.join(claim_dir, "metadata.json")


def _touch_claim_heartbeat(claim_dir: str, message: str) -> None:
    heartbeat_path = _claim_heartbeat_path(claim_dir)
    with open(heartbeat_path, "w") as f:
        f.write(message + "\n")
    now = time.time()
    os.utime(heartbeat_path, (now, now))


def _write_claim_metadata(
    claim_dir: str,
    *,
    fine_task_id: int,
    worker_id: int,
    worker_label: str,
    job_id: str,
) -> None:
    metadata = {
        "fine_task_id": int(fine_task_id),
        "worker_id": int(worker_id),
        "worker_label": worker_label,
        "job_id": job_id,
        "hostname": os.uname().nodename,
        "claimed_at_epoch": time.time(),
    }
    with open(_claim_metadata_path(claim_dir), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    _touch_claim_heartbeat(
        claim_dir,
        f"claimed fine_task_id={fine_task_id} worker={worker_label} job_id={job_id}",
    )


def _claim_stale_age_seconds(claim_dir: str) -> float:
    heartbeat_path = _claim_heartbeat_path(claim_dir)
    path = heartbeat_path if os.path.exists(heartbeat_path) else claim_dir
    return max(time.time() - os.path.getmtime(path), 0.0)


def _release_claim(claim_dir: str) -> None:
    shutil.rmtree(claim_dir, ignore_errors=True)


def _record_completed_task(
    *,
    run_dir: str,
    gpu_pool: str,
    job_id: str,
    fine_task_id: int,
    task_elapsed_seconds: float,
) -> None:
    completion_log_path = _completion_log_path(run_dir)
    os.makedirs(os.path.dirname(completion_log_path), exist_ok=True)
    line = (
        f"{time.time():.3f}\t{gpu_pool}\t{job_id}\t{int(fine_task_id)}\t"
        f"{float(task_elapsed_seconds):.6f}\n"
    )
    fd = os.open(completion_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _try_claim_task(
    *,
    task_row,
    claims_root: str,
    worker_id: int,
    worker_label: str,
    job_id: str,
    claim_stale_seconds: int,
    prepared_dir: str,
) -> str | None:
    fine_task_id = int(task_row["task_id"])
    shard_path = str(task_row["shard_path"])
    if os.path.exists(shard_path):
        return None
    reference_path = _prepared_paths(prepared_dir, fine_task_id)
    if not os.path.exists(reference_path):
        return None

    claim_dir = _claim_dir(claims_root, fine_task_id)
    try:
        os.mkdir(claim_dir)
    except FileExistsError:
        try:
            stale_age = _claim_stale_age_seconds(claim_dir)
        except FileNotFoundError:
            return None
        if stale_age <= claim_stale_seconds:
            return None
        stale_dir = f"{claim_dir}.stale_{int(time.time())}_{os.getpid()}"
        try:
            os.rename(claim_dir, stale_dir)
        except OSError:
            return None
        _release_claim(stale_dir)
        try:
            os.mkdir(claim_dir)
        except FileExistsError:
            return None

    _write_claim_metadata(
        claim_dir,
        fine_task_id=fine_task_id,
        worker_id=worker_id,
        worker_label=worker_label,
        job_id=job_id,
    )
    return claim_dir


def _count_shards(shard_dir: str) -> int:
    if not os.path.isdir(shard_dir):
        return 0
    return sum(1 for entry in os.scandir(shard_dir) if entry.is_file() and entry.name.endswith(".npz"))


def _overall_progress_label(task_row, total_fine_tasks: int) -> str:
    overall_rank = int(task_row["overall_zero_idx"]) + 1
    return f"overall {overall_rank}/{total_fine_tasks}"


def _load_shared_runtime_state(run_config: dict, model_type: str):
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=run_config["ensemble_root"],
        input_data_name=run_config["input_data_name"],
        inputs_root=run_config.get("inputs_root"),
        fold=int(run_config.get("fold", 9998)),
        fallback_num_tasks=int(run_config.get("fallback_num_tasks", 3)),
        max_members=run_config.get("max_ensemble_members"),
        member_name_prefix=run_config.get("ensemble_member_name_prefix"),
        selection_key=run_config.get("ensemble_selection_key"),
    )
    if member_dirs != run_config["member_dirs"]:
        raise ValueError(
            "[run_map_task_gpu] resolved member_dirs differ from persisted run_config member_dirs"
        )
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)

    predictor_t0 = time.perf_counter()
    predictor_states = []
    for member_idx, runtime in enumerate(runtimes, start=1):
        print(timestamped_message(
            f"[run_map_task_gpu] loading member model {member_idx}/{len(runtimes)} "
            f"({os.path.basename(member_dirs[member_idx - 1])})"
        ))
        predictor_states.append({
            "member_idx": member_idx,
            "runtime": runtime,
            "predictor": load_runtime_forward_predictor(runtime, model_type=model_type),
        })
    print(timestamped_message(
        f"[run_map_task_gpu] loaded {len(predictor_states)} predictors in "
        f"{_format_seconds(time.perf_counter() - predictor_t0)}"
    ))
    return member_dirs, runtimes, reference_runtime, predictor_states


def _process_single_task(
    *,
    task_row,
    prepared_dir: str,
    reference_runtime,
    predictor_states: list[dict],
    forward_batch_size: int,
    use_cuda_autocast: bool,
    total_fine_tasks: int,
    claim_dir: str | None,
    worker_label: str,
):
    fine_task_id = int(task_row["task_id"])
    shard_path = str(task_row["shard_path"])
    progress_label = (
        f"{worker_label} task_id={fine_task_id} tile={task_row['tile_name']} "
        f"{_overall_progress_label(task_row, total_fine_tasks)}"
    )
    if os.path.exists(shard_path):
        print(timestamped_message(
            f"[run_map_task_gpu] {progress_label} shard already exists; skipping {shard_path}"
        ))
        return 0.0

    preload_t0 = time.perf_counter()
    print(timestamped_message(
        f"[run_map_task_gpu] preloading {progress_label} "
        f"window={task_row['start_date']} to {task_row['end_date']}"
    ))
    tile_payload = load_tile_payload(task_row["tile_meta_path"])
    reference_path = _prepared_paths(prepared_dir, fine_task_id)
    try:
        reference_payload = load_prepared_tensor_payload(reference_path)
    except RuntimeError as exc:
        if "PytorchStreamReader failed reading zip archive" in str(exc):
            raise PreparedTensorNotReadyError(
                f"{progress_label} prepared tensor is not fully written yet: {reference_path}"
            ) from exc
        raise
    aggregator = initialize_running_ensemble_predictions(reference_payload["info_df"])
    preload_elapsed = time.perf_counter() - preload_t0
    print(timestamped_message(
        f"[run_map_task_gpu] preloaded {progress_label} in {_format_seconds(preload_elapsed)}"
    ))
    if claim_dir is not None:
        _touch_claim_heartbeat(claim_dir, f"{progress_label} preloaded")

    task_t0 = time.perf_counter()
    member_elapsed_seconds = []
    for member_pos, predictor_state in enumerate(predictor_states, start=1):
        mean_member_elapsed = (
            sum(member_elapsed_seconds) / len(member_elapsed_seconds)
            if member_elapsed_seconds
            else None
        )
        remaining_members = len(predictor_states) - member_pos + 1
        eta_task = (
            mean_member_elapsed * remaining_members
            if mean_member_elapsed is not None
            else None
        )
        print(timestamped_message(
            f"[run_map_task_gpu] member sweep {member_pos}/{len(predictor_states)} "
            f"for {progress_label}; elapsed={_format_seconds(time.perf_counter() - task_t0)}"
            + (
                f"; est_task_remaining={_format_seconds(eta_task)}"
                if eta_task is not None
                else ""
            )
        ))
        if claim_dir is not None:
            _touch_claim_heartbeat(
                claim_dir,
                f"{progress_label} member sweep {member_pos}/{len(predictor_states)}",
            )
        member_t0 = time.perf_counter()
        runtime = predictor_state["runtime"]
        try:
            tensor_payload = convert_tensor_payload_to_runtime(
                reference_payload,
                reference_runtime,
                runtime,
                {},
                site=f"tile_{task_row['tile_name']}_{task_row['start_date']}",
            )
        except ValueError:
            member_path = _prepared_paths(
                prepared_dir,
                fine_task_id,
                member_idx=predictor_state["member_idx"],
            )
            print(timestamped_message(
                f"[run_map_task_gpu] loading member-specific tensors {member_path} for "
                f"{progress_label} because short/long feature layout differs"
            ))
            tensor_payload = load_prepared_tensor_payload(member_path)

        pred_arrays = run_runtime_forward_loaded(
            predictor=predictor_state["predictor"],
            tensor_payload=tensor_payload,
            batch_size=forward_batch_size,
            return_info_df=False,
            use_cuda_autocast=use_cuda_autocast,
        )
        update_running_ensemble_predictions(
            aggregator,
            pred_arrays["lfmc_pred"],
        )
        member_elapsed = time.perf_counter() - member_t0
        member_elapsed_seconds.append(member_elapsed)
        print(timestamped_message(
            f"[run_map_task_gpu] member sweep {member_pos}/{len(predictor_states)} "
            f"for {progress_label} complete in {_format_seconds(member_elapsed)}"
        ))

    agg_df = finalize_running_ensemble_predictions(aggregator)
    dense_payload = densify_tile_predictions(agg_df, tile_payload)
    save_tile_shard(
        shard_path=shard_path,
        dense_payload=dense_payload,
        tile_payload=tile_payload,
        task_row=task_row,
    )
    total_elapsed = time.perf_counter() - task_t0
    print(timestamped_message(
        f"[run_map_task_gpu] {progress_label} wrote shard {shard_path} "
        f"in {_format_seconds(total_elapsed)} using {len(predictor_states)} members"
    ))
    return total_elapsed


def _run_static_gpu_task(args, run_config: dict, manifest_df: pd.DataFrame, model_type: str):
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("GPU_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if task_id < 0:
        raise ValueError("task_id must be provided or available via GPU_TASK_ID/SLURM_ARRAY_TASK_ID")

    manifest_df = manifest_df.sort_values(["gpu_job_task_id", "task_id"]).reset_index(drop=True)
    row_mask = manifest_df["gpu_job_task_id"].astype(int) == int(task_id)
    if not row_mask.any():
        raise KeyError(f"gpu_job_task_id={task_id} not found in manifest {args.manifest_path}")
    task_rows = manifest_df.loc[row_mask].copy()
    total_fine_tasks = len(manifest_df)
    task_rows = task_rows.reset_index(drop=False).rename(columns={"index": "overall_zero_idx"})

    print(timestamped_message(
        f"[run_map_task_gpu] gpu_job_task_id={task_id} processing {len(task_rows)} fine tasks"
    ))
    _, _, reference_runtime, predictor_states = _load_shared_runtime_state(run_config, model_type)

    use_cuda_autocast = bool(run_config.get("use_cuda_autocast", True))
    prepared_dir = run_config["prepared_dir"]
    forward_batch_size = _resolve_forward_batch_size(
        run_config,
        worker_label=f"gpu job {task_id}",
    )
    completed_task_count = 0
    static_job_t0 = time.perf_counter()
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        task_elapsed = _process_single_task(
            task_row=task_row,
            prepared_dir=prepared_dir,
            reference_runtime=reference_runtime,
            predictor_states=predictor_states,
            forward_batch_size=forward_batch_size,
            use_cuda_autocast=use_cuda_autocast,
            total_fine_tasks=total_fine_tasks,
            claim_dir=None,
            worker_label=f"gpu job {task_id} fine {row_idx + 1}/{len(task_rows)}",
        )
        if task_elapsed > 0:
            completed_task_count += 1

    print(timestamped_message(
        f"[run_map_task_gpu] gpu_job_task_id={task_id} complete in "
        f"{_format_seconds(time.perf_counter() - static_job_t0)}; wrote {completed_task_count} shards"
    ))


def _run_dynamic_gpu_worker(args, run_config: dict, manifest_df: pd.DataFrame, model_type: str):
    worker_id = args.worker_id
    if worker_id is None:
        worker_id = int(os.environ.get("GPU_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if worker_id < 0:
        raise ValueError("worker_id must be provided or available via GPU_TASK_ID/SLURM_ARRAY_TASK_ID")

    run_dir = os.path.dirname(args.manifest_path)
    prepared_dir = run_config["prepared_dir"]
    shard_dir = run_config["shard_dir"]
    claims_root = _claim_root(run_dir)
    os.makedirs(claims_root, exist_ok=True)

    use_cuda_autocast = bool(run_config.get("use_cuda_autocast", True))
    time_limit_seconds = _parse_time_limit_to_seconds(
        os.environ.get("GPU_JOB_TIME_LIMIT_SECONDS")
        or run_config.get("gpu_job_time_limit_seconds")
    )
    time_margin_seconds = int(
        os.environ.get("GPU_TIME_MARGIN_SECONDS")
        or run_config.get("gpu_time_margin_seconds", DEFAULT_GPU_TIME_MARGIN_SECONDS)
    )
    claim_stale_seconds = int(
        os.environ.get("GPU_CLAIM_STALE_SECONDS")
        or run_config.get("gpu_claim_stale_seconds", DEFAULT_GPU_CLAIM_STALE_SECONDS)
    )
    idle_sleep_seconds = int(
        os.environ.get("GPU_IDLE_SLEEP_SECONDS")
        or run_config.get("gpu_idle_sleep_seconds", DEFAULT_GPU_IDLE_SLEEP_SECONDS)
    )
    next_task_safety_factor = float(
        os.environ.get("GPU_NEXT_TASK_SAFETY_FACTOR")
        or run_config.get("gpu_next_task_safety_factor", DEFAULT_GPU_NEXT_TASK_SAFETY_FACTOR)
    )
    job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
    gpu_pool = str(os.environ.get("GPU_POOL", "unknown")).strip().lower()
    worker_label = f"gpu worker {worker_id} job {job_id}"
    forward_batch_size = _resolve_forward_batch_size(run_config, worker_label=worker_label)
    endgame_flag_path = _serc_only_endgame_flag_path(run_dir)

    manifest_df = (
        manifest_df.sort_values(["task_id", "start_date", "tile_name"])
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "overall_zero_idx"})
    )
    total_fine_tasks = len(manifest_df)

    print(timestamped_message(
        f"[run_map_task_gpu] {worker_label} starting dynamic queue worker "
        f"for {total_fine_tasks} fine tasks; time_limit="
        f"{_format_seconds(time_limit_seconds) if time_limit_seconds is not None else 'unbounded'}; "
        f"time_margin={_format_seconds(time_margin_seconds)}; "
        f"claim_stale={_format_seconds(claim_stale_seconds)}"
    ))
    _, _, reference_runtime, predictor_states = _load_shared_runtime_state(run_config, model_type)

    worker_t0 = time.perf_counter()
    completed_task_elapsed = []
    completed_task_count = 0
    idle_loops = 0

    while True:
        worker_elapsed = time.perf_counter() - worker_t0
        remaining_seconds = (
            max(time_limit_seconds - worker_elapsed, 0.0)
            if time_limit_seconds is not None
            else math.inf
        )
        current_shard_count = _count_shards(shard_dir)
        if current_shard_count >= total_fine_tasks:
            print(timestamped_message(
                f"[run_map_task_gpu] {worker_label} sees all shards complete "
                f"({current_shard_count}/{total_fine_tasks}); exiting"
            ))
            break

        if gpu_pool == "owners" and os.path.exists(endgame_flag_path):
            if completed_task_count > 0:
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} exiting after current shard "
                    f"because serc-only endgame is active"
                ))
            else:
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} exiting without claiming work "
                    f"because serc-only endgame is active"
                ))
            break

        estimated_next_task_seconds = None
        if completed_task_elapsed:
            mean_task_seconds = sum(completed_task_elapsed) / len(completed_task_elapsed)
            estimated_next_task_seconds = max(
                completed_task_elapsed[-1],
                mean_task_seconds,
            ) * next_task_safety_factor

        if (
            estimated_next_task_seconds is not None
            and time_limit_seconds is not None
            and remaining_seconds < (estimated_next_task_seconds + time_margin_seconds)
        ):
            print(timestamped_message(
                f"[run_map_task_gpu] {worker_label} exiting before claiming another shard: "
                f"elapsed={_format_seconds(worker_elapsed)}; "
                f"remaining={_format_seconds(remaining_seconds)}; "
                f"estimated_next_task={_format_seconds(estimated_next_task_seconds)}; "
                f"time_margin={_format_seconds(time_margin_seconds)}; "
                f"completed_tasks={completed_task_count}"
            ))
            break

        claimed_task_row = None
        claimed_dir = None
        for row_idx in range(len(manifest_df)):
            task_row = manifest_df.iloc[row_idx]
            claim_dir = _try_claim_task(
                task_row=task_row,
                claims_root=claims_root,
                worker_id=worker_id,
                worker_label=worker_label,
                job_id=job_id,
                claim_stale_seconds=claim_stale_seconds,
                prepared_dir=prepared_dir,
            )
            if claim_dir is None:
                continue
            claimed_task_row = task_row
            claimed_dir = claim_dir
            break

        if claimed_task_row is None:
            idle_loops += 1
            if time_limit_seconds is not None and remaining_seconds <= time_margin_seconds:
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} exiting idle with "
                    f"{_format_seconds(remaining_seconds)} remaining and no claimable task"
                ))
                break
            if idle_loops == 1 or idle_loops % 4 == 0:
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} found no claimable task; "
                    f"shards={current_shard_count}/{total_fine_tasks}; "
                    f"elapsed={_format_seconds(worker_elapsed)}; "
                    f"remaining={_format_seconds(remaining_seconds) if math.isfinite(remaining_seconds) else 'unbounded'}; "
                    f"sleeping {_format_seconds(idle_sleep_seconds)}"
                ))
            time.sleep(idle_sleep_seconds)
            continue

        idle_loops = 0
        fine_task_id = int(claimed_task_row["task_id"])
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} claimed fine_task_id={fine_task_id} "
            f"tile={claimed_task_row['tile_name']} {_overall_progress_label(claimed_task_row, total_fine_tasks)}"
        ))
        try:
            task_elapsed = _process_single_task(
                task_row=claimed_task_row,
                prepared_dir=prepared_dir,
                reference_runtime=reference_runtime,
                predictor_states=predictor_states,
                forward_batch_size=forward_batch_size,
                use_cuda_autocast=use_cuda_autocast,
                total_fine_tasks=total_fine_tasks,
                claim_dir=claimed_dir,
                worker_label=worker_label,
            )
        except PreparedTensorNotReadyError as exc:
            print(timestamped_message(
                f"[run_map_task_gpu] {worker_label} releasing fine_task_id={fine_task_id} "
                f"for retry because prepared tensor is not ready: {exc}"
            ))
            time.sleep(idle_sleep_seconds)
            continue
        finally:
            _release_claim(claimed_dir)

        if task_elapsed > 0:
            completed_task_elapsed.append(task_elapsed)
            completed_task_count += 1
            _record_completed_task(
                run_dir=run_dir,
                gpu_pool=gpu_pool,
                job_id=job_id,
                fine_task_id=fine_task_id,
                task_elapsed_seconds=task_elapsed,
            )
            mean_task_seconds = sum(completed_task_elapsed) / len(completed_task_elapsed)
            print(timestamped_message(
                f"[run_map_task_gpu] {worker_label} completed {completed_task_count} shard(s); "
                f"last_task={_format_seconds(task_elapsed)}; "
                f"mean_task={_format_seconds(mean_task_seconds)}; "
                f"elapsed={_format_seconds(time.perf_counter() - worker_t0)}"
            ))

    print(timestamped_message(
        f"[run_map_task_gpu] {worker_label} complete in "
        f"{_format_seconds(time.perf_counter() - worker_t0)}; "
        f"completed_tasks={completed_task_count}"
    ))


def main():
    args = get_args()
    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    manifest_df = pd.read_csv(args.manifest_path)
    model_type = (
        args.model_type
        if args.model_type is not None
        else run_config.get("model_type", DEFAULT_MODEL_TYPE)
    )
    dynamic_work_queue = args.dynamic_work_queue or (
        str(os.environ.get("DYNAMIC_GPU_WORK_QUEUE", "")).lower() == "true"
    )
    if dynamic_work_queue:
        _run_dynamic_gpu_worker(args, run_config, manifest_df, model_type)
    else:
        _run_static_gpu_task(args, run_config, manifest_df, model_type)


if __name__ == "__main__":
    main()
