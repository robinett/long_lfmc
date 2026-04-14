#!/usr/bin/env python3

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time

import pandas as pd
import torch

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    DEFAULT_SCRATCH_ROOT,
    build_static_superset_runtime,
    build_reference_raw_tensor_cache,
    convert_tensor_payload_to_runtime,
    convert_tensor_payload_to_runtime_bounded,
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
DEFAULT_GPU_BATCH_CACHE_ROOT = os.path.join(
    DEFAULT_SCRATCH_ROOT,
    "lfmc_model",
    "inference",
    "cache",
    "gpu_batch_sizes",
)
DEFAULT_GPU_BATCH_RUNTIME_DIRNAME = "worker_runtime"
DEFAULT_GPU_BATCH_MIN = 512
DEFAULT_GPU_BATCH_START = 2048
DEFAULT_GPU_BATCH_STEP = 2048
DEFAULT_GPU_BATCH_TARGET_UTILIZATION = 0.9
DEFAULT_GPU_BATCH_MIN_FREE_GB = 2.0
DEFAULT_GPU_BATCH_SELECTION_FRACTION = 0.8


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


def _round_batch_size(value: int) -> int:
    if value <= DEFAULT_GPU_BATCH_MIN:
        return DEFAULT_GPU_BATCH_MIN
    rounded = int(math.floor(value / 256.0) * 256)
    return max(rounded, DEFAULT_GPU_BATCH_MIN)


def _runtime_batch_signature(run_config: dict, reference_runtime, predictor_states: list[dict]) -> str:
    payload = {
        "input_data_name": run_config.get("input_data_name"),
        "model_type": run_config.get("model_type", DEFAULT_MODEL_TYPE),
        "member_count": len(predictor_states),
        "short_vars": list(reference_runtime["var_names"]["short_vars"]),
        "long_vars": list(reference_runtime["var_names"]["long_vars"]),
        "static_vars": list(reference_runtime["var_names"]["static_vars"]),
        "short_lag_days": [int(v) for v in reference_runtime["short_lag_days"]],
        "long_lag_days": [int(v) for v in reference_runtime["long_lag_days"]],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _current_rss_gb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
    except FileNotFoundError:
        return 0.0
    return 0.0


def _host_memory_limit_gb() -> float | None:
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if mem_per_node.isdigit():
        return float(mem_per_node) / 1024.0
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE", "").strip()
    if mem_per_cpu.isdigit() and cpus_on_node.isdigit():
        return (float(mem_per_cpu) * float(cpus_on_node)) / 1024.0
    return None


def _candidate_batch_sizes(start_batch_size: int, max_batch_size: int) -> list[int]:
    candidates = []
    current = _round_batch_size(start_batch_size)
    max_batch_size = _round_batch_size(max_batch_size)
    while current <= max_batch_size:
        candidates.append(current)
        current += DEFAULT_GPU_BATCH_STEP
    return sorted(set(candidates))


def _slice_tensor_payload_rows(tensor_payload: dict, row_indices) -> dict:
    out = dict(tensor_payload)
    out["short_tensor"] = tensor_payload["short_tensor"][row_indices]
    out["long_tensor"] = tensor_payload["long_tensor"][row_indices]
    out["static_tensor"] = tensor_payload["static_tensor"][row_indices]
    out["info_df"] = tensor_payload["info_df"].iloc[row_indices].copy()
    return out


def _sample_tensor_payload_rows(tensor_payload: dict, row_count: int) -> dict:
    total_rows = int(tensor_payload["short_tensor"].shape[0])
    row_count = int(min(max(row_count, 1), total_rows))
    if row_count >= total_rows:
        row_indices = list(range(total_rows))
    else:
        row_indices = random.sample(range(total_rows), row_count)
        row_indices.sort()
    return _slice_tensor_payload_rows(tensor_payload, row_indices=row_indices)


def _monitor_peak_rss(stop_event: threading.Event, peak_holder: list[float]) -> None:
    while not stop_event.is_set():
        peak_holder[0] = max(peak_holder[0], _current_rss_gb())
        stop_event.wait(0.05)


def _batch_cache_path(
    *,
    run_config: dict,
    reference_runtime,
    predictor_states: list[dict],
    device_name: str,
    total_memory_gb: float,
    host_memory_limit_gb: float | None,
) -> str:
    signature = _runtime_batch_signature(run_config, reference_runtime, predictor_states)
    gpu_key = _normalize_gpu_name(device_name)
    gpu_mem_key = int(round(total_memory_gb * 10.0))
    host_mem_key = "na" if host_memory_limit_gb is None else str(int(round(host_memory_limit_gb * 10.0)))
    filename = f"{gpu_key}_gpumem{gpu_mem_key}_hostmem{host_mem_key}_{signature}.json"
    return os.path.join(DEFAULT_GPU_BATCH_CACHE_ROOT, filename)


def _load_batch_cache(cache_path: str) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_batch_cache(cache_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, cache_path)


def _write_worker_batch_metadata(
    *,
    run_dir: str,
    job_id: str,
    cache_path: str,
    selected_batch_size: int,
    configured_batch_size: int,
    device_name: str,
    total_memory_gb: float,
    host_memory_limit_gb: float | None,
    calibration_info: dict | None = None,
) -> None:
    runtime_dir = os.path.join(run_dir, "gpu_work_queue", DEFAULT_GPU_BATCH_RUNTIME_DIRNAME)
    os.makedirs(runtime_dir, exist_ok=True)
    payload = {
        "job_id": str(job_id),
        "cache_path": str(cache_path),
        "selected_batch_size": int(selected_batch_size),
        "configured_batch_size": int(configured_batch_size),
        "device_name": str(device_name),
        "total_memory_gb": float(total_memory_gb),
        "host_memory_limit_gb": (
            None if host_memory_limit_gb is None else float(host_memory_limit_gb)
        ),
        "written_at_epoch": float(time.time()),
    }
    if calibration_info is not None:
        payload.update({
            "calibration_probe_limit_rows": calibration_info.get("probe_limit_rows"),
            "calibration_largest_successful_batch": calibration_info.get("largest_successful_batch"),
            "calibration_stopping_reason": calibration_info.get("stopping_reason"),
            "calibration_payload_limited": calibration_info.get("stopped_due_to_payload_limit"),
            "calibration_peak_reserved_gpu_gb": calibration_info.get("peak_reserved_gpu_gb"),
            "calibration_baseline_reserved_gpu_gb": calibration_info.get("baseline_reserved_gpu_gb"),
            "calibration_peak_rss_gb": calibration_info.get("peak_rss_gb"),
            "calibration_estimated_free_gb": calibration_info.get("estimated_free_gb"),
            "calibration_estimated_utilization": calibration_info.get("estimated_utilization"),
        })
    path = os.path.join(runtime_dir, f"job_{job_id}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _probe_batch_size(
    *,
    predictor,
    tensor_payload: dict,
    batch_size: int,
    use_cuda_autocast: bool,
) -> dict:
    row_count = int(min(batch_size, tensor_payload["short_tensor"].shape[0]))
    probe_payload = _sample_tensor_payload_rows(tensor_payload, row_count=row_count)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_reserved_gpu_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    else:
        baseline_reserved_gpu_gb = 0.0
    baseline_rss_gb = _current_rss_gb()
    peak_rss_holder = [baseline_rss_gb]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_peak_rss,
        args=(stop_event, peak_rss_holder),
        daemon=True,
    )
    monitor_thread.start()
    success = False
    oom_error = False
    runtime_error = False
    failure_message = None
    try:
        run_runtime_forward_loaded(
            predictor=predictor,
            tensor_payload=probe_payload,
            batch_size=batch_size,
            return_info_df=False,
            use_cuda_autocast=use_cuda_autocast,
        )
        success = True
    except RuntimeError as exc:
        failure_message = str(exc)
        if "out of memory" in failure_message.lower():
            oom_error = True
        else:
            runtime_error = True
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        if torch.cuda.is_available():
            peak_reserved_gpu_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            torch.cuda.empty_cache()
        else:
            peak_reserved_gpu_gb = 0.0
        gc.collect()
    return {
        "success": success,
        "oom_error": oom_error,
        "runtime_error": runtime_error,
        "failure_message": failure_message,
        "row_count": row_count,
        "baseline_rss_gb": baseline_rss_gb,
        "peak_rss_gb": peak_rss_holder[0],
        "baseline_reserved_gpu_gb": baseline_reserved_gpu_gb,
        "peak_reserved_gpu_gb": peak_reserved_gpu_gb,
    }


def _calibrate_forward_batch_size(
    *,
    run_config: dict,
    worker_label: str,
    reference_runtime,
    predictor_states: list[dict],
    reference_payload: dict,
    use_cuda_autocast: bool,
    configured_batch_size: int,
    device_name: str,
    total_memory_gb: float,
    host_memory_limit_gb: float | None,
    cache_path: str,
) -> dict:
    runtime = predictor_states[0]["runtime"]
    try:
        probe_payload = convert_tensor_payload_to_runtime(
            reference_payload,
            reference_runtime,
            runtime,
            {},
            site="gpu_batch_probe",
        )
    except ValueError:
        probe_payload = reference_payload

    probe_limit = int(probe_payload["short_tensor"].shape[0])
    if probe_limit < configured_batch_size:
        selected = _round_batch_size(probe_limit)
        return {
            "selected_batch_size": int(selected),
            "probe_limit_rows": int(probe_limit),
            "largest_successful_batch": int(probe_limit),
            "stopped_due_to_payload_limit": True,
            "stopping_reason": "payload_limit",
        }

    predictor = predictor_states[0]["predictor"]
    best_success = None
    stopping_reason = None
    for candidate in _candidate_batch_sizes(DEFAULT_GPU_BATCH_START, probe_limit):
        probe_result = _probe_batch_size(
            predictor=predictor,
            tensor_payload=probe_payload,
            batch_size=candidate,
            use_cuda_autocast=use_cuda_autocast,
        )
        probe_result["candidate_batch_size"] = int(candidate)
        estimated_free_gb = max(total_memory_gb - float(probe_result["peak_reserved_gpu_gb"]), 0.0)
        estimated_utilization = (
            float(probe_result["peak_reserved_gpu_gb"]) / float(total_memory_gb)
            if total_memory_gb > 0
            else 0.0
        )
        probe_result["estimated_free_gb"] = estimated_free_gb
        probe_result["estimated_utilization"] = estimated_utilization
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} probe batch_size={candidate} "
            f"success={probe_result['success']} "
            f"rss={probe_result['peak_rss_gb']:.2f}GB "
            f"reserved_gpu={probe_result['peak_reserved_gpu_gb']:.2f}GB "
            f"free={estimated_free_gb:.2f}GB "
            f"util={estimated_utilization:.3f}"
            + (
                f" failure={probe_result['failure_message']}"
                if probe_result.get("failure_message")
                else ""
            )
        ))
        if not probe_result["success"]:
            stopping_reason = "failure_limit"
            break
        best_success = probe_result
        if (
            estimated_utilization >= DEFAULT_GPU_BATCH_TARGET_UTILIZATION
            or estimated_free_gb < DEFAULT_GPU_BATCH_MIN_FREE_GB
        ):
            stopping_reason = "memory_threshold"
            break

    if best_success is None:
        selected_batch_size = max(DEFAULT_GPU_BATCH_MIN, configured_batch_size // 2)
        largest_successful_batch = None
        stopped_due_to_payload_limit = False
        stopping_reason = "no_success"
        peak_reserved_gpu_gb = None
        baseline_reserved_gpu_gb = None
        peak_rss_gb = None
        estimated_free_gb = None
        estimated_utilization = None
    else:
        largest_successful_batch = int(best_success["candidate_batch_size"])
        peak_reserved_gpu_gb = float(best_success["peak_reserved_gpu_gb"])
        baseline_reserved_gpu_gb = float(best_success["baseline_reserved_gpu_gb"])
        peak_rss_gb = float(best_success["peak_rss_gb"])
        estimated_free_gb = float(best_success["estimated_free_gb"])
        estimated_utilization = float(best_success["estimated_utilization"])
        selected_batch_size = _round_batch_size(
            max(
                DEFAULT_GPU_BATCH_MIN,
                int(math.floor(largest_successful_batch * DEFAULT_GPU_BATCH_SELECTION_FRACTION)),
            )
        )
        stopped_due_to_payload_limit = (
            largest_successful_batch >= probe_limit
            or (probe_limit - largest_successful_batch) < DEFAULT_GPU_BATCH_STEP
        )
        if stopped_due_to_payload_limit and stopping_reason is None:
            stopping_reason = "payload_limit"
        elif stopping_reason is None:
            stopping_reason = "candidate_exhausted"

    print(timestamped_message(
        f"[run_map_task_gpu] {worker_label} calibrated batch_size={selected_batch_size} "
        f"on gpu={device_name} memory={total_memory_gb:.1f}GB; "
        f"host_limit={host_memory_limit_gb if host_memory_limit_gb is not None else 'NA'}GB; "
        f"probe_limit={probe_limit}; "
        f"stop_reason={stopping_reason}; "
        f"largest_successful_batch={largest_successful_batch if best_success is not None else 'NA'}"
    ))
    return {
        "selected_batch_size": int(selected_batch_size),
        "probe_limit_rows": int(probe_limit),
        "largest_successful_batch": (
            None if largest_successful_batch is None else int(largest_successful_batch)
        ),
        "stopped_due_to_payload_limit": bool(stopped_due_to_payload_limit),
        "stopping_reason": str(stopping_reason),
        "peak_reserved_gpu_gb": peak_reserved_gpu_gb,
        "baseline_reserved_gpu_gb": baseline_reserved_gpu_gb,
        "peak_rss_gb": peak_rss_gb,
        "estimated_free_gb": estimated_free_gb,
        "estimated_utilization": estimated_utilization,
    }


def _resolve_forward_batch_size_info(
    run_config: dict,
    worker_label: str,
    *,
    reference_runtime=None,
    predictor_states: list[dict] | None = None,
    reference_payload: dict | None = None,
    run_dir: str | None = None,
) -> dict:
    configured_batch_size = int(run_config.get("forward_batch_size", 512))
    if not torch.cuda.is_available():
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} using configured batch_size="
            f"{configured_batch_size} because CUDA is not available"
        ))
        return {
            "selected_batch_size": int(configured_batch_size),
            "probe_limit_rows": None,
            "largest_successful_batch": None,
            "stopped_due_to_payload_limit": False,
            "stopping_reason": "cuda_unavailable",
        }

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    total_memory_gb = (
        torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    )
    host_memory_limit_gb = _host_memory_limit_gb()
    if reference_runtime is None or predictor_states is None:
        return {
            "selected_batch_size": int(configured_batch_size),
            "probe_limit_rows": None,
            "largest_successful_batch": None,
            "stopped_due_to_payload_limit": False,
            "stopping_reason": "runtime_unavailable",
        }

    cache_path = _batch_cache_path(
        run_config=run_config,
        reference_runtime=reference_runtime,
        predictor_states=predictor_states,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        host_memory_limit_gb=host_memory_limit_gb,
    )
    if reference_payload is None:
        calibration_info = {
            "selected_batch_size": int(configured_batch_size),
            "probe_limit_rows": None,
            "largest_successful_batch": None,
            "stopped_due_to_payload_limit": False,
            "stopping_reason": "no_reference_payload",
        }
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} using configured batch_size={configured_batch_size} "
            f"because no reference payload is available for startup calibration"
        ))
    else:
        calibration_info = _calibrate_forward_batch_size(
            run_config=run_config,
            worker_label=worker_label,
            reference_runtime=reference_runtime,
            predictor_states=predictor_states,
            reference_payload=reference_payload,
            use_cuda_autocast=bool(run_config.get("use_cuda_autocast", True)),
            configured_batch_size=configured_batch_size,
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            host_memory_limit_gb=host_memory_limit_gb,
            cache_path=cache_path,
        )

    if run_dir is not None:
        _write_worker_batch_metadata(
            run_dir=run_dir,
            job_id=str(os.environ.get("SLURM_JOB_ID", "local")),
            cache_path=cache_path,
            selected_batch_size=int(calibration_info["selected_batch_size"]),
            configured_batch_size=configured_batch_size,
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            host_memory_limit_gb=host_memory_limit_gb,
            calibration_info=calibration_info,
        )
    return calibration_info


def _resolve_forward_batch_size(
    run_config: dict,
    worker_label: str,
    *,
    reference_runtime=None,
    predictor_states: list[dict] | None = None,
    reference_payload: dict | None = None,
    run_dir: str | None = None,
) -> int:
    return int(
        _resolve_forward_batch_size_info(
            run_config,
            worker_label,
            reference_runtime=reference_runtime,
            predictor_states=predictor_states,
            reference_payload=reference_payload,
            run_dir=run_dir,
        )["selected_batch_size"]
    )


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


def _delete_prepared_payloads(prepared_dir: str, fine_task_id: int) -> None:
    prefix = os.path.join(prepared_dir, f"task_{fine_task_id:06d}")
    removed_count = 0
    for path in sorted(glob.glob(f"{prefix}_*.pt")):
        try:
            os.remove(path)
            removed_count += 1
        except FileNotFoundError:
            continue
    if removed_count > 0:
        print(timestamped_message(
            f"[run_map_task_gpu] deleted {removed_count} prepared payload file(s) "
            f"for fine_task_id={fine_task_id}"
        ))


def _select_calibration_task_row(task_rows, prepared_dir: str):
    candidate_rows = []
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        fine_task_id = int(task_row["task_id"])
        if os.path.exists(_prepared_paths(prepared_dir, fine_task_id)):
            candidate_rows.append(task_row)
    if len(candidate_rows) == 0:
        raise FileNotFoundError(
            f"No prepared reference tensors are available under {prepared_dir} for startup calibration"
        )
    candidate_df = pd.DataFrame(candidate_rows).reset_index(drop=True)
    if "n_pixels" in candidate_df.columns:
        candidate_df = candidate_df.sort_values(
            ["n_pixels", "task_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
        dense_pool_size = max(1, min(len(candidate_df), int(math.ceil(len(candidate_df) * 0.2))))
        dense_pool = candidate_df.iloc[:dense_pool_size].copy().reset_index(drop=True)
        selected_row = dense_pool.iloc[random.randrange(len(dense_pool))]
        print(timestamped_message(
            f"[run_map_task_gpu] startup calibration selected dense prepared shard "
            f"fine_task_id={int(selected_row['task_id'])} "
            f"n_pixels={int(selected_row['n_pixels'])} "
            f"from top {dense_pool_size}/{len(candidate_df)} prepared shards"
        ))
        return selected_row
    selected_row = candidate_df.iloc[random.randrange(len(candidate_df))]
    return selected_row


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
    delete_prepared_payload_after_shard_complete: bool,
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
        if delete_prepared_payload_after_shard_complete:
            _delete_prepared_payloads(prepared_dir, fine_task_id)
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
    raw_tensor_cache = build_reference_raw_tensor_cache(
        reference_payload,
        reference_runtime,
    )
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
            tensor_payload = convert_tensor_payload_to_runtime_bounded(
                reference_payload,
                reference_runtime,
                runtime,
                raw_tensor_cache,
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
    if delete_prepared_payload_after_shard_complete:
        _delete_prepared_payloads(prepared_dir, fine_task_id)
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
    delete_prepared_payload_after_shard_complete = bool(
        run_config.get("delete_prepared_payload_after_shard_complete", True)
    )
    prepared_dir = run_config["prepared_dir"]
    first_task_row = _select_calibration_task_row(task_rows, prepared_dir)
    reference_payload = load_prepared_tensor_payload(
        _prepared_paths(prepared_dir, int(first_task_row["task_id"]))
    )
    forward_batch_size = _resolve_forward_batch_size(
        run_config,
        worker_label=f"gpu job {task_id}",
        reference_runtime=reference_runtime,
        predictor_states=predictor_states,
        reference_payload=reference_payload,
        run_dir=os.path.dirname(args.manifest_path),
    )
    completed_task_count = 0
    static_job_t0 = time.perf_counter()
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        task_elapsed = _process_single_task(
            task_row=task_row,
            prepared_dir=prepared_dir,
            delete_prepared_payload_after_shard_complete=delete_prepared_payload_after_shard_complete,
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
    delete_prepared_payload_after_shard_complete = bool(
        run_config.get("delete_prepared_payload_after_shard_complete", True)
    )
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
    forward_batch_size = None
    recalibration_pending = False
    calibration_probe_limit_rows = None
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
    calibration_task_row = _select_calibration_task_row(manifest_df, prepared_dir)
    calibration_fine_task_id = int(calibration_task_row["task_id"])
    print(timestamped_message(
        f"[run_map_task_gpu] {worker_label} startup calibration will use "
        f"random fine_task_id={calibration_fine_task_id} tile={calibration_task_row['tile_name']}"
    ))
    calibration_reference_payload = load_prepared_tensor_payload(
        _prepared_paths(prepared_dir, calibration_fine_task_id)
    )
    calibration_info = _resolve_forward_batch_size_info(
        run_config,
        worker_label=worker_label,
        reference_runtime=reference_runtime,
        predictor_states=predictor_states,
        reference_payload=calibration_reference_payload,
        run_dir=run_dir,
    )
    forward_batch_size = int(calibration_info["selected_batch_size"])
    recalibration_pending = bool(calibration_info["stopped_due_to_payload_limit"])
    calibration_probe_limit_rows = calibration_info["probe_limit_rows"]
    if recalibration_pending:
        print(timestamped_message(
            f"[run_map_task_gpu] {worker_label} startup calibration hit payload limit "
            f"at {calibration_probe_limit_rows} rows; will allow one follow-up recalibration "
            f"on a larger claimed shard if available"
        ))

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
        if recalibration_pending:
            claimed_reference_payload = load_prepared_tensor_payload(
                _prepared_paths(prepared_dir, fine_task_id)
            )
            claimed_probe_limit_rows = int(claimed_reference_payload["short_tensor"].shape[0])
            if (
                calibration_probe_limit_rows is not None
                and claimed_probe_limit_rows > int(calibration_probe_limit_rows)
            ):
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} follow-up recalibration on "
                    f"fine_task_id={fine_task_id} with larger payload "
                    f"{claimed_probe_limit_rows} > {calibration_probe_limit_rows} rows"
                ))
                calibration_info = _resolve_forward_batch_size_info(
                    run_config,
                    worker_label=f"{worker_label} follow-up",
                    reference_runtime=reference_runtime,
                    predictor_states=predictor_states,
                    reference_payload=claimed_reference_payload,
                    run_dir=run_dir,
                )
                forward_batch_size = int(calibration_info["selected_batch_size"])
                calibration_probe_limit_rows = calibration_info["probe_limit_rows"]
            else:
                print(timestamped_message(
                    f"[run_map_task_gpu] {worker_label} skipping follow-up recalibration on "
                    f"fine_task_id={fine_task_id}; payload rows={claimed_probe_limit_rows} "
                    f"are not larger than startup probe limit={calibration_probe_limit_rows}"
                ))
            recalibration_pending = False
        try:
            task_elapsed = _process_single_task(
                task_row=claimed_task_row,
                prepared_dir=prepared_dir,
                delete_prepared_payload_after_shard_complete=delete_prepared_payload_after_shard_complete,
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
