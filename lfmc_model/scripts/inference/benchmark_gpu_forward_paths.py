#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from map_runtime_utils import (
    _effective_static_norm_arrays,
    _renormalize_tensor,
    _runtimes_share_feature_layout,
    convert_tensor_payload_to_runtime,
    densify_tile_predictions,
    finalize_running_ensemble_predictions,
    initialize_running_ensemble_predictions,
    load_prepared_tensor_payload,
    load_tile_payload,
    run_runtime_forward_loaded,
    runtimes_share_short_long_layout,
    save_tile_shard,
    update_running_ensemble_predictions,
)
from point_tool_new import predict_with_loaded_model
from run_map_task_gpu import _load_shared_runtime_state, _prepared_paths


DEFAULT_RUN_DIR = (
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/"
    "map_runs/lfmc_vh_vv_365_multisource_fusion_clim20_multiyear/years/"
    "year_2022/run_20260407_214235"
)
DEFAULT_TASK_IDS = [3320, 3474, 3521]
DEFAULT_BATCH_SIZE = 32768


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated old-vs-experimental GPU inference benchmarks on prepared map "
            "tasks without touching the active pipeline scripts."
        )
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=DEFAULT_TASK_IDS,
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/users/trobinet/long_lfmc/logs/gpu_forward_benchmarks",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/"
            "benchmark_outputs"
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest_rows(run_dir: Path, task_ids: list[int]) -> list[pd.Series]:
    manifest_path = run_dir / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = (
        manifest_df.sort_values(["task_id", "start_date", "tile_name"])
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "overall_zero_idx"})
    )
    rows = []
    for task_id in task_ids:
        matches = manifest_df.loc[manifest_df["task_id"].astype(int) == int(task_id)]
        if len(matches) == 0:
            raise KeyError(f"task_id={task_id} not found in {manifest_path}")
        rows.append(matches.iloc[0].copy())
    return rows


def gpu_name(gpu_index: int) -> str:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        universal_newlines=True,
    )
    return out.strip().splitlines()[0].strip()


class GpuSampler:
    def __init__(self, gpu_index: int, output_csv: Path, interval_seconds: float = 1.0):
        self.gpu_index = int(gpu_index)
        self.output_csv = output_csv
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        with self.output_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "gpu_index",
                    "gpu_name",
                    "utilization_gpu",
                    "memory_total_mib",
                    "memory_used_mib",
                ]
            )
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return summarize_gpu_csv(self.output_csv)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=index,name,utilization.gpu,memory.total,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    universal_newlines=True,
                ).strip()
                parts = [part.strip() for part in out.split(",")]
                if len(parts) == 5:
                    with self.output_csv.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.time()] + parts)
            except subprocess.CalledProcessError:
                pass
            self._stop.wait(self.interval_seconds)


def summarize_gpu_csv(path: Path) -> dict:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(list(reader))
    if len(rows) == 0:
        return {
            "sample_count": 0,
            "mean_gpu_utilization": None,
            "max_gpu_utilization": None,
            "mean_memory_utilization": None,
            "max_memory_utilization": None,
        }
    gpu_utils = [float(row["utilization_gpu"]) for row in rows]
    mem_utils = [
        100.0 * float(row["memory_used_mib"]) / float(row["memory_total_mib"])
        for row in rows
        if float(row["memory_total_mib"]) > 0
    ]
    return {
        "sample_count": len(rows),
        "mean_gpu_utilization": float(np.mean(gpu_utils)),
        "max_gpu_utilization": float(np.max(gpu_utils)),
        "mean_memory_utilization": float(np.mean(mem_utils)),
        "max_memory_utilization": float(np.max(mem_utils)),
    }


def _pin_if_needed(tensor: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable:
        return tensor
    if tensor.device.type != "cpu":
        return tensor
    if tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


def _denormalize_tensor_to_numpy(
    tensor: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    safe_std = np.where(np.abs(std) > 0, std, 1.0)
    reshape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    return arr * safe_std.reshape(reshape) + mean.reshape(reshape)


def _renormalize_numpy_to_tensor(
    raw_arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    safe_std = np.where(np.abs(std) > 0, std, 1.0)
    reshape = (1,) * (raw_arr.ndim - 1) + (raw_arr.shape[-1],)
    renorm = (raw_arr - mean.reshape(reshape)) / safe_std.reshape(reshape)
    return torch.tensor(renorm, dtype=dtype)


def build_reference_raw_cache(reference_payload: dict, reference_runtime: dict) -> dict:
    ref_static_vars = list(reference_runtime["var_names"]["static_vars"])
    static_ref_mean, static_ref_std = _effective_static_norm_arrays(
        ref_static_vars,
        reference_runtime["norm_params"],
    )
    return {
        "short_raw": _denormalize_tensor_to_numpy(
            reference_payload["short_tensor"],
            reference_runtime["norm_params"]["train_short_mean"],
            reference_runtime["norm_params"]["train_short_std"],
        ),
        "long_raw": _denormalize_tensor_to_numpy(
            reference_payload["long_tensor"],
            reference_runtime["norm_params"]["train_long_mean"],
            reference_runtime["norm_params"]["train_long_std"],
        ),
        "static_raw_full": _denormalize_tensor_to_numpy(
            reference_payload["static_tensor"],
            static_ref_mean,
            static_ref_std,
        ),
        "ref_static_vars": ref_static_vars,
        "ref_static_idx": {
            var_name: idx for idx, var_name in enumerate(ref_static_vars)
        },
        "static_raw_by_layout": {},
    }


def convert_tensor_payload_to_runtime_bounded(
    reference_payload: dict,
    reference_runtime: dict,
    runtime: dict,
    raw_cache: dict,
) -> dict:
    if not runtimes_share_short_long_layout(reference_runtime, runtime):
        raise ValueError("Short/long feature layout differs; full tensor rebuild required")

    short_tensor = _renormalize_numpy_to_tensor(
        raw_cache["short_raw"],
        runtime["norm_params"]["train_short_mean"],
        runtime["norm_params"]["train_short_std"],
        reference_payload["short_tensor"].dtype,
    )
    long_tensor = _renormalize_numpy_to_tensor(
        raw_cache["long_raw"],
        runtime["norm_params"]["train_long_mean"],
        runtime["norm_params"]["train_long_std"],
        reference_payload["long_tensor"].dtype,
    )

    if _runtimes_share_feature_layout(reference_runtime, runtime):
        static_raw = raw_cache["static_raw_full"]
    else:
        new_static_vars = tuple(runtime["var_names"]["static_vars"])
        static_raw = raw_cache["static_raw_by_layout"].get(new_static_vars)
        if static_raw is None:
            missing_static = [
                var_name
                for var_name in new_static_vars
                if var_name not in raw_cache["ref_static_idx"]
            ]
            if len(missing_static) > 0:
                raise ValueError(
                    f"Reference static superset is missing runtime static vars: {missing_static}"
                )
            static_indices = np.asarray(
                [raw_cache["ref_static_idx"][var_name] for var_name in new_static_vars],
                dtype=np.int64,
            )
            static_raw = raw_cache["static_raw_full"][..., static_indices]
            raw_cache["static_raw_by_layout"][new_static_vars] = static_raw

    static_new_mean, static_new_std = _effective_static_norm_arrays(
        runtime["var_names"]["static_vars"],
        runtime["norm_params"],
    )
    static_tensor = _renormalize_numpy_to_tensor(
        static_raw,
        static_new_mean,
        static_new_std,
        reference_payload["static_tensor"].dtype,
    )
    return {
        "empty": False,
        "safe_start": reference_payload["safe_start"],
        "safe_end": reference_payload["safe_end"],
        "short_tensor": short_tensor,
        "long_tensor": long_tensor,
        "static_tensor": static_tensor,
        "info_df": reference_payload["info_df"].copy(),
    }


def predict_with_loaded_model_optimized(
    short_tensor,
    long_tensor,
    static_tensor,
    model,
    device,
    norm_params,
    batch_size=512,
    use_cuda_autocast=True,
):
    n_obs = int(short_tensor.shape[0])
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    use_cuda_autocast = bool(use_cuda_autocast) and use_cuda

    preds_i = np.zeros(n_obs, dtype=np.float64)
    preds_vv = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vh = np.full(n_obs, np.nan, dtype=np.float64)
    preds_i_std = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vv_std = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vh_std = np.full(n_obs, np.nan, dtype=np.float64)

    with torch.inference_mode():
        for start_idx in range(0, n_obs, int(batch_size)):
            end_idx = min(start_idx + int(batch_size), n_obs)
            xsh_cpu = _pin_if_needed(short_tensor[start_idx:end_idx], enable=use_cuda)
            xl_cpu = _pin_if_needed(long_tensor[start_idx:end_idx], enable=use_cuda)
            xst_cpu = _pin_if_needed(static_tensor[start_idx:end_idx], enable=use_cuda)
            xsh_b = xsh_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=use_cuda,
            )
            xl_b = xl_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=use_cuda,
            )
            xst_b = xst_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=use_cuda,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_cuda_autocast,
            ):
                output = model(xsh_b, xl_b, xst_b)

            preds_i[start_idx:end_idx] = (
                output["mu_insitu"].detach().float().cpu().numpy().reshape(-1)
            )
            preds_vv[start_idx:end_idx] = (
                output["mu_vv"].detach().float().cpu().numpy().reshape(-1)
            )
            preds_vh[start_idx:end_idx] = (
                output["mu_vh"].detach().float().cpu().numpy().reshape(-1)
            )
            if "log_var_insitu" in output:
                preds_i_std[start_idx:end_idx] = np.sqrt(
                    np.exp(
                        output["log_var_insitu"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                )
            if "log_var_vv" in output:
                preds_vv_std[start_idx:end_idx] = np.sqrt(
                    np.exp(
                        output["log_var_vv"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                )
            if "log_var_vh" in output:
                preds_vh_std[start_idx:end_idx] = np.sqrt(
                    np.exp(
                        output["log_var_vh"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                )

    lfmc_mean = norm_params.get("lfmc_mean", np.nan)
    lfmc_std = norm_params.get("lfmc_std", np.nan)
    vv_mean = norm_params.get("vv_mean", np.nan)
    vv_std = norm_params.get("vv_std", np.nan)
    vh_mean = norm_params.get("vh_mean", np.nan)
    vh_std = norm_params.get("vh_std", np.nan)
    if np.isfinite(lfmc_mean) and np.isfinite(lfmc_std) and lfmc_std != 0:
        preds_i = preds_i * lfmc_std + lfmc_mean
        preds_i_std = preds_i_std * lfmc_std
    else:
        preds_i[:] = np.nan
        preds_i_std[:] = np.nan
    if np.isfinite(vv_mean) and np.isfinite(vv_std) and vv_std != 0:
        preds_vv = preds_vv * vv_std + vv_mean
        preds_vv_std = preds_vv_std * vv_std
    else:
        preds_vv[:] = np.nan
        preds_vv_std[:] = np.nan
    if np.isfinite(vh_mean) and np.isfinite(vh_std) and vh_std != 0:
        preds_vh = preds_vh * vh_std + vh_mean
        preds_vh_std = preds_vh_std * vh_std
    else:
        preds_vh[:] = np.nan
        preds_vh_std[:] = np.nan
    return {
        "lfmc_pred": preds_i.astype(np.float64, copy=False),
        "lfmc_pred_std": preds_i_std.astype(np.float64, copy=False),
        "vv_pred": preds_vv.astype(np.float64, copy=False),
        "vv_pred_std": preds_vv_std.astype(np.float64, copy=False),
        "vh_pred": preds_vh.astype(np.float64, copy=False),
        "vh_pred_std": preds_vh_std.astype(np.float64, copy=False),
    }


def run_runtime_forward_loaded_optimized(
    predictor: dict,
    tensor_payload: dict,
    batch_size: int,
    use_cuda_autocast: bool,
):
    return predict_with_loaded_model_optimized(
        tensor_payload["short_tensor"],
        tensor_payload["long_tensor"],
        tensor_payload["static_tensor"],
        model=predictor["model"],
        device=predictor["device"],
        norm_params=predictor["norm_params"],
        batch_size=batch_size,
        use_cuda_autocast=use_cuda_autocast,
    )


def run_task_mode(
    *,
    mode_name: str,
    task_row: pd.Series,
    prepared_dir: Path,
    reference_runtime,
    predictor_states: list[dict],
    batch_size: int,
    use_cuda_autocast: bool,
    reuse_tensor_cache: bool,
    use_optimized_predict: bool,
    output_path: Path,
    gpu_index: int,
    log_dir: Path,
):
    fine_task_id = int(task_row["task_id"])
    tile_payload = load_tile_payload(str(task_row["tile_meta_path"]))
    reference_payload = load_prepared_tensor_payload(
        _prepared_paths(str(prepared_dir), fine_task_id)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler(
        gpu_index=gpu_index,
        output_csv=log_dir / f"{mode_name}_task_{fine_task_id:06d}_gpu.csv",
        interval_seconds=1.0,
    )
    sampler.start()

    task_t0 = time.perf_counter()
    aggregator = initialize_running_ensemble_predictions(reference_payload["info_df"])
    tensor_cache = {}
    raw_cache = (
        build_reference_raw_cache(reference_payload, reference_runtime)
        if reuse_tensor_cache
        else None
    )
    member_rows = []
    site_name = f"tile_{task_row['tile_name']}_{task_row['start_date']}"

    for member_pos, predictor_state in enumerate(predictor_states, start=1):
        member_row = {
            "member_idx": int(member_pos),
            "runtime_member_idx": int(predictor_state["member_idx"]),
        }
        print(
            f"[{mode_name}] task_id={fine_task_id} member={member_pos}/{len(predictor_states)} convert_start",
            flush=True,
        )
        convert_t0 = time.perf_counter()
        if reuse_tensor_cache:
            tensor_payload = convert_tensor_payload_to_runtime_bounded(
                reference_payload,
                reference_runtime,
                predictor_state["runtime"],
                raw_cache,
            )
        else:
            tensor_payload = convert_tensor_payload_to_runtime(
                reference_payload,
                reference_runtime,
                predictor_state["runtime"],
                {},
                site=site_name,
            )
        member_row["convert_seconds"] = time.perf_counter() - convert_t0
        print(
            f"[{mode_name}] task_id={fine_task_id} member={member_pos}/{len(predictor_states)} "
            f"convert_done seconds={member_row['convert_seconds']:.3f}",
            flush=True,
        )

        forward_t0 = time.perf_counter()
        print(
            f"[{mode_name}] task_id={fine_task_id} member={member_pos}/{len(predictor_states)} forward_start",
            flush=True,
        )
        if use_optimized_predict:
            pred_arrays = run_runtime_forward_loaded_optimized(
                predictor=predictor_state["predictor"],
                tensor_payload=tensor_payload,
                batch_size=batch_size,
                use_cuda_autocast=use_cuda_autocast,
            )
        else:
            pred_arrays = run_runtime_forward_loaded(
                predictor=predictor_state["predictor"],
                tensor_payload=tensor_payload,
                batch_size=batch_size,
                return_info_df=False,
                use_cuda_autocast=use_cuda_autocast,
            )
        member_row["forward_seconds"] = time.perf_counter() - forward_t0
        print(
            f"[{mode_name}] task_id={fine_task_id} member={member_pos}/{len(predictor_states)} "
            f"forward_done seconds={member_row['forward_seconds']:.3f}",
            flush=True,
        )
        update_running_ensemble_predictions(aggregator, pred_arrays["lfmc_pred"])
        member_rows.append(member_row)

    post_t0 = time.perf_counter()
    print(f"[{mode_name}] task_id={fine_task_id} postprocess_start", flush=True)
    agg_df = finalize_running_ensemble_predictions(aggregator)
    dense_payload = densify_tile_predictions(agg_df, tile_payload)
    save_tile_shard(
        shard_path=str(output_path),
        dense_payload=dense_payload,
        tile_payload=tile_payload,
        task_row=task_row,
    )
    post_seconds = time.perf_counter() - post_t0
    total_seconds = time.perf_counter() - task_t0
    print(
        f"[{mode_name}] task_id={fine_task_id} postprocess_done seconds={post_seconds:.3f}",
        flush=True,
    )
    gpu_summary = sampler.stop()

    summary = {
        "mode_name": mode_name,
        "task_id": fine_task_id,
        "tile_name": str(task_row["tile_name"]),
        "start_date": str(task_row["start_date"]),
        "end_date": str(task_row["end_date"]),
        "n_pixels": int(task_row["n_pixels"]),
        "batch_size": int(batch_size),
        "use_cuda_autocast": bool(use_cuda_autocast),
        "reuse_tensor_cache": bool(reuse_tensor_cache),
        "use_optimized_predict": bool(use_optimized_predict),
        "total_seconds": float(total_seconds),
        "member_convert_seconds_total": float(
            sum(row["convert_seconds"] for row in member_rows)
        ),
        "member_forward_seconds_total": float(
            sum(row["forward_seconds"] for row in member_rows)
        ),
        "postprocess_seconds": float(post_seconds),
        "member_rows": member_rows,
        "gpu_summary": gpu_summary,
        "output_path": str(output_path),
    }
    with (log_dir / f"{mode_name}_task_{fine_task_id:06d}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def compare_npz_outputs(old_path: Path, new_path: Path) -> dict:
    with np.load(old_path, allow_pickle=False) as old_npz, np.load(
        new_path, allow_pickle=False
    ) as new_npz:
        out = {}
        for key in ["lfmc_ens_mean", "lfmc_ens_std"]:
            old_arr = np.asarray(old_npz[key], dtype=np.float64)
            new_arr = np.asarray(new_npz[key], dtype=np.float64)
            diff = new_arr - old_arr
            out[key] = {
                "shape": list(old_arr.shape),
                "max_abs_diff": float(np.nanmax(np.abs(diff))),
                "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
                "allclose_atol_1e-4_rtol_1e-4": bool(
                    np.allclose(old_arr, new_arr, atol=1e-4, rtol=1e-4, equal_nan=True)
                ),
            }
        return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    log_root = ensure_dir(Path(args.log_dir).resolve() / time.strftime("%Y%m%d_%H%M%S"))
    output_root = ensure_dir(Path(args.output_root).resolve() / log_root.name)
    old_output_dir = ensure_dir(output_root / "old")
    new_output_dir = ensure_dir(output_root / "new")

    run_config_path = run_dir / "run_config.json"
    with run_config_path.open("r") as f:
        run_config = json.load(f)

    task_rows = read_manifest_rows(run_dir, args.task_ids)
    _, _, reference_runtime, predictor_states = _load_shared_runtime_state(
        run_config,
        model_type=str(run_config.get("model_type", "standard")),
    )

    gpu_label = gpu_name(args.gpu_index)
    print(f"gpu_index={args.gpu_index}")
    print(f"gpu_name={gpu_label}")
    print(f"log_root={log_root}")
    print(f"output_root={output_root}")
    print(f"task_ids={[int(row['task_id']) for row in task_rows]}")

    prepared_dir = Path(run_config["prepared_dir"])
    benchmark_rows = []

    for task_row in task_rows:
        task_id = int(task_row["task_id"])
        print(f"running baseline task_id={task_id}")
        old_summary = run_task_mode(
            mode_name="baseline",
            task_row=task_row,
            prepared_dir=prepared_dir,
            reference_runtime=reference_runtime,
            predictor_states=predictor_states,
            batch_size=int(args.batch_size),
            use_cuda_autocast=False,
            reuse_tensor_cache=False,
            use_optimized_predict=False,
            output_path=old_output_dir / f"task_{task_id:06d}.npz",
            gpu_index=int(args.gpu_index),
            log_dir=log_root,
        )
        print(f"running experimental task_id={task_id}")
        new_summary = run_task_mode(
            mode_name="experimental",
            task_row=task_row,
            prepared_dir=prepared_dir,
            reference_runtime=reference_runtime,
            predictor_states=predictor_states,
            batch_size=int(args.batch_size),
            use_cuda_autocast=True,
            reuse_tensor_cache=True,
            use_optimized_predict=True,
            output_path=new_output_dir / f"task_{task_id:06d}.npz",
            gpu_index=int(args.gpu_index),
            log_dir=log_root,
        )
        diff_summary = compare_npz_outputs(
            old_output_dir / f"task_{task_id:06d}.npz",
            new_output_dir / f"task_{task_id:06d}.npz",
        )
        benchmark_rows.append(
            {
                "task_id": task_id,
                "tile_name": str(task_row["tile_name"]),
                "n_pixels": int(task_row["n_pixels"]),
                "baseline_seconds": old_summary["total_seconds"],
                "experimental_seconds": new_summary["total_seconds"],
                "speedup": old_summary["total_seconds"] / new_summary["total_seconds"],
                "baseline_mean_gpu_util": old_summary["gpu_summary"]["mean_gpu_utilization"],
                "experimental_mean_gpu_util": new_summary["gpu_summary"]["mean_gpu_utilization"],
                "baseline_forward_seconds": old_summary["member_forward_seconds_total"],
                "experimental_forward_seconds": new_summary["member_forward_seconds_total"],
                "baseline_convert_seconds": old_summary["member_convert_seconds_total"],
                "experimental_convert_seconds": new_summary["member_convert_seconds_total"],
                "lfmc_mean_max_abs_diff": diff_summary["lfmc_ens_mean"]["max_abs_diff"],
                "lfmc_std_max_abs_diff": diff_summary["lfmc_ens_std"]["max_abs_diff"],
                "lfmc_mean_rmse": diff_summary["lfmc_ens_mean"]["rmse"],
                "lfmc_std_rmse": diff_summary["lfmc_ens_std"]["rmse"],
                "lfmc_mean_allclose": diff_summary["lfmc_ens_mean"][
                    "allclose_atol_1e-4_rtol_1e-4"
                ],
                "lfmc_std_allclose": diff_summary["lfmc_ens_std"][
                    "allclose_atol_1e-4_rtol_1e-4"
                ],
            }
        )
        with (log_root / f"task_{task_id:06d}_diff.json").open("w") as f:
            json.dump(diff_summary, f, indent=2, sort_keys=True)
            f.write("\n")

    benchmark_path = log_root / "benchmark_summary.csv"
    with benchmark_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    rollup = {
        "gpu_index": int(args.gpu_index),
        "gpu_name": gpu_label,
        "run_dir": str(run_dir),
        "log_root": str(log_root),
        "output_root": str(output_root),
        "task_ids": [int(row["task_id"]) for row in task_rows],
        "mean_speedup": float(np.mean([row["speedup"] for row in benchmark_rows])),
        "mean_baseline_gpu_util": float(
            np.mean([row["baseline_mean_gpu_util"] for row in benchmark_rows])
        ),
        "mean_experimental_gpu_util": float(
            np.mean([row["experimental_mean_gpu_util"] for row in benchmark_rows])
        ),
    }
    with (log_root / "rollup.json").open("w") as f:
        json.dump(rollup, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(rollup, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
