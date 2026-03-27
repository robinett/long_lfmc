#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys

from map_config import default_map_config_path, get_cfg, load_map_config
here = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convenience entrypoint for the ensemble wall-to-wall map pipeline. "
            "By default it launches the interactive submission workflow that "
            "builds the manifest and submits the Slurm worker array."
        )
    )
    parser.add_argument("--config_path", type=str, default=default_map_config_path())
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--ensemble_root", type=str, default=None)
    parser.add_argument("--input_data_name", type=str, default=None)
    parser.add_argument("--ensemble_member_name_prefix", type=str, default=None)
    parser.add_argument("--inputs_root", type=str, default=None)
    parser.add_argument("--run_root", type=str, default=None)
    parser.add_argument("--grid_path", type=str, default=None)
    parser.add_argument("--validation_test", action="store_true")
    parser.add_argument("--no-validation_test", dest="validation_test", action="store_false")
    parser.set_defaults(validation_test=None)
    parser.add_argument("--interactive_validation_pipeline", action="store_true")
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--months_per_block", type=int, default=None)
    parser.add_argument("--requested_start_date", type=str, default=None)
    parser.add_argument("--requested_end_date", type=str, default=None)
    return parser.parse_args()


def _build_manifest_command(args):
    cmd = [
        sys.executable,
        os.path.join(here, "create_map_manifest.py"),
        "--config_path",
        str(args.config_path),
    ]
    if args.ensemble_root is not None:
        cmd.extend(["--ensemble_root", str(args.ensemble_root)])
    if args.input_data_name is not None:
        cmd.extend(["--input_data_name", str(args.input_data_name)])
    if args.ensemble_member_name_prefix is not None:
        cmd.extend(["--ensemble_member_name_prefix", str(args.ensemble_member_name_prefix)])
    if args.inputs_root is not None:
        cmd.extend(["--inputs_root", str(args.inputs_root)])
    if args.run_root is not None:
        cmd.extend(["--run_root", str(args.run_root)])
    if args.grid_path is not None:
        cmd.extend(["--grid_path", str(args.grid_path)])
    if args.months_per_block is not None:
        cmd.extend(["--months_per_block", str(args.months_per_block)])
    if args.requested_start_date is not None:
        cmd.extend(["--requested_start_date", str(args.requested_start_date)])
    if args.requested_end_date is not None:
        cmd.extend(["--requested_end_date", str(args.requested_end_date)])
    if args.validation_test is True:
        cmd.append("--validation_test")
    elif args.validation_test is False:
        cmd.append("--no-validation_test")
    if args.max_tiles is not None:
        cmd.extend(["--max_tiles", str(args.max_tiles)])
    return cmd


def _resolve_run_root(cfg, args):
    if args.run_root is not None:
        return str(args.run_root)
    return str(get_cfg(cfg, "paths", "run_root"))


def _latest_run_dir(run_root):
    candidates = [
        os.path.join(run_root, name)
        for name in os.listdir(run_root)
        if name.startswith("run_") and os.path.isdir(os.path.join(run_root, name))
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No run directories found under {run_root}")
    return max(candidates, key=os.path.getmtime)


def _run_local_validation_pipeline(args, cfg):
    validation_test = (
        args.validation_test
        if args.validation_test is not None
        else bool(get_cfg(cfg, "submission", "validation_test", default=True))
    )
    interactive_flag = bool(
        get_cfg(cfg, "submission", "interactive_validation_pipeline", default=False)
    )
    if not args.interactive_validation_pipeline and not interactive_flag:
        return False
    if not validation_test:
        raise ValueError(
            "Interactive validation pipeline is only allowed when validation_test is true"
        )
    if bool(get_cfg(cfg, "submission", "use_gpu_forward", default=False)):
        print(
            "[create_maps] interactive validation pipeline is running in local all-CPU mode; "
            "submission.use_gpu_forward only affects the Slurm submission path"
        )

    manifest_cmd = _build_manifest_command(args)
    print("[create_maps] Building validation manifest for local end-to-end run")
    print("[create_maps] Command:")
    print(" ".join(manifest_cmd))
    subprocess.run(manifest_cmd, check=True)

    run_root = _resolve_run_root(cfg, args)
    run_dir = _latest_run_dir(run_root)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    manifest_path = run_config["manifest_path"]
    model_type = str(run_config.get("model_type", get_cfg(cfg, "ensemble", "model_type", default="standard")))
    num_job_tasks = int(run_config["num_job_tasks"])
    num_merge_tasks = int(run_config["num_merge_tasks"])

    for job_task_id in range(num_job_tasks):
        print(
            f"[create_maps] Local worker job_task_id {job_task_id + 1}/{num_job_tasks}"
        )
        subprocess.run(
            [
                sys.executable,
                os.path.join(here, "run_map_task.py"),
                "--manifest_path",
                manifest_path,
                "--task_id",
                str(job_task_id),
                "--model_type",
                model_type,
            ],
            check=True,
        )

    print("[create_maps] Initializing merged store locally")
    subprocess.run(
        [
            sys.executable,
            os.path.join(here, "merge_map_shards.py"),
            "--manifest_path",
            manifest_path,
            "--initialize_only",
            "--overwrite",
        ],
        check=True,
    )

    for merge_task_id in range(num_merge_tasks):
        print(
            f"[create_maps] Local merge task {merge_task_id + 1}/{num_merge_tasks}"
        )
        subprocess.run(
            [
                sys.executable,
                os.path.join(here, "merge_map_shards.py"),
                "--manifest_path",
                manifest_path,
                "--merge_task_id",
                str(merge_task_id),
            ],
            check=True,
        )

    print(f"[create_maps] Running local validation for {run_dir}")
    subprocess.run(
        [
            sys.executable,
            os.path.join(here, "validate_map_outputs.py"),
            "--run_root",
            run_dir,
        ],
        check=True,
    )
    print(f"[create_maps] Interactive validation pipeline complete: {run_dir}")
    return True


def main():
    args = get_args()
    cfg = load_map_config(args.config_path)
    if _run_local_validation_pipeline(args, cfg):
        return
    if not args.manifest_only:
        submit_script = os.path.join(here, "submit_create_maps_ensemble.sh")
        print(f"[create_maps] Launching interactive submission script: {submit_script}")
        env = os.environ.copy()
        env["CONFIG_PATH"] = str(args.config_path)
        validation_test = (
            args.validation_test
            if args.validation_test is not None
            else bool(get_cfg(cfg, "submission", "validation_test", default=True))
        )
        env["VALIDATION_TEST"] = "true" if validation_test else "false"
        max_tiles = (
            args.max_tiles
            if args.max_tiles is not None
            else get_cfg(cfg, "submission", "max_tiles", default="")
        )
        if max_tiles not in {None, "", "None"}:
            env["MAX_TILES"] = str(max_tiles)
        months_per_block = (
            args.months_per_block
            if args.months_per_block is not None
            else get_cfg(cfg, "chunking", "months_per_block", default=1)
        )
        env["MONTHS_PER_BLOCK"] = str(months_per_block)
        requested_start_date = (
            args.requested_start_date
            if args.requested_start_date is not None
            else get_cfg(cfg, "data", "requested_start_date", default="")
        )
        if requested_start_date not in {None, "", "None"}:
            env["REQUESTED_START_DATE"] = str(requested_start_date)
        requested_end_date = (
            args.requested_end_date
            if args.requested_end_date is not None
            else get_cfg(cfg, "data", "requested_end_date", default="2024-12-31")
        )
        env["REQUESTED_END_DATE"] = str(requested_end_date)
        env["TIME_CHUNK_DAYS"] = str(get_cfg(cfg, "chunking", "time_chunk_days", default=31))
        env["Y_CHUNK"] = str(get_cfg(cfg, "chunking", "y_chunk", default=100))
        env["X_CHUNK"] = str(get_cfg(cfg, "chunking", "x_chunk", default=100))
        env["ARRAY_CONCURRENCY"] = str(get_cfg(cfg, "submission", "array_concurrency", default=32))
        env["TASKS_PER_JOB"] = str(get_cfg(cfg, "submission", "tasks_per_job", default=1))
        env["USE_GPU_FORWARD"] = str(
            get_cfg(cfg, "submission", "use_gpu_forward", default=False)
        ).lower()
        env["GPU_FINE_TASKS_PER_JOB"] = str(
            get_cfg(cfg, "submission", "gpu_fine_tasks_per_job", default=1)
        )
        env["GPU_MAX_JOBS"] = str(
            get_cfg(cfg, "submission", "gpu_max_jobs", default=8)
        )
        env["GPU_LOCK_DIR"] = str(
            get_cfg(
                cfg,
                "submission",
                "gpu_lock_dir",
                default="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks",
            )
        )
        env["GPU_SUBMIT_SLEEP_SECONDS"] = str(
            get_cfg(cfg, "submission", "gpu_submit_sleep_seconds", default=30)
        )
        env["GPU_TIME_LIMIT"] = str(
            get_cfg(cfg, "submission", "gpu_time_limit", default="02:00:00")
        )
        env["GPU_CPUS_PER_TASK"] = str(
            get_cfg(cfg, "submission", "gpu_cpus_per_task", default=4)
        )
        env["GPU_MEM"] = str(
            get_cfg(cfg, "submission", "gpu_mem", default="32G")
        )
        env["CLEANUP_PREPARED_TENSORS_AFTER_SUCCESS"] = str(
            get_cfg(cfg, "submission", "cleanup_prepared_tensors_after_success", default=False)
        ).lower()
        env["WAIT_FOR_VALIDATION_COMPLETION"] = str(
            get_cfg(cfg, "submission", "wait_for_validation_completion", default=False)
        ).lower()
        env["MODEL_TYPE"] = str(get_cfg(cfg, "ensemble", "model_type", default="standard"))
        if args.ensemble_root is not None:
            env["ENSEMBLE_ROOT"] = str(args.ensemble_root)
        if args.input_data_name is not None:
            env["INPUT_DATA_NAME"] = str(args.input_data_name)
        if args.ensemble_member_name_prefix is not None:
            env["ENSEMBLE_MEMBER_NAME_PREFIX"] = str(args.ensemble_member_name_prefix)
        if args.inputs_root is not None:
            env["INPUTS_ROOT"] = str(args.inputs_root)
        if args.run_root is not None:
            env["RUN_ROOT"] = str(args.run_root)
        if args.grid_path is not None:
            env["GRID_PATH"] = str(args.grid_path)
        subprocess.run(["bash", submit_script], check=True, env=env)
        return

    cmd = _build_manifest_command(args)
    print("[create_maps] Building map manifest only")
    print("[create_maps] Command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
