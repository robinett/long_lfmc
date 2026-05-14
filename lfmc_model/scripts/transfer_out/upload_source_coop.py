#!/usr/bin/env python3

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml


def timestamped_message(message: str) -> str:
    return f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def default_config_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / "source_coop_transfer_configs.yaml")


def get_args():
    parser = argparse.ArgumentParser(
        description="Upload a local dataset directory to a Source Cooperative product prefix."
    )
    parser.add_argument("--config_path", type=str, default=default_config_path())
    parser.add_argument("--dataset_key", type=str, default="example_lfmc_maps")
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--product_prefix", type=str, default=None)
    parser.add_argument("--destination_relpath", type=str, default=None)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--endpoint_url", type=str, default=None)
    parser.add_argument("--credentials_path", type=str, default=None)
    parser.add_argument("--acl", choices=["none", "bucket-owner-full-control"], default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--aws_executable", type=str, default="aws")
    parser.add_argument("--delete_extra_remote_files", action="store_true")
    parser.add_argument(
        "--no-delete_extra_remote_files",
        dest="delete_extra_remote_files",
        action="store_false",
    )
    parser.set_defaults(delete_extra_remote_files=None)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_bool(override: Optional[bool], default: bool) -> bool:
    if override is None:
        return bool(default)
    return bool(override)


def strip_slashes(value: str) -> str:
    return value.strip().strip("/")


def require_config_value(value, name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required configuration value: {name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing required configuration value: {name}")
    return text


def parse_credentials_file(credentials_path: Path) -> Dict[str, str]:
    text = credentials_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Credential file is empty: {credentials_path}")

    if text.startswith("{"):
        payload = json.loads(text)
        env = {}
        key_map = {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
            "region_name": "AWS_DEFAULT_REGION",
        }
        for src_key, env_key in key_map.items():
            value = payload.get(src_key)
            if value:
                env[env_key] = str(value)
        return env

    env = {}
    line_re = re.compile(r"^(?:export\s+)?([A-Z0-9_]+)=(.*)$")
    label_map = {
        "access key": "AWS_ACCESS_KEY_ID",
        "secret access key": "AWS_SECRET_ACCESS_KEY",
        "session token": "AWS_SESSION_TOKEN",
        "region": "AWS_DEFAULT_REGION",
        "default region": "AWS_DEFAULT_REGION",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_re.match(line)
        if match is not None:
            key = match.group(1)
            value = match.group(2).strip()
            if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                value = value[1:-1]
            env[key] = value
            continue
        if ":" not in line:
            continue
        label, value = [part.strip() for part in line.split(":", maxsplit=1)]
        env_key = label_map.get(label.lower())
        if env_key and value:
            env[env_key] = value
    return env


def local_inventory(source_path: Path, logger: Optional[TeeLogger] = None) -> Tuple[int, int]:
    file_count = 0
    total_bytes = 0
    started = time.time()
    last_log_time = started
    for path in iter_local_files(source_path):
        file_count += 1
        total_bytes += path.stat().st_size
        now = time.time()
        if logger is not None and (file_count == 1 or file_count % 100000 == 0 or now - last_log_time >= 30):
            elapsed = now - started
            rate = file_count / elapsed if elapsed > 0 else 0.0
            logger.write(
                timestamped_message(
                    "Local inventory progress: "
                    f"{file_count} files, {human_bytes(total_bytes)}, "
                    f"{rate:.0f} files/s"
                )
            )
            last_log_time = now
    return file_count, total_bytes


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_seconds(total_seconds: float) -> str:
    rounded = max(0, int(round(total_seconds)))
    hours = rounded // 3600
    minutes = (rounded % 3600) // 60
    seconds = rounded % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_progress_message(
    uploaded_files: int,
    total_files: int,
    uploaded_bytes: int,
    total_bytes: int,
    elapsed_seconds: float,
) -> str:
    percent_complete = 100.0 if total_bytes <= 0 else 100.0 * uploaded_bytes / total_bytes
    transfer_rate_bytes_per_second = 0.0 if elapsed_seconds <= 0 else uploaded_bytes / elapsed_seconds
    remaining_bytes = max(0, total_bytes - uploaded_bytes)
    eta_seconds = 0.0
    if transfer_rate_bytes_per_second > 0:
        eta_seconds = remaining_bytes / transfer_rate_bytes_per_second
    rate_text = f"{human_bytes(int(round(transfer_rate_bytes_per_second)))}/s"
    return (
        f"Upload progress: {percent_complete:5.1f}% | "
        f"files {uploaded_files}/{total_files} | "
        f"bytes {human_bytes(uploaded_bytes)}/{human_bytes(total_bytes)} | "
        f"elapsed {format_seconds(elapsed_seconds)} | "
        f"rate {rate_text} | "
        f"ETA {format_seconds(eta_seconds)}"
    )


def build_remote_uri(bucket: str, product_prefix: str, destination_relpath: str) -> str:
    bucket = strip_slashes(bucket)
    product_prefix = strip_slashes(product_prefix)
    destination_relpath = strip_slashes(destination_relpath)
    if not product_prefix:
        raise ValueError("product_prefix must not be empty")
    if not destination_relpath:
        raise ValueError("destination_relpath must not be empty")
    return f"s3://{bucket}/{product_prefix}/{destination_relpath}"


def build_remote_key_prefix(product_prefix: str, destination_relpath: str) -> str:
    product_prefix = strip_slashes(product_prefix)
    destination_relpath = strip_slashes(destination_relpath)
    return f"{product_prefix}/{destination_relpath}"


def resolve_aws_cli(aws_executable: str, env: Dict[str, str], logger: TeeLogger) -> Optional[str]:
    aws_path = shutil.which(aws_executable)
    if aws_path is None:
        logger.write(
            timestamped_message(
                f"AWS CLI executable {aws_executable!r} was not found on PATH. Falling back to boto3."
            )
        )
        return None

    version_check = subprocess.run(
        [aws_path, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    if version_check.returncode != 0:
        logger.write(
            timestamped_message(
                f"AWS CLI validation failed for {aws_path}. Falling back to boto3."
            )
        )
        for line in version_check.stdout.splitlines():
            logger.write(line)
        return None

    version_line = version_check.stdout.strip().splitlines()[0] if version_check.stdout.strip() else "unknown"
    logger.write(timestamped_message(f"AWS CLI: {aws_path}"))
    logger.write(timestamped_message(f"AWS CLI version: {version_line}"))
    return aws_path


def iter_local_files(source_path: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(source_path):
        dirs[:] = [name for name in dirs if not name.startswith(".nfs")]
        for filename in files:
            yield Path(root) / filename


def build_boto3_client(region: str, env: Dict[str, str], profile: Optional[str], endpoint_url: Optional[str]):
    import boto3

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    elif env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        session_kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        session_kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
        if env.get("AWS_SESSION_TOKEN"):
            session_kwargs["aws_session_token"] = env["AWS_SESSION_TOKEN"]
    session = boto3.Session(**session_kwargs)
    client_kwargs = {"region_name": region}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    return session.client("s3", **client_kwargs)


def list_remote_objects_boto3(client, bucket: str, key_prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    remote_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{key_prefix}/"):
        for obj in page.get("Contents", []):
            remote_objects.append(
                {
                    "Key": obj["Key"],
                    "Size": int(obj["Size"]),
                    "ETag": str(obj.get("ETag", "")).strip('"'),
                }
            )
    return remote_objects


def file_md5_hex(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_object_matches_local(
    client,
    bucket: str,
    path: Path,
    remote_obj: Dict[str, object],
    match_mode: str,
) -> bool:
    local_size = path.stat().st_size
    remote_size = int(remote_obj["Size"])
    if local_size != remote_size:
        return False
    if match_mode == "size":
        return True

    etag = str(remote_obj.get("ETag", "")).strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", etag):
        return file_md5_hex(path).lower() == etag.lower()
    if match_mode == "etag":
        return False

    try:
        head = client.head_object(Bucket=bucket, Key=str(remote_obj["Key"]))
    except Exception:
        return False
    metadata = head.get("Metadata", {})
    remote_md5 = str(metadata.get("local-md5", "")).strip().lower()
    if not remote_md5:
        return False
    return file_md5_hex(path).lower() == remote_md5


def upload_extra_args(path: Path, acl: str) -> Dict[str, object]:
    stat = path.stat()
    extra_args = {
        "Metadata": {
            "local-size": str(stat.st_size),
            "local-mtime-ns": str(stat.st_mtime_ns),
        }
    }
    if acl != "none":
        extra_args["ACL"] = acl
    return extra_args


def upload_with_boto3(
    client,
    source_path: Path,
    bucket: str,
    key_prefix: str,
    acl: str,
    logger: TeeLogger,
    dry_run: bool,
    total_bytes: int,
    max_workers: int,
    transfer_max_concurrency: int,
    match_mode: str,
    single_part_threshold_gb: int,
):
    from boto3.s3.transfer import TransferConfig

    uploaded_files = 0
    uploaded_bytes = 0
    start_time = time.time()
    last_progress_time = start_time
    progress_lock = threading.Lock()
    transfer_config = TransferConfig(
        max_concurrency=transfer_max_concurrency,
        multipart_threshold=single_part_threshold_gb * 1024**3,
        use_threads=transfer_max_concurrency > 1,
    )

    local_files = list(iter_local_files(source_path))
    total_files = len(local_files)
    remote_objects = list_remote_objects_boto3(client, bucket=bucket, key_prefix=key_prefix)
    remote_by_key = {str(obj["Key"]): obj for obj in remote_objects}
    upload_files = []
    upload_bytes = 0
    skipped_files = 0
    skipped_bytes = 0
    for path in local_files:
        rel_path = path.relative_to(source_path).as_posix()
        key = f"{key_prefix}/{rel_path}"
        file_size = path.stat().st_size
        remote_obj = remote_by_key.get(key)
        if remote_obj is not None and remote_object_matches_local(client, bucket, path, remote_obj, match_mode):
            skipped_files += 1
            skipped_bytes += file_size
        else:
            upload_files.append(path)
            upload_bytes += file_size

    logger.write(
        timestamped_message(
            f"Uploading with boto3: {len(upload_files)}/{total_files} files need transfer, "
            f"{human_bytes(upload_bytes)}/{human_bytes(total_bytes)} need transfer, "
            f"skipping {skipped_files} existing matching files ({human_bytes(skipped_bytes)}), "
            f"max_workers={max_workers}, transfer_max_concurrency={transfer_max_concurrency}, "
            f"match_mode={match_mode}, single_part_threshold_gb={single_part_threshold_gb}"
        )
    )

    def maybe_log_progress(now: float) -> None:
        nonlocal last_progress_time
        should_log_progress = (
            uploaded_files == len(upload_files)
            or uploaded_files == 1
            or uploaded_files % 100 == 0
            or (now - last_progress_time) >= 15.0
        )
        if should_log_progress:
            logger.write(
                timestamped_message(
                    format_progress_message(
                        uploaded_files=uploaded_files,
                        total_files=len(upload_files),
                        uploaded_bytes=uploaded_bytes,
                        total_bytes=upload_bytes,
                        elapsed_seconds=now - start_time,
                    )
                )
            )
            last_progress_time = now

    def upload_one(path: Path) -> Tuple[Path, int]:
        rel_path = path.relative_to(source_path).as_posix()
        key = f"{key_prefix}/{rel_path}"
        file_size = path.stat().st_size
        if dry_run:
            return path, file_size
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs=upload_extra_args(path, acl),
            Config=transfer_config,
        )
        return path, file_size

    if dry_run:
        for path in upload_files[:10]:
            rel_path = path.relative_to(source_path).as_posix()
            key = f"{key_prefix}/{rel_path}"
            logger.write(timestamped_message(f"Dry run upload candidate: s3://{bucket}/{key}"))
        logger.write(timestamped_message("Dry run enabled. Uploads were not executed."))
        return

    if not upload_files:
        logger.write(timestamped_message("No files need upload."))
        return

    def submit_next(executor, path_iter, future_to_path) -> bool:
        try:
            path = next(path_iter)
        except StopIteration:
            return False
        future = executor.submit(upload_one, path)
        future_to_path[future] = path
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        path_iter = iter(upload_files)
        future_to_path = {}
        for _ in range(max_workers):
            if not submit_next(executor, path_iter, future_to_path):
                break

        while future_to_path:
            done, _ = concurrent.futures.wait(
                future_to_path,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                path = future_to_path.pop(future)
                _, file_size = future.result()
                submit_next(executor, path_iter, future_to_path)
                with progress_lock:
                    uploaded_files += 1
                    uploaded_bytes += file_size
                    maybe_log_progress(time.time())


def delete_extra_remote_objects_boto3(
    client,
    bucket: str,
    source_path: Path,
    key_prefix: str,
    logger: TeeLogger,
    dry_run: bool,
):
    desired_keys = {f"{key_prefix}/{path.relative_to(source_path).as_posix()}" for path in iter_local_files(source_path)}
    remote_objects = list_remote_objects_boto3(client, bucket=bucket, key_prefix=key_prefix)
    delete_keys = [obj["Key"] for obj in remote_objects if obj["Key"] not in desired_keys]
    if not delete_keys:
        logger.write(timestamped_message("No extra remote objects to delete."))
        return

    logger.write(timestamped_message(f"Deleting {len(delete_keys)} extra remote objects"))
    if dry_run:
        for key in delete_keys[:10]:
            logger.write(timestamped_message(f"Dry run delete candidate: s3://{bucket}/{key}"))
        return

    for start in range(0, len(delete_keys), 1000):
        chunk = delete_keys[start:start + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
        )


def verify_with_boto3(client, bucket: str, key_prefix: str, logger: TeeLogger, dry_run: bool):
    if dry_run:
        logger.write(timestamped_message("Dry run enabled. Verification command was not executed."))
        return None, None
    remote_objects = list_remote_objects_boto3(client, bucket=bucket, key_prefix=key_prefix)
    remote_file_count = len(remote_objects)
    remote_total_bytes = sum(obj["Size"] for obj in remote_objects)
    logger.write(
        timestamped_message(
            f"Remote inventory: {remote_file_count} files, {human_bytes(remote_total_bytes)}"
        )
    )
    return remote_file_count, remote_total_bytes


def stream_command(command, env: Dict[str, str], logger: TeeLogger, dry_run: bool) -> int:
    logger.write(timestamped_message(f"Running command: {' '.join(command)}"))
    if dry_run:
        logger.write(timestamped_message("Dry run enabled. Command was not executed."))
        return 0

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.write(line.rstrip("\n"))
    return process.wait()


def parse_aws_ls_summary(output_lines) -> Tuple[Optional[int], Optional[int]]:
    file_count = None
    total_bytes = None
    for line in output_lines:
        stripped = line.strip()
        if stripped.startswith("Total Objects:"):
            file_count = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Total Size:"):
            total_bytes = int(stripped.split(":", 1)[1].strip())
    return file_count, total_bytes


def run_and_capture(command, env: Dict[str, str], logger: TeeLogger, dry_run: bool):
    logger.write(timestamped_message(f"Running command: {' '.join(command)}"))
    if dry_run:
        logger.write(timestamped_message("Dry run enabled. Verification command was not executed."))
        return 0, []

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    lines = completed.stdout.splitlines()
    for line in lines:
        logger.write(line)
    return completed.returncode, lines


def main():
    args = get_args()
    config_path = Path(args.config_path).resolve()
    config = load_config(config_path)

    source_cfg = config.get("source_coop", {})
    datasets_cfg = config.get("datasets", {})
    dataset_cfg = datasets_cfg.get(args.dataset_key, {})

    region = require_config_value(args.region or source_cfg.get("region", None), "source_coop.region")
    endpoint_url = str(args.endpoint_url or source_cfg.get("endpoint_url", "")).strip()
    bucket = require_config_value(args.bucket or source_cfg.get("bucket", None), "source_coop.bucket")
    product_prefix = require_config_value(
        args.product_prefix or source_cfg.get("product_prefix", None),
        "source_coop.product_prefix",
    )
    credentials_path = require_config_value(
        args.credentials_path or source_cfg.get("credentials_path", None),
        "source_coop.credentials_path",
    )
    destination_relpath = require_config_value(
        args.destination_relpath or dataset_cfg.get("destination_relpath", None),
        f"datasets.{args.dataset_key}.destination_relpath",
    )
    artifact_type = str(dataset_cfg.get("artifact_type", "zarr")).strip().lower()
    source_path = Path(
        require_config_value(
            args.source_path or dataset_cfg.get("source_path", None),
            f"datasets.{args.dataset_key}.source_path",
        )
    ).expanduser()
    acl = str(args.acl or source_cfg.get("acl", "none"))
    verify_after_upload = not args.skip_verify and bool(source_cfg.get("verify_after_upload", True))
    delete_extra_remote_files = resolve_bool(
        args.delete_extra_remote_files,
        bool(source_cfg.get("delete_extra_remote_files", False)),
    )
    boto3_max_workers = int(source_cfg.get("boto3_max_workers", 16))
    if boto3_max_workers < 1:
        raise ValueError("source_coop.boto3_max_workers must be at least 1")
    boto3_transfer_max_concurrency = int(source_cfg.get("boto3_transfer_max_concurrency", 1))
    if boto3_transfer_max_concurrency < 1:
        raise ValueError("source_coop.boto3_transfer_max_concurrency must be at least 1")
    boto3_match_mode = str(source_cfg.get("boto3_match_mode", "metadata")).strip().lower()
    if boto3_match_mode not in {"size", "etag", "metadata"}:
        raise ValueError("source_coop.boto3_match_mode must be one of: size, etag, metadata")
    boto3_single_part_threshold_gb = int(source_cfg.get("boto3_single_part_threshold_gb", 5))
    if boto3_single_part_threshold_gb < 1:
        raise ValueError("source_coop.boto3_single_part_threshold_gb must be at least 1")

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_path}")
    if artifact_type == "zarr":
        has_zarr_v3 = (source_path / "zarr.json").exists()
        has_zarr_v2 = (source_path / ".zgroup").exists()
        if not has_zarr_v3 and not has_zarr_v2:
            raise ValueError(
                "Expected a Zarr store with either zarr.json (Zarr v3) "
                f"or .zgroup (Zarr v2) at the top level, but found neither in {source_path}"
            )
    elif artifact_type != "directory":
        raise ValueError(
            f"Unsupported artifact_type {artifact_type!r} for dataset {args.dataset_key!r}; "
            "expected 'zarr' or 'directory'"
        )

    log_dir = Path(__file__).resolve().parent / "logs"
    log_name = f"source_coop_upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = TeeLogger(log_dir / log_name)

    try:
        logger.write(timestamped_message(f"Config path: {config_path}"))
        logger.write(timestamped_message(f"Dataset key: {args.dataset_key}"))
        logger.write(timestamped_message(f"Artifact type: {artifact_type}"))
        logger.write(timestamped_message(f"Source path: {source_path}"))

        local_file_count, local_total_bytes = local_inventory(source_path, logger=logger)
        logger.write(
            timestamped_message(
                f"Local inventory: {local_file_count} files, {human_bytes(local_total_bytes)}"
            )
        )

        remote_uri = build_remote_uri(bucket, product_prefix, destination_relpath)
        logger.write(timestamped_message(f"Remote URI: {remote_uri}"))
        logger.write(timestamped_message(f"AWS region: {region}"))
        if endpoint_url:
            logger.write(timestamped_message(f"S3 endpoint URL: {endpoint_url}"))
        logger.write(
            timestamped_message(
                "Remote delete behavior: "
                + ("enabled" if delete_extra_remote_files else "disabled")
            )
        )
        logger.write(timestamped_message(f"ACL mode: {acl}"))

        env = os.environ.copy()
        env["AWS_DEFAULT_REGION"] = region
        if args.profile:
            env["AWS_PROFILE"] = args.profile
        credentials_path = Path(str(credentials_path)).expanduser()
        credentials_env = parse_credentials_file(credentials_path)
        env.update(credentials_env)
        logger.write(
            timestamped_message(
                f"Loaded AWS credentials from {credentials_path}"
            )
        )

        aws_path = resolve_aws_cli(args.aws_executable, env=env, logger=logger)
        if aws_path is not None:
            sync_command = [aws_path, "s3", "sync", str(source_path), remote_uri]
            if endpoint_url:
                sync_command.extend(["--endpoint-url", endpoint_url])
            if acl != "none":
                sync_command.extend(["--acl", acl])
            if delete_extra_remote_files:
                sync_command.append("--delete")
            if args.dry_run:
                sync_command.append("--dryrun")

            sync_returncode = stream_command(sync_command, env=env, logger=logger, dry_run=False)
            if sync_returncode != 0:
                raise RuntimeError(f"Upload failed with exit code {sync_returncode}")

            if verify_after_upload:
                verify_command = [
                    aws_path,
                    "s3",
                    "ls",
                    remote_uri,
                    "--recursive",
                    "--summarize",
                ]
                if endpoint_url:
                    verify_command.extend(["--endpoint-url", endpoint_url])
                verify_returncode, verify_lines = run_and_capture(
                    verify_command,
                    env=env,
                    logger=logger,
                    dry_run=args.dry_run,
                )
                if verify_returncode != 0:
                    raise RuntimeError(f"Verification failed with exit code {verify_returncode}")

                remote_file_count, remote_total_bytes = parse_aws_ls_summary(verify_lines)
            else:
                remote_file_count, remote_total_bytes = None, None
        else:
            logger.write(timestamped_message("Using boto3 upload backend"))
            key_prefix = build_remote_key_prefix(product_prefix, destination_relpath)
            client = build_boto3_client(
                region=region,
                env=env,
                profile=args.profile,
                endpoint_url=endpoint_url,
            )
            upload_with_boto3(
                client=client,
                source_path=source_path,
                bucket=bucket,
                key_prefix=key_prefix,
                acl=acl,
                logger=logger,
                dry_run=args.dry_run,
                total_bytes=local_total_bytes,
                max_workers=boto3_max_workers,
                transfer_max_concurrency=boto3_transfer_max_concurrency,
                match_mode=boto3_match_mode,
                single_part_threshold_gb=boto3_single_part_threshold_gb,
            )
            if delete_extra_remote_files:
                delete_extra_remote_objects_boto3(
                    client=client,
                    bucket=bucket,
                    source_path=source_path,
                    key_prefix=key_prefix,
                    logger=logger,
                    dry_run=args.dry_run,
                )
            if verify_after_upload:
                remote_file_count, remote_total_bytes = verify_with_boto3(
                    client=client,
                    bucket=bucket,
                    key_prefix=key_prefix,
                    logger=logger,
                    dry_run=args.dry_run,
                )
            else:
                remote_file_count, remote_total_bytes = None, None

        if (
            remote_file_count is not None
            and remote_total_bytes is not None
            and not args.dry_run
            and (
                remote_file_count != local_file_count
                or remote_total_bytes != local_total_bytes
            )
        ):
            raise RuntimeError(
                "Remote inventory does not match the local inventory. "
                f"local_files={local_file_count}, remote_files={remote_file_count}, "
                f"local_bytes={local_total_bytes}, remote_bytes={remote_total_bytes}"
            )

        logger.write(timestamped_message(f"Upload workflow completed. Log saved to {logger.log_path}"))
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(timestamped_message(f"ERROR: {exc}"), file=sys.stderr, flush=True)
        raise
