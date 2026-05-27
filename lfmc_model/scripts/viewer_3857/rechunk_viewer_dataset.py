#!/usr/bin/env python3

import argparse
import concurrent.futures
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
import zarr


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "viewer_pipeline_config.yaml"
DEFAULT_DESTINATION_SUFFIX = "_rechunk_t32.zarr"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def default_destination_path(source_path: Path) -> Path:
    source_text = str(source_path)
    if source_text.endswith(".zarr"):
        return Path(source_text[:-5] + DEFAULT_DESTINATION_SUFFIX)
    return source_path.with_name(source_path.name + DEFAULT_DESTINATION_SUFFIX)


def chunk_count(shape: Sequence[int], chunks: Sequence[int]) -> int:
    total = 1
    for size, chunk in zip(shape, chunks):
        total *= int(math.ceil(int(size) / int(chunk)))
    return total


def available_inodes(path: Path) -> int:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    stat = os.statvfs(target)
    return int(stat.f_favail)


def array_names(root) -> List[str]:
    return sorted(str(name) for name in root.array_keys())


def attrs_dict(attrs) -> Dict[str, object]:
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else attrs)


def target_chunks(
    name: str,
    shape: Tuple[int, ...],
    current_chunks: Tuple[int, ...],
    *,
    time_chunk_size: int,
    spatial_chunk_size: int,
) -> Tuple[int, ...]:
    if name in {"lfmc_ens_mean", "lfmc_ens_std"}:
        return (
            min(shape[0], time_chunk_size),
            min(shape[1], spatial_chunk_size),
            min(shape[2], spatial_chunk_size),
        )
    if name in {"time", "quality_flag"}:
        return (min(shape[0], time_chunk_size),)
    if name == "dominant_landcover_code":
        return (
            min(shape[0], current_chunks[0]),
            min(shape[1], spatial_chunk_size),
            min(shape[2], spatial_chunk_size),
        )
    if name in {"lat", "lon"}:
        return (
            min(shape[0], spatial_chunk_size),
            min(shape[1], spatial_chunk_size),
        )
    return tuple(int(value) for value in current_chunks)


def estimate_target_chunks(root, *, time_chunk_size: int, spatial_chunk_size: int) -> Tuple[int, List[Tuple[str, int, Tuple[int, ...]]]]:
    details = []
    total = 0
    for name in array_names(root):
        arr = root[name]
        chunks = target_chunks(
            name,
            tuple(int(value) for value in arr.shape),
            tuple(int(value) for value in arr.chunks),
            time_chunk_size=time_chunk_size,
            spatial_chunk_size=spatial_chunk_size,
        )
        count = chunk_count(arr.shape, chunks) if arr.shape else 1
        details.append((name, count, chunks))
        total += count
    return total, details


def create_destination_arrays(
    source_path: Path,
    destination_path: Path,
    *,
    time_chunk_size: int,
    spatial_chunk_size: int,
    overwrite: bool,
) -> None:
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination exists: {destination_path}. Pass --overwrite to replace it."
            )
        log(f"Removing existing destination {destination_path}")
        shutil.rmtree(destination_path)

    log(f"Opening source metadata from {source_path}")
    source_root = zarr.open_group(str(source_path), mode="r")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dest_root = zarr.open_group(
        str(destination_path),
        mode="w",
        zarr_format=2,
        attributes=attrs_dict(source_root.attrs),
    )

    for name in array_names(source_root):
        src = source_root[name]
        chunks = target_chunks(
            name,
            tuple(int(value) for value in src.shape),
            tuple(int(value) for value in src.chunks),
            time_chunk_size=time_chunk_size,
            spatial_chunk_size=spatial_chunk_size,
        )
        log(f"Creating {name}: shape={src.shape} dtype={src.dtype} chunks={chunks}")
        dest_root.create_array(
            name,
            shape=src.shape,
            dtype=src.dtype,
            chunks=chunks,
            compressor=getattr(src, "compressor", "auto"),
            filters=getattr(src, "filters", "auto"),
            fill_value=getattr(src, "fill_value", None),
            attributes=attrs_dict(src.attrs),
            overwrite=True,
        )


def block_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterable[Tuple[slice, ...]]:
    ranges = [range(0, int(size), int(chunk)) for size, chunk in zip(shape, chunks)]
    if len(shape) == 1:
        for start0 in ranges[0]:
            yield (slice(start0, min(start0 + chunks[0], shape[0])),)
    elif len(shape) == 2:
        for start0 in ranges[0]:
            for start1 in ranges[1]:
                yield (
                    slice(start0, min(start0 + chunks[0], shape[0])),
                    slice(start1, min(start1 + chunks[1], shape[1])),
                )
    elif len(shape) == 3:
        for start0 in ranges[0]:
            for start1 in ranges[1]:
                for start2 in ranges[2]:
                    yield (
                        slice(start0, min(start0 + chunks[0], shape[0])),
                        slice(start1, min(start1 + chunks[1], shape[1])),
                        slice(start2, min(start2 + chunks[2], shape[2])),
                    )
    else:
        raise ValueError(f"Unsupported array rank {len(shape)} for block-wise copy")


def copy_array(src, dst, name: str, *, max_blocks: int = None, workers: int = 1) -> None:
    chunks = tuple(int(value) for value in dst.chunks)
    expected_blocks = chunk_count(dst.shape, chunks) if dst.shape else 1
    log(f"Copying {name}: {expected_blocks:,} destination chunks")

    if not dst.shape:
        dst[...] = src[...]
        log(f"Completed scalar array {name}")
        return

    slices_iter = block_slices(tuple(int(value) for value in dst.shape), chunks)
    if max_blocks is not None:
        slices_iter = (slices for block_idx, slices in enumerate(slices_iter) if block_idx < max_blocks)

    copied_blocks = 0
    start_time = time.time()

    def copy_one(slices: Tuple[slice, ...]) -> None:
        dst[slices] = src[slices]

    def note_progress() -> None:
        nonlocal copied_blocks
        copied_blocks += 1
        target_blocks = expected_blocks if max_blocks is None else min(max_blocks, expected_blocks)
        if copied_blocks == 1 or copied_blocks % 500 == 0 or copied_blocks == target_blocks:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = copied_blocks / elapsed
            log(
                f"{name}: copied {copied_blocks:,}/{target_blocks:,} chunks "
                f"({rate:.1f} chunks/s)"
            )

    if workers <= 1:
        for slices in slices_iter:
            copy_one(slices)
            note_progress()
    else:
        max_pending = max(workers * 2, workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            for slices in slices_iter:
                pending.add(executor.submit(copy_one, slices))
                if len(pending) >= max_pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        future.result()
                        note_progress()
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                    note_progress()

    if max_blocks is not None and copied_blocks >= max_blocks:
        log(f"Stopped {name} after --max-blocks={max_blocks}")


def copy_data(
    source_path: Path,
    destination_path: Path,
    *,
    max_blocks_per_array: int = None,
    workers: int = 1,
) -> None:
    log("Reopening source and destination after structural creation")
    source_root = zarr.open_group(str(source_path), mode="r")
    dest_root = zarr.open_group(str(destination_path), mode="a")
    for name in array_names(source_root):
        copy_array(
            source_root[name],
            dest_root[name],
            name,
            max_blocks=max_blocks_per_array,
            workers=workers,
        )


def sample_indices(size: int) -> List[int]:
    if size <= 0:
        return []
    candidates = [0, size // 2, size - 1]
    return sorted(set(int(value) for value in candidates if 0 <= value < size))


def validate_sampled_values(source_path: Path, destination_path: Path) -> None:
    log("Validating sampled values")
    source_root = zarr.open_group(str(source_path), mode="r")
    dest_root = zarr.open_group(str(destination_path), mode="r")
    source_names = array_names(source_root)
    dest_names = array_names(dest_root)
    if source_names != dest_names:
        raise AssertionError(f"Array names differ: source={source_names} dest={dest_names}")

    for name in source_names:
        src = source_root[name]
        dst = dest_root[name]
        if tuple(src.shape) != tuple(dst.shape):
            raise AssertionError(f"{name} shape differs: {src.shape} vs {dst.shape}")
        if np.dtype(src.dtype) != np.dtype(dst.dtype):
            raise AssertionError(f"{name} dtype differs: {src.dtype} vs {dst.dtype}")

        if not src.shape:
            src_values = np.asarray(src[...])
            dst_values = np.asarray(dst[...])
        elif len(src.shape) == 1:
            idx = sample_indices(src.shape[0])
            src_values = np.asarray(src[idx])
            dst_values = np.asarray(dst[idx])
        elif len(src.shape) == 2:
            ys = sample_indices(src.shape[0])
            xs = sample_indices(src.shape[1])
            src_values = np.asarray([src[y, x] for y in ys for x in xs])
            dst_values = np.asarray([dst[y, x] for y in ys for x in xs])
        elif len(src.shape) == 3:
            ts = sample_indices(src.shape[0])
            ys = sample_indices(src.shape[1])
            xs = sample_indices(src.shape[2])
            src_values = np.asarray([src[t, y, x] for t in ts for y in ys for x in xs])
            dst_values = np.asarray([dst[t, y, x] for t in ts for y in ys for x in xs])
        else:
            continue

        if not np.array_equal(src_values, dst_values, equal_nan=True):
            raise AssertionError(f"{name} sampled values differ")
        log(f"Validated {name}")


def consolidate_and_reopen(destination_path: Path) -> None:
    log(f"Consolidating metadata for {destination_path}")
    zarr.consolidate_metadata(str(destination_path))
    log("Reopening consolidated destination metadata")
    zarr.open_consolidated(str(destination_path), mode="r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rechunk the EPSG:3857 LFMC viewer Zarr store.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--time-chunk-size", type=int, default=32)
    parser.add_argument("--spatial-chunk-size", type=int, default=256)
    parser.add_argument("--min-free-inodes", type=int, default=1_500_000)
    parser.add_argument("--inode-safety-margin", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-blocks-per-array",
        type=int,
        default=None,
        help="Copy only this many destination chunks per array for smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    source_path = args.source or Path(str(cfg["output"]["viewer_dataset_path"]))
    destination_path = args.destination or default_destination_path(source_path)

    source_root = zarr.open_group(str(source_path), mode="r")
    estimated_chunks, details = estimate_target_chunks(
        source_root,
        time_chunk_size=args.time_chunk_size,
        spatial_chunk_size=args.spatial_chunk_size,
    )
    free_inodes = available_inodes(destination_path.parent)

    log(f"Source: {source_path}")
    log(f"Destination: {destination_path}")
    log(f"Target time_chunk_size={args.time_chunk_size} spatial_chunk_size={args.spatial_chunk_size}")
    log(f"Estimated destination chunk files: {estimated_chunks:,}")
    log(f"Available filesystem inodes near destination: {free_inodes:,}")
    for name, count, chunks in sorted(details, key=lambda row: row[1], reverse=True):
        log(f"  {name}: chunks={chunks} estimated_files={count:,}")

    required_inodes = estimated_chunks + args.inode_safety_margin
    if free_inodes < args.min_free_inodes:
        raise RuntimeError(
            f"Only {free_inodes:,} free inodes; require at least {args.min_free_inodes:,} before starting."
        )
    if free_inodes < required_inodes:
        raise RuntimeError(
            f"Estimated destination plus safety margin requires {required_inodes:,} inodes, "
            f"but only {free_inodes:,} are free."
        )

    if args.dry_run:
        log("Dry run complete; no data were written.")
        return

    create_destination_arrays(
        source_path,
        destination_path,
        time_chunk_size=args.time_chunk_size,
        spatial_chunk_size=args.spatial_chunk_size,
        overwrite=args.overwrite,
    )
    copy_data(
        source_path,
        destination_path,
        max_blocks_per_array=args.max_blocks_per_array,
        workers=args.workers,
    )
    if args.max_blocks_per_array is None:
        validate_sampled_values(source_path, destination_path)
        consolidate_and_reopen(destination_path)
        validate_sampled_values(source_path, destination_path)
        log("Viewer rechunk completed successfully")
    else:
        log("Partial smoke copy complete; skipping full validation and metadata consolidation")


if __name__ == "__main__":
    main()
