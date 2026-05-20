#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def timestamped_message(message: str) -> str:
    return f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    return value


def _clean_dataset_encoding(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=False)
    out.encoding = {}
    for name in out.variables:
        out[name].encoding = {}
    for name in out.coords:
        out[name].encoding = {}
    return out


def _dim_value_for_json(value, dim_name: str):
    if value is None:
        return None
    if dim_name == "time":
        return str(pd.Timestamp(value).normalize().date())
    if isinstance(value, np.generic):
        return value.item()
    return _json_safe(value)


def _selector_value(value, dim_name: str):
    if value is None:
        return None
    if dim_name == "time":
        return pd.Timestamp(value).normalize()
    return value


def _manifest_path(rollback_dir: Path, label: str) -> Path:
    return rollback_dir / f"{label}.rollback.json"


def _backup_path(rollback_dir: Path, label: str) -> Path:
    return rollback_dir / f"{label}.zarr"


def _array_dims(arr) -> list[str]:
    dims = arr.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        dims = arr.attrs.get("dimension_names")
    if dims is None:
        return []
    return [str(dim) for dim in dims]


def _read_coord_values(zarr_path: Path, dim_name: str):
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        values = ds[dim_name].values
    if dim_name == "time":
        return pd.to_datetime(values).normalize()
    return np.asarray(values)


def _dataset_dims_map(zarr_path: Path) -> dict[str, list[str]]:
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        return {name: [str(dim) for dim in ds[name].dims] for name in ds.variables}


def _contiguous_region(target_values, backup_values, dim_name: str) -> slice:
    if len(backup_values) == 0:
        return slice(0, 0)
    if dim_name == "time":
        target_raw = np.asarray(pd.to_datetime(target_values).values, dtype="datetime64[ns]")
        backup_raw = np.asarray(pd.to_datetime(backup_values).values, dtype="datetime64[ns]")
    else:
        target_raw = np.asarray(target_values)
        backup_raw = np.asarray(backup_values)
    positions = []
    for value in backup_raw:
        matches = np.where(target_raw == value)[0]
        if len(matches) != 1:
            raise ValueError(f"Could not locate unique rollback coordinate value {value} on target {dim_name}")
        positions.append(int(matches[0]))
    expected = list(range(positions[0], positions[0] + len(positions)))
    if positions != expected:
        raise ValueError(f"Rollback coordinate positions are not contiguous for {dim_name}: {positions[:10]}")
    return slice(positions[0], positions[-1] + 1)


def _range_starts(stop: int, step: int):
    return range(0, int(stop), max(1, int(step)))


def _array_chunks(arr) -> tuple[int, ...]:
    chunks = getattr(arr, "chunks", None)
    if chunks is None:
        return tuple(int(size) for size in arr.shape)
    return tuple(max(1, int(chunk)) for chunk in chunks)


def _encode_time_for_array(time_values, arr) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(time_values)).normalize()
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.asarray(times.values, dtype=arr.dtype)
    units = str(arr.attrs.get("units", "")).strip()
    if not units:
        return np.asarray(times.values, dtype="datetime64[ns]").astype(np.int64).astype(arr.dtype)
    if " since " not in units:
        raise ValueError(f"Unsupported time units while restoring rollback: {units}")
    unit_name, origin_text = units.split(" since ", 1)
    unit_aliases = {
        "nanosecond": "ns",
        "nanoseconds": "ns",
        "ns": "ns",
        "microsecond": "us",
        "microseconds": "us",
        "us": "us",
        "millisecond": "ms",
        "milliseconds": "ms",
        "ms": "ms",
        "second": "s",
        "seconds": "s",
        "s": "s",
        "minute": "m",
        "minutes": "m",
        "hour": "h",
        "hours": "h",
        "day": "D",
        "days": "D",
    }
    unit = unit_aliases.get(unit_name.strip().lower())
    if unit is None:
        raise ValueError(f"Unsupported time unit while restoring rollback: {units}")
    origin = pd.Timestamp(origin_text.strip()).tz_localize(None)
    unit_ns = pd.Timedelta(1, unit=unit).value
    deltas = (
        np.asarray(times.values, dtype="datetime64[ns]") - np.datetime64(origin.to_datetime64(), "ns")
    ).astype("timedelta64[ns]").astype(np.int64)
    if np.any(deltas % unit_ns != 0):
        raise ValueError(f"Rollback times are not exactly representable in target units: {units}")
    return (deltas // unit_ns).astype(arr.dtype)


def _copy_array_region_chunkwise(backup_arr, target_arr, dims: list[str], dim_name: str, region: slice, label: str):
    if tuple(backup_arr.shape) == tuple(0 for _ in backup_arr.shape):
        return
    dim_axis = dims.index(dim_name)
    chunks = _array_chunks(backup_arr)
    starts_by_axis = [list(_range_starts(size, chunks[axis])) for axis, size in enumerate(backup_arr.shape)]
    total_blocks = 1
    for starts in starts_by_axis:
        total_blocks *= max(1, len(starts))
    block_idx = 0
    for starts in np.ndindex(*(len(starts) for starts in starts_by_axis)):
        backup_slices = []
        target_slices = []
        for axis, start_idx in enumerate(starts):
            start = starts_by_axis[axis][start_idx]
            stop = min(int(backup_arr.shape[axis]), start + chunks[axis])
            backup_slices.append(slice(start, stop))
            if axis == dim_axis:
                target_slices.append(slice(region.start + start, region.start + stop))
            else:
                target_slices.append(slice(start, stop))
        target_arr[tuple(target_slices)] = backup_arr[tuple(backup_slices)]
        block_idx += 1
        if block_idx == 1 or block_idx == total_blocks or block_idx % 500 == 0:
            print(timestamped_message(f"  {label}: restored block {block_idx}/{total_blocks}"))


def capture_zarr_rollback(
    target_zarr: Path,
    rollback_dir: Path,
    label: str,
    dim_name: str = "time",
    window_start=None,
    window_end=None,
    reason: str = "",
) -> Path:
    target_zarr = Path(target_zarr)
    rollback_dir = Path(rollback_dir)
    rollback_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(rollback_dir, label)
    backup_path = _backup_path(rollback_dir, label)

    if manifest_path.exists():
        print(timestamped_message(f"Rollback capture already exists for {label}; preserving first baseline"))
        return manifest_path

    record = {
        "label": label,
        "target_zarr": str(target_zarr),
        "backup_zarr": str(backup_path),
        "dim_name": dim_name,
        "window_start": _dim_value_for_json(window_start, dim_name),
        "window_end": _dim_value_for_json(window_end, dim_name),
        "reason": reason,
        "captured_at": dt.datetime.now().isoformat(),
        "original_exists": target_zarr.exists(),
        "backup_count": 0,
    }

    if not target_zarr.exists():
        manifest_path.write_text(json.dumps(_json_safe(record), indent=2, sort_keys=True))
        print(timestamped_message(f"Captured missing-store rollback baseline for {label}: {target_zarr}"))
        return manifest_path

    root = zarr.open_group(str(target_zarr), mode="r")
    record["root_attrs"] = _json_safe(dict(root.attrs))

    with xr.open_zarr(target_zarr, consolidated=False) as ds:
        if dim_name not in ds.dims or dim_name not in ds.coords:
            raise ValueError(f"{target_zarr} does not expose coordinate dimension {dim_name}")
        dim_size = int(ds.sizes[dim_name])
        coord_values = ds[dim_name].values
        record["original_size"] = dim_size
        if dim_size > 0:
            record["original_start"] = _dim_value_for_json(coord_values[0], dim_name)
            record["original_end"] = _dim_value_for_json(coord_values[-1], dim_name)
        else:
            record["original_start"] = None
            record["original_end"] = None

        start_value = _selector_value(window_start, dim_name)
        end_value = _selector_value(window_end, dim_name)
        rollback_vars = [name for name in ds.variables if dim_name in ds[name].dims]
        subset = ds[rollback_vars]
        drop_coords = [name for name in subset.coords if dim_name not in subset[name].dims]
        if drop_coords:
            subset = subset.drop_vars(drop_coords, errors="ignore")
        if start_value is not None or end_value is not None:
            subset = subset.sel({dim_name: slice(start_value, end_value)})
        subset_count = int(subset.sizes.get(dim_name, 0))
        record["backup_count"] = subset_count
        if subset_count > 0:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            print(
                timestamped_message(
                    f"Capturing rollback slice for {label}: count={subset_count} path={backup_path}"
                )
            )
            _clean_dataset_encoding(subset).to_zarr(
                backup_path,
                mode="w",
                consolidated=False,
                safe_chunks=False,
            )
            zarr.consolidate_metadata(str(backup_path))

    manifest_path.write_text(json.dumps(_json_safe(record), indent=2, sort_keys=True))
    print(timestamped_message(f"Wrote rollback manifest for {label}: {manifest_path}"))
    return manifest_path


def capture_zarr_rollback_from_env(
    target_zarr: Path,
    label: str,
    dim_name: str = "time",
    window_start=None,
    window_end=None,
    reason: str = "",
) -> Path | None:
    rollback_dir = os.environ.get("LOW_LATENCY_ROLLBACK_DIR", "").strip()
    if not rollback_dir:
        return None
    return capture_zarr_rollback(
        target_zarr=Path(target_zarr),
        rollback_dir=Path(rollback_dir),
        label=label,
        dim_name=dim_name,
        window_start=window_start,
        window_end=window_end,
        reason=reason,
    )


def _restore_root_attrs(root, attrs: dict) -> None:
    current_keys = list(root.attrs.keys())
    for key in current_keys:
        if key not in attrs:
            try:
                del root.attrs[key]
            except Exception:
                pass
    root.attrs.update(attrs)


def restore_rollback_entry(manifest_path: Path, dry_run: bool = False) -> None:
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    label = record["label"]
    target_zarr = Path(record["target_zarr"])
    backup_zarr = Path(record["backup_zarr"])
    dim_name = record["dim_name"]

    if not record.get("original_exists", True):
        if target_zarr.exists():
            print(timestamped_message(f"Removing store created after rollback baseline for {label}: {target_zarr}"))
            if not dry_run:
                shutil.rmtree(target_zarr)
        return

    if not target_zarr.exists():
        raise FileNotFoundError(f"Cannot restore rollback for {label}; missing target store: {target_zarr}")

    backup_count = int(record.get("backup_count", 0))
    target_dims_map = _dataset_dims_map(target_zarr)
    if backup_count > 0:
        if not backup_zarr.exists():
            raise FileNotFoundError(f"Cannot restore rollback for {label}; missing backup store: {backup_zarr}")
        backup_dims_map = _dataset_dims_map(backup_zarr)
        target_values = _read_coord_values(target_zarr, dim_name)
        backup_values = _read_coord_values(backup_zarr, dim_name)
        region = _contiguous_region(target_values, backup_values, dim_name)
        print(
            timestamped_message(
                f"Restoring rollback slice for {label}: count={backup_count} "
                f"target_region={region.start}:{region.stop}"
            )
        )
        if not dry_run:
            backup_root = zarr.open_group(str(backup_zarr), mode="r")
            target_root = zarr.open_group(str(target_zarr), mode="a")
            for name, backup_arr in backup_root.arrays():
                dims = _array_dims(backup_arr) or backup_dims_map.get(name, [])
                if dim_name not in dims or name not in target_root:
                    continue
                target_arr = target_root[name]
                target_dims = _array_dims(target_arr) or target_dims_map.get(name, [])
                if target_dims != dims:
                    raise ValueError(f"Dimension mismatch while restoring {label}/{name}: {dims} vs {target_dims}")
                if dim_name == "time" and name == dim_name:
                    target_arr[region] = _encode_time_for_array(backup_values, target_arr)
                    print(timestamped_message(f"  {label}/{name}: restored encoded time coordinate"))
                else:
                    _copy_array_region_chunkwise(backup_arr, target_arr, dims, dim_name, region, f"{label}/{name}")

    original_size = int(record.get("original_size", 0))
    print(timestamped_message(f"Truncating {label} to original {dim_name} length {original_size}"))
    if not dry_run:
        root = zarr.open_group(str(target_zarr), mode="a")
        for name, arr in root.arrays():
            dims = _array_dims(arr) or target_dims_map.get(name, [])
            if dim_name not in dims:
                continue
            dim_axis = dims.index(dim_name)
            if int(arr.shape[dim_axis]) == original_size:
                continue
            new_shape = list(arr.shape)
            new_shape[dim_axis] = original_size
            arr.resize(tuple(new_shape))
            print(timestamped_message(f"  resized {name} to {tuple(new_shape)}"))
        _restore_root_attrs(root, record.get("root_attrs", {}))
        zarr.consolidate_metadata(str(target_zarr))


def restore_rollback_dir(rollback_dir: Path, labels: list[str] | None = None, dry_run: bool = False) -> None:
    rollback_dir = Path(rollback_dir)
    if not rollback_dir.exists():
        raise FileNotFoundError(f"Missing rollback directory: {rollback_dir}")
    manifests = sorted(rollback_dir.glob("*.rollback.json"))
    if labels:
        wanted = set(labels)
        manifests = [path for path in manifests if path.name.removesuffix(".rollback.json") in wanted]
    if not manifests:
        raise FileNotFoundError(f"No rollback manifests found in {rollback_dir}")
    print(timestamped_message(f"Restoring {len(manifests)} rollback entries from {rollback_dir}"))
    for manifest_path in manifests:
        restore_rollback_entry(manifest_path, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture or restore low-latency zarr rollback slices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--target_zarr", type=Path, required=True)
    capture.add_argument("--rollback_dir", type=Path, required=True)
    capture.add_argument("--label", type=str, required=True)
    capture.add_argument("--dim_name", type=str, default="time")
    capture.add_argument("--window_start", type=str, default=None)
    capture.add_argument("--window_end", type=str, default=None)
    capture.add_argument("--reason", type=str, default="")

    restore = subparsers.add_parser("restore")
    restore.add_argument("--rollback_dir", type=Path, required=True)
    restore.add_argument("--label", action="append", default=None)
    restore.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "capture":
        capture_zarr_rollback(
            target_zarr=args.target_zarr,
            rollback_dir=args.rollback_dir,
            label=args.label,
            dim_name=args.dim_name,
            window_start=args.window_start,
            window_end=args.window_end,
            reason=args.reason,
        )
    elif args.command == "restore":
        restore_rollback_dir(args.rollback_dir, labels=args.label, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
