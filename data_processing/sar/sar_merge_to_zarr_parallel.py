#!/usr/bin/env python3
"""Parallel-safe SAR zarr merge: init -> workers (region writes) -> finalize."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc


DEFAULT_VH = Path("/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr")
DEFAULT_VV = Path("/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full_vv.zarr")
DEFAULT_OUT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars.zarr")
DEFAULT_COORD = Path("/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_merge_queue_coord")

OUT_VARS = ["vv", "vh", "vv_minus_vh", "vv_over_vh"]
WRITE_CHUNKS = {"time": 1, "variable": 4, "y": 512, "x": 512}
COMPRESSOR = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)


def lock_file(path: Path):
    class _Lock:
        def __init__(self, p):
            self.p = p
            self.fh = None

        def __enter__(self):
            self.p.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.p, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            return self.fh

        def __exit__(self, et, ev, tb):
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()

    return _Lock(path)


def manifest_path(coord_dir: Path) -> Path:
    return coord_dir / "sar_parallel_manifest.json"


def state_path(coord_dir: Path) -> Path:
    return coord_dir / "state.json"


def todo_path(coord_dir: Path) -> Path:
    return coord_dir / "chunks.todo"


def done_path(coord_dir: Path) -> Path:
    return coord_dir / "chunks.done"


def next_seq_path(coord_dir: Path) -> Path:
    return coord_dir / "next_seq.txt"


def open_source_pair(vh_path: Path, vv_path: Path):
    vh_ds = xr.open_zarr(vh_path, consolidated=False)
    vv_ds = xr.open_zarr(vv_path, consolidated=False)
    return vh_ds, vv_ds


def validate_sources(vh_ds: xr.Dataset, vv_ds: xr.Dataset):
    if "vh_backscatter" not in vh_ds.data_vars:
        raise ValueError("vh source missing vh_backscatter")
    if "vv_backscatter" not in vv_ds.data_vars:
        raise ValueError("vv source missing vv_backscatter")
    if not np.array_equal(vh_ds["x"].values, vv_ds["x"].values):
        raise ValueError("x coordinates do not match between vh and vv")
    if not np.array_equal(vh_ds["y"].values, vv_ds["y"].values):
        raise ValueError("y coordinates do not match between vh and vv")


def build_full_time_axis(vh_ds: xr.Dataset, vv_ds: xr.Dataset) -> pd.DatetimeIndex:
    vh_times = pd.to_datetime(vh_ds["time"].values)
    vv_times = pd.to_datetime(vv_ds["time"].values)
    start = min(vh_times.min(), vv_times.min())
    end = max(vh_times.max(), vv_times.max())
    return pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")


def build_manifest(vh_ds: xr.Dataset, vv_ds: xr.Dataset, time_block_days: int):
    validate_sources(vh_ds, vv_ds)
    full_time = build_full_time_axis(vh_ds, vv_ds)
    vh_times = set(pd.to_datetime(vh_ds["time"].values).strftime("%Y-%m-%d").tolist())
    vv_times = set(pd.to_datetime(vv_ds["time"].values).strftime("%Y-%m-%d").tolist())

    chunks = []
    total_time = len(full_time)
    for chunk_id, start in enumerate(range(0, total_time, time_block_days)):
        stop = min(start + time_block_days, total_time)
        dates = full_time[start:stop]
        iso_dates = [d.strftime("%Y-%m-%d") for d in dates]
        n_vh = sum(1 for d in iso_dates if d in vh_times)
        n_vv = sum(1 for d in iso_dates if d in vv_times)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "stop": stop,
                "n_time": stop - start,
                "start_date": iso_dates[0],
                "end_date": iso_dates[-1],
                "n_vh_obs": n_vh,
                "n_vv_obs": n_vv,
            }
        )

    return {
        "version": 1,
        "time_start": full_time[0].strftime("%Y-%m-%d"),
        "time_end": full_time[-1].strftime("%Y-%m-%d"),
        "total_time": total_time,
        "time_values": [d.strftime("%Y-%m-%d") for d in full_time],
        "output_variables": list(OUT_VARS),
        "vh_path": str(vh_ds.encoding.get("source", "")),
        "vv_path": str(vv_ds.encoding.get("source", "")),
        "source_stats": {
            "vh_n_time": int(vh_ds.sizes["time"]),
            "vv_n_time": int(vv_ds.sizes["time"]),
            "vh_first": pd.to_datetime(vh_ds["time"].values[0]).strftime("%Y-%m-%d"),
            "vh_last": pd.to_datetime(vh_ds["time"].values[-1]).strftime("%Y-%m-%d"),
            "vv_first": pd.to_datetime(vv_ds["time"].values[0]).strftime("%Y-%m-%d"),
            "vv_last": pd.to_datetime(vv_ds["time"].values[-1]).strftime("%Y-%m-%d"),
        },
        "chunks": chunks,
    }


def save_manifest(coord_dir: Path, manifest: dict):
    coord_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(coord_dir).write_text(json.dumps(manifest))


def load_manifest(coord_dir: Path) -> dict:
    p = manifest_path(coord_dir)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}. Run --mode init first.")
    return json.loads(p.read_text())


def maybe_remove_output(out: Path, coord_dir: Path, overwrite_out: bool, reset_coord: bool):
    if overwrite_out and out.exists():
        print(f"Removing existing output zarr: {out}")
        shutil.rmtree(out)
    if reset_coord and coord_dir.exists():
        print(f"Removing existing coord dir: {coord_dir}")
        shutil.rmtree(coord_dir)


def make_chunk_dataset(vh_ds: xr.Dataset, vv_ds: xr.Dataset, manifest: dict, chunk_meta: dict) -> xr.Dataset:
    time_values = pd.to_datetime(manifest["time_values"][chunk_meta["start"]:chunk_meta["stop"]])
    time_index = pd.DatetimeIndex(time_values)

    vv = vv_ds["vv_backscatter"].reindex(time=time_index).astype("float32")
    vh = vh_ds["vh_backscatter"].reindex(time=time_index).astype("float32")

    both = np.isfinite(vv) & np.isfinite(vh)
    vv_minus_vh = xr.where(both, vv - vh, np.float32(np.nan)).astype("float32")
    safe_div = both & (vh != 0)
    vv_over_vh = xr.where(safe_div, vv / vh, np.float32(np.nan)).astype("float32")

    ds = xr.Dataset(
        {
            "vv": vv,
            "vh": vh,
            "vv_minus_vh": vv_minus_vh,
            "vv_over_vh": vv_over_vh,
        }
    )

    for coord_name in ["x", "y", "lat", "lon", "spatial_ref"]:
        if coord_name in vv_ds.coords:
            ds = ds.assign_coords({coord_name: vv_ds.coords[coord_name]})

    ds = ds.assign_coords(time=("time", time_index))
    return ds


def to_stacked_array(ds: xr.Dataset) -> xr.DataArray:
    ds = ds[OUT_VARS]
    arr = ds.to_array(dim="variable", name="data").transpose("time", "variable", "y", "x")
    arr = arr.assign_coords(variable=("variable", np.array(OUT_VARS, dtype=object)))
    arr = arr.chunk(
        {
            "time": WRITE_CHUNKS["time"],
            "variable": len(OUT_VARS),
            "y": WRITE_CHUNKS["y"],
            "x": WRITE_CHUNKS["x"],
        }
    )
    if "lat" in arr.coords:
        arr.coords["lat"] = arr.coords["lat"].chunk({"y": WRITE_CHUNKS["y"], "x": WRITE_CHUNKS["x"]})
    if "lon" in arr.coords:
        arr.coords["lon"] = arr.coords["lon"].chunk({"y": WRITE_CHUNKS["y"], "x": WRITE_CHUNKS["x"]})
    return arr


def zarr_encoding_for(arr: xr.DataArray):
    chunks = tuple(c[0] for c in arr.data.chunks)
    return {"data": {"compressor": COMPRESSOR, "chunks": chunks}}


def write_first(arr: xr.DataArray, out: Path):
    ds = arr.to_dataset(name="data")
    ds.to_zarr(
        out,
        mode="w",
        consolidated=False,
        zarr_format=2,
        encoding=zarr_encoding_for(arr),
        compute=True,
    )


def write_time_region_data(arr: xr.DataArray, out: Path, time_start: int):
    time_stop = time_start + int(arr.sizes["time"])
    xr.Dataset({"data": (("time", "variable", "y", "x"), arr.data)}).to_zarr(
        out,
        mode="r+",
        region={"time": slice(time_start, time_stop)},
        consolidated=False,
        zarr_format=2,
    )


def resize_time_axis_data_array(out: Path, total_time: int):
    import zarr

    root = zarr.open_group(str(out), mode="a")
    data_arr = root["data"]
    shape = tuple(data_arr.shape)
    if shape[0] != int(total_time):
        data_arr.resize((int(total_time), shape[1], shape[2], shape[3]))


def write_full_time_coordinate(out: Path, manifest: dict):
    import zarr

    times = pd.DatetimeIndex(pd.to_datetime(manifest["time_values"]))
    root = zarr.open_group(str(out), mode="a")
    time_arr = root["time"]
    if int(time_arr.shape[0]) != len(times):
        time_arr.resize((len(times),))
    # xarray region writes on datetime coords can re-encode subset blocks with local
    # units; write the encoded "days since first date" values directly.
    encoded = np.arange(len(times), dtype=time_arr.dtype)
    time_arr[:] = encoded


def init_queue(coord_dir: Path, chunk_ids, done_ids=None):
    done_ids = [str(x) for x in (done_ids or [])]
    coord_dir.mkdir(parents=True, exist_ok=True)
    with lock_file(coord_dir / "queue.lock"):
        done_set = set(done_ids)
        todo_lines = [str(cid) for cid in chunk_ids if str(cid) not in done_set]
        todo_path(coord_dir).write_text("\n".join(todo_lines) + ("\n" if todo_lines else ""))
        done_path(coord_dir).write_text("\n".join(done_ids) + ("\n" if done_ids else ""))
        next_seq_path(coord_dir).write_text("1\n")
        state_path(coord_dir).write_text(json.dumps({"last_claim": None, "last_write": None}) + "\n")


def claim_next_chunk(coord_dir: Path):
    with lock_file(coord_dir / "queue.lock"):
        todo_lines = [x for x in todo_path(coord_dir).read_text().splitlines() if x.strip()]
        if not todo_lines:
            return None
        chunk_id = int(todo_lines[0])
        todo_path(coord_dir).write_text("\n".join(todo_lines[1:]) + ("\n" if len(todo_lines) > 1 else ""))
        seq = int(next_seq_path(coord_dir).read_text().strip() or "1")
        next_seq_path(coord_dir).write_text(f"{seq+1}\n")
        try:
            s = json.loads(state_path(coord_dir).read_text() or "{}")
        except Exception:
            s = {}
        s["last_claim"] = {"chunk_id": chunk_id, "seq": seq}
        state_path(coord_dir).write_text(json.dumps(s) + "\n")
        return chunk_id, seq


def mark_done(coord_dir: Path, chunk_id: int, seq: int):
    with lock_file(coord_dir / "queue.lock"):
        done_lines = [x for x in done_path(coord_dir).read_text().splitlines() if x.strip()]
        done_lines.append(str(chunk_id))
        done_path(coord_dir).write_text("\n".join(done_lines) + "\n")
        try:
            s = json.loads(state_path(coord_dir).read_text() or "{}")
        except Exception:
            s = {}
        s["last_write"] = {"chunk_id": int(chunk_id), "seq": int(seq)}
        state_path(coord_dir).write_text(json.dumps(s) + "\n")


def consolidate(out: Path):
    import zarr

    zarr.consolidate_metadata(str(out))


def print_dry_run(manifest: dict):
    print("SAR merge dry run summary")
    print("  output time start:", manifest["time_start"])
    print("  output time end:", manifest["time_end"])
    print("  total daily timesteps:", manifest["total_time"])
    print("  variables:", manifest["output_variables"])
    print("  chunks:", len(manifest["chunks"]))
    if manifest["chunks"]:
        first = manifest["chunks"][0]
        last = manifest["chunks"][-1]
        print("  first chunk:", first)
        print("  last chunk:", last)
    print("  source stats:", manifest["source_stats"])


def init_mode(args):
    vh_ds, vv_ds = open_source_pair(args.vh_path, args.vv_path)
    try:
        manifest = build_manifest(vh_ds, vv_ds, args.time_block_days)
        save_manifest(args.coord_dir, manifest)
        if args.dry_run:
            print_dry_run(manifest)
            return

        maybe_remove_output(args.out, args.coord_dir, args.overwrite_out, reset_coord=args.overwrite_out)
        # Re-save manifest in case coord dir was deleted externally before init
        args.coord_dir.mkdir(parents=True, exist_ok=True)
        save_manifest(args.coord_dir, manifest)

        first_chunk = manifest["chunks"][0]
        print(f"Init writing first chunk {first_chunk['chunk_id']} {first_chunk['start_date']}..{first_chunk['end_date']}")
        ds_first = make_chunk_dataset(vh_ds, vv_ds, manifest, first_chunk)
        arr_first = to_stacked_array(ds_first)
        write_first(arr_first, args.out)
        print("Resizing data time axis for parallel region writes...")
        resize_time_axis_data_array(args.out, manifest["total_time"])
        print("Writing full time coordinate...")
        write_full_time_coordinate(args.out, manifest)

        chunk_ids = [c["chunk_id"] for c in manifest["chunks"]]
        init_queue(args.coord_dir, chunk_ids=chunk_ids, done_ids=[first_chunk["chunk_id"]])
        st = json.loads(state_path(args.coord_dir).read_text())
        st["last_write"] = {"chunk_id": int(first_chunk["chunk_id"]), "seq": 0}
        state_path(args.coord_dir).write_text(json.dumps(st) + "\n")
        print(f"Init complete: {len(chunk_ids)} chunks, output={args.out}")
    finally:
        vh_ds.close()
        vv_ds.close()


def worker_mode(args):
    manifest = load_manifest(args.coord_dir)
    chunk_lookup = {int(c["chunk_id"]): c for c in manifest["chunks"]}
    vh_ds, vv_ds = open_source_pair(args.vh_path, args.vv_path)
    claims_done = 0
    try:
        while True:
            if args.max_claims is not None and claims_done >= args.max_claims:
                print(f"Reached --max-claims {args.max_claims}; stopping worker")
                break
            claimed = claim_next_chunk(args.coord_dir)
            if claimed is None:
                print("No more chunks to claim.")
                break
            chunk_id, seq = claimed
            meta = chunk_lookup[chunk_id]
            print(f"[seq {seq}] Processing chunk {chunk_id}: {meta['start_date']}..{meta['end_date']}")
            t0 = time.time()
            ds_chunk = make_chunk_dataset(vh_ds, vv_ds, manifest, meta)
            arr = to_stacked_array(ds_chunk)
            print(f"[seq {seq}] Writing region time={meta['start']}:{meta['stop']}")
            write_time_region_data(arr, args.out, time_start=int(meta["start"]))
            mark_done(args.coord_dir, chunk_id, seq)
            dt = time.time() - t0
            print(f"[seq {seq}] Wrote chunk {chunk_id} in {dt:.1f}s")
            claims_done += 1
    finally:
        vh_ds.close()
        vv_ds.close()


def finalize_mode(args):
    print(f"Consolidating zarr metadata: {args.out}")
    consolidate(args.out)
    print("Finalize complete.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["init", "worker", "finalize"], default="worker")
    ap.add_argument("--vh-path", type=Path, default=DEFAULT_VH)
    ap.add_argument("--vv-path", type=Path, default=DEFAULT_VV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--coord-dir", type=Path, default=DEFAULT_COORD)
    ap.add_argument("--time-block-days", type=int, default=16)
    ap.add_argument("--overwrite-out", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-claims", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "init":
        init_mode(args)
    elif args.mode == "worker":
        worker_mode(args)
    elif args.mode == "finalize":
        finalize_mode(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
