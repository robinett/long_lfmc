#!/usr/bin/env python3
# Parallel Daymet month workers with init/worker/finalize modes.

from pathlib import Path
import argparse
import calendar
import json
import fcntl
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

from zarr_build_utils import (
    DEFAULT_COMP,
    preprocess_strip_attrs,
    open_time_batch,
    to_stacked_array,
    chunk_coords,
    write_first,
    write_time_region_data,
    resize_time_axis_data_array,
    consolidate,
    scan_daymet_regrid_month_index,
    parse_daymet_regrid_filename,
)

# --------- CONFIG ----------
ROOT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid")
OUT = Path("/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars.zarr")
WRITE_CHUNKS = {"time": 1, "variable": 9999, "y": 512, "x": 512}
CAST_FLOAT32 = True
ENGINE = "h5netcdf"
PARALLEL_OPEN = False
COMP = DEFAULT_COMP
DAYMET_VAR_WHITELIST = ["prcp", "srad", "swe", "tmax", "tmin", "vp"]
# ---------------------------


def lock_file(path: Path):
    """Context manager for an exclusive file lock (POSIX flock)."""
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


def month_index_cache_path(coord: Path) -> Path:
    return coord / "daymet_month_index.json"


def manifest_path(coord: Path) -> Path:
    return coord / "daymet_parallel_manifest.json"


def load_or_build_month_index(coord: Path, root: Path, rebuild: bool = False):
    """
    Cache the month->files mapping in coord dir so workers do not rescan the
    Daymet tree independently.
    """
    cache_path = month_index_cache_path(coord)
    lock_path = coord / "index.lock"

    with lock_file(lock_path):
        if cache_path.exists() and not rebuild:
            cached = json.loads(cache_path.read_text())
            if cached.get("root") == str(root):
                month_index = {
                    ym: [Path(p) for p in files]
                    for ym, files in cached.get("month_index", {}).items()
                }
                return month_index, cached.get("summary", {})

        month_index_str, summary = scan_daymet_regrid_month_index(root)
        payload = {
            "root": str(root),
            "month_index": month_index_str,
            "summary": summary,
        }
        cache_path.write_text(json.dumps(payload))
        month_index = {ym: [Path(p) for p in files] for ym, files in month_index_str.items()}
        return month_index, summary


def print_dry_run_summary(summary: dict, month_index: dict):
    months = sorted(month_index.keys())
    print("Daymet dry run summary")
    print("  root:", summary.get("root"))
    print("  year dirs:", len(summary.get("years_seen", [])))
    print("  empty years:", summary.get("empty_years", []))
    print("  vars:", summary.get("vars_seen", []))
    print("  months found:", len(months))
    if months:
        print("  first month:", months[0], "files:", len(month_index[months[0]]))
        print("  last month:", months[-1], "files:", len(month_index[months[-1]]))
    print("  malformed filenames:", summary.get("malformed_count", 0))
    bad_months = summary.get("problematic_months", [])
    print("  problematic months:", len(bad_months))
    for item in bad_months[:10]:
        print("   ", item)


def maybe_fill_missing_leap_dec31(ds: xr.Dataset, year: str, month: str) -> xr.Dataset:
    """
    Some leap years are missing Dec 31 in the raw regridded files. Duplicate
    the Dec 30 slice and relabel it to Dec 31 so leap years end with 366 days.
    """
    if month != "12" or not calendar.isleap(int(year)):
        return ds
    if "time" not in ds.coords:
        return ds

    times = pd.to_datetime(ds["time"].values)
    if len(times) == 0:
        return ds

    norm = pd.DatetimeIndex(times).normalize()
    dec30 = pd.Timestamp(f"{year}-12-30")
    dec31 = pd.Timestamp(f"{year}-12-31")

    if (norm == dec31).any():
        return ds
    dec30_idx = np.where(norm == dec30)[0]
    if len(dec30_idx) == 0:
        raise ValueError(f"Leap-year December missing both Dec 30 and Dec 31 for {year}")

    fill = ds.isel(time=[int(dec30_idx[-1])]).copy(deep=False)
    fill = fill.assign_coords(time=("time", [dec31.to_datetime64()]))
    print(f"Inserted synthetic {year}-12-31 from {year}-12-30")
    return xr.concat([ds, fill], dim="time").sortby("time")


def build_parallel_manifest(month_index: dict):
    """
    Build ordered month metadata with global time offsets for region writes.
    """
    months = []
    total_time = 0
    global_time_values = []
    for ym in sorted(month_index.keys()):
        year, month = ym.split("-")
        raw_dates = set()
        for f in month_index[ym]:
            _, yyyymmdd = parse_daymet_regrid_filename(f)
            if yyyymmdd is not None:
                raw_dates.add(yyyymmdd)

        if not raw_dates:
            continue

        dec31_raw = f"{year}1231"
        dec30_raw = f"{year}1230"
        synth_dec31 = (month == "12"
                       and calendar.isleap(int(year))
                       and dec31_raw not in raw_dates
                       and dec30_raw in raw_dates)
        date_list = sorted(raw_dates)
        if synth_dec31:
            date_list.append(dec31_raw)
            date_list = sorted(date_list)

        iso_dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in date_list]
        n_time = len(iso_dates)

        start = total_time
        stop = total_time + n_time
        total_time = stop
        global_time_values.extend(iso_dates)

        months.append(
            {
                "ym": ym,
                "year": year,
                "month": month,
                "start": start,
                "stop": stop,
                "n_time": n_time,
                "n_raw_dates": len(raw_dates),
                "synth_dec31": synth_dec31,
                "n_files": len(month_index[ym]),
                "dates": iso_dates,
            }
        )

    if not months:
        raise ValueError("No monthly batches found to build manifest")

    return {
        "version": 1,
        "total_time": total_time,
        "vars": list(DAYMET_VAR_WHITELIST),
        "time_values": global_time_values,
        "months": months,
    }


def save_manifest(coord: Path, manifest: dict):
    manifest_path(coord).write_text(json.dumps(manifest))


def load_manifest(coord: Path) -> dict:
    p = manifest_path(coord)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}. Run --mode init first.")
    return json.loads(p.read_text())


def manifest_month_lookup(manifest: dict) -> dict:
    return {m["ym"]: m for m in manifest["months"]}


def init_queue_for_parallel(coord: Path, month_labels, done_labels=None):
    """
    Reset queue files for a fresh parallel run. Workers claim months dynamically,
    but writes are no longer serialized by sequence.
    """
    done_labels = list(done_labels or [])
    coord.mkdir(parents=True, exist_ok=True)
    todo = coord / "months.todo"
    done = coord / "months.done"
    next_seq = coord / "next_seq.txt"
    state = coord / "state.json"

    with lock_file(coord / "queue.lock"):
        done_set = set(done_labels)
        todo_lines = [ym for ym in sorted(month_labels) if ym not in done_set]

        todo.write_text("\n".join(todo_lines) + ("\n" if todo_lines else ""))
        done.write_text("\n".join(done_labels) + ("\n" if done_labels else ""))
        next_seq.write_text("1\n")
        state_payload = {
            "last_claim": None,
            "last_write": None if not done_labels else {"ym": done_labels[-1], "seq": 0},
        }
        state.write_text(json.dumps(state_payload) + "\n")


def claim_next_month(coord: Path):
    """Atomically claim the earliest todo month and assign a sequence number."""
    todo = coord / "months.todo"
    next_seq = coord / "next_seq.txt"
    state = coord / "state.json"

    with lock_file(coord / "queue.lock"):
        todo_lines = [x for x in todo.read_text().splitlines() if x.strip()]
        if not todo_lines:
            return None

        line = todo_lines[0]
        remaining = todo_lines[1:]
        todo.write_text("\n".join(remaining) + ("\n" if remaining else ""))

        seq = int(next_seq.read_text().strip() or "1")
        next_seq.write_text(f"{seq+1}\n")

        try:
            s = json.loads(state.read_text() or "{}")
        except Exception:
            s = {}
        s["last_claim"] = {"ym": line, "seq": seq}
        state.write_text(json.dumps(s) + "\n")
        return (line, seq)


def mark_done(coord: Path, ym: str, seq: int, skipped: bool = False):
    done = coord / "months.done"
    state = coord / "state.json"
    with lock_file(coord / "queue.lock"):
        done_lines = [x for x in done.read_text().splitlines() if x.strip()]
        marker = f"{ym} (SKIPPED)" if skipped else ym
        done_lines.append(marker)
        done.write_text("\n".join(done_lines) + "\n")

        try:
            s = json.loads(state.read_text() or "{}")
        except Exception:
            s = {}
        s["last_write"] = {"ym": marker, "seq": seq}
        state.write_text(json.dumps(s) + "\n")


def write_full_time_coordinate(out: Path, manifest: dict):
    """
    Ensure the zarr store has the full global time coordinate values after the
    data array is resized to full length.
    """
    import zarr

    time_values = pd.to_datetime(manifest.get("time_values", []))
    if len(time_values) != int(manifest["total_time"]):
        raise ValueError(
            f"Manifest time_values length mismatch: {len(time_values)} vs total_time={manifest['total_time']}"
        )
    root = zarr.open_group(str(out), mode="a")
    try:
        time_arr = root["time"]
    except KeyError:
        # Some Daymet months open with `time` as an unlabeled dimension, so the
        # initial xarray write may not create a standalone time coordinate array.
        xr.Dataset(coords={"time": ("time", pd.DatetimeIndex(time_values))}).to_zarr(
            out,
            mode="a",
            consolidated=False,
            zarr_format=2,
        )
        return
    if int(time_arr.shape[0]) != len(time_values):
        time_arr.resize((len(time_values),))
    # Store encoded day offsets directly; attrs/units created by xarray on first write.
    encoded = np.arange(len(time_values), dtype=time_arr.dtype)
    time_arr[:] = encoded


def align_month_time_to_manifest(ds: xr.Dataset, month_meta: dict) -> xr.Dataset:
    """
    Force month time coordinates to match the manifest. If the manifest expects a
    leap-year synthetic Dec 31 and the dataset is one day short, duplicate the
    last slice and relabel it to the manifest's final date.
    """
    if "time" not in ds.dims:
        raise ValueError(f"{month_meta['ym']}: dataset missing time dimension")

    expected = pd.to_datetime(month_meta["dates"])
    expected_len = len(expected)
    current_len = int(ds.sizes["time"])

    if current_len == expected_len - 1 and month_meta.get("synth_dec31", False):
        last_date = expected[-1]
        fill = ds.isel(time=[-1]).copy(deep=False)
        fill = fill.assign_coords(time=("time", [last_date.to_datetime64()]))
        ds = xr.concat([ds, fill], dim="time")
        current_len = int(ds.sizes["time"])
        print(f"Inserted synthetic {last_date.strftime('%Y-%m-%d')} from prior day for {month_meta['ym']}")

    if current_len != expected_len:
        raise ValueError(
            f"{month_meta['ym']}: prepared n_time={current_len} but manifest expects {expected_len}"
        )

    ds = ds.assign_coords(time=("time", expected.to_numpy(dtype="datetime64[ns]")))
    return ds


def open_and_prepare_month(month_index: dict, ym: str, defer_stack: bool = False):
    year, month = ym.split("-")
    files = month_index.get(ym, [])
    if not files:
        return None, None

    ds = open_time_batch(
        files,
        engine=ENGINE,
        parallel_open=PARALLEL_OPEN,
        cast_float32=CAST_FLOAT32,
        preprocess=preprocess_strip_attrs,
        combine="by_coords",
        data_var_whitelist=DAYMET_VAR_WHITELIST,
    )
    print("ds opened")
    ds = maybe_fill_missing_leap_dec31(ds, year, month)

    if defer_stack:
        return ds, None

    arr = to_stacked_array(ds, WRITE_CHUNKS)
    print("arr stacked")
    arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])
    print("arr chunked")
    return ds, arr


def maybe_remove_out(out: Path):
    if not out.exists():
        return
    if out.is_dir():
        shutil.rmtree(out)
    else:
        out.unlink()


def run_init(args, coord: Path, root: Path, out: Path):
    month_index, summary = load_or_build_month_index(coord, root, rebuild=args.rebuild_index)
    month_labels = sorted(month_index.keys())
    if not month_labels:
        raise ValueError(f"No daymet monthly batches found under {root}")

    manifest = build_parallel_manifest(month_index)
    save_manifest(coord, manifest)
    lookup = manifest_month_lookup(manifest)
    first_ym = manifest["months"][0]["ym"]
    print(f"Init month: {first_ym}")
    print(f"Total planned time length: {manifest['total_time']}")

    if args.overwrite_out:
        maybe_remove_out(out)
    elif out.exists():
        raise FileExistsError(f"Output store exists: {out}. Use --overwrite-out for init.")

    ds, arr = open_and_prepare_month(month_index, first_ym)
    if ds is None:
        raise ValueError(f"No files found for initial month {first_ym}")

    try:
        first_meta = lookup[first_ym]
        if int(arr.sizes["time"]) != int(first_meta["n_time"]):
            raise ValueError(
                f"Init month time mismatch for {first_ym}: "
                f"arr={arr.sizes['time']} manifest={first_meta['n_time']}"
            )

        print(f"Writing initial store from {first_ym}…")
        write_first(arr, out, compressor=COMP)
        print("Resizing data time axis for parallel region writes…")
        resize_time_axis_data_array(out, int(manifest["total_time"]))
        print("Writing full time coordinate…")
        write_full_time_coordinate(out, manifest)
    finally:
        ds.close()

    init_queue_for_parallel(coord, month_labels, done_labels=[first_ym])
    print("Init complete:", out)
    print("Queue initialized in:", coord)


def run_worker(args, coord: Path, root: Path, out: Path):
    month_index, _ = load_or_build_month_index(coord, root, rebuild=False)
    manifest = load_manifest(coord)
    lookup = manifest_month_lookup(manifest)

    if not out.exists():
        raise FileNotFoundError(f"Output zarr store not found: {out}. Run --mode init first.")

    processed = 0
    while True:
        claim = claim_next_month(coord)
        if claim is None:
            print("No more months in queue. Worker exiting.")
            break

        ym, seq = claim
        print(f"[seq {seq}] Claimed {ym}")
        files = month_index.get(ym, [])
        if not files:
            print(f"[seq {seq}] No files for {ym}. Skipping.")
            mark_done(coord, ym, seq, skipped=True)
            continue

        month_meta = lookup.get(ym)
        if month_meta is None:
            raise KeyError(f"Claimed month {ym} not present in manifest")
        start = int(month_meta["start"])
        stop = int(month_meta["stop"])

        ds, arr = open_and_prepare_month(month_index, ym, defer_stack=True)
        if ds is None:
            print(f"[seq {seq}] No files for {ym} after open. Skipping.")
            mark_done(coord, ym, seq, skipped=True)
            continue

        try:
            ds = align_month_time_to_manifest(ds, month_meta)
            arr = to_stacked_array(ds, WRITE_CHUNKS)
            print('arr stacked (manifest-aligned)')
            arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])
            print('arr chunked (manifest-aligned)')
            n_time = int(arr.sizes["time"])
            if n_time != (stop - start):
                raise ValueError(
                    f"{ym}: prepared n_time={n_time} but manifest slice is {start}:{stop} "
                    f"(len={stop-start})"
                )
            print(f"[seq {seq}] Writing region time={start}:{stop} for {ym}…")
            write_time_region_data(arr, out, start)
            mark_done(coord, ym, seq, skipped=False)
            print(f"[seq {seq}] Wrote {ym}")
        finally:
            ds.close()

        processed += 1
        if args.max_months is not None and processed >= args.max_months:
            print(f"Worker processed {processed} months (limit). Exiting.")
            break

    print("Worker finished.")


def run_finalize(out: Path):
    if out.exists():
        print("Consolidating metadata…")
        consolidate(out)
        print("Done:", out)
    else:
        print("No store to consolidate:", out)


def parse_args():
    ap = argparse.ArgumentParser("Daymet zarr init/worker/finalize")
    ap.add_argument("--coord-dir", type=str, required=True,
                    help="Shared coordination directory for queue/cache/manifest.")
    ap.add_argument("--root", type=str, default=str(ROOT),
                    help="Daymet regridded root (year directories with daily files).")
    ap.add_argument("--out", type=str, default=str(OUT),
                    help="Output zarr store path.")
    ap.add_argument("--mode", type=str, default="worker",
                    choices=["init", "worker", "finalize"],
                    help="Execution mode.")
    ap.add_argument("--finalize", action="store_true",
                    help="Legacy alias for --mode finalize.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan files, print month/index summary, and exit.")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild of cached month index in coord dir.")
    ap.add_argument("--overwrite-out", action="store_true",
                    help="In init mode, remove an existing output zarr before rebuilding.")
    ap.add_argument("--max-months", type=int, default=None,
                    help="Optional: limit number of claimed months this worker will process.")
    return ap.parse_args()


def main():
    args = parse_args()
    global ROOT, OUT
    ROOT = Path(args.root)
    OUT = Path(args.out)
    coord = Path(args.coord_dir)
    coord.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        month_index, summary = load_or_build_month_index(coord, ROOT, rebuild=args.rebuild_index)
        print_dry_run_summary(summary, month_index)
        manifest = build_parallel_manifest(month_index)
        print("  total planned time:", manifest["total_time"])
        print("  first init month:", manifest["months"][0]["ym"])
        return

    mode = "finalize" if args.finalize else args.mode
    if mode == "init":
        run_init(args, coord, ROOT, OUT)
    elif mode == "worker":
        run_worker(args, coord, ROOT, OUT)
    elif mode == "finalize":
        run_finalize(OUT)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
