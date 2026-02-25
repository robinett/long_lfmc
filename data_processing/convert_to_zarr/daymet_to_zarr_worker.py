#!/usr/bin/env python3
# Parallel month workers with ordered writes to a single Zarr.

from pathlib import Path
import argparse
import calendar
import re
import time
import json
import fcntl
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
    append_time,
    consolidate,
    scan_daymet_regrid_month_index,
)

# --------- CONFIG (adjust paths/chunks as needed) ----------
ROOT = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid"
)
OUT = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars.zarr"
)
WRITE_CHUNKS = {"time": 1, "variable": 9999, "y": 512, "x": 512}
CAST_FLOAT32 = True
ENGINE = "h5netcdf"
PARALLEL_OPEN = False
COMP = DEFAULT_COMP
DAYMET_VAR_WHITELIST = ["prcp", "srad", "swe", "tmax", "tmin", "vp"]
# -----------------------------------------------------------

MONTH_RE = re.compile(r"^\d{2}$")
YEAR_RE  = re.compile(r"^\d{4}$")

def load_or_build_month_index(coord: Path, root: Path, rebuild: bool = False):
    """
    Cache the month->files mapping in coord dir so array workers do not rescan
    the Daymet tree independently.
    """
    cache_path = coord / "daymet_month_index.json"
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
    Some leap years are missing Dec 31 in the raw regridded files. Duplicate the
    Dec 30 slice and relabel it to Dec 31 so leap years end with 366 days.
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

def lock_file(path: Path):
    """Context manager for an exclusive file lock (POSIX flock)."""
    class _Lock:
        def __init__(self, p): self.p = p; self.fh = None
        def __enter__(self):
            self.p.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.p, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            return self.fh
        def __exit__(self, et, ev, tb):
            try: fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally: self.fh.close()
    return _Lock(path)

def init_queue_if_needed(coord: Path, month_labels):
    coord.mkdir(parents=True, exist_ok=True)
    todo      = coord / "months.todo"           # lines: YYYY-MM
    done      = coord / "months.done"           # lines: YYYY-MM
    next_seq  = coord / "next_seq.txt"          # integer: next sequence to ASSIGN
    next_write= coord / "next_write.txt"        # integer: next sequence EXPECTED to WRITE
    state     = coord / "state.json"            # optional debugging (last claim/write)

    with lock_file(coord / "queue.lock"):
        if not done.exists():
            done.write_text("")  # create empty
        done_set = set(x.strip() for x in done.read_text().splitlines() if x.strip())

        if not todo.exists():
            # fresh build of todo from all months minus done
            all_lines = sorted(month_labels)
            all_lines = [ln for ln in all_lines if ln not in done_set]
            todo.write_text("\n".join(all_lines) + ("\n" if all_lines else ""))

        if not next_seq.exists():
            next_seq.write_text("1\n")
        if not next_write.exists():
            next_write.write_text("1\n")
        if not state.exists():
            state.write_text(json.dumps({"last_claim": None, "last_write": None}) + "\n")

    return todo, done, next_seq, next_write, state

def claim_next_month(coord: Path):
    """Atomically claim the earliest todo month and assign a sequence number."""
    todo      = coord / "months.todo"
    done      = coord / "months.done"
    next_seq  = coord / "next_seq.txt"
    state     = coord / "state.json"

    with lock_file(coord / "queue.lock"):
        todo_lines = [x for x in todo.read_text().splitlines() if x.strip()]
        if not todo_lines:
            return None  # no more work

        line = todo_lines[0]  # earliest YYYY-MM
        remaining = todo_lines[1:]
        todo.write_text("\n".join(remaining) + ("\n" if remaining else ""))

        # assign sequence id
        seq = int(next_seq.read_text().strip() or "1")
        next_seq.write_text(f"{seq+1}\n")

        # update state
        try:
            s = json.loads(state.read_text() or "{}")
        except Exception:
            s = {}
        s["last_claim"] = {"ym": line, "seq": seq}
        state.write_text(json.dumps(s) + "\n")

        return (line, seq)

def write_ordered(arr, ym, seq, coord: Path):
    """
    Wait until it's this seq's turn, then write to OUT.
    Updates months.done and increments next_write.
    """
    next_write  = coord / "next_write.txt"
    done        = coord / "months.done"
    state       = coord / "state.json"
    write_lock  = coord / "write.lock"   # serialize the check+write+advance

    while True:
        with lock_file(write_lock):
            expected = int(next_write.read_text().strip())
            if seq == expected:
                # Re-check store existence while holding write lock
                if OUT.exists():
                    print(f"[seq {seq}] Appending {ym}…")
                    append_time(arr, OUT)
                else:
                    print(f"[seq {seq}] Writing first {ym}…")
                    write_first(arr, OUT, compressor=COMP)

                # mark done
                done_lines = [x for x in done.read_text().splitlines() if x.strip()]
                done_lines.append(ym)
                done.write_text("\n".join(done_lines) + "\n")

                # advance expected
                next_write.write_text(f"{expected+1}\n")

                # state debug
                try:
                    s = json.loads(state.read_text() or "{}")
                except Exception:
                    s = {}
                s["last_write"] = {"ym": ym, "seq": seq}
                state.write_text(json.dumps(s) + "\n")

                print(f"[seq {seq}] Wrote {ym}. next_write -> {expected+1}")
                return
            # else: not our turn yet
        time.sleep(2.0)  # brief backoff


def mark_done_without_write(ym, seq, coord: Path):
    """
    Advance the ordered write sequence for a claimed batch that is intentionally
    skipped (e.g., no files found) so later workers do not block forever.
    """
    next_write = coord / "next_write.txt"
    done = coord / "months.done"
    state = coord / "state.json"
    write_lock = coord / "write.lock"

    while True:
        with lock_file(write_lock):
            expected = int(next_write.read_text().strip())
            if seq == expected:
                done_lines = [x for x in done.read_text().splitlines() if x.strip()]
                done_lines.append(f"{ym} (SKIPPED)")
                done.write_text("\n".join(done_lines) + "\n")
                next_write.write_text(f"{expected+1}\n")

                try:
                    s = json.loads(state.read_text() or "{}")
                except Exception:
                    s = {}
                s["last_write"] = {"ym": f"{ym} (SKIPPED)", "seq": seq}
                state.write_text(json.dumps(s) + "\n")

                print(f"[seq {seq}] Skipped {ym}. next_write -> {expected+1}")
                return
        time.sleep(2.0)

def parse_args():
    ap = argparse.ArgumentParser("Daymet multi-worker month queue")
    ap.add_argument("--coord-dir", type=str, required=True,
                    help="Shared coordination directory for queue/locks.")
    ap.add_argument("--root", type=str, default=str(ROOT),
                    help="Daymet regridded root (year directories with daily files).")
    ap.add_argument("--out", type=str, default=str(OUT),
                    help="Output zarr store path.")
    ap.add_argument("--finalize", action="store_true",
                    help="Only consolidate metadata and exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan files, print month/index summary, and exit.")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild of cached month index in coord dir.")
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

    month_index, summary = load_or_build_month_index(coord, ROOT, rebuild=args.rebuild_index)
    month_labels = sorted(month_index.keys())

    if args.dry_run:
        print_dry_run_summary(summary, month_index)
        return

    if args.finalize:
        if OUT.exists():
            print("Consolidating metadata…")
            consolidate(OUT)
            print("Done:", OUT)
        else:
            print("No store to consolidate:", OUT)
        return

    # Build full month list and init queue files if needed
    if not month_labels:
        raise ValueError(f"No daymet monthly batches found under {ROOT}")
    init_queue_if_needed(coord, month_labels)

    processed = 0
    while True:
        claim = claim_next_month(coord)
        if claim is None:
            print("No more months in queue. Worker exiting.")
            break
        ym, seq = claim
        yyyy, mm = ym.split("-")
        print(f"[seq {seq}] Claimed {ym}")

        # Load + prep outside of write lock
        files = month_index.get(ym, [])
        if not files:
            print(f"[seq {seq}] No files for {ym}. Skipping.")
            mark_done_without_write(ym, seq, coord)
            continue

        ds = open_time_batch(
            files,
            engine=ENGINE,
            parallel_open=PARALLEL_OPEN,
            cast_float32=CAST_FLOAT32,
            preprocess=preprocess_strip_attrs,
            combine="by_coords",
            data_var_whitelist=DAYMET_VAR_WHITELIST,
        )
        print('ds opened')
        ds = maybe_fill_missing_leap_dec31(ds, yyyy, mm)

        arr = to_stacked_array(ds, WRITE_CHUNKS)
        print('arr stacked')

        arr = chunk_coords(arr, y=WRITE_CHUNKS["y"], x=WRITE_CHUNKS["x"])
        print('arr chunked')
        # Ordered write
        write_ordered(arr, ym, seq, coord)

        ds.close()

        processed += 1
        if args.max_months is not None and processed >= args.max_months:
            print(f"Worker processed {processed} months (limit). Exiting.")
            break

    print("Worker finished.")

if __name__ == "__main__":
    main()
