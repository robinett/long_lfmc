#!/usr/bin/env python3
# Parallel month workers with ordered writes to a single Zarr.

from pathlib import Path
import argparse
import re
import time
import json
import fcntl
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
)

# --------- CONFIG (adjust paths/chunks as needed) ----------
ROOT = Path(
    "/oak/stanford/groups/konings/trobinet/long_lfmc/"
    "trent_datasets/daymet/daymet_regrid"
)
OUT = Path(
    "/oak/stanford/groups/konings/trobinet/long_lfmc/"
    "trent_datasets/daymet/daymet_all_vars.zarr"
)
WRITE_CHUNKS = {"time": 1, "variable": 9999, "y": 512, "x": 512}
CAST_FLOAT32 = True
ENGINE = "h5netcdf"
PARALLEL_OPEN = False
COMP = DEFAULT_COMP
# -----------------------------------------------------------

MONTH_RE = re.compile(r"^\d{2}$")
YEAR_RE  = re.compile(r"^\d{4}$")

def find_month_keys(root: Path):
    """Return sorted [(YYYY, MM)] for all months that exist across variables."""
    keys = set()
    for var_dir in root.iterdir():
        if not var_dir.is_dir():
            continue
        for y_dir in var_dir.iterdir():
            if not (y_dir.is_dir() and YEAR_RE.match(y_dir.name)):
                continue
            for m_dir in y_dir.iterdir():
                if m_dir.is_dir() and MONTH_RE.match(m_dir.name):
                    keys.add((y_dir.name, m_dir.name))
    # chronological
    return sorted(keys, key=lambda ym: (int(ym[0]), int(ym[1])))

def files_for_month(root: Path, year: str, month: str, patterns=(".nc", ".nc4")):
    out = []
    for var_dir in root.iterdir():
        ydir = var_dir / year
        mdir = ydir / month
        if not mdir.is_dir():
            continue
        for p in patterns:
            out.extend(sorted(mdir.glob(f"*{p}")))
    return out

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

def init_queue_if_needed(coord: Path, months_all):
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
            all_lines = [f"{y}-{m}" for (y, m) in months_all]
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

        yyyy, mm = line.split("-")
        return (yyyy, mm, seq)

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

def parse_args():
    ap = argparse.ArgumentParser("Daymet multi-worker month queue")
    ap.add_argument("--coord-dir", type=str, required=True,
                    help="Shared coordination directory for queue/locks.")
    ap.add_argument("--finalize", action="store_true",
                    help="Only consolidate metadata and exit.")
    ap.add_argument("--max-months", type=int, default=None,
                    help="Optional: limit number of claimed months this worker will process.")
    return ap.parse_args()

def main():
    args = parse_args()
    coord = Path(args.coord_dir)

    if args.finalize:
        if OUT.exists():
            print("Consolidating metadata…")
            consolidate(OUT)
            print("Done:", OUT)
        else:
            print("No store to consolidate:", OUT)
        return

    # Build full month list and init queue files if needed
    months_all = find_month_keys(ROOT)
    init_queue_if_needed(coord, months_all)

    processed = 0
    while True:
        claim = claim_next_month(coord)
        if claim is None:
            print("No more months in queue. Worker exiting.")
            break
        yyyy, mm, seq = claim
        ym = f"{yyyy}-{mm}"
        print(f"[seq {seq}] Claimed {ym}")

        # Load + prep outside of write lock
        files = files_for_month(ROOT, yyyy, mm, (".nc", ".nc4"))
        if not files:
            print(f"[seq {seq}] No files for {ym}. Skipping.")
            # Still need to "complete" to advance? No: we never wrote it.
            # Put it into done to avoid infinite loop? Safer to mark done.
            write_ordered(xr.DataArray(), ym + " (EMPTY)", seq, coord)  # no-op write_first/append won't be called
            continue

        ds = open_time_batch(
            files,
            engine=ENGINE,
            parallel_open=PARALLEL_OPEN,
            cast_float32=CAST_FLOAT32,
            preprocess=preprocess_strip_attrs,
            combine="by_coords",
        )
        print('ds opened')

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
