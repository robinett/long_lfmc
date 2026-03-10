#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convenience entrypoint for the ensemble wall-to-wall map pipeline. "
            "By default it builds a manifest only. Use --submit to launch the "
            "Slurm-array workflow."
        )
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--validation_test", action="store_true")
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--months_per_block", type=int, default=1)
    parser.add_argument("--requested_start_date", type=str, default=None)
    parser.add_argument("--requested_end_date", type=str, default="2024-12-31")
    return parser.parse_args()


def main():
    args = get_args()
    if args.submit:
        submit_script = os.path.join(here, "submit_create_maps_ensemble.sh")
        print(f"[create_maps] Launching submission script: {submit_script}")
        env = os.environ.copy()
        env["VALIDATION_TEST"] = "true" if args.validation_test else "false"
        if args.max_tiles is not None:
            env["MAX_TILES"] = str(args.max_tiles)
        env["MONTHS_PER_BLOCK"] = str(args.months_per_block)
        if args.requested_start_date is not None:
            env["REQUESTED_START_DATE"] = str(args.requested_start_date)
        env["REQUESTED_END_DATE"] = str(args.requested_end_date)
        subprocess.run(["bash", submit_script], check=True, env=env)
        return

    cmd = [
        sys.executable,
        os.path.join(here, "create_map_manifest.py"),
        "--months_per_block",
        str(args.months_per_block),
        "--requested_end_date",
        str(args.requested_end_date),
    ]
    if args.requested_start_date is not None:
        cmd.extend(["--requested_start_date", str(args.requested_start_date)])
    if args.validation_test:
        cmd.append("--validation_test")
    if args.max_tiles is not None:
        cmd.extend(["--max_tiles", str(args.max_tiles)])
    print("[create_maps] Building map manifest only")
    print("[create_maps] Command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
