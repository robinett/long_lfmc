#!/usr/bin/env python3

import argparse

from paper_figure_builders import build_enabled_figures
from paper_figure_config import load_paper_figure_config


def get_args():
    parser = argparse.ArgumentParser(
        description="Build config-driven paper figures for the final LFMC models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the paper figure YAML config.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of figure keys to build (e.g. figure_1 figure_3).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    cfg = load_paper_figure_config(args.config)
    outputs = build_enabled_figures(cfg, only_figures=args.figures)
    print("Finished paper figure generation.")
    for fig_key, out_path in outputs.items():
        print(f"  {fig_key}: {out_path}")


if __name__ == "__main__":
    main()
