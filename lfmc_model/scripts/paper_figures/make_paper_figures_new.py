#!/usr/bin/env python3

import argparse

from paper_figure_builders_new import build_enabled_figures
from paper_figure_config_new import load_paper_figure_config


def get_args():
    parser = argparse.ArgumentParser(
        description="Build config-driven paper figures for the new paper workflow."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the new paper figure YAML config.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of figure keys to build (e.g. figure_1).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    cfg = load_paper_figure_config(args.config)
    outputs = build_enabled_figures(cfg, only_figures=args.figures)
    print("Finished new paper figure generation.")
    for fig_key, out_path in outputs.items():
        print(f"  {fig_key}: {out_path}")


if __name__ == "__main__":
    main()
