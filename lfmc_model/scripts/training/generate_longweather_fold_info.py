#!/usr/bin/env python3

import argparse
import json
import os

from train_multitarget_longweather_vvvh import _fold_locs_to_jsonable, create_site_split, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a canonical fold_info.json for longweather training."
    )
    parser.add_argument("--input-data-dir", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = load_data(args.input_data_dir)
    source = datasets[4]
    info = datasets[5]
    stratifier = datasets[6]

    num_insitu_obs = int((source == 0).sum().item())
    num_vv_obs = int((source == 1).sum().item())
    num_vh_obs = int((source == 2).sum().item())
    print(
        f"Generating canonical folds from {args.input_data_dir} with "
        f"split_seed={args.split_seed}, n_folds={args.n_folds}"
    )
    print(
        f"Observation counts: insitu={num_insitu_obs}, vv={num_vv_obs}, vh={num_vh_obs}"
    )

    desired_insitu_obs_per_fold = num_insitu_obs / args.n_folds
    desired_vv_obs_per_fold = num_vv_obs / args.n_folds
    desired_vh_obs_per_fold = num_vh_obs / args.n_folds

    fold_locs = {}
    used_sites = []
    for fold in range(args.n_folds):
        print(f"Generating fold {fold + 1}/{args.n_folds}")
        this_locs = create_site_split(
            info,
            source,
            desired_insitu_sample_size=int(desired_insitu_obs_per_fold),
            desired_vv_sample_size=int(desired_vv_obs_per_fold),
            desired_vh_sample_size=int(desired_vh_obs_per_fold),
            seed=int(args.split_seed),
            used_sites=used_sites,
            stratifier=stratifier,
        )
        used_sites.extend(this_locs)
        fold_locs[fold + 1] = this_locs

    remove_last = False
    for fold in sorted(fold_locs):
        if len(fold_locs[fold]) == 0 and fold != args.n_folds:
            raise ValueError(f"Fold {fold} has no locations")
        if len(fold_locs[fold]) == 0 and fold == args.n_folds:
            print(f"Fold {fold} has no locations, removing")
            remove_last = True
    if remove_last:
        del fold_locs[args.n_folds]

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(_fold_locs_to_jsonable(fold_locs), f)
    print(f"Wrote canonical fold info to {args.out_path}")


if __name__ == "__main__":
    main()
