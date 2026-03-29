#!/usr/bin/env python3

import argparse
import json
import os

from train_multitarget_longweather_vvvh import _fold_locs_to_jsonable, create_site_split, load_data
from train_multitarget_longweather_vvvh import (
    _assign_remaining_sites_to_test_folds,
    _dedupe_sites_in_order,
    _validate_sites_assigned_exactly_once,
)


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
    with open(os.path.join(args.input_data_dir, "var_names.json"), "r") as f:
        var_names = json.load(f)
    source = datasets[4]
    info = datasets[5]
    stratifier = datasets[6]
    source_np = source.detach().cpu().numpy().reshape(-1) if hasattr(source, 'detach') else source.reshape(-1)
    lfmc_mask = source_np == 0
    lfmc_info = info.loc[lfmc_mask].reset_index(drop=True)
    lfmc_stratifier = stratifier[lfmc_mask]
    lfmc_source = source_np[lfmc_mask]
    all_sites = _dedupe_sites_in_order(lfmc_info[["latitude", "longitude"]].to_numpy())

    num_insitu_obs = int((lfmc_source == 0).sum())
    num_vv_obs = 0
    num_vh_obs = 0
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
            lfmc_info,
            lfmc_source,
            desired_insitu_sample_size=int(desired_insitu_obs_per_fold),
            desired_vv_sample_size=0,
            desired_vh_sample_size=0,
            seed=int(args.split_seed),
            used_sites=used_sites,
            stratifier=lfmc_stratifier,
        )
        used_sites.extend(this_locs)
        fold_locs[fold + 1] = this_locs
    fold_locs = _assign_remaining_sites_to_test_folds(
        fold_locs=fold_locs,
        all_sites=all_sites,
        site_climate_lookup=None,
        site_stratifier_lookup=None,
        enforce_climate_train_support=False,
    )

    remove_last = False
    for fold in sorted(fold_locs):
        if len(fold_locs[fold]) == 0 and fold != args.n_folds:
            raise ValueError(f"Fold {fold} has no locations")
        if len(fold_locs[fold]) == 0 and fold == args.n_folds:
            print(f"Fold {fold} has no locations, removing")
            remove_last = True
    if remove_last:
        del fold_locs[args.n_folds]
    _validate_sites_assigned_exactly_once(fold_locs, all_sites=all_sites)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(_fold_locs_to_jsonable(fold_locs), f)
    print(f"Wrote canonical fold info to {args.out_path}")


if __name__ == "__main__":
    main()
