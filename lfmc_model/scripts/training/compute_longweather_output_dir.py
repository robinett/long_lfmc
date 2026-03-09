#!/usr/bin/env python3

import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the longweather training output directory name."
    )
    parser.add_argument("--input_data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--adam_wd", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--nhead", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--dim_feedforward", type=int, required=True)
    parser.add_argument("--long_d_model", type=int, required=True)
    parser.add_argument("--long_nhead", type=int, required=True)
    parser.add_argument("--long_num_layers", type=int, required=True)
    parser.add_argument("--long_dim_feedforward", type=int, required=True)
    parser.add_argument("--long_out_dim", type=int, required=True)
    parser.add_argument("--task_weight_type", type=str, required=True)
    parser.add_argument("--manual_task_weights", type=float, nargs="+", default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    source = torch.load(
        os.path.join(args.input_data_dir, "source.pt"),
        weights_only=False,
    )
    num_insitu_obs = int((source == 0).sum().item())
    num_vv_obs = int((source == 1).sum().item())
    num_vh_obs = int((source == 2).sum().item())

    batches_per_epoch = (num_insitu_obs + num_vv_obs + num_vh_obs) / args.batch_size * 0.7
    warmup_steps = int(3 * batches_per_epoch)

    first_task_weight_tag = ""
    if (
        args.task_weight_type == "manual"
        and args.manual_task_weights is not None
        and len(args.manual_task_weights) > 0
    ):
        first_task_weight_tag = f"_tw0{args.manual_task_weights[0]}"

    model_name = (
        f"transformer_dm{args.d_model}_nh{args.nhead}_nl{args.num_layers}_df{args.dim_feedforward}"
        f"_do{args.dropout}_bs{args.batch_size}_lr{args.lr}_warmup{warmup_steps}"
        f"_wd{args.adam_wd}_iobs{num_insitu_obs}_vvobs{num_vv_obs}_vhobs{num_vh_obs}"
        f"_dmlong{args.long_d_model}_nhlong{args.long_nhead}_nllong{args.long_num_layers}"
        f"_dflong{args.long_dim_feedforward}_outlong{args.long_out_dim}"
        f"_firstweight{first_task_weight_tag}"
    )
    if args.run_tag is not None and len(args.run_tag) > 0:
        model_name = f"{model_name}_{args.run_tag}"

    print(os.path.join(args.save_dir, model_name))


if __name__ == "__main__":
    main()
