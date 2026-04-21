#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, Optional

import yaml


def default_current_model_family_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "current_model_family.yaml")


def load_current_model_family_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or default_current_model_family_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing current model family config: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Current model family config must parse to a dict: {path}")
    cfg["_config_path"] = path
    return cfg


def resolve_current_model_family(
    variant: str = "multitask",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_current_model_family_config(config_path)
    root = cfg.get("current_model_family", {})
    if not isinstance(root, dict):
        raise ValueError(
            f"current_model_family must be a dict in {cfg['_config_path']}"
        )
    label = str(root.get("label", "")).strip()
    variant_cfg = root.get(variant, {})
    if not isinstance(variant_cfg, dict):
        raise ValueError(
            f"Missing variant {variant!r} in current model family config {cfg['_config_path']}"
        )
    outputs_root = str(variant_cfg.get("outputs_root", "")).strip()
    input_data_name = str(variant_cfg.get("input_data_name", "")).strip()
    model_num_tasks = int(variant_cfg.get("model_num_tasks", 0))
    if outputs_root == "" or input_data_name == "":
        raise ValueError(
            f"Variant {variant!r} is missing outputs_root/input_data_name in {cfg['_config_path']}"
        )
    if model_num_tasks <= 0:
        raise ValueError(
            f"Variant {variant!r} has invalid model_num_tasks={model_num_tasks} in {cfg['_config_path']}"
        )
    return {
        "label": label,
        "variant": str(variant),
        "outputs_root": outputs_root,
        "input_data_name": input_data_name,
        "model_num_tasks": model_num_tasks,
        "config_path": cfg["_config_path"],
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Resolve the active model-family defaults for operational inference."
    )
    parser.add_argument("--variant", type=str, default="multitask")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    return parser.parse_args()


def main():
    args = get_args()
    resolved = resolve_current_model_family(
        variant=args.variant,
        config_path=args.config_path,
    )
    if args.format == "lines":
        print(resolved["label"])
        print(resolved["outputs_root"])
        print(resolved["input_data_name"])
        print(str(resolved["model_num_tasks"]))
        print(resolved["config_path"])
        return
    print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
