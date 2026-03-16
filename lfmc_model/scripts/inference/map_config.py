#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional

import yaml


def default_map_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "map_configs.yaml")


def load_map_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or default_map_config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing map config: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Map config must parse to a dict: {path}")
    cfg["_config_path"] = path
    return cfg


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def config_or_override(override: Any, cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    if override is not None:
        return override
    return get_cfg(cfg, *keys, default=default)
