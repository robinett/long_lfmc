#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional

import yaml


def default_paper_figure_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "paper_figure_configs.yaml")


def _validate_model_config(model_key: str, model_cfg: Dict[str, Any]) -> None:
    required_keys = ["display_name", "outputs_root", "input_data_name", "color"]
    missing = [key for key in required_keys if key not in model_cfg]
    if len(missing) > 0:
        raise KeyError(
            f"Model '{model_key}' is missing required config keys: {missing}"
        )


def _validate_figure_model_key(fig_key: str, fig_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    model_key = fig_cfg.get("model_key")
    if model_key is None:
        return
    if model_key not in cfg["models"]:
        raise KeyError(
            f"Figure '{fig_key}' references unknown model_key '{model_key}'"
        )


def _validate_figure_model_keys(fig_key: str, fig_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    model_keys = fig_cfg.get("model_keys", [])
    missing = [key for key in model_keys if key not in cfg["models"]]
    if len(missing) > 0:
        raise KeyError(
            f"Figure '{fig_key}' references unknown model_keys: {missing}"
        )


def validate_paper_figure_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_top_level = ["paths", "models", "filters", "variability", "figures", "plotting"]
    missing = [key for key in required_top_level if key not in cfg]
    if len(missing) > 0:
        raise KeyError(f"Paper figure config is missing top-level keys: {missing}")
    cfg.setdefault("timeseries_cache", {})
    cfg["timeseries_cache"].setdefault("enabled", True)
    cfg["timeseries_cache"].setdefault("rebuild", False)
    cfg["timeseries_cache"].setdefault("cache_root", None)
    if not isinstance(cfg["models"], dict) or len(cfg["models"]) == 0:
        raise ValueError("Config 'models' must be a non-empty dict")
    for model_key, model_cfg in cfg["models"].items():
        _validate_model_config(model_key, model_cfg)
    landcover_order = cfg["filters"].get("landcover_order", [])
    if len(landcover_order) == 0:
        raise ValueError("Config 'filters.landcover_order' must be non-empty")
    if "mixed_forest" in landcover_order:
        raise ValueError(
            "Do not include 'mixed_forest' in paper landcover_order; it is excluded."
        )
    for key in ["min_obs", "min_years"]:
        if key not in cfg["variability"]:
            raise KeyError(f"Config 'variability' is missing key '{key}'")
    for fig_key, fig_cfg in cfg["figures"].items():
        if not isinstance(fig_cfg, dict):
            raise ValueError(f"Figure config '{fig_key}' must be a dict")
        _validate_figure_model_key(fig_key, fig_cfg, cfg)
        _validate_figure_model_keys(fig_key, fig_cfg, cfg)
        if "anchor_model_key" in fig_cfg and fig_cfg["anchor_model_key"] not in cfg["models"]:
            raise KeyError(
                f"Figure '{fig_key}' references unknown anchor_model_key "
                f"'{fig_cfg['anchor_model_key']}'"
            )
    return cfg


def load_paper_figure_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or default_paper_figure_config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing paper figure config: {path}")
    with open(path, "r") as file_obj:
        cfg = yaml.safe_load(file_obj)
    if not isinstance(cfg, dict):
        raise ValueError(f"Paper figure config must parse to a dict: {path}")
    cfg["_config_path"] = path
    return validate_paper_figure_config(cfg)


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
