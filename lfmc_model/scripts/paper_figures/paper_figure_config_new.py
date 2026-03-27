#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional

import yaml


def default_paper_figure_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "paper_figure_configs_new.yaml")


def validate_paper_figure_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_top_level = ["paths", "figures"]
    missing = [key for key in required_top_level if key not in cfg]
    if len(missing) > 0:
        raise KeyError(f"New paper figure config is missing top-level keys: {missing}")

    if "output_root" not in cfg["paths"]:
        raise KeyError("New paper figure config is missing 'paths.output_root'")

    cfg.setdefault("plotting", {})
    cfg.setdefault("models", {})
    cfg.setdefault("filters", {})
    cfg.setdefault("variability", {})
    cfg["filters"].setdefault(
        "landcover_order",
        ["shrub", "evergreen_forest", "deciduous_forest", "grass"],
    )
    cfg.setdefault("timeseries_selection", {})
    cfg["timeseries_selection"].setdefault("r2_percentiles", {})
    cfg["timeseries_selection"]["r2_percentiles"].setdefault("good", 95)
    cfg["timeseries_selection"]["r2_percentiles"].setdefault("average", 50)
    cfg["timeseries_selection"]["r2_percentiles"].setdefault("poor", 5)

    if len(cfg["models"]) > 0:
        for model_key, model_cfg in cfg["models"].items():
            required_model_keys = ["display_name", "outputs_root", "input_data_name", "color"]
            missing_model_keys = [key for key in required_model_keys if key not in model_cfg]
            if len(missing_model_keys) > 0:
                raise KeyError(
                    f"Model '{model_key}' is missing required config keys: {missing_model_keys}"
                )

    if not isinstance(cfg["figures"], dict) or len(cfg["figures"]) == 0:
        raise ValueError("Config 'figures' must be a non-empty dict")

    for fig_key, fig_cfg in cfg["figures"].items():
        if not isinstance(fig_cfg, dict):
            raise ValueError(f"Figure config '{fig_key}' must be a dict")
        if "enabled" not in fig_cfg:
            fig_cfg["enabled"] = False
        if bool(fig_cfg.get("enabled", False)) and "filename" not in fig_cfg:
            raise KeyError(f"Enabled figure '{fig_key}' is missing 'filename'")
        model_key = fig_cfg.get("model_key")
        if model_key is not None and model_key not in cfg["models"]:
            raise KeyError(
                f"Figure '{fig_key}' references unknown model_key '{model_key}'"
            )
        model_keys = fig_cfg.get("model_keys")
        if model_keys is not None:
            unknown_model_keys = [key for key in model_keys if key not in cfg["models"]]
            if len(unknown_model_keys) > 0:
                raise KeyError(
                    f"Figure '{fig_key}' references unknown model_keys: {unknown_model_keys}"
                )

    if "figure_3" in cfg["figures"]:
        required_variability_keys = [
            "site_min_obs",
            "monthly_min_obs",
            "monthly_min_years",
        ]
        missing_variability_keys = [
            key for key in required_variability_keys
            if key not in cfg["variability"]
        ]
        if len(missing_variability_keys) > 0:
            raise KeyError(
                "New paper figure config is missing required variability keys for figure_3: "
                f"{missing_variability_keys}"
            )

    if "figure_4" in cfg["figures"]:
        required_variability_keys = [
            "site_min_obs",
            "monthly_min_obs",
            "monthly_min_years",
        ]
        missing_variability_keys = [
            key for key in required_variability_keys
            if key not in cfg["variability"]
        ]
        if len(missing_variability_keys) > 0:
            raise KeyError(
                "New paper figure config is missing required variability keys for figure_4: "
                f"{missing_variability_keys}"
            )

    if "figure_5" in cfg["figures"]:
        required_variability_keys = [
            "site_min_obs",
            "monthly_min_obs",
            "monthly_min_years",
        ]
        missing_variability_keys = [
            key for key in required_variability_keys
            if key not in cfg["variability"]
        ]
        if len(missing_variability_keys) > 0:
            raise KeyError(
                "New paper figure config is missing required variability keys for figure_5: "
                f"{missing_variability_keys}"
            )

    return cfg


def load_paper_figure_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or default_paper_figure_config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing new paper figure config: {path}")
    with open(path, "r") as file_obj:
        cfg = yaml.safe_load(file_obj)
    if not isinstance(cfg, dict):
        raise ValueError(f"New paper figure config must parse to a dict: {path}")
    cfg["_config_path"] = path
    return validate_paper_figure_config(cfg)


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
