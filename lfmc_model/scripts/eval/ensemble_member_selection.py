import os
import re
from typing import Dict, List, Optional, Sequence

import yaml


here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENSEMBLE_SELECTION_CONFIG_PATH = os.path.abspath(
    os.path.join(here, "..", "ensemble_member_selection.yaml")
)

_SELECTION_CONFIG_CACHE: Dict[str, Dict[str, object]] = {}


def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
    if value in {None, "", "None"}:
        return None
    return str(value)


def _normalize_path(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_optional_str(value)
    if normalized is None:
        return None
    return os.path.normpath(normalized)


def _is_unset(value) -> bool:
    return value is None or value == "" or value == "None"


def _extract_training_id(member_name: str) -> Optional[int]:
    match = re.search(r"(?:^|_)ds(\d+)(?:_|$)", str(member_name))
    if match is None:
        return None
    return int(match.group(1))


def load_ensemble_selection_config(
    config_path: str = DEFAULT_ENSEMBLE_SELECTION_CONFIG_PATH,
) -> Dict[str, object]:
    config_path = os.path.abspath(config_path)
    if config_path in _SELECTION_CONFIG_CACHE:
        return _SELECTION_CONFIG_CACHE[config_path]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing ensemble member selection config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Selection config must parse to a dict: {config_path}")
    cfg["_config_path"] = config_path
    _SELECTION_CONFIG_CACHE[config_path] = cfg
    return cfg


def resolve_ensemble_member_selection(
    outputs_root: Optional[str],
    member_name_prefix: Optional[str] = None,
    selection_key: Optional[str] = None,
    config_path: str = DEFAULT_ENSEMBLE_SELECTION_CONFIG_PATH,
) -> Optional[Dict[str, object]]:
    cfg = load_ensemble_selection_config(config_path=config_path)
    selections = cfg.get("selections", {})
    if not isinstance(selections, dict):
        raise ValueError(f"Selection config 'selections' must be a dict: {cfg['_config_path']}")
    if selection_key not in {None, "", "None"}:
        if selection_key not in selections:
            raise KeyError(
                f"Unknown ensemble selection key {selection_key!r} in {cfg['_config_path']}"
            )
        selection = dict(selections[selection_key])
        selection["_selection_key"] = str(selection_key)
        selection["_config_path"] = cfg["_config_path"]
        return selection

    normalized_root = _normalize_path(outputs_root)
    normalized_prefix = _normalize_optional_str(member_name_prefix)
    matches: List[Dict[str, object]] = []
    for key, entry in selections.items():
        if not isinstance(entry, dict) or not bool(entry.get("auto_match", False)):
            continue
        if _normalize_path(entry.get("outputs_root")) != normalized_root:
            continue
        if _normalize_optional_str(entry.get("member_name_prefix")) != normalized_prefix:
            continue
        selection = dict(entry)
        selection["_selection_key"] = str(key)
        selection["_config_path"] = cfg["_config_path"]
        matches.append(selection)
    if len(matches) > 1:
        raise ValueError(
            f"Multiple auto-match ensemble selections matched outputs_root={outputs_root!r} "
            f"member_name_prefix={member_name_prefix!r} in {cfg['_config_path']}"
        )
    if len(matches) == 1:
        return matches[0]
    return None


def apply_ensemble_member_selection(
    member_dirs: Sequence[str],
    outputs_root: Optional[str],
    member_name_prefix: Optional[str] = None,
    selection_key: Optional[str] = None,
    member_name_allowlist: Optional[Sequence[str]] = None,
    member_name_suffix_allowlist: Optional[Sequence[str]] = None,
    member_training_id_allowlist: Optional[Sequence[int]] = None,
    config_path: str = DEFAULT_ENSEMBLE_SELECTION_CONFIG_PATH,
) -> List[str]:
    selection = resolve_ensemble_member_selection(
        outputs_root=outputs_root,
        member_name_prefix=member_name_prefix,
        selection_key=selection_key,
        config_path=config_path,
    )
    if _is_unset(member_name_allowlist) and selection is not None:
        member_name_allowlist = selection.get("member_names")
    if _is_unset(member_name_suffix_allowlist) and selection is not None:
        member_name_suffix_allowlist = selection.get("member_name_suffixes")
    if _is_unset(member_training_id_allowlist) and selection is not None:
        member_training_id_allowlist = selection.get("member_training_ids")

    name_allowlist = None
    if not _is_unset(member_name_allowlist):
        name_allowlist = [str(v) for v in member_name_allowlist]
    suffix_allowlist = None
    if not _is_unset(member_name_suffix_allowlist):
        suffix_allowlist = [str(v) for v in member_name_suffix_allowlist]
    training_id_allowlist = None
    if not _is_unset(member_training_id_allowlist):
        training_id_allowlist = [int(v) for v in member_training_id_allowlist]

    selected_dirs: List[str] = []
    member_names = [os.path.basename(str(path)) for path in member_dirs]
    for member_dir, member_name in zip(member_dirs, member_names):
        keep = True
        if name_allowlist is not None and member_name not in name_allowlist:
            keep = False
        if (
            keep
            and suffix_allowlist is not None
            and not any(member_name.endswith(suffix) for suffix in suffix_allowlist)
        ):
            keep = False
        if keep and training_id_allowlist is not None:
            training_id = _extract_training_id(member_name)
            if training_id is None or training_id not in training_id_allowlist:
                keep = False
        if keep:
            selected_dirs.append(str(member_dir))

    if name_allowlist is not None:
        available_names = set(member_names)
        missing = [name for name in name_allowlist if name not in available_names]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"Configured ensemble member names were not found under {outputs_root}: {missing}"
            )
    if suffix_allowlist is not None:
        missing = [
            suffix for suffix in suffix_allowlist
            if not any(member_name.endswith(suffix) for member_name in member_names)
        ]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"Configured ensemble member suffixes were not found under {outputs_root}: {missing}"
            )
    if training_id_allowlist is not None:
        available_training_ids = {
            training_id
            for training_id in [_extract_training_id(name) for name in member_names]
            if training_id is not None
        }
        missing = [
            int(training_id)
            for training_id in training_id_allowlist
            if int(training_id) not in available_training_ids
        ]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"Configured ensemble training ids were not found under {outputs_root}: {missing}"
            )
    return selected_dirs
