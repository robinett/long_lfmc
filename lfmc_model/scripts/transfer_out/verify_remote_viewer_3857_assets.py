#!/usr/bin/env python3

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


here = Path(__file__).resolve().parent
viewer_config_path = here.parent / "viewer_3857" / "viewer_config.yaml"
transfer_config_path = here / "source_coop_transfer_configs.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def join_url_parts(*parts: str) -> str:
    cleaned = [str(part).strip().strip("/") for part in parts if str(part).strip()]
    if not cleaned:
        return ""
    if str(parts[0]).startswith(("http://", "https://")):
        return str(parts[0]).rstrip("/") + "/" + "/".join(cleaned[1:])
    return "/".join(cleaned)


def load_remote_json(url: str):
    request = Request(
        url,
        headers={"User-Agent": "long-lfmc-viewer-transfer/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    viewer_cfg = load_yaml(viewer_config_path)
    transfer_cfg = load_yaml(transfer_config_path)
    assets_cfg = viewer_cfg["assets"]
    viewer_assets_cfg = transfer_cfg["datasets"]["viewer_3857_assets"]
    required_layers = [str(name) for name in viewer_assets_cfg.get("required_layers", [])]

    manifest_url = join_url_parts(
        str(assets_cfg["source_asset_base_url"]),
        str(assets_cfg["manifest_filename"]),
    )
    log(f"Opening remote viewer manifest {manifest_url}")
    started = time.time()
    manifest = load_remote_json(manifest_url)
    elapsed = time.time() - started
    log(f"Opened remote viewer manifest in {elapsed:.2f}s")

    dates = manifest.get("dates", [])
    if not dates:
        raise ValueError("Remote viewer manifest has no dates")
    layers = manifest.get("layers", {})
    missing_layers = [name for name in required_layers if name not in layers]
    if missing_layers:
        raise ValueError(f"Remote viewer manifest is missing required layers: {missing_layers}")

    for layer_name in required_layers:
        layer = layers[layer_name]
        template = str(layer.get("tile_root_template", ""))
        tile_counts = layer.get("tile_counts", {})
        if "{date}" not in template or "{z}" not in template:
            raise ValueError(f"Layer {layer_name!r} has an invalid tile template: {template!r}")
        if not tile_counts:
            raise ValueError(f"Layer {layer_name!r} has no tile_counts")
        log(
            f"Verified layer {layer_name}: label={layer.get('label')!r}, "
            f"unit={layer.get('unit')!r}, min={layer.get('min')}, max={layer.get('max')}"
        )

    log(
        "Remote viewer assets manifest verified with "
        f"{len(dates)} dates and layers {sorted(layers)}"
    )


if __name__ == "__main__":
    main()
