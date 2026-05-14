from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from keyboard_fusion.paths import CONFIGS_DIR


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the project YAML config."""
    config_path = Path(path) if path is not None else CONFIGS_DIR / "default.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must load to a dictionary: {config_path}")
    return config

