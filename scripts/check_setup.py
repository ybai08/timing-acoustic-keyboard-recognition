from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "librosa",
    "sounddevice",
    "soundfile",
    "matplotlib",
    "seaborn",
    "pynput",
]
OPTIONAL_ML_PACKAGES = [
    "torch",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    "configs/default.yaml",
    "prompts",
    "data/raw",
    "data/processed",
    "data/metadata",
    "src/keyboard_fusion",
]


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print()

    missing: list[str] = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {package}: {version}")
        except Exception as exc:  # pragma: no cover - setup diagnostic
            print(f"[MISSING] {package}: {exc}")
            missing.append(package)

    print()
    for package in OPTIONAL_ML_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] optional ML {package}: {version}")
        except Exception:
            print(f"[OPTIONAL] {package}: install with `python -m pip install -r requirements-ml.txt` for CNN training")

    print()
    for relative_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            print(f"[OK] {relative_path}")
        else:
            print(f"[MISSING] {relative_path}")
            missing.append(relative_path)

    print()
    try:
        import yaml

        config_path = PROJECT_ROOT / "configs/default.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        sample_rate = config.get("audio", {}).get("sample_rate")
        microphone = config.get("hardware", {}).get("microphone", {}).get("name", "unknown")
        print(f"[OK] config loaded: sample_rate={sample_rate}, microphone={microphone}")
    except Exception as exc:
        print(f"[MISSING] config could not be loaded: {exc}")
        missing.append("config")

    if missing:
        print()
        print("Some packages are missing. Run:")
        print("  python -m pip install -r requirements.txt")
        return 1

    print()
    print("Setup looks good. Ready for prompt/audio collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
