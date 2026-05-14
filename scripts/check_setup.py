from __future__ import annotations

import importlib
import platform
import sys


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

    if missing:
        print()
        print("Some packages are missing. Run:")
        print("  python -m pip install -r requirements.txt")
        return 1

    print()
    print("Setup looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

