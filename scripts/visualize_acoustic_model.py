from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.acoustic_visualization import (
    find_latest_acoustic_model_session,
    generate_acoustic_visualization,
)
from keyboard_fusion.paths import MODELS_DIR


def resolve_model_dir(session: str | None, model_dir: str | None, kind: str) -> Path:
    if model_dir:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Could not find model folder: {model_dir}")
        return path

    model_root = MODELS_DIR / ("acoustic_baseline" if kind == "baseline" else "acoustic_cnn")
    if session:
        path = Path(session)
        if path.exists() and path.is_dir():
            return path
        session_path = model_root / session
        if session_path.exists():
            return session_path
        raise FileNotFoundError(f"Could not find acoustic {kind} session: {session}")

    return find_latest_acoustic_model_session(kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a browser visualization for an acoustic model.")
    parser.add_argument(
        "--kind",
        choices=["cnn", "baseline"],
        default="cnn",
        help="Which model family to resolve when --model-dir is not provided. Defaults to cnn.",
    )
    parser.add_argument(
        "--session",
        help=(
            "Model session ID or folder path. Defaults to the latest folder under "
            "models/acoustic_cnn/ unless --kind baseline is used."
        ),
    )
    parser.add_argument("--model-dir", help="Path to a specific acoustic model output folder.")
    parser.add_argument("--output", type=Path, help="Optional output HTML path.")
    args = parser.parse_args(argv)

    model_dir = resolve_model_dir(args.session, args.model_dir, args.kind)
    output_path = generate_acoustic_visualization(model_dir=model_dir, output_path=args.output)

    print(f"Model folder: {model_dir}")
    print(f"Visualization: {output_path}")
    print(f"Open with: open \"{output_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
