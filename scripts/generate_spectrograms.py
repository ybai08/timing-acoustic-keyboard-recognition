from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.config import load_config
from keyboard_fusion.paths import PROCESSED_DATA_DIR
from keyboard_fusion.spectrograms import find_latest_clip_session, generate_session_spectrograms


def resolve_clip_manifest(session: str | None, clip_manifest: str | None) -> Path:
    if clip_manifest:
        path = Path(clip_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Could not find clip manifest: {clip_manifest}")
        return path

    clips_root = PROCESSED_DATA_DIR / "clips"
    if session:
        path = Path(session)
        if path.exists() and path.is_dir():
            return path / "clip_manifest.csv"
        return clips_root / session / "clip_manifest.csv"

    return find_latest_clip_session(clips_root) / "clip_manifest.csv"


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    feature_config = config.get("features", {})

    parser = argparse.ArgumentParser(description="Generate log-mel spectrogram arrays from extracted clips.")
    parser.add_argument(
        "--session",
        help=(
            "Clip session ID or folder path. Defaults to the latest folder under "
            "data/processed/clips/."
        ),
    )
    parser.add_argument("--clip-manifest", help="Path to a specific clip_manifest.csv.")
    parser.add_argument("--output-root", type=Path, default=PROCESSED_DATA_DIR / "spectrograms")
    parser.add_argument("--mel-bands", type=int, default=int(feature_config.get("mel_bands", 64)))
    parser.add_argument("--fft-window-size", type=int, default=int(feature_config.get("fft_window_size", 1024)))
    parser.add_argument("--hop-length", type=int, default=int(feature_config.get("hop_length", 256)))
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Number of clips to include in the preview HTML. Use 0 for all clips, which is the default.",
    )
    args = parser.parse_args(argv)

    clip_manifest_path = resolve_clip_manifest(args.session, args.clip_manifest)
    records, manifest_path, report_path, preview_path = generate_session_spectrograms(
        clip_manifest_path=clip_manifest_path,
        output_root=args.output_root,
        mel_bands=args.mel_bands,
        fft_window_size=args.fft_window_size,
        hop_length=args.hop_length,
        preview_count=args.preview_count,
    )

    print(f"Clip manifest: {clip_manifest_path}")
    print(f"Spectrograms generated: {len(records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(f"Preview: {preview_path}")
    if records:
        first = records[0]
        print(
            "Shape: "
            f"{first['mel_bands']} mel bands x {first['frames']} frames "
            f"(fft={first['fft_window_size']}, hop={first['hop_length']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
