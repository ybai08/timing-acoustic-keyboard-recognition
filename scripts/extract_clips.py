from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.paths import METADATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from keyboard_fusion.preprocessing import extract_session_clips, find_latest_alignment_session


def resolve_alignment_session(session: str | None) -> Path:
    alignment_root = METADATA_DIR / "alignment"
    if not session:
        return find_latest_alignment_session(alignment_root)

    path = Path(session)
    if path.exists():
        return path

    session_path = alignment_root / session
    if session_path.exists():
        return session_path

    raise FileNotFoundError(f"Could not find alignment session: {session}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut labeled keystroke WAV clips from aligned trials.")
    parser.add_argument(
        "--session",
        help=(
            "Session ID or alignment folder path to extract. Defaults to the latest "
            "folder under data/metadata/alignment/."
        ),
    )
    parser.add_argument("--raw-sessions-dir", type=Path, default=RAW_DATA_DIR / "sessions")
    parser.add_argument("--output-root", type=Path, default=PROCESSED_DATA_DIR / "clips")
    args = parser.parse_args(argv)

    alignment_session_dir = resolve_alignment_session(args.session)
    records, manifest_path, report_path = extract_session_clips(
        alignment_session_dir=alignment_session_dir,
        raw_sessions_dir=args.raw_sessions_dir,
        output_root=args.output_root,
    )

    print(f"Extracted session: {alignment_session_dir.name}")
    print(f"Clips extracted: {len(records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
