from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.alignment import align_session, find_latest_session
from keyboard_fusion.config import load_config
from keyboard_fusion.paths import METADATA_DIR, RAW_DATA_DIR


def resolve_session_dir(session: str | None) -> Path:
    if not session:
        return find_latest_session()

    path = Path(session)
    if path.exists():
        return path

    session_path = RAW_DATA_DIR / "sessions" / session
    if session_path.exists():
        return session_path

    raise FileNotFoundError(f"Could not find session: {session}")


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    segmentation_config = config.get("segmentation", {})
    default_pre_ms = float(segmentation_config.get("pre_keydown_ms", 50))
    default_post_ms = float(segmentation_config.get("post_keydown_ms", 250))

    parser = argparse.ArgumentParser(description="Map recorded key events onto WAV sample windows.")
    parser.add_argument(
        "--session",
        help=(
            "Session ID or path to align. Defaults to the latest folder under "
            "data/raw/sessions/."
        ),
    )
    parser.add_argument("--pre-ms", type=float, default=default_pre_ms)
    parser.add_argument("--post-ms", type=float, default=default_post_ms)
    parser.add_argument("--output-root", type=Path, default=METADATA_DIR / "alignment")
    args = parser.parse_args(argv)

    session_dir = resolve_session_dir(args.session)
    alignments, output_dir, report_path = align_session(
        session_dir=session_dir,
        pre_keydown_ms=args.pre_ms,
        post_keydown_ms=args.post_ms,
        output_root=args.output_root,
    )

    print(f"Aligned session: {session_dir.name}")
    print(f"Trials aligned: {len(alignments)}")
    print(f"Output folder: {output_dir}")
    print(f"Report: {report_path}")
    for alignment in alignments:
        counts = alignment["event_counts"]
        timing = alignment["trial_timing"]
        print(
            f"- {alignment['trial_id']}: "
            f"{counts['aligned_keydown_events']} keydowns, "
            f"{counts['outside_audio_keydown_events']} outside audio, "
            f"{counts['clipped_window_events']} clipped, "
            f"typed_match={alignment['typed_matches_prompt']}, "
            f"wav={timing['wav_duration_seconds']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
