from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.config import load_config
from keyboard_fusion.neural_segmentation import DEFAULT_SEGMENTER_DIR, segment_audio_file_to_clips
from keyboard_fusion.paths import PROCESSED_DATA_DIR


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    segmentation_config = config.get("segmentation", {})

    parser = argparse.ArgumentParser(
        description=(
            "Use the trained neural segmenter to convert one raw multi-key audio file "
            "into individual detected keystroke clips."
        )
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to a WAV file containing multiple keystrokes.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SEGMENTER_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "neural_segments" / "manual",
        help="Folder where detected clips and clip_manifest.csv will be written.",
    )
    parser.add_argument(
        "--expected-keys",
        type=int,
        help="Optional hard cap for detected clips. Use this when you know roughly how many keys were pressed.",
    )
    parser.add_argument("--threshold", type=float, help="Override the saved segmenter threshold.")
    parser.add_argument("--pre-ms", type=float, default=float(segmentation_config.get("pre_keydown_ms", 20)))
    parser.add_argument("--post-ms", type=float, default=float(segmentation_config.get("post_keydown_ms", 45)))
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, mps, or cuda. auto prefers Apple MPS, then CUDA, then CPU.",
    )
    args = parser.parse_args(argv)

    if not args.audio.exists():
        raise FileNotFoundError(f"Could not find audio file: {args.audio}")

    peaks, records = segment_audio_file_to_clips(
        audio_path=args.audio,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        expected_keys=args.expected_keys,
        threshold=args.threshold,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        device=args.device,
    )

    print(f"Audio: {args.audio}")
    print(f"Model: {args.model_dir}")
    print(f"Detected clips: {len(records)}")
    print(f"Output folder: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'clip_manifest.csv'}")
    for peak in peaks[:20]:
        print(f"- {peak.time_seconds:.3f}s probability={peak.strength:.3f}")
    if len(peaks) > 20:
        print(f"... {len(peaks) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
