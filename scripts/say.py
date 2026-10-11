#!/usr/bin/env python3
"""Speak arbitrary text through AlsaGeist speaker using local Piper neural TTS.

Synthesizes the given text using the local Alan Piper ONNX model in RAM
and plays it directly through the hardware audio output.

Usage:
    ./scripts/say.py "Happy Halloween, mortals!"
    ./scripts/say.py "Halt, who goes there?" --volume 0.25
    ./scripts/say.py --help
"""

import argparse
import os
import sys

from core.audio_io import AudioIO
from services.tts_service import PiperTTSService

VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Speak text through AlsaGeist using local Piper neural TTS.",
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text string for Grimjaw to say.",
    )
    parser.add_argument(
        "--volume",
        "-V",
        type=float,
        default=None,
        help="Playback volume from 0.0 to 1.0 (default: $VOLUME or 0.2).",
    )
    parser.add_argument(
        "--tempo",
        "-t",
        type=float,
        default=1.15,
        help="Speech length scale (higher = slower speech, default: 1.15).",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    return parser.parse_args()


def load_env_file(path: str = ".env") -> None:
    """Load environment variables from .env if present."""
    candidates = [
        path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path),
    ]
    target_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not target_path:
        return

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


def main() -> int:
    """Synthesize and play speech from the command line.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    load_env_file()
    args = parse_args()
    if not args.text.strip():
        print("Error: Spoken text cannot be empty.", file=sys.stderr)
        return 1

    try:
        # Resolve absolute path to Piper ONNX model
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(repo_root, "models", "piper", "en_GB-alan-medium.onnx")

        # Resolve volume from CLI flag, .env, or fallback
        vol = args.volume if args.volume is not None else float(os.environ.get("VOLUME", "0.2"))

        tts = PiperTTSService(model_path=model_path, length_scale=args.tempo)
        audio = AudioIO(volume=vol)

        print(f"💀 Grimjaw speaking (volume={vol}): \"{args.text}\"")
        samples, sr = tts.synthesize(args.text)
        audio.play_audio(samples, sample_rate=sr, blocking=True)
        return 0
    except Exception as exc:
        print(f"Failed to speak audio: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
