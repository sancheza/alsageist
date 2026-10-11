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
        default=float(os.environ.get("VOLUME", "0.2")),
        help="Playback volume from 0.0 to 1.0 (default: 0.2 or $VOLUME).",
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


def main() -> int:
    """Synthesize and play speech from the command line.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    args = parse_args()
    if not args.text.strip():
        print("Error: Spoken text cannot be empty.", file=sys.stderr)
        return 1

    try:
        tts = PiperTTSService(length_scale=args.tempo)
        audio = AudioIO(volume=args.volume)

        print(f"💀 Grimjaw speaking (volume={args.volume}): \"{args.text}\"")
        samples, sr = tts.synthesize(args.text)
        audio.play_audio(samples, sample_rate=sr, blocking=True)
        return 0
    except Exception as exc:
        print(f"Failed to speak audio: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
