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
from core.intent_builder import PersonalityProfile, find_personality_file
from core.system_utils import load_env_file, resolve_piper_model_path
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
        help="Text string for creature to say.",
    )
    parser.add_argument(
        "--personality",
        "-p",
        type=str,
        default=os.environ.get("PERSONALITY", "grimjaw"),
        help="Personality profile identifier or JSON path (default: $PERSONALITY or grimjaw).",
    )
    parser.add_argument(
        "--volume",
        "-V",
        type=float,
        default=None,
        help="Playback volume from 0.0 to 1.0 (default: $VOLUME or 0.5).",
    )
    parser.add_argument(
        "--tempo",
        "-t",
        type=float,
        default=None,
        help="Speech length scale (higher = slower speech, default: from personality or 1.15).",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=None,
        help="Path to Piper ONNX model file (default: $MODEL_PATH or resolved from personality profile).",
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
    load_env_file()
    args = parse_args()
    if not args.text.strip():
        print("Error: Spoken text cannot be empty.", file=sys.stderr)
        return 1

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile = None
        try:
            p_file = find_personality_file(args.personality)
            if os.path.isfile(p_file):
                profile = PersonalityProfile.load_from_file(p_file)
        except Exception:
            pass

        display_name = profile.display_name if profile else "Grimjaw"
        voice_id = profile.voice if profile else None
        model_path = resolve_piper_model_path(
            model_path=args.model_path or os.environ.get("MODEL_PATH"),
            voice_name=voice_id,
            repo_root=repo_root,
        )

        tempo = args.tempo
        if tempo is None:
            tempo = (
                profile.voice_settings.get("tempo", 1.15)
                if profile and profile.voice_settings
                else 1.15
            )

        vol = args.volume if args.volume is not None else float(os.environ.get("VOLUME", "0.5"))

        tts = PiperTTSService(model_path=model_path, length_scale=tempo)
        audio = AudioIO(volume=vol)

        print(f"💀 {display_name} speaking (volume={vol}): \"{args.text}\"")
        samples, sr = tts.synthesize(args.text)
        audio.play_audio(samples, sample_rate=sr, blocking=True)
        return 0
    except Exception as exc:
        print(f"Failed to speak audio: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
