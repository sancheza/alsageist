#!/usr/bin/env python3
"""Speak arbitrary text through AlsaGeist speaker using local Piper neural TTS.

Synthesizes the given text using the local Alan Piper ONNX model in RAM
and plays it directly through the hardware audio output.

Usage:
    ./scripts/say.py "Happy Halloween, mortals!"
    ./scripts/say.py "Halt, who goes there?" --volume 0.25
    ./scripts/say.py "Halt, who goes there?" --debug
    ./scripts/say.py --help
"""

import time

_PROCESS_START = time.perf_counter()  # taken before the heavy imports below so they can be timed

import argparse
from contextlib import contextmanager
import os
import sys
from typing import Iterator, List, Tuple

from core.audio_io import AudioIO
from core.env_utils import env_float
from core.intent_builder import load_personality
from core import voice_config
from core.system_utils import load_env_file
from services.tts_service import PiperTTSService

VERSION = "1.1.0"


class PhaseTimer:
    """Records how long each phase of a run takes and prints a timing table."""

    def __init__(self) -> None:
        """Start the total clock at process start."""
        self._phases: List[Tuple[str, float, str]] = []

    @contextmanager
    def phase(self, name: str, note: str = "") -> Iterator[None]:
        """Time the enclosed block.

        Args:
            name: Phase label shown in the table.
            note: Optional detail shown after the duration.

        Yields:
            None.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._phases.append((name, time.perf_counter() - start, note))

    def add(self, name: str, seconds: float, note: str = "") -> None:
        """Record a phase that was measured elsewhere."""
        self._phases.append((name, seconds, note))

    def render(self) -> str:
        """Return the timing table, with each phase's share of the total.

        Returns:
            Multi-line table text.
        """
        total = sum(seconds for _, seconds, _ in self._phases)
        width = max([len(name) for name, _, _ in self._phases] + [len("total")])
        lines = ["Timing (--debug):"]
        for name, seconds, note in self._phases:
            share = (seconds / total * 100.0) if total > 0 else 0.0
            suffix = f"  {note}" if note else ""
            lines.append(f"  {name:<{width}}  {seconds:7.2f} s  {share:5.1f}%{suffix}")
        lines.append(f"  {'total':<{width}}  {total:7.2f} s")
        return "\n".join(lines)


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
        "--debug",
        action="store_true",
        help="Print how long each phase took (imports, model load, synthesis, playback).",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Simulate speech synthesis and playback without modifying audio hardware.",
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
    timer = PhaseTimer()
    timer.add("python imports", time.perf_counter() - _PROCESS_START, "numpy, sounddevice, project modules")

    load_env_file()
    args = parse_args()
    if not args.text.strip():
        print("Error: Spoken text cannot be empty.", file=sys.stderr)
        return 1

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with timer.phase("load personality"):
            try:
                profile = load_personality(args.personality)
            except Exception as exc:
                print(f"Warning: Could not load personality '{args.personality}': {exc}", file=sys.stderr)
                profile = None

        display_name = profile.display_name if profile else args.personality
        with timer.phase("resolve voice"):
            model_path, tempo, noise = voice_config.resolve_agent_voice_config(
                profile=profile,
                model_path=args.model_path or os.environ.get("MODEL_PATH"),
                tempo=args.tempo,
                repo_root=repo_root,
            )
            vol = args.volume if args.volume is not None else env_float("VOLUME", 0.5)

        if args.dry_run:
            print(f"[DRY-RUN] {display_name} speaking (volume={vol}): \"{args.text}\"")
            print(f"  Voice Model: {model_path} (tempo={tempo}, noise={noise})")
            return 0

        with timer.phase("init audio"):
            audio = AudioIO(volume=vol)
        # The voice model (tens of seconds on a Raspberry Pi) loads only if the phrase is not cached
        tts = PiperTTSService(model_path=model_path, length_scale=tempo, noise_scale=noise, lazy_load=True)

        print(f"{display_name} speaking (volume={vol}): \"{args.text}\"")
        cached = os.path.isfile(tts.cache_path(args.text))
        started = time.perf_counter()
        try:
            samples, sr = tts.synthesize(args.text)
        finally:
            elapsed = time.perf_counter() - started
            load_sec = tts.voice_load_sec
            if tts.voice_loaded:
                timer.add("load voice model", load_sec, os.path.basename(model_path))
            else:
                timer.add("load voice model", 0.0, "skipped (phrase was cached)")
            timer.add("synthesize", max(elapsed - load_sec, 0.0), "disk cache hit" if cached else "synthesized (cache miss)")
        with timer.phase("playback", f"{len(samples) / sr:.2f} s of audio"):
            audio.play_audio(samples, sample_rate=sr, blocking=True)
        return 0
    except Exception as exc:
        print(f"Failed to speak audio: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.debug:
            print(timer.render())


if __name__ == "__main__":
    sys.exit(main())
