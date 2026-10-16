"""core/config.py - Command-line configuration and argument parsing for AlsaGeist.

Defines all CLI flags, environment variable fallbacks, and parameter bounds
for the autonomous conversational display engine.
"""

import argparse
import os
from typing import List, Optional

VERSION = "1.0.0"


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for AlsaGeist.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="AlsaGeist: Autonomous Voice-Driven Conversational Display Engine."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed diagnostic logging.",
    )
    env_llm_fallback = os.environ.get("LLM_FALLBACK", "false").strip().lower() in ("true", "1", "yes")
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        default=env_llm_fallback,
        help="Fetch custom LLM response for indeterminate prompts instead of generic canned response (default: $LLM_FALLBACK or false).",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_false",
        dest="llm_fallback",
        help="Disable LLM fallback for indeterminate prompts.",
    )
    parser.add_argument(
        "--engine",
        choices=["intent", "llm"],
        default="intent",
        help="Dialogue engine: 'intent' (fast offline banter engine) or 'llm' (neural LLM). (default: intent)",
    )
    parser.add_argument(
        "--personality",
        default=os.environ.get("PERSONALITY", "grimjaw"),
        help="Personality profile identifier or JSON path (default: $PERSONALITY or grimjaw).",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Path to Piper ONNX model file (default: $MODEL_PATH or resolved from personality profile).",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Optional input audio hardware device name or index.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Optional output audio hardware device name or index.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("VAD_THRESHOLD", 0.0025)),
        help="Minimum audio RMS energy threshold for speech detection (default: 0.0025 or $VAD_THRESHOLD). Increase if false-triggering in loud rooms.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=float(os.environ.get("VOLUME", 0.5)),
        help="Playback audio output volume multiplier between 0.0 and 1.0 (default: 0.5 or $VOLUME).",
    )
    parser.add_argument(
        "--mic-gain",
        type=float,
        default=float(os.environ.get("MIC_GAIN", 1.5)),
        help="Microphone digital input gain multiplier (default: 1.5 or $MIC_GAIN).",
    )
    parser.add_argument(
        "--min-speech-sec",
        type=float,
        default=0.15,
        help="Minimum sustained speech duration in seconds required to accept an utterance (default: 0.15s).",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=0.9,
        help="Trailing silence in seconds required before speech is considered finished (default: 0.9s).",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=float(os.environ["TEMPO"]) if "TEMPO" in os.environ else None,
        help="Piper phoneme length scale / tempo (default: from personality profile or 1.15).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=float(os.environ["NOISE"]) if "NOISE" in os.environ else None,
        help="Piper generator noise level (default: from personality profile or 0.85).",
    )
    parser.add_argument(
        "--led-pin",
        type=int,
        default=int(os.environ.get("LED_PIN", 25)),
        help="Raspberry Pi GPIO pin number for status LED (default: 25 or $LED_PIN).",
    )
    parser.add_argument(
        "--button-pin",
        type=int,
        default=int(os.environ.get("BUTTON_PIN", 23)),
        help="Raspberry Pi GPIO pin number for push button (default: 23 or $BUTTON_PIN).",
    )
    parser.add_argument(
        "--button-hold-sec",
        type=float,
        default=float(os.environ.get("BUTTON_HOLD_SEC", 5.0)),
        help="Push button hold duration in seconds to trigger sleep mode (default: 5.0s or $BUTTON_HOLD_SEC).",
    )
    parser.add_argument(
        "--button-shutdown-sec",
        type=float,
        default=float(os.environ.get("BUTTON_SHUTDOWN_SEC", 10.0)),
        help="Push button long-hold duration in seconds to trigger system shutdown (default: 10.0s or $BUTTON_SHUTDOWN_SEC).",
    )
    parser.add_argument(
        "--no-greeting",
        action="store_true",
        help="Disable spoken greeting announcement upon startup.",
    )
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = build_arg_parser()
    return parser.parse_args(args)
