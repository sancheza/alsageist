#!/usr/bin/env python3
"""AlsaGeist - Autonomous Voice-Driven Conversational Display Engine.

Autonomous voice-driven animatronic and display engine running an in-memory dialogue pipeline
with pluggable personality profiles, low-latency VAD, and local Piper neural TTS.

Usage:
    ./main.py
    ./main.py --personality snowman
    ./main.py --input-device "Microphone" --output-device "Speaker"
"""

import signal
import sys

from core.config import parse_args
from core.system_utils import load_env_file
from services.agent import AlsaGeistAgent


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    load_env_file()
    args = parse_args()

    agent = AlsaGeistAgent(args)

    def signal_handler(signum, frame):
        agent.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    agent.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
