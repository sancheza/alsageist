#!/usr/bin/env python3
"""Play and compare all generated voice mockups sequentially on macOS.

Usage:
    ./scripts/play_mockups.py
    ./scripts/play_mockups.py --persona skeleton
    ./scripts/play_mockups.py --persona witch
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

VERSION = "0.1.0"

MOCKUP_PLAYLIST: List[Dict[str, str]] = [
    # Skeleton Section
    {
        "persona": "Skeleton",
        "name": "Edge-TTS: Christopher (-25Hz Pitch, -10% Rate)",
        "file": "samples/mockups/edge_skeleton_1_christopher.mp3",
        "notes": "Deep, resonant US pitch-shifted gravel (Free / Unlimited)",
    },
    {
        "persona": "Skeleton",
        "name": "Edge-TTS: Ryan (-30Hz Pitch, -12% Rate)",
        "file": "samples/mockups/edge_skeleton_2_ryan.mp3",
        "notes": "Dark British crypt presence (Free / Unlimited)",
    },
    {
        "persona": "Skeleton",
        "name": "Edge-TTS: Connor (-20Hz Pitch, -15% Rate)",
        "file": "samples/mockups/edge_skeleton_3_connor.mp3",
        "notes": "Low, slow, grave Irish resonance (Free / Unlimited)",
    },
    {
        "persona": "Skeleton",
        "name": "Piper TTS: Alan (Local ONNX)",
        "file": "samples/mockups/piper_skeleton_alan.wav",
        "notes": "100% Offline local neural model on device",
    },
    {
        "persona": "Skeleton",
        "name": "Gemini Reference: Charon",
        "file": "samples/mockups/gemini_skeleton_charon.wav",
        "notes": "Cloud generative deep tomb vocal fry",
    },
    # Witch Section
    {
        "persona": "Wicked Witch",
        "name": "Edge-TTS: Ana (+45Hz Pitch, +12% Rate)",
        "file": "samples/mockups/edge_witch_1_ana.mp3",
        "notes": "Shrill, theatrical Wizard of Oz cackle (Free / Unlimited)",
    },
    {
        "persona": "Wicked Witch",
        "name": "Edge-TTS: Sonia (+40Hz Pitch, +15% Rate)",
        "file": "samples/mockups/edge_witch_2_sonia.mp3",
        "notes": "Theatrical British swamp hag (Free / Unlimited)",
    },
    {
        "persona": "Wicked Witch",
        "name": "Edge-TTS: Ava (+35Hz Pitch, +10% Rate)",
        "file": "samples/mockups/edge_witch_3_ava.mp3",
        "notes": "Fast, energetic, sharp cadence (Free / Unlimited)",
    },
    {
        "persona": "Wicked Witch",
        "name": "Piper TTS: Lessac (Local ONNX)",
        "file": "samples/mockups/piper_witch_lessac.wav",
        "notes": "100% Offline local neural model on device",
    },
    {
        "persona": "Wicked Witch",
        "name": "Gemini Reference: Aoede",
        "file": "samples/mockups/gemini_witch_aoede.wav",
        "notes": "Cloud generative high-pitch cackle",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Play and compare all generated voice mockups sequentially on macOS."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--persona",
        choices=["skeleton", "witch", "all"],
        default="all",
        help="Filter playlist by persona (default: all)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.5,
        help="Delay in seconds between audio tracks (default: 1.5)",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    items = []
    for item in MOCKUP_PLAYLIST:
        if args.persona == "all":
            items.append(item)
        elif args.persona == "skeleton" and item["persona"] == "Skeleton":
            items.append(item)
        elif args.persona == "witch" and item["persona"] == "Wicked Witch":
            items.append(item)

    print("=" * 75)
    print(f" ALSA-GEIST VOICE MOCKUPS PLAYLIST ({len(items)} Tracks)")
    print("=" * 75)

    for idx, item in enumerate(items, start=1):
        if not os.path.exists(item["file"]):
            continue

        print(f"\n[{idx}/{len(items)}] [{item['persona'].upper()}] {item['name']}")
        print(f"      Profile: {item['notes']}")
        print(f"      File:    {item['file']}")
        print("      Status:  ▶ Playing...", flush=True)

        subprocess.run(["afplay", item["file"]], check=True)

        if idx < len(items):
            time.sleep(args.delay)

    print("\n" + "=" * 75)
    print(" Playback complete.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
