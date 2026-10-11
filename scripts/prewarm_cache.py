#!/usr/bin/env python3
"""Pre-warm and generate audio response cache for Grimjaw dialogue.

Synthesizes character dialogue lines in advance and stores them into the
models/cache/ directory as float32 NumPy arrays. When the AlsaGeist runtime
starts or encounters these lines, they are loaded with zero latency (0ms).

Usage:
    ./scripts/prewarm_cache.py
    ./scripts/prewarm_cache.py --limit 30
    ./scripts/prewarm_cache.py --all
"""

import argparse
import os
import sys
import time

from services.intent_service import GrimjawIntentEngine
from services.tts_service import PiperTTSService

VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-warm Piper neural TTS audio cache for instant playback."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--model-path",
        default="models/piper/en_GB-alan-medium.onnx",
        help="Path to Piper ONNX model file (default: models/piper/en_GB-alan-medium.onnx).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of phrases to pre-warm (default: 25). Use --all to warm all lines.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pre-warm all response lines defined in the intent engine.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: Piper model not found at '{args.model_path}'.", file=sys.stderr)
        return 1

    print(f"Initializing Piper TTS synthesizer ({args.model_path})...")
    tts = PiperTTSService(model_path=args.model_path)
    engine = GrimjawIntentEngine()

    # Collect high-priority responses
    phrases = []

    # 1. Startup greeting, Volume announcements & Network mode announcements
    phrases.append("I am Grimjaw, sentinel of the crypt. Step forward if ye dare!")
    phrases.append("NOW IN OFFLINE MODE")
    phrases.append("NOW IN ONLINE MODE")
    for p in range(0, 101, 10):
        phrases.append(f"Volume {p} percent.")

    # 2. Fallbacks & Escalations
    phrases.extend(engine._fallbacks)
    phrases.extend(engine._escalation_lines)

    # 3. Intent responses
    if args.all:
        for _, _, responses in engine._intent_rules:
            phrases.extend(responses)
    else:
        # High-frequency intent responses (Candy rules, jokes, greetings, quotes)
        priority_intents = [
            "greeting",
            "candy_take_one",
            "candy_greedy_caught",
            "jokes_and_humor",
            "fear_scared",
            "scary_quote",
            "what_are_you",
            "who_are_you",
            "one_rib_quote",
        ]
        for intent_name, _, responses in engine._intent_rules:
            if intent_name in priority_intents:
                phrases.extend(responses)

    # Deduplicate while preserving order
    seen = set()
    unique_phrases = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)

    if not args.all:
        unique_phrases = unique_phrases[: args.limit]

    print(f"Pre-warming {len(unique_phrases)} phrases into '{tts.cache_dir}'...")

    cached_count = 0
    newly_synthesized = 0
    t0 = time.time()

    for idx, phrase in enumerate(unique_phrases, start=1):
        cache_path = tts._get_cache_path(phrase)
        if os.path.isfile(cache_path):
            cached_count += 1
            print(f"[{idx}/{len(unique_phrases)}] (Cached) \"{phrase[:50]}...\"")
            continue

        print(f"[{idx}/{len(unique_phrases)}] Synthesizing: \"{phrase[:50]}...\"")
        t_start = time.time()
        tts.synthesize(phrase)
        dur = time.time() - t_start
        print(f"   -> Finished in {dur:.2f}s")
        newly_synthesized += 1

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("Pre-warming Complete!")
    print(f"  Total checked:       {len(unique_phrases)}")
    print(f"  Already cached:      {cached_count}")
    print(f"  Newly synthesized:   {newly_synthesized}")
    print(f"  Elapsed time:        {total_time:.2f}s")
    print(f"  Cache directory:     {tts.cache_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
