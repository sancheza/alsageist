#!/usr/bin/env python3
"""Pre-warm and generate audio response cache for personality dialogue.

Synthesizes character dialogue lines in advance and stores them into the
models/cache/ directory as float32 NumPy arrays. When the AlsaGeist runtime
starts or encounters these lines, they are loaded with zero latency (0ms).

Automatically adapts to the loaded personality profile, its configured voice model,
priority intents, and system announcements.

Usage:
    ./scripts/prewarm_cache.py
    ./scripts/prewarm_cache.py --personality grimjaw --limit 30
    ./scripts/prewarm_cache.py --personality grimjaw --all
    ./scripts/prewarm_cache.py --dry-run
"""

import argparse
import os
import sys
import time
from typing import List

from core.system_utils import resolve_piper_model_path
from services.intent_service import PersonaIntentEngine
from services.tts_service import PiperTTSService

VERSION = "1.1.0"


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
        "-p",
        "--personality",
        default=os.environ.get("PERSONALITY", "grimjaw"),
        help="Personality profile identifier or path (default: $PERSONALITY or grimjaw).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Piper ONNX model file. If omitted, resolved from personality voice configuration.",
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
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Preview phrases to be synthesized without running neural TTS synthesis.",
    )
    return parser.parse_args()


def collect_phrases_to_prewarm(engine: PersonaIntentEngine, warm_all: bool, limit: int) -> List[str]:
    """Collect dialogue phrases to prewarm based on personality profile configuration.

    Args:
        engine: Initialized dialogue intent engine with loaded profile.
        warm_all: If True, include all intent responses.
        limit: Maximum number of phrases when warm_all is False.

    Returns:
        Deduplicated list of phrases to pre-warm.
    """
    profile = getattr(engine, "profile", None)
    phrases: List[str] = []

    # 1. Startup greeting
    if profile and getattr(profile, "startup_greeting", None):
        phrases.append(profile.startup_greeting)
    elif profile and getattr(profile, "display_name", None):
        phrases.append(f"I am {profile.display_name}. System online.")

    # 2. System lifecycle & mode announcements
    announcements = dict(getattr(profile, "system_announcements", {})) if profile else {}
    phrases.append(announcements.get("offline_mode", "NOW IN OFFLINE MODE"))
    phrases.append(announcements.get("online_mode", "NOW IN ONLINE MODE"))
    phrases.append(announcements.get("powering_off", "Powering off."))
    phrases.append(announcements.get("powering_down", "Powering down."))

    # 2b. Spoken low-power lines
    phrases.extend(getattr(profile, "low_power_lines", []) if profile else [])

    # 3. Volume step announcements
    volume_template = announcements.get("volume_template", "Volume {percent} percent.")
    for p in range(0, 101, 10):
        phrases.append(volume_template.format(percent=p))

    # 4. Fallbacks, escalations & costume inquiry prompts from personality profile
    phrases.extend(getattr(engine, "_fallbacks", []))
    phrases.extend(getattr(engine, "_escalation_lines", []))
    phrases.extend(getattr(engine, "_costume_prompts", []))

    # 5. Intent responses
    if warm_all:
        for _, _, responses in engine._intent_rules:
            phrases.extend(responses)
    else:
        priority_intents = list(getattr(profile, "priority_intents", [])) if profile else []
        if priority_intents:
            for intent_name, _, responses in engine._intent_rules:
                if intent_name in priority_intents:
                    phrases.extend(responses)
        else:
            for _, _, responses in engine._intent_rules:
                phrases.extend(responses)
                if len(phrases) >= limit:
                    break

    # Deduplicate preserving order
    seen = set()
    unique_phrases: List[str] = []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            unique_phrases.append(p)

    if not warm_all:
        unique_phrases = unique_phrases[:limit]

    return unique_phrases


def main() -> int:
    """Main execution function."""
    args = parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Loading personality profile '{args.personality}'...")
    engine = PersonaIntentEngine(personality_name_or_path=args.personality)
    profile = getattr(engine, "profile", None)

    # Resolve Piper model path from arguments or personality voice configuration
    voice_name = getattr(profile, "voice", "en_GB-alan-medium") if profile else "en_GB-alan-medium"
    model_path = resolve_piper_model_path(
        model_path=args.model_path,
        voice_name=voice_name,
        repo_root=repo_root,
    )

    unique_phrases = collect_phrases_to_prewarm(engine, warm_all=args.all, limit=args.limit)

    if args.dry_run:
        print("\n" + "=" * 60)
        print(f" DRY RUN: Pre-warm Cache Preview ({len(unique_phrases)} phrases)")
        print(f" Personality: {getattr(profile, 'display_name', args.personality)}")
        print(f" Voice Model: {model_path}")
        print("=" * 60)
        for idx, phrase in enumerate(unique_phrases, start=1):
            print(f"[{idx:02d}/{len(unique_phrases):02d}] \"{phrase}\"")
        print("=" * 60)
        print("Dry-run complete. No audio was synthesized.")
        return 0

    if not os.path.isfile(model_path):
        voice_id = getattr(profile, "voice", "en_GB-alan-medium") if profile else "en_GB-alan-medium"
        print(
            f"Error: Piper voice model not found at '{model_path}'.\n"
            f"Download the model using:\n"
            f"  python3 scripts/download_model.py --voice {voice_id}",
            file=sys.stderr,
        )
        return 1

    print(f"Initializing Piper TTS synthesizer ({model_path})...")
    tts = PiperTTSService(model_path=model_path)

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
    print(f"  Personality:         {getattr(profile, 'display_name', args.personality)}")
    print(f"  Total checked:       {len(unique_phrases)}")
    print(f"  Already cached:      {cached_count}")
    print(f"  Newly synthesized:   {newly_synthesized}")
    print(f"  Elapsed time:        {total_time:.2f}s")
    print(f"  Cache directory:     {tts.cache_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
