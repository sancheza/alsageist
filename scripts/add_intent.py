#!/usr/bin/env python3
"""scripts/add_intent.py - Construct, test, and attach custom dialogue intents.

Purpose:
    Constructs, tests, searches, and attaches custom dialogue intents to voice assistant
    personalities. While AlsaGeist can answer ANY prompt dynamically using an LLM,
    these intents are designed for LOCAL, IMMEDIATE responses: they execute with sub-second
    latency, require zero cloud network round-trips, function fully offline, and leverage
    pre-warmed neural audio caches.

Target Data Files Modified:
    1. Personality Profile JSON:
       data/personalities/<personality>.json (e.g., data/personalities/grimjaw.json)
       Stores intent identifiers, regex trigger patterns, responses, and fallbacks.
    2. Neural TTS Audio Cache:
       models/cache/<hash>.npy
       Pre-synthesizes NumPy audio arrays when --prewarm is enabled to eliminate
       runtime voice latency.

Usage:
    # Check if a phrase matches an existing intent:
    ./scripts/add_intent.py --check "What is your favorite movie?"

    # Search existing intents by keyword:
    ./scripts/add_intent.py --search "movie"

    # List all intents registered in a personality:
    ./scripts/add_intent.py -p grimjaw --list

    # Add a new intent with pre-warmed audio cache:
    ./scripts/add_intent.py -q "What is your favorite board game?" \\
      -r "Chess! I play with skulls instead of pawns." --prewarm

    # Append response to an existing intent on collision:
    ./scripts/add_intent.py -q "What's your favorite movie?" \\
      -r "Beetlejuice, obviously!" --append --prewarm

    # Interactive wizard:
    ./scripts/add_intent.py --interactive
    ./scripts/add_intent.py --help
"""

import argparse
import os
import sys

from core.intent_builder import (
    CustomIntent,
    IntentCollisionError,
    PersonalityProfile,
    create_custom_intent,
    find_personality_file,
)
from core.system_utils import load_env_file

VERSION = "1.1.0"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="""\
================================================================================
AlsaGeist Custom Intent Builder & Personality Manager
================================================================================
PURPOSE & ARCHITECTURE:
  AlsaGeist can answer ANY prompt dynamically using an LLM fallback. However,
  custom intents are designed specifically for LOCAL, IMMEDIATE responses.
  They bypass cloud network latency, work completely offline, and match user
  queries instantly with pre-warmed neural audio.

  This utility constructs, tests, searches, and attaches custom dialogue intents
  to personality profiles. It synthesizes robust regex trigger patterns (via Gemini
  LLM expansion or deterministic fallback), detects phrase collisions against
  existing rules, persists intents to external JSON, and pre-warms local TTS caches.

TARGET DATA FILES MODIFIED:
  1. Personality Profile JSON:
     data/personalities/<personality>.json  (e.g., data/personalities/grimjaw.json)
     Persists intent identifiers, regex trigger patterns, and dialogue response pools.
  2. Neural TTS Audio Cache:
     models/cache/<hash>.npy
     Pre-synthesizes NumPy audio arrays when --prewarm is enabled so responses
     play with zero synthesis latency at runtime.
""",
        epilog="""\
COMMON USAGE EXAMPLES:

  1. Interactive Wizard Mode (guided step-by-step prompts):
     ./scripts/add_intent.py --interactive
     ./scripts/add_intent.py

  2. Inspect Spoken Phrases (test if a query already matches an intent):
     ./scripts/add_intent.py --check "What is your favorite movie?"
     ./scripts/add_intent.py --check "Can you swim?"

  3. Search & List Existing Intents:
     ./scripts/add_intent.py --search "movie"
     ./scripts/add_intent.py -p grimjaw --list

  4. Add New Intent with Automatic Pre-Warming:
     ./scripts/add_intent.py -q "What is your favorite board game?" \\
       -r "Chess! I play with skulls instead of pawns." \\
       --prewarm

  5. Add Multiple Response Variants to an Intent:
     ./scripts/add_intent.py -q "Do you like candy corn?" \\
       -r "Candy corn? Wax pretending to be sugar!" \\
       -r "Only if it is two hundred years old!" \\
       --prewarm

  6. Append Responses to an Existing Intent (adds dialogue variety):
     ./scripts/add_intent.py -q "What is your favorite movie?" \\
       -r "Beetlejuice, obviously!" \\
       --append --prewarm

  7. Preview Intent Construction without Writing Files (Dry Run):
     ./scripts/add_intent.py -q "Can you fly?" \\
       -r "Only when catapulted!" \\
       --dry-run

  8. Target or Create an Alternative Personality Profile:
     ./scripts/add_intent.py -p snowman -q "Are you cold?" \\
       -r "I am literally made of snow!" \\
       --prewarm

  9. Offline Mode (bypass LLM expansion, use deterministic regex builder):
     ./scripts/add_intent.py -q "Where is your hat?" \\
       -r "I have no ears to hold it on!" \\
       --no-llm
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Sample trigger phrase, user question, or intent description.",
    )
    parser.add_argument(
        "-r",
        "--response",
        type=str,
        action="append",
        dest="responses",
        help="Spoken response for the intent (can be specified multiple times).",
    )
    parser.add_argument(
        "-p",
        "--personality",
        type=str,
        default=os.environ.get("PERSONALITY", "grimjaw"),
        help="Target personality profile name or file path (default: $PERSONALITY or grimjaw).",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Explicit snake_case intent name (optional, derived automatically if omitted).",
    )
    parser.add_argument(
        "-c",
        "--check",
        type=str,
        help="Test if a spoken phrase already matches an existing intent in the personality.",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        help="Search existing intents by name, pattern, or response text.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all registered intents in the personality profile.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new response(s) to an existing intent if query collision occurs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing intent if query collision occurs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force creation of a new high-priority intent even if query matches an existing intent.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM expansion and use deterministic offline rule-based regex generator.",
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Immediately synthesize audio responses into the local Piper TTS cache (.npy).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive prompt wizard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated intent name, regex patterns, and responses without saving.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    return parser.parse_args()


def handle_check(personality_name: str, phrase: str) -> int:
    """Test if a phrase triggers an existing intent."""
    profile_path = find_personality_file(personality_name)
    if not os.path.isfile(profile_path):
        print(f"Error: Personality profile '{personality_name}' not found at {profile_path}.", file=sys.stderr)
        return 1

    profile = PersonalityProfile.load_from_file(profile_path)
    print(f"\nChecking phrase against personality: '{profile.display_name}' ({profile.personality})...")
    print(f"   Query: \"{phrase}\"")

    collision = profile.check_query(phrase)
    if collision:
        intent, pattern = collision
        print("\n✅ MATCH FOUND:")
        print(f"   Intent Name: '{intent.name}'")
        print(f"   Matched Regex: {pattern}")
        print("   Current Responses:")
        for r in intent.responses:
            print(f"     \"{r}\"")
        print()
        return 0
    else:
        print("\n❌ NO MATCH: No existing intent in this profile triggers on this phrase.\n")
        return 0


def handle_search(personality_name: str, keyword: str) -> int:
    """Search intents in a personality by keyword."""
    profile_path = find_personality_file(personality_name)
    if not os.path.isfile(profile_path):
        print(f"Error: Personality profile '{personality_name}' not found at {profile_path}.", file=sys.stderr)
        return 1

    profile = PersonalityProfile.load_from_file(profile_path)
    results = profile.search_intents(keyword)
    print(f"\nSearch results for '{keyword}' in '{profile.personality}' ({len(results)} found):")
    for idx, intent in enumerate(results, start=1):
        sample_resp = intent.responses[0] if intent.responses else "No responses"
        print(f"  {idx}. [{intent.name}] (Patterns: {len(intent.patterns)})")
        print(f"     Sample Response: \"{sample_resp}\"")
    print()
    return 0


def handle_list(personality_name: str) -> int:
    """List all registered intents in a personality."""
    profile_path = find_personality_file(personality_name)
    if not os.path.isfile(profile_path):
        print(f"Error: Personality profile '{personality_name}' not found at {profile_path}.", file=sys.stderr)
        return 1

    profile = PersonalityProfile.load_from_file(profile_path)
    print(f"\nRegistered Intents in '{profile.display_name}' ({profile.personality}): {len(profile.intents)} total")
    print("-" * 70)
    for idx, intent in enumerate(profile.intents, start=1):
        sample = intent.responses[0] if intent.responses else ""
        if len(sample) > 50:
            sample = sample[:47] + "..."
        print(f"{idx:3d}. {intent.name:<32} ({len(intent.patterns):2d} pats, {len(intent.responses):2d} resps) -> \"{sample}\"")
    print("-" * 70 + "\n")
    return 0


def run_interactive_wizard(default_personality: str) -> dict:
    """Prompt the user interactively for intent details with collision detection.

    Args:
        default_personality: Default personality identifier.

    Returns:
        Dictionary of parameters for intent creation, or empty dict if cancelled.
    """
    print("\n" + "=" * 60)
    print(" CUSTOM INTENT BUILDER WIZARD")
    print("=" * 60)

    p_input = input(f"Target Personality [{default_personality}]: ").strip()
    personality = p_input if p_input else default_personality

    profile_path = find_personality_file(personality)
    profile = PersonalityProfile.load_from_file(profile_path) if os.path.isfile(profile_path) else None

    query = ""
    while not query:
        query = input("\nSample Query / Trigger phrase (e.g. 'What is your favorite book?'): ").strip()
        if not query:
            print("Query cannot be empty. Please enter a phrase.")

    collision_action = "create"
    intent_name = None

    # Check for query collision
    if profile:
        collision = profile.check_query(query)
        if collision:
            existing_intent, pattern = collision
            print("\n" + "!" * 60)
            print("QUERY COLLISION DETECTED:")
            print(f"   This phrase already matches existing intent: '{existing_intent.name}'")
            print(f"   Matched regex: {pattern}")
            print(f"   Current response(s):")
            for r in existing_intent.responses:
                print(f"     \"{r}\"")
            print("!" * 60)
            print("\nHow would you like to handle this?")
            print(f"  [1] Append new response(s) to '{existing_intent.name}' (Recommended for adding variety)")
            print(f"  [2] Overwrite '{existing_intent.name}' with a new definition")
            print(f"  [3] Force create as a separate high-priority intent")
            print(f"  [4] Cancel / Abort")

            choice = input("\nEnter choice [1-4, default=1]: ").strip()
            if choice in ("4", "cancel", "c"):
                print("Aborted.")
                return {}
            elif choice == "2":
                collision_action = "overwrite"
                intent_name = existing_intent.name
            elif choice == "3":
                collision_action = "force"
            else:
                collision_action = "append"
                intent_name = existing_intent.name

    if collision_action != "append":
        name_input = input("\nIntent identifier (optional snake_case, press Enter to auto-generate): ").strip()
        if name_input:
            intent_name = name_input

    responses = []
    action_desc = "new response(s) to append" if collision_action == "append" else "candidate responses"
    print(f"\nEnter {action_desc} (press Enter with empty line when finished):")
    idx = 1
    while True:
        resp = input(f"  Response {idx}: ").strip()
        if not resp:
            break
        responses.append(resp)
        idx += 1

    prewarm_input = input("\nPre-warm TTS audio cache now? [Y/n]: ").strip().lower()
    prewarm = prewarm_input not in ("n", "no")

    return {
        "query": query,
        "responses": responses,
        "personality": personality,
        "name": intent_name,
        "prewarm": prewarm,
        "collision_action": collision_action,
    }


def main() -> int:
    """Main execution entry point.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    load_env_file()
    args = parse_args()

    # 1. Handle inspection / search commands
    if args.check:
        return handle_check(args.personality, args.check.strip())
    if args.search:
        return handle_search(args.personality, args.search.strip())
    if args.list:
        return handle_list(args.personality)

    # 2. Determine interactive vs flag mode
    interactive_mode = args.interactive or (not args.query and not args.responses)

    if interactive_mode:
        wizard_data = run_interactive_wizard(default_personality=args.personality)
        if not wizard_data:
            return 0
        query = wizard_data["query"]
        responses = wizard_data["responses"]
        personality = wizard_data["personality"]
        intent_name = wizard_data["name"]
        prewarm = wizard_data["prewarm"]
        collision_action = wizard_data.get("collision_action", "create")
        dry_run = args.dry_run
        use_llm = not args.no_llm
    else:
        query = args.query.strip()
        responses = args.responses or []
        personality = args.personality
        intent_name = args.name
        prewarm = args.prewarm
        dry_run = args.dry_run
        use_llm = not args.no_llm

        if args.append:
            collision_action = "append"
        elif args.overwrite:
            collision_action = "overwrite"
        elif args.force:
            collision_action = "force"
        else:
            collision_action = "raise"

    print(f"\nProcessing intent for personality: '{personality}'...")

    try:
        intent, profile = create_custom_intent(
            query_or_text=query,
            responses=responses,
            personality_name_or_path=personality,
            name=intent_name,
            prewarm=prewarm and not dry_run,
            dry_run=dry_run,
            use_llm=use_llm,
            check_collision=True,
            collision_action=collision_action,
        )

        target_path = find_personality_file(personality)
        print("\n" + "-" * 60)
        if collision_action == "append":
            print(f"✅ Appended Response(s) to Existing Intent: '{intent.name}'")
        else:
            print(f"✅ Intent Constructed: '{intent.name}'")
        print("-" * 60)
        print("Match Patterns:")
        for pat in intent.patterns:
            print(f"  • {pat}")
        print("\nCurrent Responses:")
        for resp in intent.responses:
            print(f"  \"{resp}\"")

        if dry_run:
            print(f"\n[DRY RUN] Would save to '{target_path}' (no changes made).")
        else:
            print(f"\nSaved to personality file: '{target_path}'")
            print(f"Total intents in profile '{profile.personality}': {len(profile.intents)}")
            if prewarm and responses:
                print("Pre-warming audio cache complete.")
        print("-" * 60 + "\n")
        return 0

    except IntentCollisionError as col_err:
        print("\n" + "!" * 65, file=sys.stderr)
        print(f"INTENT COLLISION DETECTED for query: \"{query}\"", file=sys.stderr)
        print(f"   Matches existing intent: '{col_err.existing_intent.name}'", file=sys.stderr)
        print(f"   Via pattern: {col_err.matched_pattern}", file=sys.stderr)
        print("   Existing responses:", file=sys.stderr)
        for r in col_err.existing_intent.responses:
            print(f"     \"{r}\"", file=sys.stderr)
        print("!" * 65, file=sys.stderr)
        print("\nTo resolve this collision, re-run with one of these flags:", file=sys.stderr)
        print(f"  --append     Append your new response(s) to '{col_err.existing_intent.name}'", file=sys.stderr)
        print(f"  --overwrite  Overwrite '{col_err.existing_intent.name}' with new patterns/responses", file=sys.stderr)
        print(f"  --force      Force create a separate high-priority intent", file=sys.stderr)
        print(f"Or use --interactive to choose interactively.\n", file=sys.stderr)
        return 1

    except Exception as exc:
        print(f"\nError creating custom intent: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
