#!/usr/bin/env python3
"""scripts/summarize_personality.py - Dynamically summarize Q&A supported by a personality profile.

Purpose:
    Reads any AlsaGeist personality profile JSON (e.g. data/personalities/grimjaw.json)
    and dynamically extracts, cleans, and categorizes all supported user questions
    and character answers into Markdown, plain text, or JSON formats.

Usage:
    # Summarize Grimjaw in Markdown format (stdout):
    ./scripts/summarize_personality.py -p grimjaw

    # Generate a Markdown file catalog:
    ./scripts/summarize_personality.py -p grimjaw -o docs/grimjaw_responses.md

    # Display in plain text for terminal reading:
    ./scripts/summarize_personality.py -p grimjaw --format text

    # Filter to costume recognition intents:
    ./scripts/summarize_personality.py -p grimjaw -c "Costumes"

    # Export structured JSON:
    ./scripts/summarize_personality.py -p grimjaw --format json
"""

import argparse
from collections import defaultdict
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from core.intent_builder import PersonalityProfile, find_personality_file

VERSION = "1.0.0"

QUESTION_STARTERS = (
    "what", "how", "why", "where", "when", "who", "which", "whose",
    "can", "could", "do", "does", "did", "are", "is", "will", "would",
    "am", "have", "has", "should"
)


def clean_regex_pattern(pat: str) -> str:
    """Transform a regular expression trigger pattern into a natural sample question.

    Args:
        pat: Raw regex pattern string from personality profile intent.

    Returns:
        Cleaned, human-readable example prompt or question.
    """
    s = pat

    # Strip word boundaries, start/end anchors
    s = re.sub(r"\\[bBAZz]", "", s)
    s = re.sub(r"^[\^]+|[\$]+$", "", s)

    # Replace escaped characters
    s = s.replace(r"\ ", " ").replace(r"\?", "?").replace(r"\!", "!").replace(r"\'", "'")
    s = s.replace(r"\-", "-").replace(r"\/", "/")

    # Canonicalize pronouns: (your|his) -> your, (you|him) -> you
    s = re.sub(r"\(\s*your\s*\|\s*his\s*\)", "your", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*you\s*\|\s*him\s*\)", "you", s, flags=re.IGNORECASE)

    # Iteratively resolve innermost regex groups (a|b|c) -> a
    max_loops = 6
    while "(" in s and ")" in s and max_loops > 0:
        max_loops -= 1

        def _pick_group(m: re.Match) -> str:
            content = m.group(1).strip()
            if not content:
                return " "
            parts = [p.strip() for p in content.split("|") if p.strip()]
            return f" {parts[0]} " if parts else " "

        s = re.sub(r"\(([^()]+)\)", _pick_group, s)

    # Strip character classes like [\s-] or [^a-z]
    s = re.sub(r"\[\s*\\s\s*-\s*\]", "-", s)
    s = re.sub(r"\[[^\]]+\]", " ", s)

    # Strip wildcards and regex spacing tokens
    s = re.sub(r"\.\*|\.\+|\\[sS][*+]|\\[dDwW][*+]", " ", s)

    # Strip single character quantifiers like s? -> s
    s = re.sub(r"(\w)\?", r"\1", s)

    # Strip remaining standalone regex operators
    s = re.sub(r"[?*+()|]", "", s)

    # Clean up whitespace
    s = " ".join(s.split())
    if not s:
        return ""

    # Capitalize first character
    s = s[0].upper() + s[1:]

    # Add question mark if starting with standard interrogative words
    first_word = s.split()[0].lower()
    if first_word in QUESTION_STARTERS and not s.endswith(("?", ".", "!")):
        s += "?"

    return s


def extract_sample_questions(patterns: List[str], max_questions: int = 3) -> List[str]:
    """Extract and deduplicate human-readable sample questions from regex patterns.

    Args:
        patterns: List of regex trigger patterns for an intent.
        max_questions: Maximum number of sample questions to retain.

    Returns:
        List of cleaned, unique sample question strings.
    """
    seen = set()
    cleaned_questions = []

    for pat in patterns:
        cleaned = clean_regex_pattern(pat)
        if not cleaned:
            continue
        key = cleaned.lower().rstrip("?!.")
        if key not in seen:
            seen.add(key)
            cleaned_questions.append(cleaned)
            if len(cleaned_questions) >= max_questions:
                break

    return cleaned_questions


def categorize_intent(intent_name: str) -> str:
    """Map intent snake_case identifier to a human-readable topic category.

    Args:
        intent_name: Snake-case intent identifier.

    Returns:
        Human-readable category title string.
    """
    name = intent_name.lower()
    if name.startswith("costume_"):
        return "Costumes & Character Recognition"
    if name.startswith("candy_") or any(k in name for k in ("candy", "bowl", "treat", "hungry", "eat_candy")):
        return "Candy Bowl & Porch Rules"
    if name.startswith("dogs_") or any(k in name for k in ("dog", "hound", "puppy", "canine")):
        return "Dogs & Pets"
    if name.startswith("jokes_") or any(k in name for k in ("riddle", "humor", "joke")):
        return "Jokes & Humor"
    if name.startswith("favorite_"):
        return "Favorites & Preferences"
    if any(k in name for k in ("movie", "quote", "pop_culture", "song", "monologue", "poem", "seinfeld", "blade_runner", "sucka")):
        return "Pop Culture, Movies & Quotes"
    if any(k in name for k in ("bone", "skeleton", "anatomy", "naked", "skin", "heart", "organ", "tummy", "stomach", "skull", "teeth", "hair", "eyes", "ears", "body", "muscle")):
        return "Skeleton Anatomy, Bones & Calcium"
    if any(k in name for k in ("weather", "time", "clock", "cold", "temperature", "sunburnt")):
        return "Time, Weather & Environment"
    if any(k in name for k in ("greeting", "farewell", "goodbye", "thank", "hello", "hi", "bye", "who_are_you", "what_is_your_name")):
        return "Greetings, Identity & Farewells"
    if any(k in name for k in ("fear", "scared", "afraid", "fight", "insult", "hug", "selfie", "dance", "magic", "smart", "dumb", "ugly", "mean", "attitude", "laugh")):
        return "Interactive Banter & Rebuttals"
    return "Lore, Questions & Banter"


def generate_markdown_summary(
    profile: PersonalityProfile,
    max_questions: int = 3,
    category_filter: Optional[str] = None,
    include_system: bool = True,
) -> str:
    """Generate a GitHub-flavored Markdown document cataloging questions and answers.

    Args:
        profile: Instantiated PersonalityProfile dataclass.
        max_questions: Maximum questions per intent to display.
        category_filter: Optional substring filter for category names.
        include_system: Whether to include greetings, fallbacks, and environment responses.

    Returns:
        Rendered Markdown document string.
    """
    lines: List[str] = []
    lines.append(f"# {profile.display_name} ({profile.personality}) Prompt Catalog\n")
    lines.append(f"> {profile.description}\n")

    lines.append("## Overview\n")
    lines.append(f"- **Voice Model:** `{profile.voice}`")
    if profile.voice_settings:
        settings_str = ", ".join(f"`{k}={v}`" for k, v in sorted(profile.voice_settings.items()))
        lines.append(f"- **Voice Settings:** {settings_str}")
    lines.append(f"- **Total Registered Intents:** {len(profile.intents)}")
    lines.append("")

    if include_system:
        lines.append("## System & Dynamic Responses\n")
        if profile.startup_greeting:
            lines.append(f"- **Startup Greeting:** \"{profile.startup_greeting}\"")
        if profile.time_responses:
            lines.append("- **Local Time Handler:** Dynamically speaks accurate local time with bedtime warning:")
            for k, v in sorted(profile.time_responses.items()):
                lines.append(f"  - *{k.capitalize()}:* \"{v}\"")
        if profile.weather_responses:
            lines.append("- **Live Weather Handler:** Dynamically queries ambient temperature via wttr.in:")
            for k, v in sorted(profile.weather_responses.items()):
                lines.append(f"  - *{k.capitalize()}:* \"{v}\"")
        if profile.costume_prompts:
            lines.append("- **Costume Inquiry Prompts:**")
            for prompt in profile.costume_prompts:
                lines.append(f"  - \"{prompt}\"")
        if profile.fallbacks:
            lines.append("- **Offline / Indeterminate Fallbacks:**")
            for fb in profile.fallbacks:
                lines.append(f"  - \"{fb}\"")
        if profile.escalation_lines:
            lines.append("- **Loitering Escalation Warnings:**")
            for esc in profile.escalation_lines:
                lines.append(f"  - \"{esc}\"")
        lines.append("")

    # Group intents by category
    categorized = defaultdict(list)
    for intent in profile.intents:
        cat = categorize_intent(intent.name)
        if category_filter and category_filter.lower() not in cat.lower() and category_filter.lower() not in intent.name.lower():
            continue
        categorized[cat].append(intent)

    lines.append("## Conversational Intent Catalog\n")
    for cat_name, intents in sorted(categorized.items()):
        lines.append(f"### {cat_name} ({len(intents)} intents)\n")
        for intent in intents:
            title = intent.name.replace("_", " ").title()
            lines.append(f"#### {title} (`{intent.name}`)\n")

            sample_qs = extract_sample_questions(intent.patterns, max_questions=max_questions)
            if sample_qs:
                lines.append("**Supported Trigger Questions / Prompts:**")
                for q in sample_qs:
                    lines.append(f"- *\"{q}\"*")
                lines.append("")

            lines.append("**Spoken Responses:**")
            for r in intent.responses:
                lines.append(f"- \"{r}\"")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_text_summary(
    profile: PersonalityProfile,
    max_questions: int = 3,
    category_filter: Optional[str] = None,
    include_system: bool = True,
) -> str:
    """Generate a clean terminal-friendly text summary of questions and answers.

    Args:
        profile: Instantiated PersonalityProfile dataclass.
        max_questions: Maximum questions per intent to display.
        category_filter: Optional category substring filter.
        include_system: Whether to include system responses.

    Returns:
        Formatted terminal text string.
    """
    lines: List[str] = []
    divider = "=" * 76
    subdivider = "-" * 76

    lines.append(divider)
    lines.append(f"  {profile.display_name.upper()} ({profile.personality})")
    lines.append(f"  {profile.description}")
    lines.append(f"  Voice: {profile.voice} | Total Intents: {len(profile.intents)}")
    lines.append(divider)

    if include_system and profile.startup_greeting:
        lines.append(f"\n[Startup Greeting]: \"{profile.startup_greeting}\"")

    categorized = defaultdict(list)
    for intent in profile.intents:
        cat = categorize_intent(intent.name)
        if category_filter and category_filter.lower() not in cat.lower() and category_filter.lower() not in intent.name.lower():
            continue
        categorized[cat].append(intent)

    for cat_name, intents in sorted(categorized.items()):
        lines.append(f"\n{cat_name.upper()} ({len(intents)} intents)")
        lines.append(subdivider)
        for idx, intent in enumerate(intents, start=1):
            sample_qs = extract_sample_questions(intent.patterns, max_questions=max_questions)
            lines.append(f"  {idx}. [{intent.name}]")
            for q in sample_qs:
                lines.append(f"     ❓ \"{q}\"")
            for r in intent.responses:
                lines.append(f"     \"{r}\"")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_json_summary(
    profile: PersonalityProfile,
    max_questions: int = 3,
    category_filter: Optional[str] = None,
    include_system: bool = True,
) -> str:
    """Generate a structured JSON summary of supported questions and answers.

    Args:
        profile: Instantiated PersonalityProfile dataclass.
        max_questions: Maximum questions per intent to display.
        category_filter: Optional category substring filter.
        include_system: Whether to include system responses.

    Returns:
        Serialized JSON string.
    """
    intents_data = []
    for intent in profile.intents:
        cat = categorize_intent(intent.name)
        if category_filter and category_filter.lower() not in cat.lower() and category_filter.lower() not in intent.name.lower():
            continue
        intents_data.append({
            "name": intent.name,
            "category": cat,
            "sample_questions": extract_sample_questions(intent.patterns, max_questions=max_questions),
            "patterns_count": len(intent.patterns),
            "responses": intent.responses,
        })

    data: Dict[str, Any] = {
        "personality": profile.personality,
        "display_name": profile.display_name,
        "description": profile.description,
        "voice": profile.voice,
        "voice_settings": profile.voice_settings,
        "total_intents": len(profile.intents),
        "intents": intents_data,
    }

    if include_system:
        data["system"] = {
            "startup_greeting": profile.startup_greeting,
            "costume_prompts": profile.costume_prompts,
            "fallbacks": profile.fallbacks,
            "escalation_lines": profile.escalation_lines,
            "time_responses": profile.time_responses,
            "weather_responses": profile.weather_responses,
            "system_announcements": profile.system_announcements,
        }

    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Dynamically summarize questions and answers supported by an AlsaGeist personality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--personality",
        default=os.environ.get("PERSONALITY", "grimjaw"),
        help="Personality profile identifier or JSON path (default: $PERSONALITY or grimjaw).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["markdown", "text", "json"],
        default="markdown",
        help="Output format: 'markdown', 'text', or 'json' (default: markdown).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional destination file path (default: write to standard output).",
    )
    parser.add_argument(
        "-c",
        "--category",
        default=None,
        help="Filter summary to a specific category or intent keyword.",
    )
    parser.add_argument(
        "-q",
        "--max-questions",
        type=int,
        default=3,
        help="Maximum sample questions to render per intent (default: 3).",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Exclude system greetings, fallbacks, and environment responses.",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Simulate summary generation without writing output files.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    return parser.parse_args()


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    args = parse_args()

    try:
        profile_path = find_personality_file(args.personality)
        if not os.path.isfile(profile_path):
            print(f"Error: Personality profile file not found for '{args.personality}' at {profile_path}.", file=sys.stderr)
            return 1

        profile = PersonalityProfile.load_from_file(profile_path)
    except Exception as exc:
        print(f"Error loading personality '{args.personality}': {exc}", file=sys.stderr)
        return 1

    include_system = not args.no_system
    if args.dry_run:
        destination = f" and write to {args.output}" if args.output else ""
        print(f"[DRY-RUN] Would generate {args.format} summary for '{profile.display_name}' ({len(profile.intents)} intents){destination}")
        return 0

    if args.format == "markdown":
        output_text = generate_markdown_summary(
            profile=profile,
            max_questions=args.max_questions,
            category_filter=args.category,
            include_system=include_system,
        )
    elif args.format == "json":
        output_text = generate_json_summary(
            profile=profile,
            max_questions=args.max_questions,
            category_filter=args.category,
            include_system=include_system,
        )
    else:
        output_text = generate_text_summary(
            profile=profile,
            max_questions=args.max_questions,
            category_filter=args.category,
            include_system=include_system,
        )

    if args.output:
        try:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Successfully generated summary for '{profile.display_name}' ({len(profile.intents)} intents) -> {args.output}")
            return 0
        except Exception as exc:
            print(f"Error writing to output file '{args.output}': {exc}", file=sys.stderr)
            return 1

    sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
