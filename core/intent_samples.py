"""Human-readable sample questions derived from intent regex patterns."""

import re
from typing import List

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
