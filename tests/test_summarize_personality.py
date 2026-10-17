"""Unit tests for scripts/summarize_personality.py."""

import json
import os
import subprocess
import sys
import tempfile
import pytest

from core.intent_builder import CustomIntent, PersonalityProfile
from core.intent_samples import clean_regex_pattern, extract_sample_questions
from scripts.summarize_personality import (
    categorize_intent,
    generate_markdown_summary,
    generate_text_summary,
    generate_json_summary,
)


def test_clean_regex_pattern():
    """Verify regex patterns are transformed into natural sample questions."""
    # Standard interrogative sentence
    pat1 = r"\bwhat\ is\ your\ favorite\ board\ game\b"
    assert clean_regex_pattern(pat1) == "What is your favorite board game?"

    # Alternatives and group resolution
    pat2 = r"\bhow much (is|for|costs?) (one|a) rib\b"
    assert clean_regex_pattern(pat2) == "How much is one rib?"

    # Pronoun normalization
    pat3 = r"\b(do you|does he) miss (your|his) skin\b"
    assert clean_regex_pattern(pat3) == "Do you miss your skin?"

    # Non-question phrase
    pat4 = r"\bborrow a rib\b"
    assert clean_regex_pattern(pat4) == "Borrow a rib"

    # Quantifiers and optional tokens
    pat5 = r"\b(are you|is he)\s*(scared|afraid) of (dogs|puppies)\b"
    assert clean_regex_pattern(pat5) == "Are you scared of dogs?"


def test_extract_sample_questions_deduplication():
    """Verify questions are deduplicated and limit is enforced."""
    patterns = [
        r"\bwhat is your name\b",
        r"\bwhat\ is\ your\ name\b",
        r"\bwho are you\b",
        r"\bwhat are you called\b",
        r"\bwho goes there\b",
    ]
    qs = extract_sample_questions(patterns, max_questions=2)
    assert len(qs) == 2
    assert qs[0] == "What is your name?"
    assert qs[1] == "Who are you?"


def test_categorize_intent():
    """Verify intent categories are accurately mapped."""
    assert categorize_intent("costume_mario") == "Costumes & Character Recognition"
    assert categorize_intent("candy_take_one") == "Candy Bowl & Porch Rules"
    assert categorize_intent("dogs_borrow_rib") == "Dogs & Pets"
    assert categorize_intent("jokes_and_humor") == "Jokes & Humor"
    assert categorize_intent("favorite_board_game") == "Favorites & Preferences"
    assert categorize_intent("seinfeld_quote") == "Pop Culture, Movies & Quotes"
    assert categorize_intent("heart_and_organs") == "Skeleton Anatomy, Bones & Calcium"
    assert categorize_intent("cold_and_weather") == "Time, Weather & Environment"
    assert categorize_intent("greeting") == "Greetings, Identity & Farewells"
    assert categorize_intent("random_lore_topic") == "Lore, Questions & Banter"


@pytest.fixture
def sample_profile():
    """Create a sample profile fixture for summary generation tests."""
    return PersonalityProfile(
        personality="test_sentinel",
        display_name="Test Sentinel",
        description="A test sentinel guarding the gate.",
        voice="test_voice",
        voice_settings={"tempo": 1.2, "noise": 0.8},
        startup_greeting="Greetings, traveler!",
        intents=[
            CustomIntent(
                name="greeting",
                patterns=[r"\bhello there\b", r"\bhi\b"],
                responses=["Hello!", "Greetings!"],
            ),
            CustomIntent(
                name="favorite_food",
                patterns=[r"\bwhat is your favorite food\b"],
                responses=["I enjoy calcium broth."],
            ),
        ],
        time_responses={"standard": "The time is {time}."},
        weather_responses={"cold": "It is cold."},
    )


def test_generate_markdown_summary(sample_profile):
    """Verify Markdown summary rendering."""
    md = generate_markdown_summary(sample_profile, max_questions=2)
    assert "# Test Sentinel (test_sentinel) Prompt Catalog" in md
    assert "- **Voice Model:** `test_voice`" in md
    assert "### Greetings, Identity & Farewells" in md
    assert "#### Greeting (`greeting`)" in md
    assert "*\"Hello there\"*" in md
    assert "\"Hello!\"" in md


def test_generate_text_summary(sample_profile):
    """Verify plain text summary rendering."""
    text = generate_text_summary(sample_profile, max_questions=2)
    assert "TEST SENTINEL (test_sentinel)" in text
    assert "[greeting]" in text
    assert "❓ \"Hello there\"" in text
    assert '     "Hello!"' in text


def test_generate_json_summary(sample_profile):
    """Verify structured JSON summary rendering."""
    json_str = generate_json_summary(sample_profile, max_questions=2)
    data = json.loads(json_str)
    assert data["personality"] == "test_sentinel"
    assert data["total_intents"] == 2
    assert len(data["intents"]) == 2
    assert data["intents"][0]["name"] == "greeting"
    assert "Hello there" in data["intents"][0]["sample_questions"]



def test_cli_invocation_help():
    """Verify script responds to --help with all options."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    res = subprocess.run(
        [sys.executable, "scripts/summarize_personality.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "--personality" in res.stdout
    assert "--format" in res.stdout
    assert "--output" in res.stdout


def test_cli_invocation_file_output():
    """Verify script generates an output file when -o/--output is provided."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        res = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_personality.py",
                "-p",
                "grimjaw",
                "-o",
                tmp_path,
                "-c",
                "Dogs",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0
        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Grimjaw the Crypt Skeleton" in content
        assert "Dogs & Pets" in content
        assert "dogs_borrow_rib" in content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
