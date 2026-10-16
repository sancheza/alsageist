"""Automated test suite verifying that all 100 child statements/questions are recognized."""

import json
import os
import pytest
from services.intent_service import GrimjawIntentEngine

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "grimjaw_test_utterances.json")


@pytest.fixture(scope="module")
def child_questions():
    """Load the 100 child question and statement test vectors from JSON fixture."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["child_questions_and_statements_100"]


def test_all_100_questions_and_statements(child_questions):
    """Verify that every child question/statement resolves to an intent and not a fallback."""
    engine = GrimjawIntentEngine()
    fallbacks = set(engine._fallbacks)

    unmatched = []
    for idx, prompt in enumerate(child_questions, start=1):
        engine.reset_conversation()
        response = engine.generate_response(prompt)
        assert isinstance(response, str)
        assert len(response) > 5

        if response in fallbacks:
            unmatched.append((idx, prompt, response))

    assert len(unmatched) == 0, f"Unmatched prompts ({len(unmatched)}): {unmatched}"
