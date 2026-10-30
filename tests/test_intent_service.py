"""Unit tests for the Grimjaw Intent & Banter Engine.

Validates intent classification and response consistency against the loaded
personality JSON profile using structured evaluation fixtures from
tests/fixtures/grimjaw_test_utterances.json.
"""

import json
import os
import pytest
from services.intent_service import PersonaIntentEngine

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "grimjaw_test_utterances.json")


@pytest.fixture
def engine():
    """Fixture providing an initialized PersonaIntentEngine instance."""
    return PersonaIntentEngine()


@pytest.fixture
def intent_map(engine):
    """Fixture providing a dictionary mapping intent names to their valid response sets."""
    return {intent.name: set(intent.responses) for intent in engine.profile.intents}


@pytest.fixture(scope="module")
def utterances():
    """Fixture providing golden verification test vectors from JSON fixture."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_intent_engine_categories(engine, intent_map, utterances):
    """Verify that primary conversation categories classify into their expected intent and return profile responses."""
    for case in utterances["category_intent_cases"]:
        query = case["query"]
        expected_intent = case["expected_intent"]
        engine.reset_conversation()
        response, matched_intent = engine.classify_and_respond(query)
        assert isinstance(response, str)
        assert len(response) > 5
        assert matched_intent == expected_intent, f"Query '{query}' classified as '{matched_intent}', expected '{expected_intent}'"
        assert response in intent_map[expected_intent], f"Response '{response}' for query '{query}' is not defined in profile responses for '{expected_intent}'"


def test_intent_engine_prolonged_conversation(engine, utterances):
    """Verify that the engine gives proper relevant answers even after 10+ turns."""
    for q in utterances["prolonged_conversation_queries"]:
        resp = engine.generate_response(q)
        assert len(resp) > 5


def test_intent_engine_third_person_and_expanded_queries(engine, intent_map, utterances):
    """Verify that third-person phrased questions and expanded categories map dynamically."""
    for case in utterances["third_person_and_expanded_cases"]:
        query = case["query"]
        expected_intent = case["expected_intent"]
        engine.reset_conversation()
        response, matched_intent = engine.classify_and_respond(query)
        assert isinstance(response, str)
        assert len(response) > 5
        assert matched_intent == expected_intent, f"Query '{query}' classified as '{matched_intent}', expected '{expected_intent}'"
        assert response in intent_map[expected_intent], f"Response '{response}' for query '{query}' not in profile responses for '{expected_intent}'"


def test_intent_engine_user_session_queries(engine, intent_map, utterances):
    """Verify queries reported from real session transcripts match expected intent categories and profile responses."""
    for case in utterances["session_transcript_cases"]:
        query = case["query"]
        expected_intent = case["expected_intent"]
        engine.reset_conversation()
        response, matched_intent = engine.classify_and_respond(query)
        assert isinstance(response, str)
        assert len(response) > 5
        assert matched_intent == expected_intent, f"Query '{query}' classified as '{matched_intent}', expected '{expected_intent}'"
        assert response in intent_map[expected_intent], f"Response '{response}' for query '{query}' not in profile responses for '{expected_intent}'"


def test_intent_engine_reset(engine):
    """Verify conversational turn counter resets properly."""
    engine.generate_response("hello")
    assert engine._turn_count == 1
    engine.reset_conversation()
    assert engine._turn_count == 0


def test_costume_and_quote_specific_matching(engine, intent_map, utterances):
    """Verify that specific character costumes match their exact profile intent categories."""
    for case in utterances["costume_and_quote_cases"]:
        prompt = case["query"]
        expected_intent = case["expected_intent"]
        resp, intent = engine.classify_and_respond(prompt)
        assert intent == expected_intent, f"Prompt '{prompt}' classified as '{intent}', expected '{expected_intent}'"
        assert resp in intent_map[expected_intent], f"Response '{resp}' not in profile responses for '{expected_intent}'"


def test_new_intents_hobby_pets_costume_seinfeld(engine, intent_map, utterances):
    """Verify dynamic mapping for hobbies, pets, favorite costume, and Seinfeld quotes."""
    for case in utterances["hobby_pets_seinfeld_cases"]:
        prompt = case["query"]
        expected_intent = case["expected_intent"]
        resp, intent = engine.classify_and_respond(prompt)
        assert intent == expected_intent, f"Prompt '{prompt}' classified as '{intent}', expected '{expected_intent}'"
        assert resp in intent_map[expected_intent], f"Response '{resp}' not in profile responses for '{expected_intent}'"


def test_time_and_weather_intent_classification(engine, utterances):
    """Verify time and weather questions map to time_and_clock and weather_and_temperature intents."""
    for q in utterances["time_queries"]:
        _, intent = engine.classify_and_respond(q)
        assert intent == "time_and_clock", f"Failed for '{q}': got intent '{intent}'"

    for q in utterances["weather_queries"]:
        _, intent = engine.classify_and_respond(q)
        assert intent == "weather_and_temperature", f"Failed for '{q}': got intent '{intent}'"


def test_random_content_intent(engine, utterances):
    """Verify that asking for something random yields a line from the general pool and never costume feedback."""
    for q in utterances["random_content_queries"]:
        resp, intent = engine.classify_and_respond(q)
        assert intent == "random_content", f"Failed for '{q}': got intent '{intent}'"
        assert len(resp) > 5
        assert resp in engine._random_content_pool, f"Response '{resp}' not in random content pool"


def test_pick_unique_evicts_oldest_phrase_first(engine):
    """When the recent list is full the oldest phrase is dropped, never the one just chosen."""
    engine._recently_used.clear()
    engine._max_recent = 3
    choices = [f"line {i}" for i in range(20)]

    picked = []
    for _ in range(6):
        picked.append(engine._pick_unique(choices))

    # The memory holds the last three picks; the oldest is forgotten first, never the newest
    assert list(engine._recently_used) == picked[-3:]
    # A pick never repeats one of the three before it (older lines may legitimately come back)
    for index, line in enumerate(picked):
        assert line not in picked[max(0, index - 3):index]


def test_pick_unique_resets_when_every_choice_was_used(engine):
    """With every choice used recently the engine starts over instead of failing."""
    engine._recently_used.clear()
    choices = ["a", "b"]
    first_round = {engine._pick_unique(choices), engine._pick_unique(choices)}
    assert first_round == {"a", "b"}
    assert engine._pick_unique(choices) in choices


@pytest.mark.parametrize("query, expected_intent", [
    ("G.I. Joe.", "costume_gi_joe"),
    ("gi joe", "costume_gi_joe"),
    ("I am G.I. Joe", "costume_gi_joe"),
    ("I'm dressed as Evie.", "costume_eevee"),
    ("Evie", "costume_eevee"),
    ("Eevee", "costume_eevee"),
    ("I'm dressed as a cactus", "costume"),
    ("I am dressed as Batman", "costume"),
    ("my costume is a pumpkin", "costume"),
])
def test_costume_gi_joe_evie_and_generic_matching(engine, query, expected_intent):
    """Verify GI Joe, Evie/Eevee, and generic 'dressed as' phrases match expected intents."""
    resp, intent = engine.classify_and_respond(query)
    assert intent == expected_intent
    assert isinstance(resp, str) and len(resp) > 5


@pytest.mark.parametrize("query", [
    "tell me a scary story",
    "tell a spooky story",
    "tell me a horror story",
    "tell me a ghost story",
    "can you tell me a scary story",
    "do you know any scary stories",
    "tell us a creepy tale",
    "scary story",
    "spooky stories",
    "tell me something scary",
])
def test_scary_story_intent_matching(engine, query):
    """Verify scary and spooky story requests match the scary_story intent and return abridged stories."""
    resp, intent = engine.classify_and_respond(query)
    assert intent == "scary_story"
    assert isinstance(resp, str) and len(resp) > 30

