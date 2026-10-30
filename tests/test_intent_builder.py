"""Unit tests for core.intent_builder module."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from core.intent_authoring import IntentCollisionError, IntentParser, create_custom_intent
from core.intent_builder import CustomIntent, PersonalityProfile, find_personality_file



def test_custom_intent_creation_and_validation():
    """Verify CustomIntent validates name formatting and regex pattern compilation."""
    intent = CustomIntent(
        name="Favorite Dinosaur!",
        patterns=[r"\bfavorite dinosaur\b", r"\bwhat('s| is) your favorite dino\b"],
        responses=["The Velociraptor, naturally!"],
    )
    assert intent.name == "favorite_dinosaur"
    assert len(intent.patterns) == 2
    assert len(intent.responses) == 1

    # Invalid regex pattern should raise ValueError
    with pytest.raises(ValueError, match="Invalid regular expression"):
        CustomIntent(
            name="broken_regex",
            patterns=[r"[\b(unclosed_group"],
            responses=["Should fail"],
        )


def test_custom_intent_serialization():
    """Verify serialization to and from dictionary."""
    intent = CustomIntent(
        name="test_intent",
        patterns=[r"\btest\b"],
        responses=["Test response"],
        metadata={"author": "user"},
    )
    data = intent.to_dict()
    assert data["name"] == "test_intent"
    assert data["metadata"] == {"author": "user"}

    restored = CustomIntent.from_dict(data)
    assert restored.name == intent.name
    assert restored.patterns == intent.patterns
    assert restored.responses == intent.responses
    assert restored.metadata == intent.metadata


def test_personality_profile_persistence():
    """Verify PersonalityProfile can be saved to and loaded from a JSON file."""
    profile = PersonalityProfile(
        personality="snowman",
        display_name="Frosty the Snarky Snowman",
        description="A chillingly sarcastic snowman display.",
        voice="en_GB-alan-medium",
        startup_greeting="Brrr! Who disturbed my snowdrift?",
        fallbacks=["I'm melting here, talk faster!"],
        intents=[
            CustomIntent(
                name="melt_question",
                patterns=[r"\bwill you melt\b"],
                responses=["Not if the freezer holds out!"],
            )
        ],
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        profile.save_to_file(tmp_path)
        loaded = PersonalityProfile.load_from_file(tmp_path)

        assert loaded.personality == "snowman"
        assert loaded.display_name == "Frosty the Snarky Snowman"
        assert loaded.startup_greeting == "Brrr! Who disturbed my snowdrift?"
        assert len(loaded.intents) == 1
        assert loaded.intents[0].name == "melt_question"
        assert loaded.fallbacks == ["I'm melting here, talk faster!"]

        rules = loaded.compile_rules()
        assert len(rules) == 1
        assert rules[0][0] == "melt_question"
        assert rules[0][1] == [r"\bwill you melt\b"]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_personality_profile_docstring():
    """Verify PersonalityProfile loads and serializes docstrings and parses grimjaw.json doc."""
    # Test grimjaw.json has docstring loaded
    grimjaw_path = find_personality_file("grimjaw")
    grimjaw_profile = PersonalityProfile.load_from_file(grimjaw_path)
    assert grimjaw_profile.docstring is not None
    assert "AlsaGeist Personality Profile" in grimjaw_profile.doc
    assert "1. Purpose & Overview" in grimjaw_profile.doc

    # Test serialization preserves _doc
    profile = PersonalityProfile(
        personality="witch",
        display_name="Hexcrow",
        description="Wicked witch",
        docstring=["Line 1", "Line 2"],
    )
    data = profile.to_dict()
    assert "_doc" in data
    assert data["_doc"] == ["Line 1", "Line 2"]
    assert profile.doc == "Line 1\nLine 2"

    restored = PersonalityProfile.from_dict(data)
    assert restored.docstring == ["Line 1", "Line 2"]
    assert restored.doc == "Line 1\nLine 2"


def test_personality_add_and_overwrite_intent():
    """Verify adding intents and updating existing intents."""
    profile = PersonalityProfile(
        personality="witch",
        display_name="Madame Hexcrow",
        description="A wicked porch witch.",
    )

    intent1 = CustomIntent(
        name="potion_recipe",
        patterns=[r"\bmake a potion\b"],
        responses=["Eye of newt and toe of frog!"],
    )
    profile.add_intent(intent1)
    assert len(profile.intents) == 1

    # Overwrite intent with updated response
    intent2 = CustomIntent(
        name="potion_recipe",
        patterns=[r"\bmake a potion\b", r"\bbrew a potion\b"],
        responses=["Double, double toil and trouble!"],
    )
    profile.add_intent(intent2, overwrite=True)
    assert len(profile.intents) == 1
    assert profile.get_intent("potion_recipe").responses == ["Double, double toil and trouble!"]


def test_intent_parser_rule_based_fallback():
    """Verify offline rule-based parser derives clean name and regexes."""
    parser = IntentParser()

    intent = parser.parse(
        text="What is your favorite dinosaur?",
        responses=["The Velociraptor, naturally!"],
        use_llm=False,
    )

    assert "dinosaur" in intent.name
    assert len(intent.patterns) >= 1
    assert any(r"\b" in pat for pat in intent.patterns)
    assert intent.responses == ["The Velociraptor, naturally!"]


def test_intent_parser_llm_flow():
    """Verify LLM parsing path with mocked LLM response."""
    parser = IntentParser(gemini_api_key="mock_key")

    mock_llm_json = json.dumps({
        "name": "favorite_planet",
        "patterns": [
            r"\bwhat('s| is) (your|his) favorite planet\b",
            r"\bfavorite planet\b",
        ],
        "responses": [
            "Pluto! Because like me, it was unjustly discarded!",
        ],
    })

    with patch.object(parser, "_call_gemini", return_value=mock_llm_json):
        intent = parser.parse(
            text="What is your favorite planet in space?",
            use_llm=True,
        )

        assert intent.name == "favorite_planet"
        assert len(intent.patterns) == 2
        assert "Pluto!" in intent.responses[0]
        assert intent.metadata.get("source") == "llm_expanded"


def test_create_custom_intent_dry_run():
    """Verify create_custom_intent honors dry_run=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile_path = os.path.join(tmp_dir, "test_hero.json")
        profile = PersonalityProfile(
            personality="test_hero",
            display_name="Test Hero",
            description="A test hero",
        )
        profile.save_to_file(profile_path)

        intent, updated_profile = create_custom_intent(
            query_or_text="Can you fly?",
            responses=["Only when thrown by a giant!"],
            personality_name_or_path=profile_path,
            dry_run=True,
            use_llm=False,
        )

        assert intent.name is not None
        # Since dry_run=True, disk file should still have 0 intents
        disk_profile = PersonalityProfile.load_from_file(profile_path)
        assert len(disk_profile.intents) == 0


def test_create_custom_intent_persists_to_disk():
    """Verify create_custom_intent saves to disk when dry_run=False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile_path = os.path.join(tmp_dir, "test_hero.json")
        profile = PersonalityProfile(
            personality="test_hero",
            display_name="Test Hero",
            description="A test hero",
        )
        profile.save_to_file(profile_path)

        intent, updated_profile = create_custom_intent(
            query_or_text="Can you fly?",
            responses=["Only when thrown by a giant!"],
            personality_name_or_path=profile_path,
            dry_run=False,
            use_llm=False,
        )

        disk_profile = PersonalityProfile.load_from_file(profile_path)
        assert len(disk_profile.intents) == 1
        assert disk_profile.intents[0].responses == ["Only when thrown by a giant!"]


def test_personality_check_query_collision():
    """Verify check_query accurately detects matching existing intents."""
    profile = PersonalityProfile(
        personality="skeleton",
        display_name="Grimjaw",
        description="A skeleton",
        intents=[
            CustomIntent(
                name="favorite_movie",
                patterns=[r"\b(favorite|best)\b.*\b(movie|film)\b"],
                responses=["The Nightmare Before Christmas!"],
            )
        ],
    )

    # Matching query
    collision = profile.check_query("What is your favorite movie?")
    assert collision is not None
    matched_intent, matched_pattern = collision
    assert matched_intent.name == "favorite_movie"
    assert r"\b(favorite|best)\b.*\b(movie|film)\b" in matched_pattern

    # Non-matching query
    assert profile.check_query("What is your favorite board game?") is None


def test_personality_search_and_append_responses():
    """Verify search_intents and append_responses functionality."""
    profile = PersonalityProfile(
        personality="skeleton",
        display_name="Grimjaw",
        description="A skeleton",
        intents=[
            CustomIntent(
                name="favorite_movie",
                patterns=[r"\bfavorite movie\b"],
                responses=["Nightmare Before Christmas"],
            )
        ],
    )

    # Search by intent name
    assert len(profile.search_intents("movie")) == 1
    # Search by response keyword
    assert len(profile.search_intents("Christmas")) == 1
    # Non-matching search
    assert len(profile.search_intents("spaceship")) == 0

    # Append response
    res = profile.append_responses("favorite_movie", ["Beetlejuice!"])
    assert res is True
    intent = profile.get_intent("favorite_movie")
    assert len(intent.responses) == 2
    assert "Beetlejuice!" in intent.responses

    # Deduplication on append
    profile.append_responses("favorite_movie", ["Beetlejuice!"])
    assert len(intent.responses) == 2


def test_create_custom_intent_collision_handling():
    """Verify create_custom_intent handles collisions according to collision_action."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile_path = os.path.join(tmp_dir, "test_personality.json")
        profile = PersonalityProfile(
            personality="test_personality",
            display_name="Test Personality",
            description="Testing collisions",
            intents=[
                CustomIntent(
                    name="favorite_movie",
                    patterns=[r"\bfavorite movie\b"],
                    responses=["Nightmare Before Christmas"],
                )
            ],
        )
        profile.save_to_file(profile_path)

        # 1. Collision action: raise (default)
        with pytest.raises(IntentCollisionError) as exc_info:
            create_custom_intent(
                query_or_text="What is your favorite movie?",
                responses=["Hocus Pocus!"],
                personality_name_or_path=profile_path,
                collision_action="raise",
                dry_run=True,
                use_llm=False,
            )
        assert exc_info.value.existing_intent.name == "favorite_movie"

        # 2. Collision action: append
        intent, updated_profile = create_custom_intent(
            query_or_text="What is your favorite movie?",
            responses=["Hocus Pocus!"],
            personality_name_or_path=profile_path,
            collision_action="append",
            dry_run=False,
            use_llm=False,
        )
        assert intent.name == "favorite_movie"
        reloaded = PersonalityProfile.load_from_file(profile_path)
        assert len(reloaded.get_intent("favorite_movie").responses) == 2
        assert "Hocus Pocus!" in reloaded.get_intent("favorite_movie").responses

        # 3. Collision action: overwrite
        create_custom_intent(
            query_or_text="What is your favorite movie?",
            responses=["Ghostbusters!"],
            personality_name_or_path=profile_path,
            collision_action="overwrite",
            dry_run=False,
            use_llm=False,
        )
        reloaded = PersonalityProfile.load_from_file(profile_path)
        assert reloaded.get_intent("favorite_movie").responses == ["Ghostbusters!"]


def test_add_intent_cli_help(capsys):
    """Verify add_intent.py --help provides detailed purpose, architecture, and examples."""
    import subprocess
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "add_intent.py")
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "PURPOSE & ARCHITECTURE:" in result.stdout
    assert "AlsaGeist can answer ANY prompt dynamically using an LLM fallback" in result.stdout
    assert "LOCAL, IMMEDIATE responses" in result.stdout
    assert "TARGET DATA FILES MODIFIED:" in result.stdout
    assert "COMMON USAGE EXAMPLES:" in result.stdout





def test_build_system_prompt_names_persona():
    """The LLM system prompt names the persona and carries the dialogue rules."""
    profile = PersonalityProfile(personality="demo", display_name="Demo Bot", description="A test persona.")
    prompt = profile.build_system_prompt()
    assert prompt.startswith("You are Demo Bot. A test persona.")
    assert "under 2 sentences" in prompt


def test_announcement_falls_back_to_default():
    """announcement() returns the profile text or the supplied default."""
    profile = PersonalityProfile(
        personality="demo",
        display_name="Demo Bot",
        description="",
        system_announcements={"powering_off": "Goodnight."},
    )
    assert profile.announcement("powering_off", "Powering off.") == "Goodnight."
    assert profile.announcement("online_mode", "NOW IN ONLINE MODE") == "NOW IN ONLINE MODE"


def test_load_personality_by_name_and_missing():
    """load_personality loads a shipped profile and raises for an unknown one."""
    from core.intent_builder import load_personality

    assert load_personality("grimjaw").personality == "grimjaw"
    with pytest.raises(FileNotFoundError, match="nonexistent_persona_xyz"):
        load_personality("nonexistent_persona_xyz")


def test_resolve_voice_settings_precedence():
    """Explicit values beat profile settings, which beat the built-in defaults."""
    from core.voice_config import resolve_voice_settings

    profile = PersonalityProfile(
        personality="demo",
        display_name="Demo",
        description="",
        voice_settings={"tempo": 1.3, "noise": 0.6},
    )
    assert resolve_voice_settings(profile) == (1.3, 0.6)
    assert resolve_voice_settings(profile, tempo=1.0) == (1.0, 0.6)
    assert resolve_voice_settings(profile, noise=0.1) == (1.3, 0.1)
    assert resolve_voice_settings(None) == (1.15, 0.85)
    assert resolve_voice_settings(PersonalityProfile("d", "D", "")) == (1.15, 0.85)


def test_intent_parser_groq_uses_shared_model_and_params(monkeypatch):
    """The intent parser asks Groq for the shared default model with its reasoning setting."""

    monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
    parser = IntentParser(groq_api_key="g")
    with patch("core.intent_authoring.cloud_api.groq_chat", return_value="{}") as chat:
        parser._call_groq("expand this")
    payload = chat.call_args.args[1]
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}


def test_low_power_lines_round_trip_and_are_omitted_when_empty():
    """low_power_lines is saved and loaded, and left out of the file when a profile has none."""
    with_lines = PersonalityProfile(personality="d", display_name="D", description="", low_power_lines=["I am weak."])
    assert PersonalityProfile.from_dict(with_lines.to_dict()).low_power_lines == ["I am weak."]
    assert "low_power_lines" not in PersonalityProfile(personality="d", display_name="D", description="").to_dict()


def test_grimjaw_low_power_lines_are_speakable():
    """The shipped persona has several short lines with no emoji or em-dash (they are spoken aloud)."""
    from core.intent_builder import load_personality

    lines = load_personality("grimjaw").low_power_lines
    assert len(lines) >= 4 and len(set(lines)) == len(lines)
    for line in lines:
        assert 20 < len(line) <= 110, line
        assert "—" not in line and not any(ord(ch) > 0x2000 for ch in line), line


def test_personality_apply_override_intents_replace_and_append():
    """PersonalityProfile.apply_override updates responses for existing intents and appends new ones."""
    profile = PersonalityProfile(
        personality="skeleton",
        display_name="Skeleton Guardian",
        description="Guards the crypt",
        startup_greeting="Halt!",
        intents=[
            CustomIntent(name="favorite_food", patterns=[r"\bfood\b"], responses=["Bone broth!"]),
            CustomIntent(name="greeting", patterns=[r"\bhello\b"], responses=["Greetings, mortal!"]),
        ],
    )

    override_data = {
        "startup_greeting": "Welcome to my crypt!",
        "intents": [
            {
                "name": "favorite_food",
                "responses": ["Candy corn and roasted pumpkins!"],
            },
            {
                "name": "secret_password",
                "patterns": [r"\bopen sesame\b"],
                "responses": ["Door unlocked!"],
            },
        ],
    }

    merged = profile.apply_override(override_data)
    assert merged.startup_greeting == "Welcome to my crypt!"
    assert len(merged.intents) == 3

    intent_map = {i.name: i for i in merged.intents}
    assert intent_map["favorite_food"].responses == ["Candy corn and roasted pumpkins!"]
    assert intent_map["favorite_food"].patterns == [r"\bfood\b"]
    assert intent_map["greeting"].responses == ["Greetings, mortal!"]
    assert intent_map["secret_password"].responses == ["Door unlocked!"]


def test_personality_apply_override_append_responses():
    """PersonalityProfile.apply_override appends responses when append=True is set."""
    profile = PersonalityProfile(
        personality="skeleton",
        display_name="Skeleton",
        description="",
        intents=[
            CustomIntent(name="joke", patterns=[r"\bjoke\b"], responses=["Why did the skeleton cross the road?"]),
        ],
    )

    override_data = {
        "intents": [
            {
                "name": "joke",
                "append": True,
                "responses": ["To get to the body shop!"],
            }
        ]
    }

    merged = profile.apply_override(override_data)
    assert len(merged.intents) == 1
    assert merged.intents[0].responses == [
        "Why did the skeleton cross the road?",
        "To get to the body shop!",
    ]


def test_personality_apply_override_dictionaries_shallow_merge():
    """PersonalityProfile.apply_override merges dictionary fields preserving unspecified keys."""
    profile = PersonalityProfile(
        personality="skeleton",
        display_name="Skeleton",
        description="",
        system_announcements={
            "offline_mode": "NOW OFFLINE",
            "online_mode": "NOW ONLINE",
        },
        time_responses={
            "bedtime": "Time to sleep!",
            "standard": "The time is {time}.",
        },
    )

    override_data = {
        "system_announcements": {
            "offline_mode": "DISCONNECTED FROM WORLD",
            "muted": "AUDIO MUTED",
        },
        "time_responses": {
            "bedtime": "Way past your curfew!",
        },
    }

    merged = profile.apply_override(override_data)
    assert merged.system_announcements["offline_mode"] == "DISCONNECTED FROM WORLD"
    assert merged.system_announcements["online_mode"] == "NOW ONLINE"
    assert merged.system_announcements["muted"] == "AUDIO MUTED"
    assert merged.time_responses["bedtime"] == "Way past your curfew!"
    assert merged.time_responses["standard"] == "The time is {time}."


def test_load_personality_with_override_file(tmp_path):
    """load_personality loads base profile and automatically merges <name>.override.json."""
    from core.intent_builder import load_personality

    base_file = tmp_path / "phantom.json"
    override_file = tmp_path / "phantom.override.json"

    base_data = {
        "personality": "phantom",
        "display_name": "The Phantom",
        "description": "Haunts the theater",
        "startup_greeting": "I am here.",
        "intents": [
            {"name": "favorite_food", "patterns": [r"\bfood\b"], "responses": ["Ectoplasm."]},
        ],
    }
    base_file.write_text(json.dumps(base_data), encoding="utf-8")

    override_data = {
        "startup_greeting": "The music of the night begins!",
        "intents": [
            {"name": "favorite_food", "responses": ["Popcorn at the opera!"]},
        ],
    }
    override_file.write_text(json.dumps(override_data), encoding="utf-8")

    # With overrides enabled (default)
    loaded = load_personality(str(base_file))
    assert loaded.startup_greeting == "The music of the night begins!"
    assert loaded.intents[0].responses == ["Popcorn at the opera!"]

    # With overrides disabled explicitly
    loaded_raw = load_personality(str(base_file), apply_overrides=False)
    assert loaded_raw.startup_greeting == "I am here."
    assert loaded_raw.intents[0].responses == ["Ectoplasm."]

