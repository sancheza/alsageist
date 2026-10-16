"""Unit tests for scripts/prewarm_cache.py."""

import json
import os
import subprocess
import tempfile
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "prewarm_cache.py"))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = {**os.environ, "PYTHONPATH": REPO_ROOT}


def test_prewarm_cache_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True, env=ENV)
    assert result.returncode == 0
    assert "usage: prewarm_cache.py" in result.stdout
    assert "--personality" in result.stdout
    assert "--dry-run" in result.stdout


def test_prewarm_cache_version():
    """Verify -v / --version prints script version."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True, env=ENV)
    assert result.returncode == 0
    assert "prewarm_cache.py 1.1.0" in result.stdout


def test_prewarm_cache_dry_run():
    """Verify --dry-run prints phrases to be warmed without invoking Piper synthesis."""
    result = subprocess.run([SCRIPT_PATH, "--dry-run", "--limit", "15"], capture_output=True, text=True, env=ENV)
    assert result.returncode == 0
    assert "DRY RUN: Pre-warm Cache Preview (15 phrases)" in result.stdout
    assert "Personality: Grimjaw the Crypt Skeleton" in result.stdout
    assert "en_GB-alan-medium.onnx" in result.stdout
    assert "Dry-run complete. No audio was synthesized." in result.stdout


def test_prewarm_cache_custom_personality():
    """Verify prewarm_cache dynamically adapts to a custom personality JSON with unique voice and announcements."""
    custom_profile = {
        "personality": "bramble_witch",
        "display_name": "Bramble the Swamp Witch",
        "description": "An ancient witch brewing potions in the bog.",
        "voice": "en_US-lessac-medium",
        "startup_greeting": "Cackle! Who dares disturb my cauldron?",
        "fallbacks": ["Speak up, mortal, the swamp bubbles drown your voice!"],
        "escalation_lines": ["Away with you before I turn you into a newt!"],
        "costume_prompts": ["What kind of familiar are you dressed as?"],
        "priority_intents": ["potion_recipe"],
        "system_announcements": {
            "offline_mode": "The ether is silent. Offline mode engaged.",
            "online_mode": "The spirits have connected. Online mode active.",
            "powering_off": "Extinguishing the cauldron flames.",
            "volume_template": "Witch whisper level {percent} percent."
        },
        "intents": [
            {
                "name": "potion_recipe",
                "patterns": ["potion", "brew"],
                "responses": ["Eye of newt and toe of frog!"]
            },
            {
                "name": "swamp_history",
                "patterns": ["history", "bog"],
                "responses": ["This bog has stood for ten centuries."]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = os.path.join(tmpdir, "bramble_witch.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(custom_profile, f)

        result = subprocess.run(
            [SCRIPT_PATH, "--personality", profile_path, "--dry-run", "--all"],
            capture_output=True,
            text=True,
            env=ENV,
        )
        assert result.returncode == 0
        assert "Personality: Bramble the Swamp Witch" in result.stdout
        assert "en_US-lessac-medium.onnx" in result.stdout
        assert "Cackle! Who dares disturb my cauldron?" in result.stdout
        assert "The ether is silent. Offline mode engaged." in result.stdout
        assert "Witch whisper level 50 percent." in result.stdout
        assert "Eye of newt and toe of frog!" in result.stdout
        assert "This bog has stood for ten centuries." in result.stdout
