"""Unit tests for voice commands parsing (shutdown, mode toggle, mode query).

Verifies offline-safe speech patterns with unpunctuated lowercase text,
typographic apostrophes, phonetic word splits ('off line'), and conversational prefixes.
"""

import pytest

from core.voice_commands import (
    COMMAND_MODE_AUTO,
    COMMAND_MODE_OFFLINE,
    COMMAND_MODE_ONLINE,
    COMMAND_MODE_QUERY,
    COMMAND_SHUTDOWN,
    match_voice_command,
)


@pytest.mark.parametrize("phrase", [
    # Basic shutdown commands
    "shut down",
    "shutdown",
    "power off",
    "power down",
    # With targets
    "shut down system",
    "shut down the system",
    "power off system",
    "power off the system",
    "power down the device",
    "turn off system",
    "turn off the display",
    "shut down alsageist",
    # With conversational prefixes and wake words
    "grimjaw shut down",
    "hey grimjaw power off",
    "please shut down",
    "can you power off",
    "could you shut down the system",
    # Vosk unpunctuated offline formats
    "shut down",
    "power off",
    "grimjaw power down",
])
def test_match_voice_command_shutdown(phrase):
    """Verify shutdown voice commands match accurately."""
    assert match_voice_command(phrase) == COMMAND_SHUTDOWN


@pytest.mark.parametrize("phrase", [
    # Offline mode commands
    "go offline",
    "go off line",
    "switch to offline",
    "switch to off line",
    "switch to offline mode",
    "switch to off line mode",
    "set mode offline",
    "set offline mode",
    "offline mode",
    "off line mode",
    "change to offline mode",
    "enter offline mode",
    "turn offline",
    "disconnect",
    "disconnect from internet",
    # With conversational prefixes
    "grimjaw go offline",
    "please switch to offline mode",
    "can you go off line",
    "hey grimjaw offline mode",
])
def test_match_voice_command_offline(phrase):
    """Verify offline mode switch commands match across variations."""
    assert match_voice_command(phrase) == COMMAND_MODE_OFFLINE


@pytest.mark.parametrize("phrase", [
    # Online mode commands
    "go online",
    "go on line",
    "switch to online",
    "switch to on line",
    "switch to online mode",
    "switch to on line mode",
    "set mode online",
    "set online mode",
    "online mode",
    "on line mode",
    "change to online mode",
    "enter online mode",
    "turn online",
    "connect",
    "reconnect",
    "reconnect to internet",
    # With conversational prefixes
    "grimjaw go online",
    "please go online",
    "can you switch to online mode",
])
def test_match_voice_command_online(phrase):
    """Verify online mode switch commands match across variations."""
    assert match_voice_command(phrase) == COMMAND_MODE_ONLINE


@pytest.mark.parametrize("phrase", [
    # Auto mode commands
    "auto mode",
    "automatic mode",
    "set mode auto",
    "switch to auto mode",
    "switch to automatic mode",
    "go auto",
])
def test_match_voice_command_auto(phrase):
    """Verify automatic mode commands match."""
    assert match_voice_command(phrase) == COMMAND_MODE_AUTO


@pytest.mark.parametrize("phrase", [
    # Query mode commands
    "what mode are you in",
    "what mode is this",
    "what mode is it",
    "what is your mode",
    "what's your mode",
    "what is the mode",
    "what mode",
    "which mode are you in",
    "which mode",
    "are you online or offline",
    "are you offline or online",
    "are you online",
    "are you offline",
    "are you on line",
    "are you off line",
    "current mode",
    "check mode",
    "status mode",
    "system mode",
    "mode status",
    "mode check",
    # With conversational prefixes
    "grimjaw what mode are you in",
    "please tell me what mode are you in",
    "tell me your mode",
    "can you check mode",
    "hey grimjaw are you online or offline",
])
def test_match_voice_command_query(phrase):
    """Verify mode status queries match accurately."""
    assert match_voice_command(phrase) == COMMAND_MODE_QUERY


@pytest.mark.parametrize("phrase", [
    # Empty / whitespace
    "",
    "   ",
    # Casual speech that should NOT trigger commands
    "why did you shut down yesterday",
    "don't shut down",
    "turn off the lights",
    "why are you offline",
    "tell me a joke about offline",
    "what is the meaning of life",
    "hello grimjaw",
    "can i have candy",
    "what time is it",
    "what is the weather",
])
def test_match_voice_command_ignores_unrelated(phrase):
    """Verify non-command speech and conversational phrases do not false-trigger."""
    assert match_voice_command(phrase) is None
