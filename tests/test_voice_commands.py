"""Unit tests for voice commands parsing (shutdown, mode toggle, mode query).

Verifies offline-safe speech patterns with unpunctuated lowercase text,
typographic apostrophes, phonetic word splits ('off line'), and conversational prefixes.
"""

import pytest

from core.voice_commands import (
    COMMAND_ACCESS_DENIED,
    COMMAND_ENGINE_QUERY,
    COMMAND_FALLBACK_DISABLE,
    COMMAND_FALLBACK_ENABLE,
    COMMAND_FALLBACK_QUERY,
    COMMAND_MODE_AUTO,
    COMMAND_MODE_OFFLINE,
    COMMAND_MODE_ONLINE,
    COMMAND_MODE_QUERY,
    COMMAND_MUTE,
    COMMAND_SHUTDOWN,
    COMMAND_VOICE_QUERY,
    PRIVILEGED_COMMANDS,
    is_verbal_password,
    match_voice_command,
    match_voice_command_detailed,
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
    # Query voice model commands
    "what is your voice model",
    "what's your voice model",
    "what is the voice model",
    "what voice model are you using",
    "which voice model are you using",
    "what voice model is this",
    "what voice are you using",
    "which voice are you using",
    "what is your voice",
    "which voice model",
    "what voice model",
    "voice model",
    "current voice model",
    "check voice model",
    "voice model status",
    # With conversational prefixes
    "hey grimjaw what voice model are you using",
    "please tell me what is your voice model",
    "can you tell me which voice are you using",
    "grimjaw voice model",
])
def test_match_voice_command_voice_query(phrase):
    """Verify voice model queries match accurately."""
    assert match_voice_command(phrase) == COMMAND_VOICE_QUERY


@pytest.mark.parametrize("phrase", [
    # Query dialogue engine commands
    "what is your dialogue engine",
    "what is your dialog engine",
    "what's your dialogue engine",
    "what is the dialogue engine",
    "which dialogue engine are you using",
    "which dialog engine are you using",
    "what dialogue engine are you running",
    "what dialog engine are you running",
    "what engine are you using",
    "which engine are you using",
    "what engine is running",
    "what is your engine",
    "dialogue engine",
    "dialog engine",
    "current engine",
    "are you using intent or llm",
    "are you using llm or intent",
    "check dialogue engine",
    "dialogue engine status",
    # With conversational prefixes
    "hey grimjaw what engine are you using",
    "please tell me what is your dialogue engine",
    "can you tell me are you using intent or llm",
    "grimjaw dialogue engine",
])
def test_match_voice_command_engine_query(phrase):
    """Verify dialogue engine queries match accurately."""
    assert match_voice_command(phrase) == COMMAND_ENGINE_QUERY


@pytest.mark.parametrize("phrase", [
    "what is your fallback status",
    "what is the fallback status",
    "fallback status",
    "llm fallback status",
    "is fallback enabled",
    "is fallback on",
    "is llm fallback enabled",
    "is llm fallback on",
    "check fallback status",
    "query fallback status",
    "are you using fallback",
    "are you using llm fallback",
    "hey grimjaw is llm fallback enabled",
    "grimjaw fallback status",
])
def test_match_voice_command_fallback_query(phrase):
    """Verify fallback status queries match accurately."""
    assert match_voice_command(phrase) == COMMAND_FALLBACK_QUERY


@pytest.mark.parametrize("phrase", [
    "enable fallback",
    "enable llm fallback",
    "turn on fallback",
    "turn on llm fallback",
    "activate fallback",
    "activate llm fallback",
    "fallback on",
    "llm fallback on",
    "switch on fallback",
    "switch on llm fallback",
    "hey grimjaw enable llm fallback",
    "grimjaw fallback on",
    "set llm fallback to enabled",
    "set fallback to enabled",
    "set fallback enabled",
    "set llm fallback enabled",
    "set fallback on",
    "turn fallback on",
    "turn llm fallback on",
    "switch fallback on",
])
def test_match_voice_command_fallback_enable(phrase):
    """Verify fallback enable commands match accurately."""
    assert match_voice_command(phrase) == COMMAND_FALLBACK_ENABLE


@pytest.mark.parametrize("phrase", [
    "disable fallback",
    "disable llm fallback",
    "turn off fallback",
    "turn off llm fallback",
    "deactivate fallback",
    "deactivate llm fallback",
    "fallback off",
    "llm fallback off",
    "switch off fallback",
    "switch off llm fallback",
    "hey grimjaw disable llm fallback",
    "grimjaw fallback off",
    "set llm fallback to disabled",
    "set fallback to disabled",
    "set fallback disabled",
    "set llm fallback disabled",
    "set fallback off",
    "turn fallback off",
    "turn llm fallback off",
    "switch fallback off",
])
def test_match_voice_command_fallback_disable(phrase):
    """Verify fallback disable commands match accurately."""
    assert match_voice_command(phrase) == COMMAND_FALLBACK_DISABLE


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


@pytest.mark.parametrize("phrase, expected_cmd", [
    ("alsageist shut down", COMMAND_SHUTDOWN),
    ("grimjaw alsageist power off", COMMAND_SHUTDOWN),
    ("shut down alsageist", COMMAND_SHUTDOWN),
    ("Shut down. Password, Alphageist.", COMMAND_SHUTDOWN),
    ("shut down password alsageist", COMMAND_SHUTDOWN),
    ("Auf die Geist shut down", COMMAND_SHUTDOWN),
    ("shut down auf die geist", COMMAND_SHUTDOWN),
    ("alsageist switch to offline mode", COMMAND_MODE_OFFLINE),
    ("grimjaw alsageist go online", COMMAND_MODE_ONLINE),
    ("alsageist automatic mode", COMMAND_MODE_AUTO),
    ("alsageist enable llm fallback", COMMAND_FALLBACK_ENABLE),
    ("alsageist disable fallback", COMMAND_FALLBACK_DISABLE),
])
def test_privileged_commands_succeed_with_valid_password(phrase, expected_cmd):
    """Privileged voice commands succeed when the verbal password or phonetic alias is provided."""
    assert match_voice_command(phrase, password="alsageist") == expected_cmd


@pytest.mark.parametrize("phrase", [
    "shut down",
    "grimjaw shut down",
    "power off",
    "grimjaw power off",
    "turn off the system",
    "switch to offline mode",
    "grimjaw go offline",
    "go online",
    "grimjaw go online",
    "automatic mode",
])
def test_privileged_commands_rejected_without_password(phrase):
    """Privileged voice commands return COMMAND_ACCESS_DENIED without the verbal password."""
    assert match_voice_command(phrase, password="alsageist") == COMMAND_ACCESS_DENIED


@pytest.mark.parametrize("phrase, expected_cmd", [
    ("what mode are you in", COMMAND_MODE_QUERY),
    ("are you online", COMMAND_MODE_QUERY),
    ("what is your voice model", COMMAND_VOICE_QUERY),
    ("what engine are you using", COMMAND_ENGINE_QUERY),
    ("is fallback enabled", COMMAND_FALLBACK_QUERY),
    ("enable fallback", COMMAND_FALLBACK_ENABLE),
    ("set llm fallback to enabled", COMMAND_FALLBACK_ENABLE),
    ("disable fallback", COMMAND_FALLBACK_DISABLE),
    ("set llm fallback to disabled", COMMAND_FALLBACK_DISABLE),
    ("alsageist what mode are you in", COMMAND_MODE_QUERY),
])
def test_unprivileged_commands_succeed_without_password(phrase, expected_cmd):
    """Query and fallback toggle commands do not require a verbal password."""
    assert match_voice_command(phrase, password="alsageist") == expected_cmd


def test_custom_verbal_password():
    """Custom verbal password configured via parameter is verified."""
    assert match_voice_command("sesame shut down", password="sesame") == COMMAND_SHUTDOWN
    assert match_voice_command("alsageist shut down", password="sesame") == COMMAND_ACCESS_DENIED


@pytest.mark.parametrize("pwd", [None, "", "none", "off", "false"])
def test_disabled_verbal_password(pwd):
    """Password protection can be disabled by setting password to None, empty, or 'none'."""
    assert match_voice_command("shut down", password=pwd) == COMMAND_SHUTDOWN
    assert match_voice_command("switch to offline mode", password=pwd) == COMMAND_MODE_OFFLINE


@pytest.mark.parametrize("utterance", [
    "alsageist",
    "alsa geist",
    "alphageist",
    "alpha geist",
    "Auf die Geist.",
    "auf geist",
    "Geist.",
    "Password, Alphageist.",
    "the password is alsageist",
    "verbal password alsageist",
])
def test_is_verbal_password_phonetic_variations(utterance):
    """Verifies that speech recognizer transcriptions for alsageist are recognized."""
    assert is_verbal_password(utterance, password="alsageist") is True


@pytest.mark.parametrize("utterance", [
    "hello",
    "shut down",
    "give me candy",
    "wrong password",
    "sesame",
])
def test_is_verbal_password_rejects_non_passwords(utterance):
    """Non-password utterances return False."""
    assert is_verbal_password(utterance, password="alsageist") is False


@pytest.mark.parametrize("phrase", [
    "mute",
    "mute yourself",
    "grimjaw mute",
    "grimjaw mute yourself",
    "hey grimjaw mute",
    "please mute",
    "mute the microphone",
    "mute microphone",
    "mute the mic",
    "mute mic",
    "silence yourself",
    "turn off the mic",
    "turn off microphone",
    "go to sleep",
    "enter sleep",
    "standby",
    "enter standby",
    "set mode to mute",
    "set mode mute",
    "set mode to muted",
    "set mode muted",
    "set to mute",
    "switch to mute",
    "switch to mute mode",
    "mute mode",
    "turn on mute",
    "turn on mute mode",
    "enable mute",
])
def test_match_voice_command_mute(phrase):
    """Verify mute voice commands match accurately."""
    assert match_voice_command(phrase) == COMMAND_MUTE


@pytest.mark.parametrize("phrase, expected_cmd", [
    ("mute", COMMAND_ACCESS_DENIED),
    ("mute yourself", COMMAND_ACCESS_DENIED),
    ("grimjaw mute", COMMAND_ACCESS_DENIED),
    ("set mode to mute", COMMAND_ACCESS_DENIED),
    ("alsageist mute", COMMAND_MUTE),
    ("alsageist set mode to mute", COMMAND_MUTE),
    ("grimjaw mute password alsageist", COMMAND_MUTE),
    ("mute yourself alsageist", COMMAND_MUTE),
    ("Auf die Geist mute", COMMAND_MUTE),
])
def test_match_voice_command_mute_privileged(phrase, expected_cmd):
    """Verify mute is treated as a privileged command requiring verbal password."""
    assert match_voice_command(phrase, password="alsageist") == expected_cmd
