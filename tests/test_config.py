"""Unit tests for core/config.py."""

import pytest
from core.config import VERSION, build_arg_parser, parse_args


def test_config_version():
    """Verify config version string."""
    assert VERSION == "1.0.0"


def test_config_defaults(monkeypatch):
    """Verify default CLI arguments when no flags are supplied."""
    monkeypatch.delenv("PERSONALITY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK", raising=False)
    args = parse_args([])
    assert args.engine == "intent"
    assert args.personality == "grimjaw"
    assert args.llm_fallback is False
    assert args.volume == 0.5
    assert args.mic_gain == 1.5
    assert args.threshold == 0.0035
    assert args.led_pin == 25
    assert args.button_pin == 23
    assert args.voice_command_password == "alsageist"


def test_config_voice_command_password_env(monkeypatch):
    """Verify VOICE_COMMAND_PASSWORD environment variable overrides default."""
    monkeypatch.setenv("VOICE_COMMAND_PASSWORD", "sesame")
    args = parse_args([])
    assert args.voice_command_password == "sesame"


def test_config_invalid_env_falls_back_to_defaults(monkeypatch):
    """Verify invalid environment variables fall back cleanly to defaults without crashing."""
    monkeypatch.setenv("VAD_THRESHOLD", "not_a_float")
    monkeypatch.setenv("VOLUME", "invalid")
    monkeypatch.setenv("LED_PIN", "abc")
    args = parse_args([])
    assert args.threshold == 0.0035
    assert args.volume == 0.5
    assert args.led_pin == 25


def test_config_explicit_flags():
    """Verify explicit arguments override defaults."""
    args = parse_args([
        "--personality", "snowman",
        "--engine", "llm",
        "--llm-fallback",
        "--volume", "0.8",
        "--led-pin", "18",
        "--button-pin", "24",
        "--no-greeting",
        "--voice-command-password", "abracadabra",
    ])
    assert args.personality == "snowman"
    assert args.engine == "llm"
    assert args.llm_fallback is True
    assert args.volume == 0.8
    assert args.led_pin == 18
    assert args.button_pin == 24
    assert args.no_greeting is True
    assert args.voice_command_password == "abracadabra"
