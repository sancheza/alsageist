"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch):
    """Keep cloud API keys from the developer's shell out of every test.

    Tests that need a key pass it explicitly, so no test can reach a real service.
    """
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _weather_enabled(monkeypatch):
    """Enable weather lookups (off by default without a location); tests mock the network."""
    monkeypatch.setenv("WEATHER_ENABLED", "true")
    monkeypatch.delenv("WEATHER_LOCATION", raising=False)
