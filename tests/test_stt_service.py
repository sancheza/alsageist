"""Unit tests for SpeechToTextService parameterization."""

import pytest
from services.stt_service import SpeechToTextService


def test_stt_service_initialization():
    """Verify STT service parameter defaults and explicit overrides."""
    stt = SpeechToTextService(
        groq_api_key="fake_groq",
        groq_model="whisper-custom-model",
    )
    assert stt.groq_api_key == "fake_groq"
    assert stt.groq_model == "whisper-custom-model"


def test_stt_service_env_overrides(monkeypatch):
    """Verify STT service respects GROQ_STT_MODEL and GEMINI_STT_MODEL."""
    monkeypatch.setenv("GROQ_STT_MODEL", "whisper-env-model")
    monkeypatch.setenv("GEMINI_STT_MODEL", "gemini-env-model")

    stt = SpeechToTextService(groq_api_key="fake_groq", gemini_api_key="fake_gemini")
    assert stt.groq_model == "whisper-env-model"
    assert stt.gemini_model == "gemini-env-model"


def test_stt_service_missing_keys_raises(monkeypatch):
    """Verify ValueError is raised if no backend is configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    with pytest.raises(ValueError, match="configured for Speech-to-Text"):
        SpeechToTextService(groq_api_key=None, gemini_api_key=None, vosk_model_path="/nonexistent")

