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
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    with pytest.raises(ValueError, match="configured for Speech-to-Text"):
        SpeechToTextService(groq_api_key=None, gemini_api_key=None, vosk_model_path="/nonexistent")



def test_stt_groq_failure_falls_back_to_gemini():
    """A Groq failure transcribes through Gemini."""
    import numpy as np
    from unittest.mock import patch

    stt = SpeechToTextService(groq_api_key="g", gemini_api_key="k")
    audio = np.zeros(1600, dtype=np.float32)
    with patch("services.stt_service.cloud_api.groq_transcribe", side_effect=OSError("down")), \
         patch(
             "services.stt_service.cloud_api.gemini_generate_parts",
             return_value=[{"text": "open the door"}],
         ):
        assert stt.transcribe(audio) == "open the door"


def test_stt_groq_receives_timeout():
    """Groq transcription always passes a timeout."""
    import numpy as np
    from unittest.mock import patch

    stt = SpeechToTextService(groq_api_key="g")
    with patch("services.stt_service.cloud_api.groq_transcribe", return_value="hi") as call:
        assert stt.transcribe(np.zeros(1600, dtype=np.float32)) == "hi"
    assert call.call_args.kwargs["timeout"] > 0


def _audio():
    import numpy as np
    return np.zeros(1600, dtype=np.float32)


def test_stt_gemini_total_failure_raises_instead_of_returning_silence():
    """When every Gemini model errors, transcription fails loudly rather than reporting silence."""
    from unittest.mock import patch

    stt = SpeechToTextService(gemini_api_key="k")
    stt._vosk_model = None  # a Vosk model installed on this machine must not rescue the call
    with patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")):
        with pytest.raises(RuntimeError, match="All transcription backends failed"):
            stt.transcribe(_audio())


def test_stt_gemini_failure_falls_back_to_vosk():
    """A Gemini outage hands the audio to the offline Vosk model."""
    from unittest.mock import MagicMock, patch

    stt = SpeechToTextService(gemini_api_key="k")
    stt._vosk_model = MagicMock()
    with patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")), \
         patch.object(stt, "transcribe_offline", return_value="offline words") as offline:
        assert stt.transcribe(_audio()) == "offline words"
    offline.assert_called_once()


def test_stt_gemini_no_speech_is_empty_not_error():
    """A model that answers with no text means silence and does not trigger fallbacks."""
    from unittest.mock import patch

    stt = SpeechToTextService(gemini_api_key="k")
    with patch("services.stt_service.cloud_api.gemini_generate_parts", return_value=[]):
        assert stt.transcribe(_audio()) == ""


def test_stt_gemini_transcript_limit_allows_long_utterances():
    """The transcript token cap leaves room for a full-length utterance."""
    from unittest.mock import patch

    stt = SpeechToTextService(gemini_api_key="k")
    with patch("services.stt_service.cloud_api.gemini_generate_parts", return_value=[{"text": "hi"}]) as call:
        stt.transcribe(_audio())
    assert call.call_args.args[2]["generationConfig"]["maxOutputTokens"] >= 200


def test_stt_error_lists_every_failed_backend():
    """The final error names each backend that failed."""
    from unittest.mock import patch

    stt = SpeechToTextService(groq_api_key="g", gemini_api_key="k")
    stt._vosk_model = None
    with patch("services.stt_service.cloud_api.groq_transcribe", side_effect=OSError("groq down")), \
         patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=OSError("gemini down")):
        with pytest.raises(RuntimeError) as exc:
            stt.transcribe(_audio())
    assert "Groq Whisper: groq down" in str(exc.value)
    assert "Gemini" in str(exc.value)
