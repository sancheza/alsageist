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


# ---- last_cloud_failed: lets the agent notice a dead network ----

def _stt_with(groq=False, gemini=False, vosk=False):
    from unittest.mock import MagicMock

    stt = SpeechToTextService(groq_api_key="placeholder", gemini_api_key="placeholder",
                              vosk_model_path="/nonexistent")
    stt.groq_api_key = "g" if groq else None
    stt.gemini_api_key = "k" if gemini else None
    stt._vosk_model = MagicMock() if vosk else None
    return stt


def test_cloud_success_is_not_a_cloud_failure():
    """A cloud backend that answers leaves last_cloud_failed False."""
    from unittest.mock import patch

    stt = _stt_with(groq=True, gemini=True, vosk=True)
    with patch("services.stt_service.cloud_api.groq_transcribe", return_value="hi"):
        stt.transcribe(_audio())
    assert stt.last_cloud_failed is False


def test_fallback_to_second_cloud_backend_is_not_a_total_cloud_failure():
    """Groq failing while Gemini answers means the network is up."""
    from unittest.mock import patch

    stt = _stt_with(groq=True, gemini=True, vosk=True)
    with patch("services.stt_service.cloud_api.groq_transcribe", side_effect=OSError("403")), \
         patch("services.stt_service.cloud_api.gemini_generate_parts", return_value=[{"text": "hi"}]):
        assert stt.transcribe(_audio()) == "hi"
    assert stt.last_cloud_failed is False


def test_vosk_rescue_after_every_cloud_backend_failed_is_flagged():
    """Reaching Vosk after all cloud backends failed sets last_cloud_failed."""
    from unittest.mock import patch

    stt = _stt_with(groq=True, gemini=True, vosk=True)
    with patch("services.stt_service.cloud_api.groq_transcribe", side_effect=OSError("down")), \
         patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")), \
         patch.object(stt, "transcribe_offline", return_value="offline words"):
        assert stt.transcribe(_audio()) == "offline words"
    assert stt.last_cloud_failed is True


def test_single_cloud_backend_failure_with_vosk_rescue_is_flagged():
    """With only one cloud backend configured, its failure followed by Vosk is still flagged."""
    from unittest.mock import patch

    stt = _stt_with(groq=True, vosk=True)
    with patch("services.stt_service.cloud_api.groq_transcribe", side_effect=OSError("down")), \
         patch.object(stt, "transcribe_offline", return_value="offline words"):
        stt.transcribe(_audio())
    assert stt.last_cloud_failed is True


def test_total_failure_with_cloud_attempts_is_flagged():
    """When every backend fails and cloud was tried, the flag is set even though the call raises."""
    from unittest.mock import patch

    stt = _stt_with(gemini=True)
    with patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")):
        with pytest.raises(RuntimeError):
            stt.transcribe(_audio())
    assert stt.last_cloud_failed is True


def test_offline_only_use_is_never_a_cloud_failure():
    """No cloud keys, or an explicit offline request, never sets the flag."""
    from unittest.mock import patch

    stt = _stt_with(vosk=True)
    with patch.object(stt, "transcribe_offline", return_value="words"):
        stt.transcribe(_audio())
        assert stt.last_cloud_failed is False

    stt = _stt_with(groq=True, vosk=True)
    stt.last_cloud_failed = True  # stale value from an earlier call
    with patch.object(stt, "transcribe_offline", return_value="words"):
        stt.transcribe(_audio(), offline=True)
    assert stt.last_cloud_failed is False


def test_stt_warns_when_vosk_missing_with_cloud_keys(monkeypatch, capsys):
    """When cloud keys exist but Vosk model is missing, a warning is printed."""
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    SpeechToTextService(groq_api_key="g", gemini_api_key="k")
    captured = capsys.readouterr().out
    assert "Offline Vosk STT model is not installed" in captured


def test_stt_gemini_stops_iteration_on_empty_answer(monkeypatch):
    """When a Gemini model answers successfully with empty text, subsequent models are not called."""
    from unittest.mock import patch
    import numpy as np

    stt = SpeechToTextService(gemini_api_key="k")
    called_models = []

    def mock_generate(api_key, model, payload, timeout):
        called_models.append(model)
        return [{"text": ""}]

    with patch("services.stt_service.cloud_api.gemini_generate_parts", side_effect=mock_generate):
        result = stt.transcribe(np.zeros(1600, dtype=np.float32))
        assert result == ""
        assert len(called_models) == 1

