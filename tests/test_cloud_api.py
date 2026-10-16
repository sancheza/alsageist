"""Unit tests for core/cloud_api.py request construction and response parsing."""

from unittest.mock import patch

from core import cloud_api


def test_gemini_generate_parts_returns_first_candidate_parts():
    """Parts come from the first candidate; the key travels in a header."""
    response = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}, {"content": {"parts": []}}]}
    with patch("core.cloud_api.http_client.post_json", return_value=response) as post:
        parts = cloud_api.gemini_generate_parts("KEY", "model-x", {"contents": []}, timeout=3)

    assert parts == [{"text": "hi"}]
    url = post.call_args.args[0]
    assert url.endswith("/models/model-x:generateContent")
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "KEY"}
    assert post.call_args.kwargs["timeout"] == 3


def test_gemini_generate_parts_empty_without_candidates():
    """No candidates yields an empty list."""
    with patch("core.cloud_api.http_client.post_json", return_value={}):
        assert cloud_api.gemini_generate_parts("KEY", "m", {}, timeout=1) == []


def test_groq_chat_returns_stripped_content():
    """Chat returns the first choice content, stripped."""
    response = {"choices": [{"message": {"content": "  hello  "}}]}
    with patch("core.cloud_api.http_client.post_json", return_value=response) as post:
        assert cloud_api.groq_chat("KEY", {"model": "m"}, timeout=2) == "hello"
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer KEY"}


def test_groq_chat_empty_without_choices():
    """No choices yields an empty string."""
    with patch("core.cloud_api.http_client.post_json", return_value={"choices": []}):
        assert cloud_api.groq_chat("KEY", {}, timeout=2) == ""


def test_groq_transcribe_uploads_wav_with_model():
    """Transcription uploads the WAV as 'file' with the model field."""
    with patch("core.cloud_api.http_client.post_multipart", return_value={"text": " hello world "}) as post:
        text = cloud_api.groq_transcribe("KEY", "whisper-x", b"WAV", timeout=5)

    assert text == "hello world"
    assert post.call_args.kwargs["fields"] == {"model": "whisper-x"}
    assert post.call_args.kwargs["files"] == {"file": ("audio.wav", "audio/wav", b"WAV")}
    assert post.call_args.kwargs["timeout"] == 5
