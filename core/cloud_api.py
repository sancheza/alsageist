"""Groq and Gemini API calls shared by dialogue, speech-to-text, and intent tooling.

Wraps endpoints, authentication headers, and response parsing on top of
core.http_client so callers pass a payload and receive text or parts.
"""

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.error

from core import http_client

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
GEMINI_DEFAULT_FLASH_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]
GEMINI_DEFAULT_LLM_MODELS = GEMINI_DEFAULT_FLASH_MODELS
GEMINI_DEFAULT_STT_MODELS = GEMINI_DEFAULT_FLASH_MODELS


# Models the API reported as not found (retired or misspelled). They are skipped for the rest of
# the process so every request does not pay a round trip to rediscover it.
_unavailable_models: set = set()
_unavailable_lock = threading.Lock()


def _usable(models: List[str]) -> List[str]:
    """Drop models already known to be unavailable; keep all if that would leave none."""
    with _unavailable_lock:
        remaining = [m for m in models if m not in _unavailable_models]
    return remaining or models


def gemini_llm_models(override: Optional[str] = None) -> List[str]:
    """Return prioritized candidate models for Gemini text generation."""
    explicit = override or os.environ.get("GEMINI_LLM_MODEL")
    return [explicit] if explicit else _usable(list(GEMINI_DEFAULT_LLM_MODELS))


def gemini_stt_models(override: Optional[str] = None) -> List[str]:
    """Return prioritized candidate models for Gemini audio transcription."""
    explicit = override or os.environ.get("GEMINI_STT_MODEL")
    return [explicit] if explicit else _usable(list(GEMINI_DEFAULT_STT_MODELS))


def groq_llm_model() -> str:
    """Return the Groq chat model: $GROQ_LLM_MODEL, else the default."""
    return os.environ.get("GROQ_LLM_MODEL", GROQ_DEFAULT_LLM_MODEL)


def groq_model_params(model: str) -> Dict[str, Any]:
    """Return extra chat request fields a Groq model needs.

    The gpt-oss models spend max_tokens on hidden reasoning, and with a short limit
    they often return an empty reply. Low reasoning effort leaves room for the answer.

    Args:
        model: Groq model name.

    Returns:
        Fields to merge into the chat completions request body.
    """
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


def gemini_generate_parts(
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    timeout: float,
) -> List[Dict[str, Any]]:
    """Call Gemini generateContent and return the first candidate's content parts.

    Args:
        api_key: Gemini API key.
        model: Gemini model name.
        payload: generateContent request body.
        timeout: Request timeout in seconds.

    Returns:
        Content parts of the first candidate, or an empty list when there is none.
    """
    try:
        result = http_client.post_json(
            GEMINI_URL.format(model=model),
            payload,
            headers={"x-goog-api-key": str(api_key)},
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            with _unavailable_lock:
                _unavailable_models.add(model)
        raise
    candidates = result.get("candidates", [])
    if not candidates:
        return []
    return candidates[0].get("content", {}).get("parts", [])


def gemini_generate(
    api_key: str,
    models: List[str],
    payload: Dict[str, Any],
    timeout: float,
    extract: Callable[[List[Dict[str, Any]]], str],
    deadline: Optional[float] = None,
    stop_on_answer: bool = False,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> str:
    """Try Gemini models in order and return the first usable text.

    Args:
        api_key: Gemini API key.
        models: Candidate models, most preferred first.
        payload: generateContent request body.
        timeout: Per-request timeout in seconds, shortened to what is left of the deadline.
        extract: Reduces a candidate's content parts to text; an empty string means no usable text.
        deadline: Optional time.monotonic() value after which no further model is tried.
        stop_on_answer: If True, a model that answers with no usable text ends the search (an
            empty transcript of silent audio is a valid answer, so further models are not tried).
        on_error: Optional callback receiving (model, error) for each failed model.

    Returns:
        The extracted text, or an empty string when no model produced any.

    Raises:
        RuntimeError: If every model attempted failed.
    """
    last_error: Optional[Exception] = None
    answered = False
    for model in models:
        remaining = timeout if deadline is None else min(timeout, deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            parts = gemini_generate_parts(api_key, model, payload, timeout=remaining)
        except Exception as exc:  # isolation point: one bad model must not stop the others
            last_error = exc
            if on_error is not None:
                on_error(model, exc)
            continue
        answered = True
        text = extract(parts) if parts else ""
        if text:
            return text
        if stop_on_answer:
            return ""
    if not answered and last_error is not None:
        raise RuntimeError(f"all Gemini models failed: {last_error}") from last_error
    return ""


def groq_chat(api_key: str, payload: Dict[str, Any], timeout: float) -> str:
    """Call the Groq chat completions endpoint.

    Args:
        api_key: Groq API key.
        payload: Chat completions request body.
        timeout: Request timeout in seconds.

    Returns:
        Stripped content of the first choice, or an empty string when there is none.
    """
    result = http_client.post_json(
        GROQ_CHAT_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    choices = result.get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def groq_transcribe(api_key: str, model: str, wav_bytes: bytes, timeout: float) -> str:
    """Transcribe WAV audio with the Groq Whisper endpoint.

    Args:
        api_key: Groq API key.
        model: Whisper model name.
        wav_bytes: WAV file contents.
        timeout: Request timeout in seconds.

    Returns:
        Stripped transcript text.
    """
    result = http_client.post_multipart(
        GROQ_TRANSCRIBE_URL,
        fields={"model": model},
        files={"file": ("audio.wav", "audio/wav", wav_bytes)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return result.get("text", "").strip()
