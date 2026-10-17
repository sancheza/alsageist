"""Groq and Gemini API calls shared by dialogue, speech-to-text, and intent tooling.

Wraps endpoints, authentication headers, and response parsing on top of
core.http_client so callers pass a payload and receive text or parts.
"""

import os
from typing import Any, Dict, List

from core import http_client

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


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
    result = http_client.post_json(
        GEMINI_URL.format(model=model),
        payload,
        headers={"x-goog-api-key": str(api_key)},
        timeout=timeout,
    )
    candidates = result.get("candidates", [])
    if not candidates:
        return []
    return candidates[0].get("content", {}).get("parts", [])


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
    return choices[0].get("message", {}).get("content", "").strip()


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
