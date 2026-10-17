"""Conversational LLM service managing persona dialogue reasoning.

Generates terse, character-accurate banter in response to visitors
using Groq (default model gpt-oss) or Gemini Flash.
"""

import os
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from core import cloud_api, env_utils
from core.intent_builder import PROMPT_RULES


DEFAULT_SYSTEM_PROMPT = f"You are a witty, dry-humored conversational voice assistant. {PROMPT_RULES}"
GEMINI_TIMEOUT_SEC = 6.0
GROQ_TIMEOUT_SEC = 8.0
TOTAL_TIMEOUT_SEC = 12.0  # all backends together; a visitor should not wait longer for a reply
MAX_REPLY_CHARS = 300
_STAGE_DIRECTION = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_SPEECH = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U0000200D]"
)
DEFAULT_FALLBACK_RESPONSE = "Sorry, I lost my train of thought. Please ask me again."


class ConversationalLLMService:
    """Dialogue reasoning engine for the conversational display."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 6,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        fallback_response: Union[str, Callable[[], str], None] = None,
    ) -> None:
        """Initialize the conversational LLM service.

        Args:
            groq_api_key: Optional Groq API key for Llama.
            gemini_api_key: Optional Gemini API key for Gemini Flash text.
            system_prompt: Character persona and behavioral constraints.
            max_history_turns: Max back-and-forth dialogue turns to retain.
            groq_model: Optional Groq LLM model (default: $GROQ_LLM_MODEL or cloud_api.GROQ_DEFAULT_LLM_MODEL).
            gemini_model: Optional Gemini model override (default: $GEMINI_LLM_MODEL).
            fallback_response: Optional dialogue string, or a function returning one, spoken when
                every backend fails (default: $LLM_FALLBACK_RESPONSE or a neutral apology).
        """
        self.groq_api_key = env_utils.groq_api_key(groq_api_key)
        self.gemini_api_key = env_utils.gemini_api_key(gemini_api_key)
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.groq_model = groq_model or cloud_api.groq_llm_model()
        self.gemini_model = gemini_model or os.environ.get("GEMINI_LLM_MODEL")
        self._fallback_source = (
            fallback_response
            or os.environ.get("LLM_FALLBACK_RESPONSE")
            or DEFAULT_FALLBACK_RESPONSE
        )
        self.history: List[Dict[str, str]] = []
        self._history_lock = threading.Lock()

        if not self.groq_api_key and not self.gemini_api_key:
            raise ValueError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY is configured for LLM dialogue."
            )

    @property
    def fallback_response(self) -> str:
        """Return the line to speak when every backend fails (a function source is called each time)."""
        source = self._fallback_source
        return source() if callable(source) else source

    def generate_response(self, user_text: str) -> str:
        """Generate creature reply to user speech.

        Args:
            user_text: Transcribed user speech.

        Returns:
            Creature dialogue string.
        """
        if not user_text or not user_text.strip():
            return ""

        deadline = time.monotonic() + TOTAL_TIMEOUT_SEC
        reply = ""
        backends = []
        if self.groq_api_key:
            backends.append(("Groq", self._generate_groq))
        if self.gemini_api_key:
            backends.append(("Gemini", self._generate_gemini))
        for name, generate in backends:
            if time.monotonic() >= deadline:
                print(f"  (LLM {name} skipped: reply deadline of {TOTAL_TIMEOUT_SEC:.0f}s used up)")
                break
            try:
                reply = generate(user_text, deadline)
            except Exception as exc:
                print(f"  (LLM backend {name} failed: {exc})")
                continue
            if reply:
                break
            print(f"  (LLM backend {name} returned an empty reply)")

        if not reply:
            # Every backend failed or answered with nothing: speak the canned line,
            # and keep it out of the conversation history.
            return self.fallback_response

        with self._history_lock:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            if len(self.history) > self.max_history_turns * 2:
                self.history = self.history[-(self.max_history_turns * 2):]

        return reply

    def reset_history(self) -> None:
        """Clear conversation history for the next visitor."""
        with self._history_lock:
            self.history.clear()

    def reset_conversation(self) -> None:
        """Reset conversation history for the next visitor (implements DialogueEngine)."""
        self.reset_history()

    def classify_and_respond(self, user_text: str) -> Tuple[str, str]:
        """Generate response and return with intent tag (implements DialogueEngine).

        Args:
            user_text: Transcribed input string from the user.

        Returns:
            Tuple of (response_text, "llm_generated").
        """
        return self.generate_response(user_text), "llm_generated"

    def _generate_gemini(self, user_text: str, deadline: Optional[float] = None) -> str:
        """Generate reply using fast Gemini Flash text API.

        Returns:
            Cleaned reply text, or an empty string when every model fails or answers with nothing.
        """
        models_to_try = cloud_api.gemini_llm_models(self.gemini_model)

        with self._history_lock:
            history = list(self.history)
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "generationConfig": {
                "temperature": 0.8,
                "thinkingConfig": {"thinkingBudget": 0},
                "maxOutputTokens": 120,
            },
        }

        last_error: Optional[Exception] = None
        for model in models_to_try:
            timeout = self._timeout(GEMINI_TIMEOUT_SEC, deadline)
            if timeout <= 0:
                break
            try:
                parts = cloud_api.gemini_generate_parts(
                    self.gemini_api_key, model, payload, timeout=timeout
                )
                if not parts:
                    continue
                cleaned = self._clean_reply(parts[0].get("text", ""))
                if cleaned:
                    return cleaned
            except Exception as exc:
                last_error = exc
                print(f"  (Gemini model {model} failed: {exc})")

        if last_error is not None:
            raise RuntimeError(f"all Gemini models failed: {last_error}") from last_error
        return ""

    def _generate_groq(self, user_text: str, deadline: Optional[float] = None) -> str:
        """Generate reply using Groq Llama endpoint."""
        messages = [{"role": "system", "content": self.system_prompt}]
        with self._history_lock:
            messages.extend(dict(msg) for msg in self.history)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 150,
            **cloud_api.groq_model_params(self.groq_model),
        }

        text = cloud_api.groq_chat(
            self.groq_api_key, payload, timeout=max(0.5, self._timeout(GROQ_TIMEOUT_SEC, deadline))
        )
        return self._clean_reply(text)

    @staticmethod
    def _timeout(default_sec: float, deadline: Optional[float]) -> float:
        """Return the per-request timeout, shortened to what is left of the reply deadline."""
        if deadline is None:
            return default_sec
        return min(default_sec, deadline - time.monotonic())

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Reduce a model reply to plain spoken text.

        Removes markdown asterisks, bracketed stage directions, emoji and line breaks, caps the
        length, and drops a trailing unfinished sentence when a complete one precedes it.

        Args:
            text: Raw model output.

        Returns:
            Cleaned reply, or an empty string if nothing speakable remains.
        """
        cleaned = _NON_SPEECH.sub("", _STAGE_DIRECTION.sub(" ", text))
        cleaned = " ".join(cleaned.replace("*", "").split())
        if len(cleaned) > MAX_REPLY_CHARS:
            cleaned = cleaned[:MAX_REPLY_CHARS]
        if cleaned and cleaned[-1] not in ".!?\"'":
            last_end = max(cleaned.rfind(mark) for mark in ".!?")
            if last_end > 0:
                cleaned = cleaned[: last_end + 1]
        return cleaned.strip()
