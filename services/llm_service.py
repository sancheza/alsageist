"""Conversational LLM service managing persona dialogue reasoning.

Generates terse, character-accurate banter in response to visitors
using Groq Llama 3.3 or Gemini Flash.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from core import cloud_api


DEFAULT_SYSTEM_PROMPT = (
    "You are a witty, dry-humored conversational voice assistant. "
    "Rules you must strictly obey:\n"
    "1. Keep every response under 2 sentences maximum.\n"
    "2. Be witty, dry, and stay in character.\n"
    "3. Never output emojis, markdown formatting, asterisks, stage directions, or sound effect tags (like *laughs* or [cackles]).\n"
    "4. Return plain spoken dialogue only."
)
GEMINI_TIMEOUT_SEC = 6.0
GROQ_TIMEOUT_SEC = 8.0
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
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = (
            gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
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

        reply = ""
        if self.groq_api_key:
            try:
                reply = self._generate_groq(user_text)
                if not reply:
                    raise RuntimeError("Groq returned an empty reply")
            except Exception:
                if self.gemini_api_key:
                    try:
                        reply = self._generate_gemini(user_text)
                    except Exception:
                        reply = ""
                else:
                    reply = ""
        else:
            try:
                reply = self._generate_gemini(user_text)
            except Exception:
                reply = ""

        if not reply:
            # Every backend failed or answered with nothing: speak the canned line,
            # and keep it out of the conversation history.
            return self.fallback_response

        # Update history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

        # Trim history
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

        return reply

    def reset_history(self) -> None:
        """Clear conversation history for the next visitor."""
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

    def _generate_gemini(self, user_text: str) -> str:
        """Generate reply using fast Gemini Flash text API.

        Returns:
            Cleaned reply text, or an empty string when every model fails or answers with nothing.
        """
        models_to_try = cloud_api.gemini_llm_models(self.gemini_model)

        contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "generationConfig": {
                "temperature": 0.8,
                "thinkingConfig": {"thinkingBudget": 0},
                "maxOutputTokens": 80,
            },
        }

        for model in models_to_try:
            try:
                parts = cloud_api.gemini_generate_parts(
                    self.gemini_api_key, model, payload, timeout=GEMINI_TIMEOUT_SEC
                )
                if not parts:
                    continue
                cleaned = self._clean_reply(parts[0].get("text", ""))
                if cleaned:
                    return cleaned
            except Exception:
                continue

        return ""

    def _generate_groq(self, user_text: str) -> str:
        """Generate reply using Groq Llama endpoint."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 100,
            **cloud_api.groq_model_params(self.groq_model),
        }

        text = cloud_api.groq_chat(self.groq_api_key, payload, timeout=GROQ_TIMEOUT_SEC)
        return self._clean_reply(text)

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Strip markdown asterisks and line breaks so the reply is plain spoken text."""
        return text.strip().replace("*", "").replace("\n", " ").strip()
