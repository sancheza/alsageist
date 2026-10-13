"""Conversational LLM service managing persona dialogue reasoning.

Generates terse, character-accurate banter in response to visitors
using Groq Llama 3.3 or Gemini Flash.
"""

import json
import os
from typing import Dict, List, Optional
import urllib.error
import urllib.request


GRIMJAW_SYSTEM_PROMPT = (
    "You are Grimjaw, a sarcastic, ancient skeletal guardian stationed on the porch guarding "
    "the Halloween candy bowl. "
    "Rules you must strictly obey:\n"
    "1. Keep every response under 2 sentences maximum.\n"
    "2. Be witty, dry, and tease trick-or-treaters playfully about their costumes, bravery, or greed for candy.\n"
    "3. Never output emojis, markdown formatting, asterisks, stage directions, or sound effect tags (like *laughs* or [cackles]).\n"
    "4. Return plain spoken dialogue only."
)


class ConversationalLLMService:
    """Dialogue reasoning engine for the conversational display."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        system_prompt: str = GRIMJAW_SYSTEM_PROMPT,
        max_history_turns: int = 6,
    ) -> None:
        """Initialize the conversational LLM service.

        Args:
            groq_api_key: Optional Groq API key for Llama 3.3.
            gemini_api_key: Optional Gemini API key for Gemini Flash text.
            system_prompt: Character persona and behavioral constraints.
            max_history_turns: Max back-and-forth dialogue turns to retain.
        """
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = (
            gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

        if not self.groq_api_key and not self.gemini_api_key:
            raise ValueError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY is configured for LLM dialogue."
            )

    def generate_response(self, user_text: str) -> str:
        """Generate creature reply to user speech.

        Args:
            user_text: Transcribed user speech.

        Returns:
            Creature dialogue string.
        """
        if not user_text or not user_text.strip():
            return ""

        if self.groq_api_key:
            try:
                reply = self._generate_groq(user_text)
            except Exception as exc:
                if self.gemini_api_key:
                    reply = self._generate_gemini(user_text)
                else:
                    raise RuntimeError(f"Groq LLM generation failed: {exc}") from exc
        else:
            reply = self._generate_gemini(user_text)

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

    def _generate_gemini(self, user_text: str) -> str:
        """Generate reply using fast Gemini Flash text API."""
        models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]

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
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.gemini_api_key}"
            )
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=6) as response:
                    result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue
                text = parts[0].get("text", "").strip()
                cleaned = text.replace("*", "").replace("\n", " ").strip()
                if cleaned:
                    return cleaned
            except Exception:
                continue

        return "Halt, little mortal! Reach into my bowl if you dare, but leave before I take your bones!"

    def _generate_groq(self, user_text: str) -> str:
        """Generate reply using Groq Llama 3.3 endpoint."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 100,
        }

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        choices = result.get("choices", [])
        if not choices:
            return ""

        text = choices[0].get("message", {}).get("content", "").strip()
        cleaned = text.replace("*", "").replace("\n", " ").strip()
        return cleaned
