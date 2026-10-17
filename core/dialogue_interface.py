"""Dialogue engine protocol defining the standard conversational reasoning interface."""

from typing import Protocol, Tuple, runtime_checkable


@runtime_checkable
class DialogueEngine(Protocol):
    """Protocol for conversational dialogue engines (intent-based or LLM-based)."""

    def generate_response(self, user_text: str) -> str:
        """Generate a response text given visitor user text."""
        ...

    def classify_and_respond(self, user_text: str) -> Tuple[str, str]:
        """Classify user intent and return a response tuple: (response_text, matched_intent)."""
        ...

    def reset_conversation(self) -> None:
        """Reset conversation context / history for the next visitor."""
        ...
