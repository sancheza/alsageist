"""Structured conversation and interaction logger for AlsaGeist.

Records every audio capture, transcription, dialogue match, and response
to JSONL and plain-text log files for dynamic evaluation and fine-tuning.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict

DEFAULT_MAX_MB = 5.0
DEFAULT_BACKUPS = 3
REDACTED_TEXT = "[not stored]"


class InteractionLogger:
    """Logs conversation events to JSONL and human-readable text files."""

    def __init__(
        self,
        log_dir: str = "logs",
        persona_name: str = "Assistant",
        max_bytes: int = int(DEFAULT_MAX_MB * 1024 * 1024),
        backup_count: int = DEFAULT_BACKUPS,
        store_text: bool = True,
    ) -> None:
        """Initialize the interaction logger.

        Args:
            log_dir: Directory where log files are stored.
            persona_name: Display name of the persona written to the text log.
            max_bytes: Size at which a log file is rotated; 0 or less disables rotation.
            backup_count: Rotated copies kept per log file (name.1 is the newest).
            store_text: If False, visitor and persona text is replaced by a placeholder
                and only timing, intent and audio statistics are recorded.
        """
        self.log_dir = log_dir
        self.persona_name = persona_name
        self.max_bytes = max_bytes
        self.backup_count = max(0, backup_count)
        self.store_text = store_text
        os.makedirs(self.log_dir, mode=0o700, exist_ok=True)

        self.jsonl_path = os.path.join(self.log_dir, "interactions.jsonl")
        self.text_path = os.path.join(self.log_dir, "conversations.log")
        self._turn_id = 0

    @classmethod
    def from_env(cls, log_dir: str, persona_name: str = "Assistant") -> "InteractionLogger":
        """Create a logger configured from the environment.

        Reads LOG_MAX_MB (default 5), LOG_BACKUPS (default 3) and LOG_TRANSCRIPTS
        (default true; set false to stop storing what visitors and the persona say).

        Args:
            log_dir: Directory where log files are stored.
            persona_name: Display name of the persona written to the text log.

        Returns:
            Configured InteractionLogger.
        """
        store_text = os.environ.get("LOG_TRANSCRIPTS", "true").strip().lower() not in ("false", "0", "no")
        return cls(
            log_dir=log_dir,
            persona_name=persona_name,
            max_bytes=int(float(os.environ.get("LOG_MAX_MB", DEFAULT_MAX_MB)) * 1024 * 1024),
            backup_count=int(os.environ.get("LOG_BACKUPS", DEFAULT_BACKUPS)),
            store_text=store_text,
        )

    def _rotate_if_needed(self, path: str) -> None:
        """Rotate a log file that reached max_bytes, keeping backup_count older copies."""
        if self.max_bytes <= 0:
            return
        try:
            if os.path.getsize(path) < self.max_bytes:
                return
        except OSError:
            return

        if self.backup_count == 0:
            os.remove(path)
            return
        for index in range(self.backup_count - 1, 0, -1):
            older = f"{path}.{index}"
            if os.path.exists(older):
                os.replace(older, f"{path}.{index + 1}")
        os.replace(path, f"{path}.1")

    def _append(self, path: str, text: str) -> None:
        """Append text to a log file that only the current user can read."""
        self._rotate_if_needed(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(text)

    def log_interaction(
        self,
        transcribed_text: str,
        response_text: str,
        engine: str,
        matched_intent: str,
        audio_duration_sec: float,
        audio_rms: float,
        latencies_ms: Dict[str, int],
        status: str = "PROCESSED",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record an interaction event.

        Args:
            transcribed_text: What the STT engine transcribed.
            response_text: What the persona replied.
            engine: 'intent' or 'llm'.
            matched_intent: Specific intent rule name or category.
            audio_duration_sec: Duration of captured PCM audio.
            audio_rms: Root Mean Square energy of captured PCM audio.
            latencies_ms: Dict of latency measurements (stt, dialogue, tts, total).
            status: 'PROCESSED', 'FILTERED_SILENCE', 'FILTERED_ARTIFACT', 'ERROR'.
            notes: Optional diagnostic notes.

        Returns:
            The structured record dictionary.
        """
        self._turn_id += 1
        timestamp = datetime.now().isoformat()
        if not self.store_text:
            transcribed_text = REDACTED_TEXT if transcribed_text else ""
            response_text = REDACTED_TEXT if response_text else ""

        record = {
            "turn_id": self._turn_id,
            "timestamp": timestamp,
            "status": status,
            "engine": engine,
            "audio_duration_sec": round(audio_duration_sec, 3),
            "audio_rms": round(audio_rms, 5),
            "transcribed_text": transcribed_text,
            "matched_intent": matched_intent,
            "response_text": response_text,
            "latencies_ms": latencies_ms,
            "notes": notes,
        }

        # Append to JSONL
        try:
            self._append(self.jsonl_path, json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[LOGGER ERROR] Failed writing JSONL: {exc}")

        # Append to human-readable text log
        if status == "PROCESSED":
            line = (
                f"[{timestamp}] [Turn {self._turn_id}] "
                f"User: \"{transcribed_text}\" | "
                f"{self.persona_name} ({matched_intent}): \"{response_text}\" | "
                f"Total: {latencies_ms.get('total', 0)}ms\n"
            )
        else:
            line = (
                f"[{timestamp}] [Turn {self._turn_id}] "
                f"[{status}] RMS: {audio_rms:.4f}, Raw: \"{transcribed_text}\"\n"
            )
        try:
            self._append(self.text_path, line)
        except OSError as exc:
            print(f"[LOGGER ERROR] Failed writing log: {exc}")

        return record
