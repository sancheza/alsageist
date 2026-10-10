"""Structured conversation and interaction logger for AlsaGeist.

Records every audio capture, transcription, dialogue match, and response
to JSONL and plain-text log files for dynamic evaluation and fine-tuning.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, Optional


class InteractionLogger:
    """Logs conversation events to JSONL and human-readable text files."""

    def __init__(self, log_dir: str = "logs") -> None:
        """Initialize the interaction logger.

        Args:
            log_dir: Directory where log files are stored.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.log_dir, "interactions.jsonl")
        self.text_path = os.path.join(self.log_dir, "conversations.log")
        self._turn_id = 0

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
            response_text: What Grimjaw replied.
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
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[LOGGER ERROR] Failed writing JSONL: {exc}")

        # Append to human-readable text log
        try:
            with open(self.text_path, "a", encoding="utf-8") as f:
                if status == "PROCESSED":
                    f.write(
                        f"[{timestamp}] [Turn {self._turn_id}] "
                        f"User: \"{transcribed_text}\" | "
                        f"Grimjaw ({matched_intent}): \"{response_text}\" | "
                        f"Total: {latencies_ms.get('total', 0)}ms\n"
                    )
                else:
                    f.write(
                        f"[{timestamp}] [Turn {self._turn_id}] "
                        f"[{status}] RMS: {audio_rms:.4f}, Raw: \"{transcribed_text}\"\n"
                    )
        except Exception as exc:
            print(f"[LOGGER ERROR] Failed writing log: {exc}")

        return record
