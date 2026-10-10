"""Unit tests for the InteractionLogger module."""

import json
import os
import shutil
import tempfile
import pytest
from core.logger import InteractionLogger


def test_interaction_logger_writes_jsonl_and_text():
    temp_dir = tempfile.mkdtemp()
    try:
        logger = InteractionLogger(log_dir=temp_dir)
        record = logger.log_interaction(
            transcribed_text="How old are you?",
            response_text="Two hundred years!",
            engine="intent",
            matched_intent="age_and_death",
            audio_duration_sec=1.5,
            audio_rms=0.025,
            latencies_ms={"stt": 250, "dialogue": 2, "tts": 100, "total": 352},
            status="PROCESSED",
        )

        assert record["turn_id"] == 1
        assert record["transcribed_text"] == "How old are you?"
        assert record["matched_intent"] == "age_and_death"

        # Verify JSONL file exists and is valid JSON
        assert os.path.exists(logger.jsonl_path)
        with open(logger.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["transcribed_text"] == "How old are you?"
        assert data["response_text"] == "Two hundred years!"

        # Verify human-readable text log
        assert os.path.exists(logger.text_path)
        with open(logger.text_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "How old are you?" in content
        assert "Two hundred years!" in content
    finally:
        shutil.rmtree(temp_dir)
