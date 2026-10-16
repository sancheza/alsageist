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


def _log(logger, text="hello", reply="halt"):
    return logger.log_interaction(
        transcribed_text=text,
        response_text=reply,
        engine="intent",
        matched_intent="greeting",
        audio_duration_sec=1.0,
        audio_rms=0.02,
        latencies_ms={"total": 10},
    )


def test_logger_files_are_owner_only(tmp_path):
    """Log files contain visitor speech, so only the owner can read them."""
    import stat

    logger = InteractionLogger(log_dir=str(tmp_path / "logs"))
    _log(logger)
    for path in (logger.jsonl_path, logger.text_path):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(logger.log_dir).st_mode) == 0o700


def test_logger_rotates_and_keeps_limited_backups(tmp_path):
    """A full log file is rotated and only backup_count older copies are kept."""
    logger = InteractionLogger(log_dir=str(tmp_path), max_bytes=400, backup_count=2)
    for index in range(40):
        _log(logger, text=f"utterance {index}")

    names = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("interactions.jsonl"))
    assert names == ["interactions.jsonl", "interactions.jsonl.1", "interactions.jsonl.2"]
    for name in names:
        assert (tmp_path / name).stat().st_size < 400 + 600  # a file can pass the limit by one record
    newest = [json.loads(l) for l in (tmp_path / "interactions.jsonl").read_text().splitlines()]
    assert newest[-1]["transcribed_text"] == "utterance 39"


def test_logger_rotation_with_no_backups_discards_old_data(tmp_path):
    """backup_count=0 deletes a full file instead of keeping a copy."""
    logger = InteractionLogger(log_dir=str(tmp_path), max_bytes=300, backup_count=0)
    for index in range(30):
        _log(logger, text=f"utterance {index}")
    assert sorted(p.name for p in tmp_path.iterdir() if "interactions" in p.name) == ["interactions.jsonl"]


def test_logger_can_skip_storing_text(tmp_path):
    """With store_text=False neither file contains what was said, but timings and intents remain."""
    logger = InteractionLogger(log_dir=str(tmp_path), store_text=False)
    record = _log(logger, text="my name is Sam", reply="Hello Sam")

    assert record["transcribed_text"] == "[not stored]"
    assert record["response_text"] == "[not stored]"
    assert record["matched_intent"] == "greeting"
    combined = (tmp_path / "interactions.jsonl").read_text() + (tmp_path / "conversations.log").read_text()
    assert "Sam" not in combined
    assert '"total": 10' in combined


def test_logger_from_env(tmp_path, monkeypatch):
    """LOG_MAX_MB, LOG_BACKUPS and LOG_TRANSCRIPTS configure the logger."""
    monkeypatch.setenv("LOG_MAX_MB", "2")
    monkeypatch.setenv("LOG_BACKUPS", "5")
    monkeypatch.setenv("LOG_TRANSCRIPTS", "false")
    logger = InteractionLogger.from_env(str(tmp_path))
    assert (logger.max_bytes, logger.backup_count, logger.store_text) == (2 * 1024 * 1024, 5, False)

    monkeypatch.delenv("LOG_TRANSCRIPTS")
    assert InteractionLogger.from_env(str(tmp_path)).store_text is True
