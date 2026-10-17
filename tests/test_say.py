"""Unit tests for scripts/say.py argument parsing and personality resolution."""

import os
import re
import subprocess
import sys
import time
import pytest


def test_say_help():
    """Verify say.py responds to --help with personality and options."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    res = subprocess.run(
        [sys.executable, "scripts/say.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "--personality" in res.stdout
    assert "--model-path" in res.stdout
    assert "--tempo" in res.stdout
    assert "--volume" in res.stdout


def test_say_empty_text():
    """Verify say.py rejects empty text."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    res = subprocess.run(
        [sys.executable, "scripts/say.py", "   "],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "Error: Spoken text cannot be empty." in res.stderr


# ---- --debug phase timing ----

from unittest.mock import MagicMock, patch

import numpy as np

from scripts import say


def test_say_help_lists_debug():
    """--debug appears in the help text."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run(
        [sys.executable, "scripts/say.py", "--help"],
        cwd=repo_root, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": repo_root},
    )
    assert "--debug" in res.stdout


def test_phase_timer_records_phases_and_shares():
    """The table lists each phase with its duration and percentage of the total."""
    timer = say.PhaseTimer()
    timer.add("alpha", 3.0, "first")
    timer.add("beta", 1.0)
    text = timer.render()

    assert "Timing (--debug):" in text
    assert "alpha" in text and "3.00 s" in text and " 75.0%" in text and "first" in text
    assert "beta" in text and " 25.0%" in text
    assert "total" in text and "4.00 s" in text


def test_phase_timer_context_manager_measures_elapsed_time():
    """phase() records the time spent in the block, even when the block raises."""
    timer = say.PhaseTimer()
    with timer.phase("sleepy"):
        time.sleep(0.05)
    with pytest.raises(RuntimeError):
        with timer.phase("boom"):
            raise RuntimeError("fail")

    names = [name for name, _, _ in timer._phases]
    assert names == ["sleepy", "boom"]
    assert timer._phases[0][1] >= 0.05


def _run_main(monkeypatch, argv, tts_side_effect=None, voice_loaded=False, voice_load_sec=0.0):
    tts = MagicMock()
    tts.cache_path.return_value = "/nonexistent/cache.npy"
    tts.voice_loaded = voice_loaded
    tts.voice_load_sec = voice_load_sec
    if tts_side_effect:
        tts.synthesize.side_effect = tts_side_effect
    else:
        tts.synthesize.return_value = (np.zeros(22050, dtype=np.float32), 22050)
    monkeypatch.setattr(sys, "argv", ["say.py"] + argv)
    with patch.object(say, "PiperTTSService", return_value=tts), \
         patch.object(say, "AudioIO") as audio_cls:
        return say.main(), tts, audio_cls.return_value


def test_say_debug_prints_every_phase(monkeypatch, capsys):
    """--debug prints a timing line for each phase of a successful run."""
    code, _, audio = _run_main(monkeypatch, ["--debug", "hello"])
    out = capsys.readouterr().out

    assert code == 0
    for label in ("python imports", "load personality", "resolve voice", "init audio",
                  "load voice model", "synthesize", "playback", "total"):
        assert label in out, label
    assert "synthesized (cache miss)" in out
    assert "1.00 s of audio" in out
    audio.play_audio.assert_called_once()


def test_say_without_debug_prints_no_timing(monkeypatch, capsys):
    """Timing output appears only with --debug."""
    code, _, _ = _run_main(monkeypatch, ["hello"])
    assert code == 0
    assert "Timing (--debug)" not in capsys.readouterr().out


def test_say_debug_reports_cache_hit(monkeypatch, capsys):
    """The synthesize line says when the clip came from the disk cache."""
    with patch("scripts.say.os.path.isfile", return_value=True):
        _run_main(monkeypatch, ["--debug", "hello"])
    assert "disk cache hit" in capsys.readouterr().out


def test_say_debug_prints_partial_timings_on_failure(monkeypatch, capsys):
    """A failure still prints the phases that completed before it."""
    code, _, _ = _run_main(monkeypatch, ["--debug", "hello"], tts_side_effect=RuntimeError("piper failed"))
    captured = capsys.readouterr()

    assert code == 1
    assert "piper failed" in captured.err
    assert "load voice model" in captured.out
    assert "synthesize" in captured.out
    assert "playback" not in captured.out


def test_say_creates_tts_with_lazy_voice_loading(monkeypatch):
    """say.py asks for lazy loading so a cached phrase never pays for the voice model."""
    monkeypatch.setattr(sys, "argv", ["say.py", "hello"])
    with patch.object(say, "PiperTTSService") as tts_cls, patch.object(say, "AudioIO"):
        tts_cls.return_value.synthesize.return_value = (np.zeros(2205, dtype=np.float32), 22050)
        tts_cls.return_value.voice_load_sec = 0.0
        tts_cls.return_value.voice_loaded = False
        say.main()
    assert tts_cls.call_args.kwargs["lazy_load"] is True


def test_say_debug_reports_skipped_voice_load_for_cached_phrase(monkeypatch, capsys):
    """A cached phrase shows the voice model as skipped."""
    _run_main(monkeypatch, ["--debug", "hello"], voice_loaded=False)
    out = capsys.readouterr().out
    assert "load voice model" in out and "skipped (phrase was cached)" in out


def test_say_debug_separates_voice_load_from_synthesis(monkeypatch, capsys):
    """When the voice loads during synthesis its time is reported on its own line."""
    def slow_synth(_text):
        time.sleep(0.3)
        return np.zeros(22050, dtype=np.float32), 22050

    tts = MagicMock()
    tts.cache_path.return_value = "/nonexistent/cache.npy"
    tts.voice_loaded = True
    tts.voice_load_sec = 0.25
    tts.synthesize.side_effect = slow_synth
    monkeypatch.setattr(sys, "argv", ["say.py", "--debug", "hello"])
    with patch.object(say, "PiperTTSService", return_value=tts), patch.object(say, "AudioIO"):
        say.main()

    out = capsys.readouterr().out
    load = re.search(r"load voice model\s+(\d+\.\d+) s.*en_GB-alan-medium\.onnx", out)
    synth = re.search(r"synthesize\s+(\d+\.\d+) s", out)
    assert load and float(load.group(1)) == 0.25
    # About 0.3 s was spent in synthesize(); the 0.25 s of voice loading is reported separately
    assert synth and 0.0 <= float(synth.group(1)) < 0.2
