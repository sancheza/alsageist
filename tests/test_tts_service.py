"""Unit tests for Piper TTS Service."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from services.tts_service import PiperTTSService


def test_piper_tts_synthesize_in_memory():
    model_path = "models/piper/en_GB-alan-medium.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Piper model not downloaded.")

    tts = PiperTTSService(model_path=model_path)
    audio_data, sample_rate = tts.synthesize("Boo!")

    assert isinstance(audio_data, np.ndarray)
    assert audio_data.dtype == np.float32
    assert sample_rate == 22050
    assert len(audio_data) > 0
    assert np.max(np.abs(audio_data)) <= 1.0


def test_piper_tts_missing_model():
    with pytest.raises(FileNotFoundError):
        PiperTTSService(model_path="nonexistent_model.onnx")


def test_piper_tts_synthesize_with_ellipsis_pauses():
    model_path = "models/piper/en_GB-alan-medium.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Piper model not downloaded.")

    tts = PiperTTSService(model_path=model_path, pause_duration_sec=0.5)
    text = "Hello mortal... take your candy... scram!"
    audio_data, sample_rate = tts.synthesize(text)

    assert isinstance(audio_data, np.ndarray)
    assert audio_data.dtype == np.float32
    assert sample_rate == 22050
    # Expected duration is speech + at least 2 x 0.5s pauses (1.0s of silence)
    assert len(audio_data) / sample_rate >= 2.0


def test_piper_tts_phonetic_normalization():
    model_path = "models/piper/en_GB-alan-medium.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Piper model not downloaded.")

    tts = PiperTTSService(model_path=model_path)
    # Ensure phrases with phonetic overrides synthesize without errors
    audio_data, sample_rate = tts.synthesize("Hasta la vista, baby! Tannhäuser Gate! Freddy Krueger! Mwahahaha!")

    assert isinstance(audio_data, np.ndarray)
    assert len(audio_data) > 0


def test_piper_tts_accepts_bare_voice_id():
    """Verify a bare voice ID resolves to the model under models/piper."""
    if not os.path.exists("models/piper/en_GB-alan-medium.onnx"):
        pytest.skip("Piper model not downloaded.")

    tts = PiperTTSService(model_path="en_GB-alan-medium")
    assert tts.model_path.endswith(os.path.join("models", "piper", "en_GB-alan-medium.onnx"))


class _FakeVoiceTTS(PiperTTSService):
    """PiperTTSService whose synthesis returns a deterministic clip without a Piper model."""

    synth_calls = 0

    def _synthesize_chunk(self, chunk_text):
        type(self).synth_calls += 1
        return np.full(1000, 0.1, dtype=np.float32)


@pytest.fixture
def make_tts(tmp_path):
    """Build fake-voice TTS services that share a temp cache directory."""
    model = tmp_path / "voice-a.onnx"
    model.write_text("model")
    other = tmp_path / "voice-b.onnx"
    other.write_text("model")
    _FakeVoiceTTS.synth_calls = 0

    def _make(model_name="voice-a.onnx", **kwargs):
        kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
        return _FakeVoiceTTS(model_path=str(tmp_path / model_name), **kwargs)

    return _make


def test_cache_key_includes_voice_and_settings(make_tts):
    """Different voice models or settings never share a cache file."""
    base = make_tts()._get_cache_path("hello")
    assert make_tts("voice-b.onnx")._get_cache_path("hello") != base
    assert make_tts(length_scale=1.5)._get_cache_path("hello") != base
    assert make_tts(noise_scale=0.3)._get_cache_path("hello") != base
    assert make_tts()._get_cache_path("hello") == base


def test_voice_change_does_not_reuse_other_voice_audio(make_tts):
    """A second voice synthesizes its own audio instead of reading the first voice's cache."""
    make_tts().synthesize("hello")
    make_tts("voice-b.onnx").synthesize("hello")
    assert _FakeVoiceTTS.synth_calls == 2


def test_disk_cache_hit_skips_synthesis(make_tts):
    """A second service instance reads the saved clip from disk."""
    make_tts().synthesize("hello")
    audio, rate = make_tts().synthesize("hello")
    assert _FakeVoiceTTS.synth_calls == 1
    assert len(audio) == 1000 and rate == 22050


def test_memory_cache_evicts_least_recently_used(make_tts):
    """The RAM cache keeps only max_memory_entries phrases, dropping the least recently used."""
    tts = make_tts(max_memory_entries=2)
    tts.synthesize("one")
    tts.synthesize("two")
    tts.synthesize("one")  # refresh "one"
    tts.synthesize("three")
    assert list(tts._cache) == ["one", "three"]


def test_disk_cache_prunes_oldest_files(make_tts, tmp_path):
    """The disk cache deletes the least recently used files beyond its size limit."""
    clip_bytes = 1000 * 4 + 128  # float32 samples plus .npy header
    tts = make_tts(max_disk_cache_mb=(clip_bytes * 2.5) / (1024 * 1024))
    paths = []
    for index, phrase in enumerate(["one", "two", "three"]):
        tts.synthesize(phrase)
        path = tts._get_cache_path(phrase)
        os.utime(path, (1000 + index, 1000 + index))  # deterministic age order
        paths.append(path)
        tts._prune_disk_cache()

    remaining = [p for p in paths if os.path.exists(p)]
    assert paths[0] not in remaining
    assert paths[2] in remaining


def test_corrupt_disk_cache_file_is_resynthesized(make_tts):
    """A truncated cache file is ignored and the phrase is synthesized again."""
    tts = make_tts()
    path = tts._get_cache_path("hello")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not a numpy file")

    audio, _ = tts.synthesize("hello")
    assert len(audio) == 1000
    assert _FakeVoiceTTS.synth_calls == 1


def _subprocess_tts(tmp_path):
    """Return a service whose in-process voice failed to load, so it uses the piper command."""
    model = tmp_path / "voice.onnx"
    model.write_text("model")
    tts = PiperTTSService(model_path=str(model), cache_dir=str(tmp_path / "cache"))
    tts._voice = None
    tts._load_error = ImportError("No module named 'piper'")
    return tts


def test_piper_load_failure_is_reported(tmp_path, capsys):
    """A voice that cannot load in-process says why before falling back to the command."""
    model = tmp_path / "voice.onnx"
    model.write_text("model")
    with patch.dict("sys.modules", {"piper": None}):  # makes `import piper` fail
        PiperTTSService(model_path=str(model), cache_dir=str(tmp_path / "cache"))
    out = capsys.readouterr().out
    assert "could not load" in out and "falling back to the 'piper' command" in out


def test_piper_missing_command_error_names_both_failures(tmp_path):
    """With no in-process voice and no piper command the error explains both."""
    tts = _subprocess_tts(tmp_path)
    with patch("services.tts_service.subprocess.Popen", side_effect=FileNotFoundError("piper")):
        with pytest.raises(RuntimeError) as exc:
            tts._synthesize_chunk("hello")
    assert "No module named 'piper'" in str(exc.value)
    assert "'piper' command could not be run" in str(exc.value)


def test_piper_command_failure_includes_stderr(tmp_path):
    """A failing piper command reports its exit code and error output."""
    tts = _subprocess_tts(tmp_path)
    process = MagicMock(returncode=2)
    process.communicate.return_value = (b"", b"Error: invalid model config")
    with patch("services.tts_service.subprocess.Popen", return_value=process):
        with pytest.raises(RuntimeError, match=r"exit code 2\): Error: invalid model config"):
            tts._synthesize_chunk("hello")


# ---- lazy voice loading ----

import sys
import types


@pytest.fixture
def fake_piper(monkeypatch):
    """Install a fake piper module and return its PiperVoice.load mock."""
    module = types.ModuleType("piper")
    module.PiperVoice = MagicMock()
    module.PiperVoice.load.return_value.synthesize.return_value = []
    module.SynthesisConfig = MagicMock()
    monkeypatch.setitem(sys.modules, "piper", module)
    return module.PiperVoice.load


def _model(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_text("model")
    return str(model)


def test_eager_loading_is_the_default(fake_piper, tmp_path):
    """The default constructor loads the voice immediately, as the service needs."""
    tts = PiperTTSService(model_path=_model(tmp_path), cache_dir=str(tmp_path / "c"))
    fake_piper.assert_called_once()
    assert tts.voice_loaded is True


def test_lazy_loading_defers_the_voice_until_a_cache_miss(fake_piper, tmp_path):
    """A lazy service loads nothing at construction and nothing for a cached phrase."""
    cache_dir = str(tmp_path / "c")
    warm = PiperTTSService(model_path=_model(tmp_path), cache_dir=cache_dir)
    warm._synthesize_chunk = lambda text: np.full(100, 0.1, dtype=np.float32)
    warm.synthesize("cached phrase")
    fake_piper.reset_mock()

    lazy = PiperTTSService(model_path=_model(tmp_path), cache_dir=cache_dir, lazy_load=True)
    assert lazy.voice_loaded is False
    audio, _ = lazy.synthesize("cached phrase")

    assert len(audio) == 100
    fake_piper.assert_not_called()
    assert lazy.voice_loaded is False and lazy.voice_load_sec == 0.0


def test_lazy_loading_loads_once_on_first_uncached_phrase(fake_piper, tmp_path):
    """The first uncached phrase loads the voice; later phrases reuse it."""
    lazy = PiperTTSService(model_path=_model(tmp_path), cache_dir=str(tmp_path / "c"), lazy_load=True)
    fake_piper.assert_not_called()

    lazy.synthesize("first new phrase")
    lazy.synthesize("second new phrase")

    fake_piper.assert_called_once()
    assert lazy.voice_loaded is True
    assert lazy.voice_load_sec > 0.0


def test_lazy_load_failure_falls_back_to_command_once(tmp_path, monkeypatch, capsys):
    """A voice that cannot load is reported once and the piper command is used."""
    monkeypatch.setitem(sys.modules, "piper", None)  # `import piper` raises ImportError
    lazy = PiperTTSService(model_path=_model(tmp_path), cache_dir=str(tmp_path / "c"), lazy_load=True)
    assert capsys.readouterr().out == ""  # nothing printed until the voice is needed

    with patch("services.tts_service.subprocess.Popen", side_effect=FileNotFoundError("piper")):
        for _ in range(2):
            with pytest.raises(RuntimeError):
                lazy._synthesize_chunk("hello")
    assert capsys.readouterr().out.count("could not load") == 1


def test_piper_cli_timeout_kills_process(tmp_path, monkeypatch):
    """Piper CLI fallback raises RuntimeError when process times out."""
    import subprocess
    import sys

    monkeypatch.setitem(sys.modules, "piper", None)
    lazy = PiperTTSService(model_path=_model(tmp_path), cache_dir=str(tmp_path / "c"), lazy_load=True)

    mock_proc = MagicMock()
    mock_proc.communicate.side_effect = [subprocess.TimeoutExpired(cmd="piper", timeout=10.0), (b"", b"")]

    with patch("services.tts_service.subprocess.Popen", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="timed out"):
            lazy._synthesize_chunk("hello")
        mock_proc.kill.assert_called_once()


def test_tts_service_ram_cache_hit_updates_disk_mtime(tmp_path):
    """Accessing an item from RAM cache updates its disk file mtime."""
    import time

    cache_dir = str(tmp_path / "cache")
    tts = PiperTTSService(model_path=_model(tmp_path), cache_dir=cache_dir)
    tts._synthesize_chunk = lambda text: np.full(100, 0.1, dtype=np.float32)

    # First synthesis writes to disk and stores in RAM
    tts.synthesize("hello world")
    cache_path = tts._get_cache_path("hello world")
    assert os.path.isfile(cache_path)

    # Set old mtime on disk
    old_time = time.time() - 3600
    os.utime(cache_path, (old_time, old_time))
    assert os.stat(cache_path).st_mtime < time.time() - 100

    # RAM hit must update disk mtime
    tts.synthesize("hello world")
    assert os.stat(cache_path).st_mtime > time.time() - 10


