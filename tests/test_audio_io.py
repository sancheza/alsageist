"""Unit tests for AudioIO cross-platform capture, playback, and resampling."""

import numpy as np
import pytest
from core.audio_io import AudioIO


def test_audio_resampling():
    """Verify in-memory linear interpolation resampling from 22.05kHz to 44.1kHz."""
    orig_sr = 22050
    target_sr = 44100
    duration_sec = 0.5
    num_samples = int(orig_sr * duration_sec)

    # 440 Hz test sine wave
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    sine_wave = np.sin(2 * np.pi * 440 * t)

    resampled = AudioIO.resample_audio(sine_wave, orig_sr, target_sr)

    assert len(resampled) == int(target_sr * duration_sec)
    assert resampled.dtype == np.float32
    assert np.max(np.abs(resampled)) <= 1.05


def test_audio_resampling_identical_rate():
    """Verify that resampling to the identical sample rate returns original data."""
    data = np.ones(100, dtype=np.float32)
    result = AudioIO.resample_audio(data, 16000, 16000)
    assert np.array_equal(data, result)


def test_audio_io_mute_unmute_state():
    """Verify mute_capture and unmute_capture state toggles and queue behavior."""
    audio = AudioIO(sample_rate=16000, channels=1)
    assert not audio._is_muted

    audio.mute_capture()
    assert audio._is_muted

    # Simulate callback frame arriving while muted
    dummy_frame = np.ones((480, 1), dtype=np.float32)
    audio._is_capturing = True
    audio._input_callback(dummy_frame, 480, {}, 0)

    # Queue must remain empty because audio is muted
    assert audio.read_frame(timeout=0.01) is None

    # Unmute and send frame
    audio.unmute_capture()
    assert not audio._is_muted
    audio._input_callback(dummy_frame, 480, {}, 0)

    received_frame = audio.read_frame(timeout=0.1)
    assert received_frame is not None
    assert len(received_frame) == 480


def test_audio_io_stereo_downmix_callback():
    """Verify that multi-channel input (AIY Voice HAT stereo Knowles MEMS) is downmixed to mono."""
    audio = AudioIO(sample_rate=16000, channels=2)
    audio._is_capturing = True
    audio._is_muted = False

    # Stereo frame where Left = 0.5, Right = -0.5 -> Mean = 0.0
    stereo_frame = np.zeros((480, 2), dtype=np.float32)
    stereo_frame[:, 0] = 0.5
    stereo_frame[:, 1] = -0.5

    audio._input_callback(stereo_frame, 480, {}, 0)

    mono_result = audio.read_frame(timeout=0.1)
    assert mono_result is not None
    assert mono_result.ndim == 1
    assert len(mono_result) == 480
    assert np.allclose(mono_result, 0.0, atol=1e-5)


def test_audio_io_mic_gain():
    """Verify that input audio callback scales incoming samples by mic_gain."""
    audio = AudioIO(sample_rate=16000, channels=1, mic_gain=2.5)
    audio._is_capturing = True
    audio._is_muted = False

    raw_frame = np.full((480, 1), 0.2, dtype=np.float32)
    audio._input_callback(raw_frame, 480, {}, 0)

    result = audio.read_frame(timeout=0.1)
    assert result is not None
    assert np.allclose(result, 0.5, atol=1e-5)


def test_audio_io_volume_clamping_and_scaling(monkeypatch):
    """Verify that volume attribute clamps between 0.0 and 1.0 and scales output."""
    audio = AudioIO(sample_rate=16000, channels=1, volume=1.5)
    assert audio.volume == 1.0

    audio_low = AudioIO(sample_rate=16000, channels=1, volume=-0.2)
    assert audio_low.volume == 0.0

    audio_normal = AudioIO(sample_rate=16000, channels=1, volume=0.8)
    assert audio_normal.volume == 0.8



def test_audio_io_remembers_working_output_rate(monkeypatch):
    """After 22050 Hz is rejected once, later clips go straight to 44100 Hz."""
    audio = AudioIO(sample_rate=16000, channels=1, volume=1.0)
    played_rates = []

    def fake_play(data, samplerate, device, blocking):
        if samplerate == 22050:
            raise RuntimeError("Invalid sample rate")
        played_rates.append(samplerate)

    monkeypatch.setattr("core.audio_io.sd.play", fake_play)
    monkeypatch.setattr("core.audio_io.sd.wait", lambda: None)
    clip = np.zeros(2205, dtype=np.float32)

    audio.play_audio(clip, sample_rate=22050)
    audio.play_audio(clip, sample_rate=22050)

    assert played_rates == [44100, 44100]
    assert audio._output_rate_fallbacks == {22050: 44100}


def test_audio_io_non_22050_rate_error_propagates(monkeypatch):
    """A rejected rate other than 22050 Hz raises instead of resampling."""
    audio = AudioIO(sample_rate=16000, channels=1)
    monkeypatch.setattr(
        "core.audio_io.sd.play",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad device")),
    )
    with pytest.raises(RuntimeError):
        audio.play_audio(np.zeros(160, dtype=np.float32), sample_rate=16000)


def test_audio_io_drain_input_queue():
    """drain_input_queue discards every buffered frame."""
    audio = AudioIO(sample_rate=16000, channels=1)
    audio._is_capturing = True
    frame = np.ones((480, 1), dtype=np.float32)
    for _ in range(5):
        audio._input_callback(frame, 480, {}, 0)

    audio.drain_input_queue()
    assert audio.read_frame(timeout=0.01) is None


def test_audio_io_lock_is_reentrant():
    """AudioIO._lock is re-entrant to prevent signal-handler deadlocks."""
    import threading

    audio = AudioIO(sample_rate=16000, channels=1)
    assert isinstance(audio._lock, type(threading.RLock()))
    with audio._lock:
        # Re-entering while holding the lock must not deadlock
        audio.stop_capture()
        audio.drain_input_queue()


def test_audio_io_input_queue_is_bounded():
    """_input_queue has a maxsize and drops frames without raising when full."""
    audio = AudioIO(sample_rate=16000, channels=1)
    assert audio._input_queue.maxsize == 128
    audio._is_capturing = True
    frame = np.ones((480, 1), dtype=np.float32)
    for _ in range(200):
        audio._input_callback(frame, 480, {}, 0)
    assert audio._input_queue.qsize() == 128


def test_audio_io_failed_stream_closed_before_stereo_retry():
    """When mono stream fails, it is closed before retrying stereo."""
    from unittest.mock import MagicMock, patch

    audio = AudioIO(sample_rate=16000, channels=1)
    mock_mono_stream = MagicMock()
    mock_mono_stream.start.side_effect = RuntimeError("Mono rejected")
    mock_stereo_stream = MagicMock()

    streams = [mock_mono_stream, mock_stereo_stream]

    with patch("core.audio_io.sd.InputStream", side_effect=streams):
        audio.start_capture()
        mock_mono_stream.close.assert_called_once()
        assert audio.channels == 2
        mock_stereo_stream.start.assert_called_once()
        assert audio.is_capturing is True


def test_audio_io_stereo_failure_resets_capturing_and_stream():
    """When stereo stream fails, _is_capturing is False, stream is None, and retry is allowed."""
    from unittest.mock import MagicMock, patch

    audio = AudioIO(sample_rate=16000, channels=1)
    mock_mono = MagicMock()
    mock_mono.start.side_effect = RuntimeError("Mono rejected")
    mock_stereo = MagicMock()
    mock_stereo.start.side_effect = RuntimeError("Stereo rejected")

    with patch("core.audio_io.sd.InputStream", side_effect=[mock_mono, mock_stereo]):
        with pytest.raises(RuntimeError, match="Stereo rejected"):
            audio.start_capture()

    assert audio.is_capturing is False
    assert audio._input_stream is None
    mock_stereo.close.assert_called_once()

    # Retry with working stream succeeds rather than returning early on a dead stream
    working_stream = MagicMock()
    with patch("core.audio_io.sd.InputStream", return_value=working_stream):
        audio.start_capture()
        assert audio.is_capturing is True
        assert audio._input_stream is working_stream
        working_stream.start.assert_called_once()


def test_audio_io_initial_stereo_failure_resets_capturing():
    """When initial channels=2 fails, _is_capturing is False and stream is None."""
    from unittest.mock import MagicMock, patch

    audio = AudioIO(sample_rate=16000, channels=2)
    mock_stream = MagicMock()
    mock_stream.start.side_effect = RuntimeError("Stereo failed")

    with patch("core.audio_io.sd.InputStream", return_value=mock_stream):
        with pytest.raises(RuntimeError, match="Stereo failed"):
            audio.start_capture()

    assert audio.is_capturing is False
    assert audio._input_stream is None
    mock_stream.close.assert_called_once()


