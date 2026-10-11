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
