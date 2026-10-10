"""Unit tests for Voice Activity Detector."""

import numpy as np
import pytest
from core.vad import VoiceActivityDetector


def test_vad_initialization():
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    assert vad.frame_size == 480
    assert not vad.is_speaking


def test_calculate_rms_silence():
    vad = VoiceActivityDetector()
    silence_frame = np.zeros(480, dtype=np.float32)
    rms = vad.calculate_rms(silence_frame)
    assert rms == 0.0


def test_calculate_rms_signal():
    vad = VoiceActivityDetector()
    sine_wave = np.sin(np.linspace(0, 2 * np.pi, 480)).astype(np.float32) * 0.5
    rms = vad.calculate_rms(sine_wave)
    assert rms > 0.3


def test_vad_speech_detection_and_completion():
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        energy_threshold_multiplier=2.0,
        min_speech_frames=2,
        silence_timeout_sec=0.1,  # Fast timeout for test
        min_energy_threshold=0.01,
        min_active_speech_sec=0.3,
    )

    silence = np.zeros(480, dtype=np.float32)
    tone = (np.sin(np.linspace(0, 2 * np.pi, 480)) * 0.5).astype(np.float32)

    # Ambient silence
    for _ in range(5):
        is_speech, completed, _ = vad.process_frame(silence)
        assert not is_speech
        assert not completed

    # Speech onset (20 frames of tone = 0.6s to exceed min duration 0.3s)
    for _ in range(20):
        vad.process_frame(tone)
    assert vad.is_speaking

    # Trailing silence triggering completion
    completed_utterance = None
    for _ in range(10):
        is_speech, completed, utterance = vad.process_frame(silence)
        if completed:
            completed_utterance = utterance
            break

    assert completed_utterance is not None
    assert len(completed_utterance) > 0
    assert not vad.is_speaking


def test_vad_rejects_brief_transient_noise():
    """Verify that transient background clicks (< min active speech) are rejected."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        energy_threshold_multiplier=2.0,
        min_speech_frames=2,
        silence_timeout_sec=0.1,
        min_energy_threshold=0.02,
        min_active_speech_sec=0.5,  # Requires at least ~17 frames of speech
    )

    silence = np.zeros(480, dtype=np.float32)
    click = (np.sin(np.linspace(0, 2 * np.pi, 480)) * 0.5).astype(np.float32)

    # Only 3 frames of click (90ms)
    for _ in range(3):
        vad.process_frame(click)

    # Trailing silence
    completed_utterance = None
    for _ in range(10):
        _, completed, utterance = vad.process_frame(silence)
        if completed:
            completed_utterance = utterance

    # Should be rejected because total active speech was only 90ms
    assert completed_utterance is None
    assert not vad.is_speaking
