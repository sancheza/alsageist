"""Unit tests for Piper TTS Service."""

import os
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
