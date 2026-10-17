"""Unit tests validating refactoring fixes, security bounds, and edge cases."""

import io
import math
import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np
import pytest

from core import cloud_api, http_client
from core.audio_io import AudioIO
from core.control_server import ControlServer
from core.intent_builder import PersonalityProfile
from core.led_controller import LEDState, RaspberryPiLEDController
from core.system_utils import load_env_file
from core.vad import VoiceActivityDetector
from services.agent import AlsaGeistAgent
from services.basic_info_service import BasicInfoService
from services.intent_service import PersonaIntentEngine
from services.tts_service import PiperTTSService
from tests.agent_helpers import make_bare_agent


def test_piper_tts_concurrency_lock():
    """Verify PiperTTSService initializes an RLock for thread-safe synthesis."""
    with patch("os.path.exists", return_value=True):
        with patch("services.tts_service.PiperVoice", create=True):
            tts = PiperTTSService(model_path="dummy.onnx", lazy_load=True)
            assert hasattr(tts, "_lock")
            assert isinstance(tts._lock, type(threading.RLock()))

            # Verify lock acquires cleanly during synthesize
            with patch.object(tts, "_synthesize_chunk", return_value=np.zeros(100, dtype=np.float32)):
                samples, sr = tts.synthesize("test lock")
                assert len(samples) == 100


def test_control_server_semaphore_bounding():
    """Verify ControlServer uses a BoundedSemaphore to limit concurrent clients."""
    agent = make_bare_agent()
    server = ControlServer(agent=agent, socket_path="/tmp/test_alsageist_bound.sock")
    assert hasattr(server, "_client_semaphore")
    assert server._client_semaphore._value == 8

    # Simulate client connection acquisition and release
    mock_client = MagicMock()
    mock_client.recv.return_value = b""  # Immediate disconnect
    assert server._client_semaphore.acquire(blocking=False)
    assert server._client_semaphore._value == 7
    server._handle_client(mock_client)
    # Semaphore value must be restored back to 8
    assert server._client_semaphore._value == 8


def test_persona_intent_empty_fallbacks_safe():
    """Verify PersonaIntentEngine handles empty fallbacks and collections gracefully."""
    profile = PersonalityProfile(
        personality="test",
        display_name="TestPersona",
        description="Test persona description",
        voice="alan",
        intents=[],
        fallbacks=[],
        costume_prompts=[],
        escalation_lines=[],
    )
    engine = PersonaIntentEngine(personality_name_or_path=profile)
    # Classify unmatched prompt with empty fallbacks
    response, intent = engine.classify_and_respond("unmatched random string 12345")
    assert intent == "fallback_unrecognized"
    assert response == "I am not sure how to answer that."

    # Test turn count >= 2 with empty costume prompts (should not raise)
    engine._turn_count = 3
    resp2, intent2 = engine.classify_and_respond("another phrase")
    assert intent2 == "fallback_unrecognized"
    assert resp2 == "I am not sure how to answer that."

    # Test turn count >= 8 with empty escalation lines (should not raise)
    engine._turn_count = 10
    resp3, intent3 = engine.classify_and_respond("late turn phrase")
    assert intent3 == "fallback_unrecognized"
    assert resp3 == "I am not sure how to answer that."


def test_vad_calculate_rms_nan_and_empty():
    """Verify VAD handles empty frames and NaN samples safely without corrupting noise floor."""
    vad = VoiceActivityDetector(sample_rate=16000)

    # Empty frame
    assert vad.calculate_rms(np.array([], dtype=np.float32)) == 0.0

    # NaN frame
    nan_frame = np.array([np.nan, 0.5, np.nan], dtype=np.float32)
    assert vad.calculate_rms(nan_frame) == 0.0

    # Normal frame
    normal_frame = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
    rms = vad.calculate_rms(normal_frame)
    assert rms > 0.0
    assert not math.isnan(rms)


def test_rpi_led_controller_cleanup_and_step_bound():
    """Verify RaspberryPiLEDController handles device closed and stop joins worker thread."""
    with patch("core.led_controller.Button", create=True):
        with patch("core.led_controller.PWMLED", create=True):
            controller = RaspberryPiLEDController(led_pin=25, button_pin=23)
            controller._backend = "gpiozero"

            # Mock worker thread
            mock_thread = MagicMock()
            controller._thread = mock_thread

            controller.stop()
            mock_thread.join.assert_called_once_with(timeout=1.0)
            assert controller._thread is None

            # Simulate closed device duty cycle failure handling
            mock_led = MagicMock()
            type(mock_led).value = MagicMock(side_effect=RuntimeError("Device closed"))
            controller._gpio_led = mock_led
            # Should not raise exception
            controller._set_duty_cycle(0.8)


def test_http_client_surfaces_http_error_body_and_json_err():
    """Verify http_client surfaces error body on HTTPError and context on json decode errors."""
    mock_http_err = urllib.error.HTTPError(
        url="http://example.com/api",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=io.BytesIO(b'{"error": "Invalid API key provided"}'),
    )

    with patch("urllib.request.urlopen", side_effect=mock_http_err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            http_client.get_json("http://example.com/api")
        assert "Invalid API key provided" in str(exc_info.value)

    # Test invalid JSON response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"<html>Not JSON</html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            http_client.get_json("http://example.com/api")
        assert "Invalid JSON received" in str(exc_info.value)
        assert "http://example.com/api" in str(exc_info.value)


def test_cloud_api_unified_models():
    """Verify cloud_api provides centralized Gemini LLM and STT models."""
    llm_models = cloud_api.gemini_llm_models()
    assert len(llm_models) >= 2
    assert "gemini-2.5-flash" in llm_models

    stt_models = cloud_api.gemini_stt_models()
    assert len(stt_models) >= 2
    assert "gemini-2.5-flash" in stt_models


def test_audio_io_resample_2x_fast_path():
    """Verify AudioIO fast 2x linear interpolation matches expected output."""
    orig_sr = 22050
    target_sr = 44100
    samples = np.array([0.0, 0.4, 0.8, 1.0], dtype=np.float32)

    resampled = AudioIO.resample_audio(samples, orig_sr, target_sr)
    assert len(resampled) == 8
    assert resampled[0] == 0.0
    assert abs(resampled[1] - 0.2) < 1e-5
    assert resampled[2] == 0.4
    assert abs(resampled[3] - 0.6) < 1e-5
    assert resampled[4] == 0.8
    assert abs(resampled[5] - 0.9) < 1e-5
    assert resampled[6] == 1.0
    assert resampled[7] == 1.0


def test_load_env_file_export_and_inline_comments():
    """Verify load_env_file handles leading export and unquoted inline comments."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write(
            "export TEST_KEY_1=value1 # inline comment\n"
            "TEST_KEY_2=\"value2 # with hash\"\n"
            "export TEST_KEY_3='value3 # with hash'\n"
            "TEST_KEY_4=value4\n"
        )
        tf_name = tf.name

    try:
        os.environ.pop("TEST_KEY_1", None)
        os.environ.pop("TEST_KEY_2", None)
        os.environ.pop("TEST_KEY_3", None)
        os.environ.pop("TEST_KEY_4", None)

        load_env_file(tf_name)

        assert os.environ.get("TEST_KEY_1") == "value1"
        assert os.environ.get("TEST_KEY_2") == "value2 # with hash"
        assert os.environ.get("TEST_KEY_3") == "value3 # with hash"
        assert os.environ.get("TEST_KEY_4") == "value4"
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)
        os.environ.pop("TEST_KEY_1", None)
        os.environ.pop("TEST_KEY_2", None)
        os.environ.pop("TEST_KEY_3", None)
        os.environ.pop("TEST_KEY_4", None)


def test_basic_info_service_cache_preservation():
    """Verify BasicInfoService can receive and expose existing weather cache."""
    cache_data = {"temp_f": "72", "temp_c": "22", "desc": "sunny"}
    timestamp = time.time()
    service = BasicInfoService(
        cached_weather=cache_data,
        cached_weather_time=timestamp,
    )
    assert service.cached_weather == cache_data
    assert service.cached_weather_time == timestamp

    # Weather response uses preserved cache without network call
    resp = service.get_weather_response(allow_network=False)
    assert "72 degrees" in resp


def test_agent_play_speech_audio_nonblocking_thread():
    """Verify non-blocking speech playback dispatches to worker thread."""
    agent = make_bare_agent()
    with patch("threading.Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        agent._play_speech_audio(np.zeros(100, dtype=np.float32), 22050, blocking=False)

        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()


def test_say_script_dry_run():
    """Verify say.py --dry-run exits 0 without calling audio playback."""
    from scripts import say
    test_args = ["say.py", "Hello test", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        ret = say.main()
        assert ret == 0


def test_summarize_personality_dry_run():
    """Verify summarize_personality.py --dry-run exits 0."""
    from scripts import summarize_personality
    test_args = ["summarize_personality.py", "-p", "grimjaw", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        ret = summarize_personality.main()
        assert ret == 0
