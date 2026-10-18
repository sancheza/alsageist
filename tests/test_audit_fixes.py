"""Tests covering the audit fixes for bugs, security, architecture, and efficiency."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core import cloud_api
from core.control_server import ControlServer
from core.intent_builder import PersonalityProfile
from core.led_controller import ButtonGestureTracker, create_led_controller
from core.system_utils import load_env_file
from core.voice_config import resolve_agent_voice_config
from core.vad import VoiceActivityDetector
from services.agent import AlsaGeistAgent
from services.tts_service import PiperTTSService
from tests.agent_helpers import make_bare_agent


# --- 1. Control Server Volume Parsing Fix ---

def test_volume_one_percent_versus_full_volume():
    """Verify that volume 1 represents 1% and 1.0 represents 100%."""
    mock_agent = MagicMock()
    mock_agent.volume = 0.5
    mock_agent.set_volume.side_effect = lambda v: v

    server = ControlServer(agent=mock_agent)

    # Percentage scale: "1" must be 1% (0.01), not 100% (1.0)
    resp = server.dispatch_command("volume", "1")
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(0.01)

    # Integer 1 must also be 1%
    resp = server.dispatch_command("volume", 1)
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(0.01)

    # 100% scale
    resp = server.dispatch_command("volume", "100")
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(1.0)

    # Decimal fraction 0.5 and "0.5" must be 50%
    resp = server.dispatch_command("volume", 0.5)
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(0.5)

    resp = server.dispatch_command("volume", "0.5")
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(0.5)

    # Float 1.0 or string "1.0" with decimal must be 100%
    resp = server.dispatch_command("volume", 1.0)
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(1.0)

    resp = server.dispatch_command("volume", "1.0")
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(1.0)

    # 0 must be 0%
    resp = server.dispatch_command("volume", "0")
    assert resp["status"] == "ok"
    mock_agent.set_volume.assert_called_with(0.0)


# --- 2. Button Gesture Tracker Fix ---

def test_button_hold_fires_only_on_release_before_long_hold():
    """Verify hold fires only if released between hold_time_sec and long_hold_time_sec."""
    import time
    from core.led_controller import ButtonGestureTracker, ConsoleLEDController

    led = ConsoleLEDController(verbose=False)
    events = []
    led.set_button_callbacks(
        on_press=lambda: events.append("press"),
        on_hold=lambda: events.append("hold"),
        on_long_hold=lambda: events.append("long_hold"),
        hold_time_sec=0.08,
        long_hold_time_sec=0.35,
    )
    tracker = ButtonGestureTracker(led)

    # 1. Quick press (< 0.08s) -> press only
    tracker.pressed()
    time.sleep(0.02)
    tracker.released()
    assert events == ["press"]

    # 2. Hold between 0.08s and 0.35s and release -> hold only
    events.clear()
    tracker.pressed()
    time.sleep(0.15)
    # While still pressed, hold should not have fired yet
    assert events == []
    tracker.released()
    assert events == ["hold"]

    # 3. Hold past 0.35s -> long_hold fires on timeout; release does not fire hold
    events.clear()
    tracker.pressed()
    time.sleep(0.45)
    assert events == ["long_hold"]
    tracker.released()
    assert events == ["long_hold"]  # No duplicate "hold" on release


# --- 3. Env Loader Quoted Value Parsing Fix ---

def test_load_env_file_handles_internal_quotes_and_comments(tmp_path, monkeypatch):
    """Verify quoted strings with internal quotes or apostrophes are not truncated."""
    env_content = (
        'PROMPT_ONE="You are a assistant, saying \\"hello\\" to all"\n'
        "PROMPT_TWO='It is a\\'ight'\n"
        'QUOTED_COMMENT="hello world" # trailing comment\n'
        'PLAIN_COMMENT=sample_val # another comment\n'
        'SIMPLE_STR="simple"\n'
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content, encoding="utf-8")

    monkeypatch.delenv("PROMPT_ONE", raising=False)
    monkeypatch.delenv("PROMPT_TWO", raising=False)
    monkeypatch.delenv("QUOTED_COMMENT", raising=False)
    monkeypatch.delenv("PLAIN_COMMENT", raising=False)
    monkeypatch.delenv("SIMPLE_STR", raising=False)

    load_env_file(str(env_file))

    assert os.environ.get("PROMPT_ONE") == 'You are a assistant, saying \\"hello\\" to all'
    assert os.environ.get("PROMPT_TWO") == "It is a\\'ight"
    assert os.environ.get("QUOTED_COMMENT") == "hello world"
    assert os.environ.get("PLAIN_COMMENT") == "sample_val"
    assert os.environ.get("SIMPLE_STR") == "simple"


# --- 4. Unhandled ValueError in LLM Mode Fix ---

def test_llm_mode_missing_keys_exits_cleanly(monkeypatch):
    """Verify --engine llm without API keys raises SystemExit with clear guidance."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    args = MagicMock()
    args.engine = "llm"
    args.llm_fallback = False

    with patch.object(AlsaGeistAgent, "__init__", return_value=None):
        agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
        agent.args = args

        with pytest.raises(SystemExit) as exc_info:
            agent._build_dialogue(profile=None, personality="grimjaw")

        assert "Dialogue Engine Configuration Error" in str(exc_info.value)
        assert "GROQ_API_KEY" in str(exc_info.value)


# --- 5. VAD NaN and Inf Audio Samples Fix ---

def test_vad_calculate_rms_handles_inf_and_nan():
    """Verify calculate_rms returns 0.0 for inf and nan samples without breaking."""
    vad = VoiceActivityDetector()

    # Infinite sample
    inf_frame = np.array([float("inf"), 0.1, -0.1], dtype=np.float32)
    assert vad.calculate_rms(inf_frame) == 0.0

    # NaN sample
    nan_frame = np.array([float("nan"), 0.1, -0.1], dtype=np.float32)
    assert vad.calculate_rms(nan_frame) == 0.0

    # Normal valid sample
    normal_frame = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    assert vad.calculate_rms(normal_frame) > 0.0


# --- 6. TTS RAM Cache Hit Pure In-Memory Return Fix ---

def test_tts_model_size_cached(tmp_path):
    """Verify that PiperTTSService caches model_size and does not call getsize repeatedly."""
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"x" * 500)
    fake_json = tmp_path / "model.onnx.json"
    fake_json.write_text('{"audio": {"sample_rate": 22050}}')

    tts = PiperTTSService(model_path=str(fake_model), cache_dir=str(tmp_path / "cache"), lazy_load=True)

    assert tts._model_size == 500
    with patch("os.path.getsize") as mock_getsize:
        path1 = tts._get_cache_path("phrase 1")
        path2 = tts._get_cache_path("phrase 2")
        mock_getsize.assert_not_called()
        assert path1 != path2
        assert str(500) not in path1  # hash contains size in key preimage


# --- 7. Voice and Model Configuration Resolver Fix ---

def test_resolve_agent_voice_config_precedence(monkeypatch, tmp_path):
    """Verify voice configuration precedence: CLI > Env > Profile."""
    model_dir = tmp_path / "models" / "piper"
    model_dir.mkdir(parents=True)
    custom_model = model_dir / "custom.onnx"
    custom_model.touch()

    profile = PersonalityProfile(
        personality="test",
        display_name="Test Persona",
        description="Testing",
        voice="custom",
        voice_settings={"tempo": 1.25, "noise": 0.75},
    )

    # 1. Profile defaults
    path, tempo, noise = resolve_agent_voice_config(profile=profile, repo_root=str(tmp_path))
    assert path == str(custom_model)
    assert tempo == 1.25
    assert noise == 0.75

    # 2. CLI flag overrides
    path, tempo, noise = resolve_agent_voice_config(
        profile=profile,
        tempo=0.9,
        noise=0.6,
        repo_root=str(tmp_path),
    )
    assert tempo == 0.9
    assert noise == 0.6


# --- 8. Gemini Shared Model List ---

def test_cloud_api_gemini_models_shared():
    """Verify GEMINI_DEFAULT_LLM_MODELS and GEMINI_DEFAULT_STT_MODELS share definition."""
    assert cloud_api.GEMINI_DEFAULT_LLM_MODELS == cloud_api.GEMINI_DEFAULT_STT_MODELS
    assert "gemini-2.0-flash" in cloud_api.GEMINI_DEFAULT_FLASH_MODELS


# --- 9. Spurious Button Release Fix ---

def test_button_gesture_tracker_ignores_spurious_release():
    """Verify ButtonGestureTracker ignores release edges when press_time is 0.0."""
    controller = MagicMock()
    controller.hold_time_sec = 5.0
    controller.long_hold_time_sec = 10.0
    tracker = ButtonGestureTracker(controller)

    # Calling released without pressed should not invoke any callback
    tracker.released()
    controller.button_callback.assert_not_called()
    controller.button_hold_callback.assert_not_called()
    controller.button_long_hold_callback.assert_not_called()


# --- 10. Silent Volume Cycle at 0% Fix ---

def test_button_volume_cycle_audible_at_zero():
    """Verify that cycling volume to 0% speaks the announcement audibly before restoring 0.0."""
    agent = make_bare_agent()
    agent.audio.volume = 1.0

    volumes_during_speak = []

    def mock_speak(text, **kwargs):
        volumes_during_speak.append(agent.audio.volume)
        return True

    agent.speak = MagicMock(side_effect=mock_speak)

    agent._on_button_press()

    assert agent.audio.volume == 0.0
    assert len(volumes_during_speak) == 1
    assert volumes_during_speak[0] == 0.5  # Boosted volume during announcement
    agent.speak.assert_called_once()
    assert "0 percent" in agent.speak.call_args[0][0]


# --- 11. TTS Chunk Synthesis Resilience & Format Handling ---

def test_tts_synthesize_chunk_handles_chunk_formats_and_fallback(tmp_path):
    """Verify TTS chunk synthesis supports audio_int16_array and falls back on exception."""
    model = tmp_path / "voice.onnx"
    model.write_text("model")
    tts = PiperTTSService(model_path=str(model), cache_dir=str(tmp_path / "cache"), lazy_load=True)
    tts._voice_loaded = True

    # 1. Test audio_int16_array conversion
    mock_chunk = MagicMock()
    mock_chunk.audio_int16_array = np.array([0, 16384, -16384], dtype=np.int16)
    del mock_chunk.audio_float_array
    del mock_chunk.audio_bytes

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [mock_chunk]
    tts._voice = mock_voice

    audio = tts._synthesize_chunk("Testing audio int16 format")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-4)

    # 2. Test in-process exception falls through to CLI subprocess
    mock_voice.synthesize.side_effect = RuntimeError("Phonemizer failed")
    raw_pcm = np.array([0, 16384], dtype=np.int16).tobytes()

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = (raw_pcm, b"")
    mock_proc.returncode = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        audio_fb = tts._synthesize_chunk("Testing fallback")
        assert isinstance(audio_fb, np.ndarray)
        assert len(audio_fb) == 2


# --- 12. Raspberry Pi 5 Sysfs Device Tree Model Detection ---

def test_create_led_controller_detects_pi_from_sys_firmware_devicetree():
    """Verify create_led_controller detects Pi from /sys/firmware/devicetree/base/model."""
    from io import StringIO

    def fake_open(path, *args, **kwargs):
        if path == "/sys/firmware/devicetree/base/model":
            return StringIO("Raspberry Pi 5 Model B Rev 1.0\x00")
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=fake_open), \
         patch("core.led_controller.RaspberryPiLEDController") as mock_rpi_cls:
        ctrl = create_led_controller(verbose=False)
        mock_rpi_cls.assert_called_once()


# --- 13. Dynamic Info Decoupling in Dialogue Resolution ---

def test_resolve_dialogue_response_unconditionally_checks_basic_info():
    """Verify live dynamic info (time/weather) queries are handled unconditionally."""
    agent = make_bare_agent(engine="intent")
    agent.basic_info.match_intent.return_value = "time_and_clock"
    agent.basic_info.respond.return_value = "The time is 3:00 PM."

    reply, intent = agent._resolve_dialogue_response("what time is it")

    assert reply == "The time is 3:00 PM."
    assert intent == "time_and_clock"
    agent.basic_info.match_intent.assert_called_once_with("what time is it")
    agent.basic_info.respond.assert_called_once_with("time_and_clock", allow_network=agent.is_online)
    agent.dialogue_engine.classify_and_respond.assert_not_called()

