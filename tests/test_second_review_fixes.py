"""Regression tests for the second code review: matching, caching, shutdown, threading, privacy."""

import os
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core import cloud_api, intent_authoring, system_utils
from core.control_server import ControlServer
from core.intent_builder import CustomIntent, PersonalityProfile, normalize_query
from core.led_controller import ButtonGestureTracker, ConsoleLEDController, RaspberryPiLEDController
from core.vad import VoiceActivityDetector
from services.basic_info_service import BasicInfoService
from services.intent_service import PersonaIntentEngine
from services.llm_service import ConversationalLLMService
from services.stt_service import SpeechToTextService
from services.tts_service import PiperTTSService
from tests.agent_helpers import make_bare_agent


# --- intent matching ---------------------------------------------------------------------

def test_normalize_query_maps_typographic_apostrophes():
    assert normalize_query("What’s  YOUR name?") == "what's your name"
    assert normalize_query("it‘s `fine`") == "it's 'fine'"


def test_engine_matches_curly_apostrophe_like_straight_one():
    profile = PersonalityProfile(
        "t", "T", "",
        intents=[CustomIntent("identity", [r"\bwhat('s| is) your name\b"], ["I am T."])],
    )
    engine = PersonaIntentEngine(profile)
    assert engine.classify_and_respond("what’s your name")[1] == "identity"
    assert engine.classify_and_respond("what's your name")[1] == "identity"


# --- audio caching -----------------------------------------------------------------------

def _turn(agent, intent):
    agent.stt.transcribe.return_value = "tell me something"
    agent.vad.calculate_rms.return_value = 0.05
    agent.dialogue_engine.classify_and_respond.return_value = ("reply", intent)
    agent._play_speech_audio = MagicMock()
    agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    return agent.tts.synthesize.call_args.kwargs["cache"]


@pytest.mark.parametrize("intent,cached", [
    ("llm_generated", False),
    ("llm_fallback_custom", False),
    ("time_and_clock", False),
    ("identity", True),
    ("fallback_unrecognized", True),
])
def test_only_fixed_profile_lines_are_cached(intent, cached):
    assert _turn(make_bare_agent(), intent) is cached


def test_cache_hit_does_not_wait_for_synthesis_in_progress(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    tts = PiperTTSService(model_path=str(model), cache_dir=str(tmp_path), lazy_load=True)
    tts._remember("hello", np.ones(10, dtype=np.float32))
    done = []
    with tts._lock:  # another thread is busy synthesizing
        worker = threading.Thread(target=lambda: done.append(tts.synthesize("hello")))
        worker.start()
        worker.join(timeout=2)
    assert done and len(done[0][0]) == 10


def test_memory_cache_is_bounded_by_size(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    tts = PiperTTSService(model_path=str(model), cache_dir=str(tmp_path), lazy_load=True, max_memory_mb=1)
    for i in range(10):
        tts._remember(f"p{i}", np.zeros(100_000, dtype=np.float32))  # 400 kB each
    assert tts._cache_bytes <= 1024 * 1024
    assert "p9" in tts._cache and "p0" not in tts._cache


def test_cache_hit_rewrites_timestamp_at_most_once_an_hour(tmp_path):
    path = tmp_path / "a.npy"
    path.write_bytes(b"x")
    with patch("services.tts_service.os.utime") as utime:
        PiperTTSService._touch(str(path))
        assert not utime.called  # fresh file
        old = time.time() - 7200
        os.utime(path, (old, old))
        PiperTTSService._touch(str(path))
        assert utime.call_count == 1


# --- shutdown and button -----------------------------------------------------------------

def test_control_shutdown_runs_even_when_button_shutdown_is_disabled(monkeypatch):
    monkeypatch.setenv("BUTTON_SHUTDOWN_ENABLED", "false")
    agent = make_bare_agent()
    with patch("services.agent.system_utils.trigger_host_shutdown", return_value=True) as shutdown:
        agent._power_off_host()
    shutdown.assert_called_once()
    assert agent.audio.volume == 0.5


def test_shutdown_announcement_keeps_a_zero_volume_setting():
    agent = make_bare_agent()
    agent.audio.volume = 0.0
    seen = []
    agent.audio.play_audio.side_effect = lambda *a, **k: seen.append(agent.audio.volume)
    with patch("services.agent.system_utils.trigger_host_shutdown", return_value=True):
        agent._power_off_host()
    assert seen == [0.5] and agent.audio.volume == 0.0


def test_long_press_without_long_hold_callback_is_an_ordinary_hold():
    led = ConsoleLEDController(verbose=False)
    events = []
    led.set_button_callbacks(
        on_press=lambda: events.append("press"),
        on_hold=lambda: events.append("hold"),
        on_long_hold=None,
        hold_time_sec=0.05,
        long_hold_time_sec=0.1,
    )
    tracker = ButtonGestureTracker(led)
    tracker.pressed()
    time.sleep(0.25)
    tracker.released()
    assert events == ["hold"]


def test_long_hold_timer_firing_after_release_does_nothing():
    led = ConsoleLEDController(verbose=False)
    fired = []
    led.set_button_callbacks(on_long_hold=lambda: fired.append(1))
    tracker = ButtonGestureTracker(led)
    tracker.pressed()
    tracker.released()
    tracker._on_long_hold()  # the timer thread was already running when release arrived
    assert fired == [] and tracker._long_hold_fired is False


# --- control server ----------------------------------------------------------------------

@pytest.fixture
def server():
    agent = MagicMock()
    agent.volume = 0.5
    agent.set_volume.side_effect = lambda v: round(min(1.0, max(0.0, v)), 2)
    return ControlServer(agent=agent, socket_path="/tmp/unused.sock"), agent


@pytest.mark.parametrize("value", ["nan", "inf", "+nan", "-inf"])
def test_volume_rejects_non_finite_values(server, value):
    srv, agent = server
    assert srv.dispatch_command("volume", value)["status"] == "error"
    agent.set_volume.assert_not_called()


@pytest.mark.parametrize("value", ["/etc/passwd", "../x", "a/b.json", "x" * 80])
def test_personality_command_accepts_names_only(server, value):
    srv, agent = server
    assert srv.dispatch_command("personality", value)["status"] == "error"
    agent.switch_personality.assert_not_called()


def test_personality_command_switches_by_name(server):
    srv, agent = server
    agent.switch_personality.return_value = MagicMock(display_name="Snow", personality="snowman")
    assert srv.dispatch_command("personality", "snowman")["status"] == "ok"
    agent.switch_personality.assert_called_once_with("snowman")


# --- threading ---------------------------------------------------------------------------

def test_vad_survives_concurrent_resets():
    vad = VoiceActivityDetector(min_energy_threshold=0.001)
    errors = []
    stop = threading.Event()

    def reset_loop():
        while not stop.is_set():
            vad.reset()

    thread = threading.Thread(target=reset_loop)
    thread.start()
    try:
        rng = np.random.default_rng(1)
        for i in range(3000):
            level = 0.2 if (i // 40) % 2 == 0 else 0.0
            try:
                vad.process_frame((rng.standard_normal(480) * level).astype(np.float32))
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)
    finally:
        stop.set()
        thread.join()
    assert errors == []


# --- speech recognition and offline mode -------------------------------------------------

def test_automatic_offline_mode_uses_cloud_when_no_offline_model():
    agent = make_bare_agent(is_online=False, _consecutive_filtered_turns=0)
    agent.stt.offline_available = False
    agent.stt.transcribe.return_value = ""
    agent.vad.calculate_rms.return_value = 0.05
    agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    assert agent.stt.transcribe.call_args.kwargs["offline"] is False


def test_forced_offline_mode_never_uses_cloud():
    agent = make_bare_agent(is_online=False, _mode_forced=False, _consecutive_filtered_turns=0)
    agent.stt.offline_available = False
    agent.stt.transcribe.return_value = ""
    agent.vad.calculate_rms.return_value = 0.05
    agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    assert agent.stt.transcribe.call_args.kwargs["offline"] is True


def test_offline_available_reflects_the_vosk_model(monkeypatch):
    stt = SpeechToTextService(groq_api_key="k")
    assert stt.offline_available is False
    stt._vosk_model = object()
    assert stt.offline_available is True


# --- turn timeout ------------------------------------------------------------------------

def test_slow_speech_recognition_is_logged_as_a_timeout():
    agent = make_bare_agent()
    agent.vad.calculate_rms.return_value = 0.05
    agent.stt.transcribe.return_value = "what is the airspeed of a swallow"
    clock = iter([0.0, 0.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0])
    with patch("services.agent.time.time", side_effect=lambda: next(clock)):
        agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    assert agent.logger.log_interaction.call_args.kwargs["status"] == "TIMEOUT"
    agent.tts.synthesize.assert_not_called()


# --- LLM fallback ------------------------------------------------------------------------

@pytest.mark.parametrize("intent", ["fallback_unrecognized", "costume_inquiry", "lingerer_escalation"])
def test_llm_fallback_answers_every_unrecognized_outcome(intent):
    agent = make_bare_agent(llm_fallback_enabled=True)
    agent.dialogue_engine.classify_and_respond.return_value = ("canned", intent)
    agent.llm_fallback_service = MagicMock()
    agent.llm_fallback_service.generate_response.return_value = "model answer"
    assert agent._resolve_dialogue_response("hmm") == ("model answer", "llm_fallback_custom")


def test_recognized_intents_are_not_sent_to_the_llm():
    agent = make_bare_agent(llm_fallback_enabled=True)
    agent.dialogue_engine.classify_and_respond.return_value = ("I am T.", "identity")
    agent.llm_fallback_service = MagicMock()
    assert agent._resolve_dialogue_response("who are you") == ("I am T.", "identity")
    agent.llm_fallback_service.generate_response.assert_not_called()


def test_unsuitable_llm_reply_is_replaced_by_the_fallback_line():
    service = ConversationalLLMService(groq_api_key="k", fallback_response="Ask me again.")
    with patch.object(service, "_generate_groq", return_value="Well, shit happens."):
        assert service.generate_response("hi") == "Ask me again."
    assert service.history == []


def test_blocked_words_can_be_extended(monkeypatch):
    monkeypatch.setenv("LLM_BLOCKED_WORDS", "pumpkinhead, voldemort")
    assert ConversationalLLMService._is_unsuitable("You said Voldemort!")
    assert not ConversationalLLMService._is_unsuitable("Happy Halloween, friend.")


# --- cloud API ---------------------------------------------------------------------------

def test_gemini_generate_returns_first_usable_text():
    calls = []

    def fake(key, model, payload, timeout):
        calls.append(model)
        return [] if model == "a" else [{"text": "hello"}]

    with patch.object(cloud_api, "gemini_generate_parts", side_effect=fake):
        text = cloud_api.gemini_generate("k", ["a", "b"], {}, 1.0, lambda parts: parts[0]["text"])
    assert text == "hello" and calls == ["a", "b"]


def test_gemini_generate_stops_on_an_empty_answer_when_asked():
    with patch.object(cloud_api, "gemini_generate_parts", return_value=[]) as call:
        text = cloud_api.gemini_generate("k", ["a", "b"], {}, 1.0, lambda p: "", stop_on_answer=True)
    assert text == "" and call.call_count == 1


def test_gemini_generate_raises_when_every_model_fails():
    with patch.object(cloud_api, "gemini_generate_parts", side_effect=OSError("down")):
        with pytest.raises(RuntimeError, match="all Gemini models failed"):
            cloud_api.gemini_generate("k", ["a", "b"], {}, 1.0, lambda p: "x")


def test_gemini_generate_reports_each_failed_model():
    seen = []
    with patch.object(cloud_api, "gemini_generate_parts", side_effect=OSError("down")):
        with pytest.raises(RuntimeError):
            cloud_api.gemini_generate("k", ["a", "b"], {}, 1.0, lambda p: "x", on_error=lambda m, e: seen.append(m))
    assert seen == ["a", "b"]


def test_groq_chat_treats_null_content_as_empty():
    with patch.object(cloud_api.http_client, "post_json", return_value={"choices": [{"message": {"content": None}}]}):
        assert cloud_api.groq_chat("k", {}, 1.0) == ""


# --- .env parsing ------------------------------------------------------------------------

def test_env_value_keeps_hash_inside_tokens_and_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        'KEY_ONE=abc#def\nKEY_TWO="a # b" # note\nKEY_THREE=plain # note\nKEY_FOUR=# nothing\n',
        encoding="utf-8",
    )
    for name in ("KEY_ONE", "KEY_TWO", "KEY_THREE", "KEY_FOUR"):
        monkeypatch.delenv(name, raising=False)
    system_utils.load_env_file(str(env))
    assert os.environ["KEY_ONE"] == "abc#def"
    assert os.environ["KEY_TWO"] == "a # b"
    assert os.environ["KEY_THREE"] == "plain"
    assert os.environ["KEY_FOUR"] == ""
    for name in ("KEY_ONE", "KEY_TWO", "KEY_THREE", "KEY_FOUR"):
        monkeypatch.delenv(name, raising=False)


def test_relative_env_file_prefers_the_repository_copy_over_the_working_directory(tmp_path, monkeypatch):
    stray = tmp_path / "ALSAGEIST_STRAY.env"
    stray.write_text("STRAY_VALUE=from_cwd\n", encoding="utf-8")
    repo_copy = os.path.join(os.path.dirname(os.path.dirname(system_utils.__file__)), "ALSAGEIST_STRAY.env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STRAY_VALUE", raising=False)
    try:
        with open(repo_copy, "w", encoding="utf-8") as handle:
            handle.write("STRAY_VALUE=from_repo\n")
        system_utils.load_env_file("ALSAGEIST_STRAY.env")
        assert os.environ["STRAY_VALUE"] == "from_repo"
    finally:
        os.remove(repo_copy)
        monkeypatch.delenv("STRAY_VALUE", raising=False)


# --- privacy defaults and weather --------------------------------------------------------

def test_weather_is_off_without_a_location(monkeypatch):
    monkeypatch.delenv("WEATHER_ENABLED", raising=False)
    monkeypatch.delenv("WEATHER_LOCATION", raising=False)
    with patch("services.basic_info_service.http_client.get_json") as get:
        service = BasicInfoService()
        service.prefetch_weather()
        assert service.get_weather_data() is None
    get.assert_not_called()


def test_weather_is_on_with_a_location(monkeypatch):
    monkeypatch.delenv("WEATHER_ENABLED", raising=False)
    monkeypatch.setenv("WEATHER_LOCATION", "Boston")
    assert BasicInfoService()._enabled is True


def test_weather_can_be_forced_on_or_off(monkeypatch):
    monkeypatch.setenv("WEATHER_LOCATION", "Boston")
    monkeypatch.setenv("WEATHER_ENABLED", "false")
    assert BasicInfoService()._enabled is False


def test_stale_weather_refresh_waits_out_the_negative_cache():
    service = BasicInfoService(cache_ttl_sec=1.0, cached_weather={"temp_f": "50", "temp_c": "10", "desc": ""},
                               cached_weather_time=time.time() - 100)
    service._last_fetch_attempt_time = time.time()  # a refresh just failed
    with patch.object(service, "prefetch_weather") as prefetch:
        assert service.get_weather_data()["temp_f"] == "50"
    prefetch.assert_not_called()


def test_failed_weather_lookup_is_reported(capsys):
    service = BasicInfoService()
    with patch("services.basic_info_service.http_client.get_json", side_effect=OSError("down")):
        assert service._fetch_weather() is None
    assert "Weather lookup failed" in capsys.readouterr().out


# --- intent authoring --------------------------------------------------------------------

@pytest.mark.parametrize("pattern", [r"(a+)+$", r"\b(\w+\s?)*end\b", r"(x*)*", "a" * 400])
def test_backtracking_risks_are_detected(pattern):
    assert intent_authoring.is_backtracking_risk(pattern)


@pytest.mark.parametrize("pattern", [r"\b(you|he)( are)?.*\b(jerk|clown)\b", r"\bwhat('s| is) your name\b"])
def test_ordinary_patterns_are_accepted(pattern):
    assert not intent_authoring.is_backtracking_risk(pattern)


def test_new_intent_that_shadows_existing_questions_is_refused(tmp_path):
    profile = PersonalityProfile(
        "t", "T", "",
        intents=[CustomIntent("movie", [r"\bwhat is your favorite movie\b"], ["Beetlejuice."])],
    )
    path = tmp_path / "t.json"
    profile.save_to_file(str(path))
    with pytest.raises(intent_authoring.IntentShadowError) as err:
        intent_authoring.create_custom_intent(
            "your favorite", ["Hmm."], personality_name_or_path=str(path), use_llm=False, dry_run=True,
        )
    assert err.value.shadowed == ["movie"]


def test_shadowing_intent_is_allowed_with_force(tmp_path):
    profile = PersonalityProfile(
        "t", "T", "",
        intents=[CustomIntent("movie", [r"\bwhat is your favorite movie\b"], ["Beetlejuice."])],
    )
    path = tmp_path / "t.json"
    profile.save_to_file(str(path))
    intent, _ = intent_authoring.create_custom_intent(
        "your favorite", ["Hmm."], personality_name_or_path=str(path), use_llm=False, dry_run=True,
        collision_action="force",
    )
    assert intent.name


def test_failed_llm_expansion_is_reported(capsys):
    parser = intent_authoring.IntentParser(gemini_api_key="k")
    with patch.object(parser, "_parse_with_llm", side_effect=RuntimeError("quota")):
        intent = parser.parse("do you like pie", ["Yes."])
    assert intent.metadata["source"] == "rule_based"
    assert "quota" in capsys.readouterr().out


# --- hardware ----------------------------------------------------------------------------

def test_gpio_failure_names_the_cause_and_releases_the_pin(capsys, monkeypatch):
    led_device = MagicMock()
    fake = types.ModuleType("gpiozero")
    fake.PWMLED = MagicMock(return_value=led_device)
    fake.Button = MagicMock(side_effect=RuntimeError("pin 23 busy"))
    monkeypatch.setitem(sys.modules, "gpiozero", fake)
    monkeypatch.setitem(sys.modules, "RPi", None)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", None)
    controller = RaspberryPiLEDController()
    controller.start()
    assert "pin 23 busy" in capsys.readouterr().err
    led_device.close.assert_called_once()


def test_unknown_audio_device_name_is_reported(capsys):
    from core.audio_io import AudioIO
    devices = [{"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0}]
    with patch("core.audio_io.sd.query_devices", return_value=devices):
        audio = AudioIO(device_input="Missing")
    assert audio.device_input is None
    assert "no input audio device matches 'Missing'" in capsys.readouterr().out


# --- voice resolution --------------------------------------------------------------------

def test_voice_config_resolves_without_importing_the_intent_module():
    from core import voice_config
    assert voice_config.resolve_voice_settings(None) == (1.15, 0.85)
    path, tempo, noise = voice_config.resolve_agent_voice_config(repo_root="/repo")
    assert path.endswith("en_GB-alan-medium.onnx") and (tempo, noise) == (1.15, 0.85)
