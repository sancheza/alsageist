"""Regression tests for the code-review fixes (resilience, security, duplication)."""

import json
import os
import socket
import threading
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from core import cloud_api, env_utils
from core.control_server import ControlServer
from core.intent_builder import PersonalityProfile
from core.led_controller import LEDState
from core.logger import InteractionLogger
from core.utterance_filter import classify_utterance
from services.basic_info_service import BasicInfoService, safe_format
from services.llm_service import ConversationalLLMService
from tests.agent_helpers import make_args, make_bare_agent


# --- env_utils ---------------------------------------------------------------------------

def test_env_parsers_fall_back_on_malformed_values(monkeypatch):
    monkeypatch.setenv("X_NUM", "abc")
    monkeypatch.setenv("X_BOOL", "maybe")
    assert env_utils.env_float("X_NUM", 1.5) == 1.5
    assert env_utils.env_int("X_NUM", 7) == 7
    assert env_utils.env_bool("X_BOOL", True) is True
    monkeypatch.setenv("X_BOOL", "OFF")
    assert env_utils.env_bool("X_BOOL", True) is False


def test_gemini_key_falls_back_to_google_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert env_utils.gemini_api_key() == "g"
    assert env_utils.gemini_api_key("explicit") == "explicit"


# --- agent resilience --------------------------------------------------------------------

def test_playback_error_does_not_end_the_turn():
    agent = make_bare_agent()
    agent.stt.transcribe.return_value = "hello there"
    agent.vad.calculate_rms.return_value = 0.05
    agent.dialogue_engine.classify_and_respond.return_value = ("hi", "greeting")
    agent._play_speech_audio = MagicMock(side_effect=RuntimeError("device unplugged"))
    with patch("services.agent.time.sleep"):
        agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    assert agent.led.current_state == LEDState.ERROR


def test_main_loop_survives_an_unexpected_error():
    agent = make_bare_agent()
    agent.led.start = MagicMock()
    agent.audio.start_capture = MagicMock()
    agent.control_server = MagicMock()
    agent.power_monitor = None
    agent.logger.jsonl_path = "x"
    agent.is_online = True
    agent.basic_info = MagicMock()
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("boom")
        agent.running = False

    agent._loop_once = flaky
    with patch("services.agent.time.sleep"):
        agent.start()
    assert calls["n"] == 2


def test_failed_voice_load_keeps_current_voice():
    agent = make_bare_agent()
    profile = PersonalityProfile(personality="x", display_name="X", description="d", voice="missing-voice")
    original_tts = agent.tts
    agent._load_profile = MagicMock(return_value=profile)
    agent._build_dialogue = MagicMock(return_value=(MagicMock(), None))
    agent._tts_settings = MagicMock(return_value=("/no/such.onnx", 1.0, 0.5))
    agent._build_tts = MagicMock(side_effect=FileNotFoundError("no model"))
    with pytest.raises(FileNotFoundError):
        agent.switch_personality("x")
    assert agent.tts is original_tts


def test_user_mute_survives_speech_playback():
    agent = make_bare_agent()
    agent.set_mic_muted(True)
    agent._play_speech_audio(b"\x00", 16000, blocking=True, cooldown_sec=0)
    assert agent.audio.is_muted is True
    agent.set_mic_muted(False)
    assert agent.audio.is_muted is False


def test_wake_does_not_unmute_a_user_muted_mic():
    agent = make_bare_agent()
    agent.set_mic_muted(True)
    agent.disable()
    agent.enable()
    assert agent.audio.is_muted is True


def test_llm_engine_answers_time_from_the_clock():
    agent = make_bare_agent(args=make_args(engine="llm"))
    agent.basic_info = BasicInfoService()
    reply, intent = agent._resolve_dialogue_response("what time is it")
    assert intent == "time_and_clock"
    agent.dialogue_engine.classify_and_respond.assert_not_called()
    assert ":" in reply


def test_dynamic_answers_are_not_cached():
    agent = make_bare_agent()
    agent.stt.transcribe.return_value = "what time is it"
    agent.vad.calculate_rms.return_value = 0.05
    agent.dialogue_engine.classify_and_respond.return_value = ("x", "time_and_clock")
    agent.basic_info = BasicInfoService()
    agent._play_speech_audio = MagicMock()
    agent._handle_utterance(MagicMock(__len__=lambda self: 16000))
    assert agent.tts.synthesize.call_args.kwargs["cache"] is False


def test_button_shutdown_can_be_disabled(monkeypatch):
    monkeypatch.setenv("BUTTON_SHUTDOWN_ENABLED", "false")
    agent = make_bare_agent()
    with patch("services.agent.system_utils.trigger_host_shutdown") as shutdown:
        agent._on_button_long_hold()
    shutdown.assert_not_called()
    assert agent.is_disabled is False


# --- control server ----------------------------------------------------------------------

@pytest.fixture
def server():
    agent = MagicMock()
    agent.volume = 0.5
    agent.is_disabled = False
    agent.set_volume.side_effect = lambda v: round(min(1.0, max(0.0, v)), 2)
    return ControlServer(agent=agent, socket_path="/tmp/unused.sock"), agent


def test_shutdown_requires_confirmation(server):
    srv, agent = server
    assert srv.dispatch_command("shutdown")["status"] == "error"
    agent.request_shutdown.assert_not_called()
    assert srv.dispatch_command("shutdown", "confirm")["status"] == "ok"


def test_say_rejects_overlong_phrase(server):
    srv, agent = server
    resp = srv.dispatch_command("say", "x" * 501)
    assert resp["status"] == "error"
    agent.speak.assert_not_called()


def test_relative_volume_is_percentage_points(server):
    srv, agent = server
    srv.dispatch_command("volume", "+1")
    assert agent.set_volume.call_args.args[0] == pytest.approx(0.51)
    srv.dispatch_command("volume", "-10")
    assert agent.set_volume.call_args.args[0] == pytest.approx(0.40)


def test_slow_client_is_cut_off_by_total_deadline(server, monkeypatch):
    srv, _ = server
    monkeypatch.setattr("core.control_server.REQUEST_DEADLINE_SEC", 0.5)
    a, b = socket.socketpair()
    srv._client_semaphore.acquire()
    thread = threading.Thread(target=srv._handle_client, args=(a,))
    thread.start()
    started = time.monotonic()
    for _ in range(4):  # drip bytes, each well inside a per-recv timeout
        try:
            b.sendall(b" ")
        except OSError:
            break  # the server already gave up on this client
        time.sleep(0.2)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert time.monotonic() - started < 2.5
    assert "timed out" in b.recv(1024).decode()
    b.close()


def test_non_object_request_gets_an_error(server):
    srv, _ = server
    a, b = socket.socketpair()
    srv._client_semaphore.acquire()
    b.sendall(b"[1, 2]\n")
    srv._handle_client(a)
    assert json.loads(b.recv(1024))["status"] == "error"
    b.close()


# --- LLM service -------------------------------------------------------------------------

def test_clean_reply_removes_directions_emoji_and_partial_sentence():
    raw = "Ha *ha* [cackles] that is funny \U0001F600. And then the"
    assert ConversationalLLMService._clean_reply(raw) == "Ha ha that is funny ."


def test_clean_reply_caps_length():
    assert len(ConversationalLLMService._clean_reply("word. " * 200)) <= 300


def test_llm_deadline_bounds_total_wait(capsys, monkeypatch):
    llm = ConversationalLLMService(groq_api_key="g", gemini_api_key="k")
    monkeypatch.setattr("services.llm_service.TOTAL_TIMEOUT_SEC", 0.0)
    llm._generate_groq = MagicMock()
    assert llm.generate_response("hello") == llm.fallback_response
    llm._generate_groq.assert_not_called()
    assert "deadline" in capsys.readouterr().out


def test_llm_failure_reason_is_printed(capsys):
    llm = ConversationalLLMService(groq_api_key="g")
    llm._generate_groq = MagicMock(side_effect=RuntimeError("HTTP 401"))
    llm.generate_response("hello")
    assert "HTTP 401" in capsys.readouterr().out


# --- cloud_api ---------------------------------------------------------------------------

def test_retired_gemini_model_is_skipped_afterwards(monkeypatch):
    monkeypatch.delenv("GEMINI_LLM_MODEL", raising=False)
    cloud_api._unavailable_models.clear()
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    with patch("core.cloud_api.http_client.post_json", side_effect=err):
        with pytest.raises(urllib.error.HTTPError):
            cloud_api.gemini_generate_parts("k", "gemini-1.5-flash", {}, 1.0)
    assert "gemini-1.5-flash" not in cloud_api.gemini_llm_models()
    cloud_api._unavailable_models.clear()


# --- misc --------------------------------------------------------------------------------

def test_safe_format_falls_back_on_bad_placeholder():
    assert safe_format("It is {time} {unknown}", "fallback", time="1:00") == "fallback"
    assert safe_format("It is {time}", "fallback", time="1:00") == "It is 1:00"


def test_logger_text_log_escapes_newlines(tmp_path):
    log = InteractionLogger(log_dir=str(tmp_path))
    log.log_interaction("hi\n[FAKE] entry", "ok", "intent", "x", 1.0, 0.1, {"total": 1})
    assert len((tmp_path / "conversations.log").read_text().splitlines()) == 1


def test_profile_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "p.json"
    PersonalityProfile(personality="p", display_name="P", description="d").save_to_file(str(path))
    assert json.loads(path.read_text())["personality"] == "p"
    assert os.listdir(tmp_path) == ["p.json"]


def test_utterance_filter():
    assert classify_utterance("", 0.1)[2] == "FILTERED_SILENCE"
    assert classify_utterance("Thank you.", 0.005)[2] == "FILTERED_ARTIFACT"
    assert classify_utterance("thank you", 0.05)[0] is False
    assert classify_utterance("0:42", 0.05)[2] == "FILTERED_ARTIFACT"


def test_tts_cache_key_changes_with_pronunciation_rules(tmp_path):
    from services.tts_service import PiperTTSService

    model = tmp_path / "v.onnx"
    model.write_bytes(b"x")
    tts = PiperTTSService(model_path=str(model), cache_dir=str(tmp_path / "c"), lazy_load=True)
    before = tts.cache_path("hello")
    with patch("services.tts_service._RULES_FINGERPRINT", "other"):
        assert tts.cache_path("hello") != before
    assert not tts.is_cached("hello")
