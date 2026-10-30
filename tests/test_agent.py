"""Unit tests for services/agent.py: construction wiring, control API, and main.py entry point."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.intent_builder import PersonalityProfile, find_personality_file
from core.led_controller import LEDState
from services.agent import AlsaGeistAgent
from services.intent_service import PersonaIntentEngine
from tests.agent_helpers import make_args, make_bare_agent


@pytest.fixture
def built_agent(tmp_path, monkeypatch):
    """Construct a real AlsaGeistAgent through __init__ with hardware subsystems mocked."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_PATH", raising=False)
    args = make_args(led_pin=25, button_pin=23, button_hold_sec=5.0, button_shutdown_sec=10.0)
    with patch("services.agent.AudioIO") as audio_cls, \
         patch("services.agent.SpeechToTextService") as stt_cls, \
         patch("services.agent.PiperTTSService") as tts_cls, \
         patch("services.agent.create_led_controller") as led_factory, \
         patch("services.agent.ControlServer") as server_cls, \
         patch.object(AlsaGeistAgent, "_check_internet", return_value=True):
        agent = AlsaGeistAgent(args)
        yield agent, {
            "audio": audio_cls, "stt": stt_cls, "tts": tts_cls,
            "led": led_factory, "server": server_cls,
        }


def test_constructor_builds_intent_engine_from_loaded_profile(built_agent):
    """The default intent engine is created from the loaded profile object."""
    agent, _ = built_agent
    assert isinstance(agent.dialogue_engine, PersonaIntentEngine)
    assert agent.dialogue_engine.profile is agent.profile
    assert agent.persona_name == agent.profile.display_name
    assert agent.logger.persona_name == agent.profile.display_name


def test_constructor_resolves_profile_voice_id_to_model_file(built_agent):
    """A bare voice ID in the profile resolves to a models/piper/<id>.onnx path."""
    _, mocks = built_agent
    model_path = mocks["tts"].call_args.kwargs["model_path"]
    assert model_path.endswith("models/piper/en_GB-alan-medium.onnx")


def test_constructor_registers_button_callbacks(built_agent):
    """Button press, hold and long-hold callbacks are wired to the agent."""
    agent, mocks = built_agent
    kwargs = mocks["led"].return_value.set_button_callbacks.call_args.kwargs
    assert kwargs["on_press"] == agent._on_button_press
    assert kwargs["on_hold"] == agent._on_button_hold
    assert kwargs["on_long_hold"] == agent._on_button_long_hold


def test_stt_configuration_error_exits_with_guidance():
    """A missing STT configuration exits with a message naming the required keys."""
    with patch("services.agent.SpeechToTextService", side_effect=ValueError("no backend")):
        with pytest.raises(SystemExit) as exc:
            AlsaGeistAgent._build_stt()
    assert "GROQ_API_KEY" in str(exc.value)


def test_load_profile_missing_non_strict_returns_none(capsys):
    """A missing profile returns None with a visible warning when not strict."""
    assert AlsaGeistAgent._load_profile("nonexistent_persona_xyz") is None
    assert "nonexistent_persona_xyz" in capsys.readouterr().out


def test_load_profile_missing_strict_raises():
    """A missing profile raises when strict."""
    with pytest.raises(FileNotFoundError):
        AlsaGeistAgent._load_profile("nonexistent_persona_xyz", strict=True)


def test_build_llm_uses_profile_prompt():
    """The LLM service receives the persona's system prompt."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    with patch("services.agent.ConversationalLLMService") as llm_cls:
        AlsaGeistAgent._build_llm(profile)
    llm_cls.assert_called_once()
    assert llm_cls.call_args.kwargs["system_prompt"] == profile.build_system_prompt()


def test_build_llm_failure_lines_come_from_the_persona(monkeypatch):
    """When every backend fails the spoken line is one of the persona's own fallback lines."""
    monkeypatch.delenv("LLM_FALLBACK_RESPONSE", raising=False)
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    with patch("services.agent.ConversationalLLMService") as llm_cls:
        AlsaGeistAgent._build_llm(profile)
    fallback = llm_cls.call_args.kwargs["fallback_response"]
    assert callable(fallback)
    assert {fallback() for _ in range(50)} <= set(profile.fallbacks)


def test_build_llm_env_override_beats_persona_fallback_lines(monkeypatch):
    """An explicit LLM_FALLBACK_RESPONSE is not replaced by the persona's lines."""
    monkeypatch.setenv("LLM_FALLBACK_RESPONSE", "Custom line.")
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    with patch("services.agent.ConversationalLLMService") as llm_cls:
        AlsaGeistAgent._build_llm(profile)
    assert llm_cls.call_args.kwargs["fallback_response"] is None


def test_build_tts_prefers_cli_overrides_over_profile():
    """Command-line tempo and noise override profile voice settings."""
    agent = make_bare_agent(args=make_args(tempo=1.4, noise=0.5))
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    with patch("services.agent.PiperTTSService") as tts_cls:
        agent._build_tts(profile)
    assert tts_cls.call_args.kwargs["length_scale"] == 1.4
    assert tts_cls.call_args.kwargs["noise_scale"] == 0.5


def test_set_volume_clamps():
    """Volume is clamped to 0.0 to 1.0 and rounded."""
    agent = make_bare_agent()
    assert agent.set_volume(1.7) == 1.0
    assert agent.set_volume(-0.2) == 0.0
    assert agent.set_volume(0.456) == 0.46
    assert agent.audio.volume == 0.46


def test_toggle_round_trip():
    """toggle() disables then re-enables and reports the new state."""
    agent = make_bare_agent()
    assert agent.toggle() is True
    assert agent.is_disabled is True
    assert agent.audio.is_muted is True
    assert agent.toggle() is False
    assert agent.is_disabled is False
    assert agent.audio.is_muted is False


def test_speak_refused_while_disabled():
    """speak() does nothing while the agent is in sleep mode."""
    agent = make_bare_agent()
    agent.disable()
    assert agent.speak("hello") is False
    agent.tts.synthesize.assert_not_called()


def test_speak_returns_false_on_synthesis_error():
    """speak() reports synthesis failures instead of raising."""
    agent = make_bare_agent()
    agent.tts.synthesize.side_effect = RuntimeError("piper failed")
    assert agent.speak("hello") is False


def test_concurrent_speech_plays_one_clip_at_a_time():
    """Two threads speaking at once never overlap playback."""
    agent = make_bare_agent()
    active = 0
    max_active = 0
    guard = threading.Lock()

    def fake_play(*args, **kwargs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1

    agent.audio.play_audio.side_effect = fake_play
    threads = [
        threading.Thread(target=agent.speak, args=("hello",), kwargs={"cooldown_sec": 0.0})
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert agent.audio.play_audio.call_count == 4
    assert max_active == 1


def test_button_hold_toggles_and_names_persona(capsys):
    """The 5-second hold toggles sleep mode and prints the active persona name."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    agent._on_button_hold()
    assert agent.is_disabled is True
    assert profile.display_name in capsys.readouterr().out


def test_mode_announcement_uses_profile_text():
    """Mode announcements come from the profile's system_announcements."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    profile.system_announcements["offline_mode"] = "Custom offline line."
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    agent._announce_mode(online=False)
    agent.tts.synthesize.assert_called_with("Custom offline line.")


def test_main_runs_agent_and_installs_signal_handlers():
    """main() builds the agent, installs SIGINT/SIGTERM handlers, and starts it."""
    import main as main_module

    with patch.object(main_module, "load_env_file"), \
         patch.object(main_module, "parse_args", return_value=make_args()), \
         patch.object(main_module, "AlsaGeistAgent") as agent_cls, \
         patch.object(main_module.signal, "signal") as install:
        assert main_module.main() == 0

    agent_cls.return_value.start.assert_called_once()
    assert install.call_count == 2


def _agent_for_network_polling():
    agent = make_bare_agent()
    agent.network.last_check_time = 0.0
    agent.network.check_interval_sec = 10.0
    agent.network._thread = None
    agent.network._result = None
    return agent


def test_network_poll_runs_probe_on_background_thread():
    """The connectivity probe does not run on the audio-loop thread."""
    agent = _agent_for_network_polling()
    probe_threads = []

    def fake_check():
        probe_threads.append(threading.current_thread())
        return False

    with patch.object(AlsaGeistAgent, "_check_internet", side_effect=fake_check):
        agent._poll_network(now=100.0)
        agent.network._thread.join(timeout=2)

    assert probe_threads and probe_threads[0] is not threading.current_thread()
    assert agent.is_online is True  # result applies on the next poll


def test_network_poll_applies_result_and_announces_change():
    """A published result flips the mode once and announces it."""
    agent = _agent_for_network_polling()
    agent.network.last_check_time = 100.0
    agent.network._result = False
    agent._announce_mode = MagicMock()

    agent._poll_network(now=101.0)
    assert agent.is_online is False
    agent._announce_mode.assert_called_once_with(online=False)
    assert agent.network._result is None

    agent._poll_network(now=102.0)
    agent._announce_mode.assert_called_once()


def test_network_poll_does_not_start_second_probe_while_running():
    """No new probe starts while the previous one is still alive."""
    agent = _agent_for_network_polling()
    release = threading.Event()

    def slow_check():
        release.wait(timeout=2)
        return True

    with patch.object(AlsaGeistAgent, "_check_internet", side_effect=slow_check):
        agent._poll_network(now=100.0)
        first = agent.network._thread
        agent.network.last_check_time = 0.0
        agent._poll_network(now=200.0)
        assert agent.network._thread is first
        release.set()
        first.join(timeout=2)


def test_weather_intent_uses_cache_only_when_offline():
    """Offline mode answers weather questions without contacting the network."""
    agent = make_bare_agent(is_online=False)
    agent.dialogue_engine.classify_and_respond.return_value = ("canned", "weather_and_temperature")
    agent._resolve_dialogue_response("what's the weather")
    agent.basic_info.respond.assert_called_once_with("weather_and_temperature", allow_network=False)


def test_idle_reset_fires_once_after_idle_period():
    """The conversation resets once after IDLE_RESET_SEC and not on every later check."""
    agent = make_bare_agent()
    agent._last_activity_time = 100.0
    agent.reset_dialogue = MagicMock()

    agent._reset_if_idle(105.0)
    agent.reset_dialogue.assert_not_called()

    agent._reset_if_idle(111.0)
    agent._reset_if_idle(150.0)
    agent.reset_dialogue.assert_called_once()


def test_activity_rearms_idle_reset():
    """New activity allows another idle reset later."""
    agent = make_bare_agent()
    agent._last_activity_time = 0.0
    agent.reset_dialogue = MagicMock()
    agent._reset_if_idle(100.0)
    assert agent.reset_dialogue.call_count == 1

    agent._note_activity()
    agent._reset_if_idle(agent._last_activity_time + 1.0)
    assert agent.reset_dialogue.call_count == 1
    agent._reset_if_idle(agent._last_activity_time + 11.0)
    assert agent.reset_dialogue.call_count == 2


def test_main_loop_resets_idle_conversation_while_frames_keep_arriving():
    """With continuous silent frames (never a read timeout) the loop still resets the conversation."""
    import numpy as np

    agent = make_bare_agent()
    agent.network.last_check_time = time.time()
    agent.network.check_interval_sec = 3600.0
    agent.network._thread = None
    agent.network._result = None
    agent.logger.jsonl_path = "unused"
    resets = threading.Event()
    agent.reset_dialogue = MagicMock(side_effect=lambda: resets.set())

    def silent_frame(timeout=0.05):
        time.sleep(0.005)
        return np.zeros(480, dtype=np.float32)

    agent.audio.read_frame.side_effect = silent_frame
    agent.vad.process_frame.return_value = (False, False, None)
    agent.vad.is_speaking = False

    with patch("services.agent.IDLE_RESET_SEC", 0.05):
        thread = threading.Thread(target=agent.start, daemon=True)
        thread.start()
        fired = resets.wait(timeout=2.0)
        agent.running = False
        thread.join(timeout=2.0)

    assert fired
    assert agent.reset_dialogue.call_count == 1


def test_forced_mode_survives_periodic_connectivity_check():
    """A mode set by a control command is not undone by the next connectivity result."""
    agent = _agent_for_network_polling()
    agent.set_online(False)
    assert agent.mode_forced is True

    agent.network._result = True  # probe says the network is up
    agent._announce_mode = MagicMock()
    agent._poll_network(now=agent.network.last_check_time + 1.0)

    assert agent.is_online is False
    agent._announce_mode.assert_not_called()
    assert agent.network._thread is None  # no probe is started while forced


def test_auto_mode_resumes_detection():
    """set_online(None) clears the override and schedules an immediate probe."""
    agent = _agent_for_network_polling()
    agent.set_online(False)
    agent.network.last_check_time = 12345.0

    agent.set_online(None)

    assert agent.mode_forced is False
    assert agent.network.last_check_time == 0.0


def test_failed_shutdown_returns_agent_to_normal_operation(capsys):
    """If no poweroff method works the display wakes up instead of staying asleep."""
    agent = make_bare_agent()
    agent.audio.volume = 0.5
    with patch("services.agent.system_utils.trigger_host_shutdown", return_value=False):
        agent._on_button_long_hold()

    assert agent.is_disabled is False
    assert "Shutdown failed" in capsys.readouterr().out


def test_personality_switch_does_not_block_other_commands_while_building():
    """A slow voice load happens outside the state lock, so toggle() stays responsive."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    loading = threading.Event()
    release = threading.Event()

    def slow_build(_profile):
        loading.set()
        release.wait(timeout=5)
        return MagicMock()

    agent._build_tts = slow_build
    switcher = threading.Thread(target=agent.switch_personality, args=("grimjaw",), daemon=True)
    switcher.start()
    assert loading.wait(timeout=2)

    started = time.time()
    agent.toggle()  # would wait for the whole build if the state lock were held
    assert time.time() - started < 1.0
    assert agent.is_disabled is True

    release.set()
    switcher.join(timeout=5)
    assert not switcher.is_alive()


def test_personality_switch_reuses_voice_with_same_settings():
    """The running TTS is kept when the new persona resolves to the same model and settings."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    model_path, tempo, noise = agent._tts_settings(profile)
    agent.tts = MagicMock(model_path=model_path, length_scale=tempo, noise_scale=noise)
    agent._build_tts = MagicMock()

    agent.switch_personality("grimjaw")

    agent._build_tts.assert_not_called()
    assert agent.tts.model_path == model_path


def test_personality_switch_builds_new_voice_when_settings_differ():
    """A different voice model or setting loads a new TTS."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    agent.tts = MagicMock(model_path="/other/voice.onnx", length_scale=1.0, noise_scale=0.5)
    new_tts = MagicMock()
    agent._build_tts = MagicMock(return_value=new_tts)

    agent.switch_personality("grimjaw")

    agent._build_tts.assert_called_once()
    assert agent.tts is new_tts


def test_agent_set_llm_fallback_enables_and_disables():
    """set_llm_fallback initializes the LLM service on demand and toggles state."""
    agent = make_bare_agent()
    mock_llm = MagicMock()
    agent._build_llm = MagicMock(return_value=mock_llm)

    assert agent.llm_fallback_enabled is False
    assert agent.llm_fallback_service is None

    result = agent.set_llm_fallback(True)
    assert result is True
    assert agent.llm_fallback_enabled is True
    assert agent.llm_fallback_service is mock_llm
    agent._build_llm.assert_called_once_with(agent.profile)

    # Disabling preserves the service but turns off the flag
    result = agent.set_llm_fallback(False)
    assert result is False
    assert agent.llm_fallback_enabled is False

    # Enabling again reuses existing service
    agent._build_llm.reset_mock()
    result = agent.set_llm_fallback(True)
    assert result is True
    assert agent.llm_fallback_enabled is True
    agent._build_llm.assert_not_called()


def test_agent_set_llm_fallback_raises_without_api_keys(monkeypatch):
    """set_llm_fallback propagates ValueError when no LLM API keys are configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    agent = make_bare_agent()
    with pytest.raises(ValueError, match="Neither GROQ_API_KEY nor GEMINI_API_KEY"):
        agent.set_llm_fallback(True)


def test_agent_starts_and_stops_power_monitor():
    """The power monitor runs for the life of the agent loop and is stopped on shutdown."""
    agent = make_bare_agent()
    agent.power_monitor = MagicMock()
    agent.audio.read_frame.return_value = None
    agent.logger.jsonl_path = "unused"
    agent.network.last_check_time = time.time()
    agent.network.check_interval_sec = 3600.0
    agent.network._thread = None
    agent.network._result = None

    runner = threading.Thread(target=agent.start, daemon=True)
    runner.start()
    for _ in range(100):
        if agent.power_monitor.start.called:
            break
        time.sleep(0.02)
    agent.running = False
    runner.join(timeout=3)

    agent.power_monitor.start.assert_called_once()
    agent.power_monitor.stop.assert_called_once()


class _CooldownGate:
    """Replaces time.sleep inside the agent so a test decides exactly when a cooldown ends."""

    def __init__(self):
        self.entered = threading.Event()
        self.open = threading.Event()

    def sleep(self, _seconds):
        self.entered.set()
        assert self.open.wait(timeout=5), "cooldown gate was never opened"


def test_cooldown_does_not_hold_the_audio_lock():
    """Other callers can take the audio lock while a finished clip is cooling down."""
    agent = make_bare_agent()
    gate = _CooldownGate()

    with patch("services.agent.time.sleep", gate.sleep):
        cooling = threading.Thread(
            target=agent._play_speech_audio, args=(b"", 22050), kwargs={"cooldown_sec": 1.0}, daemon=True
        )
        cooling.start()
        assert gate.entered.wait(timeout=3)          # the clip has played and is now cooling down

        got_lock = agent._audio_lock.acquire(timeout=1.0)  # would time out if the cooldown held the lock
        if got_lock:
            agent._audio_lock.release()
        gate.open.set()
        cooling.join(timeout=3)

    assert got_lock is True


def test_microphone_stays_muted_until_the_latest_clip_finishes():
    """A clip that starts during another's cooldown keeps the mic muted until it ends."""
    agent = make_bare_agent()
    gate = _CooldownGate()
    second_playing = threading.Event()
    release_second = threading.Event()
    calls = {"n": 0}

    def play(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            second_playing.set()
            assert release_second.wait(timeout=5)

    agent.audio.play_audio.side_effect = play

    with patch("services.agent.time.sleep", gate.sleep):
        first = threading.Thread(
            target=agent._play_speech_audio, args=(b"", 22050), kwargs={"cooldown_sec": 1.0}, daemon=True
        )
        first.start()
        assert gate.entered.wait(timeout=3)          # first clip played; it is cooling down

        second = threading.Thread(
            target=agent._play_speech_audio, args=(b"", 22050), kwargs={"cooldown_sec": 0.0}, daemon=True
        )
        second.start()
        assert second_playing.wait(timeout=3)        # second clip is playing during the first's cooldown

        gate.open.set()                              # the first cooldown ends
        first.join(timeout=3)
        assert not first.is_alive()
        agent.audio.unmute_capture.assert_not_called()
        assert agent.audio.is_muted is True

        release_second.set()
        second.join(timeout=3)

    agent.audio.unmute_capture.assert_called_once()
    assert agent.audio.is_muted is False


def test_microphone_unmutes_after_a_failed_playback():
    """An error while playing still ends in an unmuted microphone and an idle LED."""
    agent = make_bare_agent()
    agent.audio.play_audio.side_effect = RuntimeError("device busy")

    with pytest.raises(RuntimeError):
        agent._play_speech_audio(b"", 22050, cooldown_sec=0.0)

    agent.audio.unmute_capture.assert_called_once()
    assert agent.led.current_state == LEDState.IDLE


def test_disabled_agent_is_not_unmuted_after_playback():
    """If the agent is put to sleep during playback the microphone stays muted."""
    agent = make_bare_agent()
    agent.audio.play_audio.side_effect = lambda *a, **k: setattr(agent, "is_disabled", True)

    agent._play_speech_audio(b"", 22050, cooldown_sec=0.0)

    agent.audio.unmute_capture.assert_not_called()


# ---- go offline promptly when cloud speech recognition fails ----

def _agent_with_failed_cloud_stt(**overrides):
    agent = _agent_for_network_polling()
    for name, value in overrides.items():
        setattr(agent, name, value)
    agent.network.last_check_time = 1000.0
    agent.stt.transcribe.return_value = "what time is it"
    agent.stt.last_cloud_failed = True
    agent.vad.calculate_rms.return_value = 0.05
    agent.dialogue_engine.classify_and_respond.return_value = ("reply", "greeting")
    agent.tts.synthesize.return_value = (np.zeros(100, dtype=np.float32), 22050)
    return agent


def test_failed_cloud_stt_requests_an_immediate_connectivity_check():
    """When every cloud STT backend failed the next loop pass probes the network right away."""
    agent = _agent_with_failed_cloud_stt()
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.network.last_check_time == 0.0


def test_healthy_cloud_stt_does_not_request_a_check():
    """Normal operation leaves the 15-minute schedule alone."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.last_cloud_failed = False
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.network.last_check_time == 1000.0


def test_stt_total_failure_also_requests_a_check():
    """A transcription that raises after cloud attempts still asks for a probe."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.transcribe.side_effect = RuntimeError("All transcription backends failed")
    with patch("services.agent.time.sleep"):
        agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.network.last_check_time == 0.0


def test_check_not_requested_when_mode_is_forced_or_already_offline():
    """A forced mode is respected, and an offline agent waits for the periodic check to come back."""
    forced = _agent_with_failed_cloud_stt()
    forced.network.forced = True
    forced._request_network_check()
    assert forced.network.last_check_time == 1000.0

    offline = _agent_with_failed_cloud_stt(is_online=False)
    offline._request_network_check()
    assert offline.network.last_check_time == 1000.0


def test_check_requests_are_rate_limited():
    """Repeated failures inside 30 seconds do not keep re-triggering probes."""
    agent = _agent_with_failed_cloud_stt()
    agent._request_network_check()
    assert agent.network.last_check_time == 0.0

    agent.network.last_check_time = 5000.0       # the probe ran and updated the time
    agent._request_network_check()           # a second failure right after
    assert agent.network.last_check_time == 5000.0

    agent.network._last_request_time -= 31          # 31 seconds later a new request is allowed
    agent._request_network_check()
    assert agent.network.last_check_time == 0.0


def test_cloud_failure_with_network_down_goes_offline_and_announces():
    """End to end: failed cloud STT, probe finds no network, mode flips offline once."""
    agent = _agent_with_failed_cloud_stt()
    agent._announce_mode = MagicMock()
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    with patch.object(AlsaGeistAgent, "_check_internet", return_value=False):
        agent._poll_network(now=time.time())          # the loop starts the probe
        agent.network._thread.join(timeout=2)
        agent._poll_network(now=time.time())          # the next pass applies the result

    assert agent.is_online is False
    agent._announce_mode.assert_called_once_with(online=False)


def test_cloud_failure_with_network_up_stays_online():
    """If only the cloud API is down (probe succeeds) the agent does not go offline."""
    agent = _agent_with_failed_cloud_stt()
    agent._announce_mode = MagicMock()
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    with patch.object(AlsaGeistAgent, "_check_internet", return_value=True):
        agent._poll_network(now=time.time())
        agent.network._thread.join(timeout=2)
        agent._poll_network(now=time.time())

    assert agent.is_online is True
    agent._announce_mode.assert_not_called()


# ---- spoken low-power line ----

def test_low_power_line_is_one_of_the_personas_lines(capsys):
    """The agent speaks one of the profile's low-power lines when nobody is talking to it."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    agent._last_activity_time = time.time() - 60

    assert agent._announce_low_power(MagicMock()) is True

    spoken = agent.tts.synthesize.call_args.args[0]
    assert spoken in profile.low_power_lines
    assert "(low power)" in capsys.readouterr().out


def test_low_power_line_waits_while_a_conversation_is_active():
    """During or right after a conversation the agent declines, so the monitor retries later."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(profile=profile, dialogue_engine=PersonaIntentEngine(profile))
    agent._note_activity()

    assert agent._announce_low_power(MagicMock()) is False
    agent.tts.synthesize.assert_not_called()


def test_low_power_line_not_spoken_while_asleep():
    """A disabled (sleeping) agent stays silent."""
    agent = make_bare_agent()
    agent._last_activity_time = time.time() - 60
    agent.is_disabled = True
    assert agent._announce_low_power(MagicMock()) is False
    agent.tts.synthesize.assert_not_called()


def test_low_power_line_uses_neutral_default_without_persona_lines():
    """A profile with no low-power lines gets a plain, persona-free sentence."""
    from services.agent import DEFAULT_LOW_POWER_LINE

    agent = make_bare_agent()
    agent._last_activity_time = time.time() - 60
    assert agent._announce_low_power(MagicMock()) is True
    agent.tts.synthesize.assert_called_once_with(DEFAULT_LOW_POWER_LINE)
    for word in ("crypt", "bones", "mortal", "children"):
        assert word not in DEFAULT_LOW_POWER_LINE.lower()


def _construct_agent(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_PATH", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    args = make_args(led_pin=25, button_pin=23, button_hold_sec=5.0, button_shutdown_sec=10.0)
    with patch("services.agent.AudioIO"), patch("services.agent.SpeechToTextService"), \
         patch("services.agent.PiperTTSService"), patch("services.agent.create_led_controller"), \
         patch("services.agent.ControlServer"), patch("services.agent.PowerMonitor.from_env") as from_env, \
         patch.object(AlsaGeistAgent, "_check_internet", return_value=True):
        agent = AlsaGeistAgent(args)
    return agent, from_env


def test_power_monitor_is_created_by_default(tmp_path, monkeypatch):
    """Without LOW_POWER_MONITOR the agent runs a power monitor."""
    monkeypatch.delenv("LOW_POWER_MONITOR", raising=False)
    agent, from_env = _construct_agent(tmp_path, monkeypatch)
    assert agent.power_monitor is from_env.return_value


def test_power_monitor_can_be_turned_off_completely(tmp_path, monkeypatch):
    """LOW_POWER_MONITOR=false creates no monitor at all, so nothing is polled, logged, or spoken."""
    for value in ("false", "0", "no", "off", "FALSE"):
        agent, from_env = _construct_agent(tmp_path, monkeypatch, LOW_POWER_MONITOR=value)
        from_env.assert_not_called()
        assert agent.power_monitor is None, value


def test_agent_runs_and_stops_without_a_power_monitor():
    """With the monitor off, start() and stop() work and nothing tries to start or stop one."""
    agent = make_bare_agent()
    agent.power_monitor = None
    agent.logger.jsonl_path = "unused"
    agent.network.last_check_time = time.time()
    agent.network.check_interval_sec = 3600.0
    agent.network._thread = None
    agent.network._result = None
    agent.audio.read_frame.return_value = None

    runner = threading.Thread(target=agent.start, daemon=True)
    runner.start()
    time.sleep(0.3)
    agent.running = False
    runner.join(timeout=3)

    assert not runner.is_alive()
    agent.audio.stop_capture.assert_called_once()


def test_low_power_announcement_is_on_by_default(tmp_path, monkeypatch):
    """Without LOW_POWER_ANNOUNCE the monitor is given the agent's announce method."""
    monkeypatch.delenv("LOW_POWER_ANNOUNCE", raising=False)
    agent, from_env = _construct_agent(tmp_path, monkeypatch)
    assert from_env.call_args.kwargs["announce"] == agent._announce_low_power


def test_low_power_announcement_can_be_turned_off(tmp_path, monkeypatch):
    """LOW_POWER_ANNOUNCE=false keeps the log warnings but removes the spoken line."""
    for value in ("false", "0", "no", "off", "FALSE"):
        _, from_env = _construct_agent(tmp_path, monkeypatch, LOW_POWER_ANNOUNCE=value)
        assert from_env.call_args.kwargs["announce"] is None, value


def test_power_monitor_starts_after_the_startup_greeting():
    """A low-power line can never be spoken before the greeting or during startup."""
    agent = make_bare_agent()
    agent.args = make_args(no_greeting=False)
    agent.logger.jsonl_path = "unused"
    agent.network.last_check_time = time.time()
    agent.network.check_interval_sec = 3600.0
    agent.network._thread = None
    agent.network._result = None
    agent.audio.read_frame.return_value = None
    order = []
    agent._play_startup_greeting = MagicMock(side_effect=lambda: order.append("greeting"))
    agent.power_monitor = MagicMock()
    agent.power_monitor.start.side_effect = lambda: order.append("monitor")

    runner = threading.Thread(target=agent.start, daemon=True)
    runner.start()
    for _ in range(100):
        if "monitor" in order:
            break
        time.sleep(0.02)
    agent.running = False
    runner.join(timeout=3)

    assert order == ["greeting", "monitor"]
    # and the quiet period has just been restarted, so the monitor's first check would not speak
    assert time.time() - agent._last_activity_time < 5


def test_agent_stop_is_idempotent():
    """Calling stop() multiple times cleans up exactly once and does not raise."""
    agent = make_bare_agent()
    agent.running = True
    agent.power_monitor = MagicMock()
    agent.control_server = MagicMock()

    agent.stop()
    assert agent.running is False
    assert agent._stopped is True
    assert agent.audio.stop_playback.call_count == 1
    assert agent.audio.stop_capture.call_count == 1

    # Second call must be a no-op
    agent.stop()
    assert agent.audio.stop_playback.call_count == 1
    assert agent.audio.stop_capture.call_count == 1


def test_agent_redacts_transcripts_when_logging_disabled(capsys):
    """When store_text is False, plain transcripts are redacted from stdout."""
    agent = _agent_with_failed_cloud_stt()
    agent.logger.store_text = False
    agent.stt.transcribe.return_value = "secret visitor question"
    agent.dialogue_engine.classify_and_respond.return_value = ("secret persona answer", "test_intent")

    agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    captured = capsys.readouterr().out
    assert "secret visitor question" not in captured
    assert "secret persona answer" not in captured
    assert "[TRANSCRIPT REDACTED]" in captured


def test_agent_dialogue_error_is_logged():
    """A dialogue engine error logs an interaction record with status ERROR."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.transcribe.return_value = "hello"
    agent.dialogue_engine.classify_and_respond.side_effect = RuntimeError("reasoning failure")

    with patch("services.agent.time.sleep"):
        agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    agent.logger.log_interaction.assert_called_once()
    kwargs = agent.logger.log_interaction.call_args.kwargs
    assert kwargs["status"] == "ERROR"
    assert kwargs["matched_intent"] == "dialogue_error"
    assert "reasoning failure" in kwargs["notes"]


def test_agent_tts_error_is_logged():
    """A TTS synthesis failure logs an interaction record with status ERROR."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.transcribe.return_value = "hello"
    agent.dialogue_engine.classify_and_respond.return_value = ("reply text", "greeting")
    agent.tts.synthesize.side_effect = RuntimeError("synthesis failure")

    with patch("services.agent.time.sleep"):
        agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    agent.logger.log_interaction.assert_called_once()
    kwargs = agent.logger.log_interaction.call_args.kwargs
    assert kwargs["status"] == "ERROR"
    assert "synthesis failure" in kwargs["notes"]




def test_filtered_utterance_keeps_led_off_while_asleep():
    """A turn that is filtered after the agent went to sleep does not light the LED."""
    agent = make_bare_agent(is_disabled=True)
    agent.stt.transcribe.return_value = ""
    agent.vad.calculate_rms.return_value = 0.05
    agent.led.set_state(LEDState.OFF)
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.led.current_state != LEDState.IDLE


def test_agent_voice_command_switch_to_offline():
    """Speaking 'switch to offline mode' forces offline mode and returns the offline announcement."""
    agent = make_bare_agent(is_online=True)
    assert agent.is_online is True
    reply, intent = agent._resolve_dialogue_response("switch to offline mode")
    assert agent.is_online is False
    assert agent.mode_forced is True
    assert intent == "system_mode_offline"
    assert reply == "NOW IN OFFLINE MODE"


def test_agent_voice_command_switch_to_online():
    """Speaking 'go online' forces online mode and returns the online announcement."""
    agent = make_bare_agent(is_online=False)
    assert agent.is_online is False
    reply, intent = agent._resolve_dialogue_response("go online")
    assert agent.is_online is True
    assert agent.mode_forced is True
    assert intent == "system_mode_online"
    assert reply == "NOW IN ONLINE MODE"


def test_agent_voice_command_query_mode_offline_and_online():
    """Speaking 'what mode are you in' returns current mode status reliably."""
    agent = make_bare_agent(is_online=False)
    reply, intent = agent._resolve_dialogue_response("what mode are you in")
    assert intent == "system_mode_query"
    assert reply == "I am in offline mode."

    agent.set_online(True)
    reply, intent = agent._resolve_dialogue_response("are you online or offline")
    assert intent == "system_mode_query"
    assert reply == "I am in online mode."


def test_agent_voice_command_query_voice_model():
    """Speaking 'what is your voice model' returns the active voice model announcement."""
    agent = make_bare_agent()
    agent.tts = MagicMock(model_path="models/piper/en_GB-alan-medium.onnx")
    reply, intent = agent._resolve_dialogue_response("what is your voice model")
    assert intent == "system_voice_query"
    assert reply == "My voice model is en GB alan medium."


def test_agent_voice_command_query_dialogue_engine():
    """Speaking 'what engine are you using' returns the active dialogue engine status."""
    agent = make_bare_agent()
    agent.args.engine = "intent"
    reply, intent = agent._resolve_dialogue_response("what engine are you using")
    assert intent == "system_engine_query"
    assert reply == "The dialogue engine is intent."

    agent.args.engine = "llm"
    reply, intent = agent._resolve_dialogue_response("what is your dialogue engine")
    assert intent == "system_engine_query"
    assert reply == "The dialogue engine is neural L L M."


def test_agent_voice_command_fallback_query_and_toggle():
    """Speaking fallback voice commands queries, enables, and disables fallback."""
    agent = make_bare_agent(llm_fallback_enabled=False)
    mock_llm = MagicMock()
    agent._build_llm = MagicMock(return_value=mock_llm)

    # 1. Query while disabled
    reply, intent = agent._resolve_dialogue_response("what is your fallback status")
    assert intent == "system_fallback_query"
    assert reply == "L L M fallback is disabled."

    # 2. Enable
    reply, intent = agent._resolve_dialogue_response("enable llm fallback")
    assert intent == "system_fallback_enable"
    assert reply == "L L M fallback is enabled."
    assert agent.llm_fallback_enabled is True

    # 3. Query while enabled
    reply, intent = agent._resolve_dialogue_response("is fallback enabled")
    assert intent == "system_fallback_query"
    assert reply == "L L M fallback is enabled."

    # 4. Disable
    reply, intent = agent._resolve_dialogue_response("disable llm fallback")
    assert intent == "system_fallback_disable"
    assert reply == "L L M fallback is disabled."
    assert agent.llm_fallback_enabled is False


def test_agent_voice_command_shutdown_triggers_request_shutdown():
    """Speaking 'shut down' in an utterance triggers host shutdown."""
    agent = make_bare_agent()
    agent.stt.transcribe.return_value = "grimjaw shut down"
    agent.vad.calculate_rms.return_value = 0.05
    agent.request_shutdown = MagicMock()

    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    agent.request_shutdown.assert_called_once()
    agent.logger.log_interaction.assert_called_once()
    kwargs = agent.logger.log_interaction.call_args.kwargs
    assert kwargs["matched_intent"] == "system_shutdown"


def test_agent_voice_command_two_step_password_challenge_response():
    """Speaking shutdown triggers challenge, following up with password completes the command."""
    agent = make_bare_agent(voice_command_password="alsageist")
    # Step 1: User says "shut down"
    reply1, intent1 = agent._resolve_dialogue_response("shut down")
    assert intent1 == "system_access_denied"
    assert "access denied" in reply1.lower()
    assert agent._pending_privileged_command is not None

    # Step 2: User responds with speech recognizer phonetic alias "Auf die Geist."
    reply2, intent2 = agent._resolve_dialogue_response("Auf die Geist.")
    assert intent2 == "system_shutdown"
    assert agent._pending_privileged_command is None


def test_agent_voice_command_combined_password_phrase():
    """User speaking 'Shut down. Password, Alphageist.' succeeds in a single utterance."""
    agent = make_bare_agent(voice_command_password="alsageist")
    reply, intent = agent._resolve_dialogue_response("Shut down. Password, Alphageist.")
    assert intent == "system_shutdown"
    assert agent._pending_privileged_command is None
    assert kwargs["status"] == "PROCESSED"


def test_agent_voice_command_privileged_access_denied():
    """Privileged voice command without password returns access denied announcement."""
    agent = make_bare_agent(is_online=True, voice_command_password="alsageist")
    reply, intent = agent._resolve_dialogue_response("switch to offline mode")
    assert intent == "system_access_denied"
    assert "access denied" in reply.lower()
    assert agent.is_online is True


def test_agent_voice_command_privileged_with_password():
    """Privileged voice command with valid password succeeds."""
    agent = make_bare_agent(is_online=True, voice_command_password="alsageist")
    reply, intent = agent._resolve_dialogue_response("alsageist switch to offline mode")
    assert intent == "system_mode_offline"
    assert agent.is_online is False


def test_agent_voice_command_shutdown_denied_without_password():
    """Speaking shutdown without password does not trigger host shutdown."""
    agent = make_bare_agent(voice_command_password="alsageist")
    agent.stt.transcribe.return_value = "grimjaw shut down"
    agent.vad.calculate_rms.return_value = 0.05
    agent.request_shutdown = MagicMock()

    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    agent.request_shutdown.assert_not_called()
    agent.logger.log_interaction.assert_called_once()
    kwargs = agent.logger.log_interaction.call_args.kwargs
    assert kwargs["matched_intent"] == "system_access_denied"


def test_agent_voice_command_shutdown_allowed_with_password():
    """Speaking shutdown with password triggers host shutdown."""
    agent = make_bare_agent(voice_command_password="alsageist")
    agent.stt.transcribe.return_value = "grimjaw alsageist shut down"
    agent.vad.calculate_rms.return_value = 0.05
    agent.request_shutdown = MagicMock()

    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    agent.request_shutdown.assert_called_once()
    agent.logger.log_interaction.assert_called_once()
    kwargs = agent.logger.log_interaction.call_args.kwargs
    assert kwargs["matched_intent"] == "system_shutdown"


def test_agent_voice_command_two_step_password_challenge_response():
    """Speaking shutdown triggers challenge, following up with password completes the command."""
    agent = make_bare_agent(voice_command_password="alsageist")
    # Step 1: User says "shut down"
    reply1, intent1 = agent._resolve_dialogue_response("shut down")
    assert intent1 == "system_access_denied"
    assert "access denied" in reply1.lower()
    assert agent._pending_privileged_command is not None

    # Step 2: User responds with speech recognizer phonetic alias "Auf die Geist."
    reply2, intent2 = agent._resolve_dialogue_response("Auf die Geist.")
    assert intent2 == "system_shutdown"
    assert agent._pending_privileged_command is None


def test_agent_voice_command_combined_password_phrase():
    """User speaking 'Shut down. Password, Alphageist.' succeeds in a single utterance."""
    agent = make_bare_agent(voice_command_password="alsageist")
    reply, intent = agent._resolve_dialogue_response("Shut down. Password, Alphageist.")
    assert intent == "system_shutdown"
    assert agent._pending_privileged_command is None


def test_build_startup_announcement_verbose_online_with_ssid_and_ip():
    """Verify verbose on-boot announcement includes product, version, SSID, IP, volume, and greeting."""
    agent = make_bare_agent()
    agent.profile = PersonalityProfile(
        personality="grimjaw",
        display_name="Grimjaw",
        description="Gargoyle guardian",
        voice="alan",
        startup_greeting="Grimjaw is awake. What do you want?",
    )
    agent.args.verbose_boot = True
    agent.is_online = True
    agent.audio.volume = 0.8

    with patch("core.wifi_manager.get_current_wifi_ssid", return_value="SpookyNet"), \
         patch("core.system_utils.get_local_ip", return_value="192.168.1.150"), \
         patch("services.agent.read_undervoltage_alarm", return_value=False), \
         patch("services.agent.read_power_status", return_value=None):
        text = agent._build_startup_announcement()

    assert "AlsaGeist version 1.1.0." in text
    assert f"{agent.persona_name} persona active." in text
    assert "Connected to Wi-Fi SpookyNet." in text
    assert "IP address 192.168.1.150." in text
    assert "Volume 80 percent." in text
    assert "Warning: Power supply under-voltage detected." not in text
    assert "Grimjaw is awake. What do you want?" in text


def test_build_startup_announcement_verbose_online_no_ssid():
    """Verify verbose on-boot announcement handles online state without visible SSID (e.g. Ethernet)."""
    agent = make_bare_agent()
    agent.args.verbose_boot = True
    agent.is_online = True
    agent.audio.volume = 0.5

    with patch("core.wifi_manager.get_current_wifi_ssid", return_value=None), \
         patch("core.system_utils.get_local_ip", return_value="10.0.0.22"), \
         patch("services.agent.read_undervoltage_alarm", return_value=False), \
         patch("services.agent.read_power_status", return_value=None):
        text = agent._build_startup_announcement()

    assert "AlsaGeist version 1.1.0." in text
    assert "Connected to network." in text
    assert "IP address 10.0.0.22." in text
    assert "Volume 50 percent." in text


def test_build_startup_announcement_verbose_offline():
    """Verify verbose on-boot announcement states offline mode when offline."""
    agent = make_bare_agent()
    agent.args.verbose_boot = True
    agent.is_online = False
    agent.audio.volume = 0.65

    with patch("services.agent.read_undervoltage_alarm", return_value=False), \
         patch("services.agent.read_power_status", return_value=None):
        text = agent._build_startup_announcement()

    assert "AlsaGeist version 1.1.0." in text
    assert "Operating in offline mode." in text
    assert "Connected to Wi-Fi" not in text
    assert "IP address" not in text
    assert "Volume 65 percent." in text


def test_build_startup_announcement_verbose_undervoltage_warning():
    """Verify power under-voltage warning is included if under-voltage is detected at boot."""
    agent = make_bare_agent()
    agent.args.verbose_boot = True
    agent.is_online = True

    with patch("core.wifi_manager.get_current_wifi_ssid", return_value="HomeNet"), \
         patch("core.system_utils.get_local_ip", return_value="192.168.1.10"), \
         patch("services.agent.read_undervoltage_alarm", return_value=True):
        text = agent._build_startup_announcement()

    assert "Warning: Power supply under-voltage detected." in text


def test_build_startup_announcement_non_verbose():
    """Verify non-verbose mode returns only the concise character greeting."""
    agent = make_bare_agent()
    agent.profile = PersonalityProfile(
        personality="grimjaw",
        display_name="Grimjaw",
        description="Gargoyle guardian",
        voice="alan",
        startup_greeting="Grimjaw is awake. What do you want?",
    )
    agent.args.verbose_boot = False
    agent.is_online = True

    text = agent._build_startup_announcement()
    assert "AlsaGeist version" not in text
    assert "Connected to" not in text
    assert "Volume" not in text
    assert text == "Grimjaw is awake. What do you want?"


def test_play_startup_greeting_speaks_built_announcement():
    """Verify _play_startup_greeting passes announcement to speak()."""
    agent = make_bare_agent()
    agent._build_startup_announcement = MagicMock(return_value="Test startup message.")
    agent.speak = MagicMock()

    agent._play_startup_greeting()
    agent.speak.assert_called_once_with("Test startup message.", blocking=True, cooldown_sec=0.0)


def test_startup_offline_announcement_verbose_vs_non_verbose():
    """Verify offline mode is announced in startup greeting when verbose, or via _announce_mode when non-verbose."""
    agent = make_bare_agent()
    agent.args = make_args(no_greeting=False)
    agent.args.verbose_boot = True
    agent.is_online = False
    agent.network.is_online = False
    agent.network.apply_first_result = MagicMock()
    agent.network.poll = MagicMock(return_value=None)
    agent.logger.jsonl_path = "unused"
    agent._handle_wifi_startup_fallback = MagicMock()
    agent._play_startup_greeting = MagicMock()
    agent._announce_mode = MagicMock()
    agent.audio.read_frame.return_value = None

    def stop_agent():
        time.sleep(0.05)
        agent.running = False

    threading.Thread(target=stop_agent, daemon=True).start()
    agent.start()

    agent._play_startup_greeting.assert_called_once()
    agent._announce_mode.assert_not_called()

    agent.running = False
    agent.args.verbose_boot = False
    agent._play_startup_greeting.reset_mock()
    agent._announce_mode.reset_mock()

    threading.Thread(target=stop_agent, daemon=True).start()
    agent.start()

    agent._play_startup_greeting.assert_called_once()
    agent._announce_mode.assert_called_once_with(online=False)


def test_agent_voice_command_mute_execution():
    """Verify executing COMMAND_MUTE mutes microphone capture and sets user_muted state."""
    agent = make_bare_agent(voice_command_password="alsageist")
    reply, intent = agent._resolve_dialogue_response("alsageist mute yourself")
    assert intent == "system_mute"
    assert reply == "Microphone muted."
    assert agent.user_muted is True
    assert agent.mic_muted is True


def test_agent_voice_command_mute_two_step_challenge():
    """Speaking 'mute' triggers challenge, following up with password mutes the microphone."""
    agent = make_bare_agent(voice_command_password="alsageist")
    # Step 1: User says "grimjaw mute" without password
    reply1, intent1 = agent._resolve_dialogue_response("grimjaw mute")
    assert intent1 == "system_access_denied"
    assert "access denied" in reply1.lower()
    assert agent._pending_privileged_command == ("system_mute", agent._pending_privileged_command[1])
    assert agent.user_muted is False

    # Step 2: User responds with password
    reply2, intent2 = agent._resolve_dialogue_response("alsageist")
    assert intent2 == "system_mute"
    assert reply2 == "Microphone muted."
    assert agent.user_muted is True
    assert agent.mic_muted is True


def test_build_startup_announcement_cache_status_reporting():
    """Verify startup announcement announces pre-warmed audio cache status when engine and tts exist."""
    agent = make_bare_agent()
    agent.args.verbose_boot = True
    agent.is_online = False
    agent.audio.volume = 0.5

    # Case 1: Audio cache fully pre-warmed (0 missing out of 100)
    with patch.object(agent, "_check_audio_cache_status", return_value=(0, 100)), \
         patch("services.agent.read_undervoltage_alarm", return_value=False), \
         patch("services.agent.read_power_status", return_value=None):
        text = agent._build_startup_announcement()
        assert "Audio cache fully pre-warmed." in text
        assert "not pre-warmed" not in text

    # Case 2: Audio cache has unwarmed responses (5 missing out of 100)
    with patch.object(agent, "_check_audio_cache_status", return_value=(5, 100)), \
         patch("services.agent.read_undervoltage_alarm", return_value=False), \
         patch("services.agent.read_power_status", return_value=None):
        text = agent._build_startup_announcement()
        assert "Warning: 5 responses not pre-warmed." in text
        assert "fully pre-warmed" not in text




