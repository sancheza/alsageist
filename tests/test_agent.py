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
    agent.last_net_check_time = 0.0
    agent.net_check_interval = 10.0
    agent._net_check_thread = None
    agent._net_check_result = None
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
        agent._net_check_thread.join(timeout=2)

    assert probe_threads and probe_threads[0] is not threading.current_thread()
    assert agent.is_online is True  # result applies on the next poll


def test_network_poll_applies_result_and_announces_change():
    """A published result flips the mode once and announces it."""
    agent = _agent_for_network_polling()
    agent.last_net_check_time = 100.0
    agent._net_check_result = False
    agent._announce_mode = MagicMock()

    agent._poll_network(now=101.0)
    assert agent.is_online is False
    agent._announce_mode.assert_called_once_with(online=False)
    assert agent._net_check_result is None

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
        first = agent._net_check_thread
        agent.last_net_check_time = 0.0
        agent._poll_network(now=200.0)
        assert agent._net_check_thread is first
        release.set()
        first.join(timeout=2)


def test_weather_intent_uses_cache_only_when_offline():
    """Offline mode answers weather questions without contacting the network."""
    agent = make_bare_agent(is_online=False)
    agent.dialogue_engine.classify_and_respond.return_value = ("canned", "weather_and_temperature")
    agent._resolve_dialogue_response("what's the weather")
    agent.basic_info.get_weather_response.assert_called_once_with(allow_network=False)


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
    agent.last_net_check_time = time.time()
    agent.net_check_interval = 3600.0
    agent._net_check_thread = None
    agent._net_check_result = None
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

    agent._net_check_result = True  # probe says the network is up
    agent._announce_mode = MagicMock()
    agent._poll_network(now=agent.last_net_check_time + 1.0)

    assert agent.is_online is False
    agent._announce_mode.assert_not_called()
    assert agent._net_check_thread is None  # no probe is started while forced


def test_auto_mode_resumes_detection():
    """set_online(None) clears the override and schedules an immediate probe."""
    agent = _agent_for_network_polling()
    agent.set_online(False)
    agent.last_net_check_time = 12345.0

    agent.set_online(None)

    assert agent.mode_forced is False
    assert agent.last_net_check_time == 0.0


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


def test_agent_starts_and_stops_power_monitor():
    """The power monitor runs for the life of the agent loop and is stopped on shutdown."""
    agent = make_bare_agent()
    agent.power_monitor = MagicMock()
    agent.audio.read_frame.return_value = None
    agent.logger.jsonl_path = "unused"
    agent.last_net_check_time = time.time()
    agent.net_check_interval = 3600.0
    agent._net_check_thread = None
    agent._net_check_result = None

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
    agent.last_net_check_time = 1000.0
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
    assert agent.last_net_check_time == 0.0


def test_healthy_cloud_stt_does_not_request_a_check():
    """Normal operation leaves the 15-minute schedule alone."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.last_cloud_failed = False
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.last_net_check_time == 1000.0


def test_stt_total_failure_also_requests_a_check():
    """A transcription that raises after cloud attempts still asks for a probe."""
    agent = _agent_with_failed_cloud_stt()
    agent.stt.transcribe.side_effect = RuntimeError("All transcription backends failed")
    with patch("services.agent.time.sleep"):
        agent._handle_utterance(np.zeros(16000, dtype=np.float32))
    assert agent.last_net_check_time == 0.0


def test_check_not_requested_when_mode_is_forced_or_already_offline():
    """A forced mode is respected, and an offline agent waits for the periodic check to come back."""
    forced = _agent_with_failed_cloud_stt(_mode_forced=True)
    forced._request_network_check()
    assert forced.last_net_check_time == 1000.0

    offline = _agent_with_failed_cloud_stt(is_online=False)
    offline._request_network_check()
    assert offline.last_net_check_time == 1000.0


def test_check_requests_are_rate_limited():
    """Repeated failures inside 30 seconds do not keep re-triggering probes."""
    agent = _agent_with_failed_cloud_stt()
    agent._request_network_check()
    assert agent.last_net_check_time == 0.0

    agent.last_net_check_time = 5000.0       # the probe ran and updated the time
    agent._request_network_check()           # a second failure right after
    assert agent.last_net_check_time == 5000.0

    agent._last_check_request -= 31          # 31 seconds later a new request is allowed
    agent._request_network_check()
    assert agent.last_net_check_time == 0.0


def test_cloud_failure_with_network_down_goes_offline_and_announces():
    """End to end: failed cloud STT, probe finds no network, mode flips offline once."""
    agent = _agent_with_failed_cloud_stt()
    agent._announce_mode = MagicMock()
    agent._handle_utterance(np.zeros(16000, dtype=np.float32))

    with patch.object(AlsaGeistAgent, "_check_internet", return_value=False):
        agent._poll_network(now=time.time())          # the loop starts the probe
        agent._net_check_thread.join(timeout=2)
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
        agent._net_check_thread.join(timeout=2)
        agent._poll_network(now=time.time())

    assert agent.is_online is True
    agent._announce_mode.assert_not_called()
