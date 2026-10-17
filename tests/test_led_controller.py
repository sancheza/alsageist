"""Unit tests for LED and Button Controller."""

from core.led_controller import ConsoleLEDController, LEDState, create_led_controller


def test_console_led_state_transitions():
    led = ConsoleLEDController(verbose=False)
    led.start()
    assert led.current_state == LEDState.OFF

    led.set_state(LEDState.IDLE)
    assert led.current_state == LEDState.IDLE

    led.set_state(LEDState.LISTENING)
    assert led.current_state == LEDState.LISTENING

    led.set_state(LEDState.SPEAKING)
    assert led.current_state == LEDState.SPEAKING

    led.stop()
    assert led.current_state == LEDState.OFF


def test_button_callback_registration():
    led = ConsoleLEDController(verbose=False)
    called = False

    def on_click():
        nonlocal called
        called = True

    led.set_button_callback(on_click)
    assert led.button_callback is not None
    led.button_callback()
    assert called


def test_factory_creation():
    controller = create_led_controller(verbose=False)
    assert controller is not None


def test_console_led_flash():
    led = ConsoleLEDController(verbose=False)
    led.start()
    led.set_state(LEDState.LISTENING)
    assert led.current_state == LEDState.LISTENING

    led.flash(count=3)
    assert led.flash_count == 3
    assert led.current_state == LEDState.THINKING


def test_pi_led_flash_fallback():
    from core.led_controller import RaspberryPiLEDController
    led = RaspberryPiLEDController(led_pin=25, button_pin=23)
    led.start()
    led.set_state(LEDState.LISTENING)
    assert led.current_state == LEDState.LISTENING

    led.flash(count=3, on_time=0.001, off_time=0.001)
    assert led.current_state == LEDState.THINKING
    assert not led._flashing
    led.stop()


def test_button_press_and_hold_callbacks():
    led = ConsoleLEDController(verbose=False)
    press_called = False
    hold_called = False

    def on_press():
        nonlocal press_called
        press_called = True

    def on_hold():
        nonlocal hold_called
        hold_called = True

    led.set_button_callbacks(on_press=on_press, on_hold=on_hold, hold_time_sec=5.0)
    assert led.button_callback is not None
    assert led.button_hold_callback is not None
    assert led.hold_time_sec == 5.0

    led.trigger_button_press()
    assert press_called
    assert not hold_called

    led.trigger_button_hold()
    assert hold_called


def test_rpi_led_controller_hold_setup():
    from core.led_controller import RaspberryPiLEDController
    led = RaspberryPiLEDController(led_pin=25, button_pin=23, hold_time_sec=5.0)
    assert led.hold_time_sec == 5.0

    press_called = False
    hold_called = False

    def on_p():
        nonlocal press_called
        press_called = True

    def on_h():
        nonlocal hold_called
        hold_called = True

    led.set_button_callbacks(on_press=on_p, on_hold=on_h, hold_time_sec=5.0)
    assert led.button_callback is not None
    assert led.button_hold_callback is not None


def test_agent_mute_toggle_on_button_hold():
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import argparse

    args = argparse.Namespace(
        engine="intent",
        input_device=None,
        output_device=None,
        volume=0.5,
        mic_gain=1.0,
        silence_timeout=0.6,
        threshold=0.003,
        min_speech_sec=0.2,
        tempo=1.0,
        noise=0.667,
        model_path=None,
        no_greeting=True,
    )

    agent = make_bare_agent()
    agent.args = args
    agent.running = False
    agent.is_disabled = False
    agent.audio = MagicMock()
    agent.audio.volume = 0.5
    agent.vad = MagicMock()
    agent.dialogue_engine = MagicMock()
    agent.tts = MagicMock()
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)
    agent.led.set_state(LEDState.IDLE)

    # 1. First 5-second hold: mutes and disables Grimjaw
    agent._on_button_hold()
    assert agent.is_disabled is True
    assert agent.led.current_state == LEDState.OFF
    agent.audio.stop_playback.assert_called_once()
    agent.audio.mute_capture.assert_called_once()

    # 2. While disabled, short press should be ignored
    agent.audio.volume = 0.5
    agent._on_button_press()
    assert agent.audio.volume == 0.5  # volume did not cycle

    # 3. Second 5-second hold: unmutes and re-enables Grimjaw
    agent._on_button_hold()
    assert agent.is_disabled is False
    assert agent.led.current_state == LEDState.IDLE
    agent.audio.unmute_capture.assert_called_once()


def test_console_led_thinking_state():
    led = ConsoleLEDController(verbose=False)
    led.start()
    led.set_state(LEDState.THINKING)
    assert led.current_state == LEDState.THINKING
    led.stop()


def test_agent_llm_fallback_custom_response():
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import numpy as np

    agent = make_bare_agent()
    agent.args = MagicMock(engine="intent")
    agent.running = False
    agent.is_disabled = False
    agent.is_online = True
    agent.llm_fallback_enabled = True
    agent.audio = MagicMock()
    agent.audio.sample_rate = 16000
    agent.vad = MagicMock()
    agent.vad.calculate_rms.return_value = 0.05
    agent.stt = MagicMock()
    agent.stt.transcribe.return_value = "What is the airspeed velocity of an unladen swallow?"
    agent.dialogue_engine = MagicMock()
    agent.dialogue_engine.classify_and_respond.return_value = ("Generic fallback", "fallback_unrecognized")
    agent.llm_fallback_service = MagicMock()
    agent.llm_fallback_service.generate_response.return_value = "African or European swallow, meatbag?!"
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (np.zeros(1600, dtype=np.float32), 22050)
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)

    utterance = np.zeros(16000, dtype=np.float32)
    agent._handle_utterance(utterance)

    # Verified custom response fetched and passed to TTS
    agent.llm_fallback_service.generate_response.assert_called_once_with(
        "What is the airspeed velocity of an unladen swallow?"
    )
    agent.tts.synthesize.assert_called_once_with("African or European swallow, meatbag?!", cache=True)
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "llm_fallback_custom"


def test_agent_llm_fallback_disabled_uses_generic():
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import numpy as np

    agent = make_bare_agent()
    agent.args = MagicMock(engine="intent")
    agent.running = False
    agent.is_disabled = False
    agent.is_online = True
    agent.llm_fallback_enabled = False
    agent.audio = MagicMock()
    agent.audio.sample_rate = 16000
    agent.vad = MagicMock()
    agent.vad.calculate_rms.return_value = 0.05
    agent.stt = MagicMock()
    agent.stt.transcribe.return_value = "Random unknown query"
    agent.dialogue_engine = MagicMock()
    agent.dialogue_engine.classify_and_respond.return_value = ("Generic fallback joke", "fallback_unrecognized")
    agent.llm_fallback_service = MagicMock()
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (np.zeros(1600, dtype=np.float32), 22050)
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)

    utterance = np.zeros(16000, dtype=np.float32)
    agent._handle_utterance(utterance)

    # LLM service should not be called when fallback is disabled
    agent.llm_fallback_service.generate_response.assert_not_called()
    agent.tts.synthesize.assert_called_once_with("Generic fallback joke", cache=True)
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "fallback_unrecognized"


def test_agent_llm_fallback_offline_degradation():
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import numpy as np

    agent = make_bare_agent()
    agent.args = MagicMock(engine="intent")
    agent.running = False
    agent.is_disabled = False
    agent.is_online = False  # OFFLINE
    agent.llm_fallback_enabled = True
    agent.audio = MagicMock()
    agent.audio.sample_rate = 16000
    agent.vad = MagicMock()
    agent.vad.calculate_rms.return_value = 0.05
    agent.stt = MagicMock()
    agent.stt.transcribe.return_value = "Unknown offline query"
    agent.dialogue_engine = MagicMock()
    agent.dialogue_engine.classify_and_respond.return_value = ("Offline fallback line", "fallback_unrecognized")
    agent.llm_fallback_service = MagicMock()
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (np.zeros(1600, dtype=np.float32), 22050)
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)

    utterance = np.zeros(16000, dtype=np.float32)
    agent._handle_utterance(utterance)

    # When offline, LLM should not be called and generic fallback is used
    agent.llm_fallback_service.generate_response.assert_not_called()
    agent.tts.synthesize.assert_called_once_with("Offline fallback line", cache=True)


def test_agent_llm_fallback_error_degradation():
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import numpy as np

    agent = make_bare_agent()
    agent.args = MagicMock(engine="intent")
    agent.running = False
    agent.is_disabled = False
    agent.is_online = True
    agent.llm_fallback_enabled = True
    agent.audio = MagicMock()
    agent.audio.sample_rate = 16000
    agent.vad = MagicMock()
    agent.vad.calculate_rms.return_value = 0.05
    agent.stt = MagicMock()
    agent.stt.transcribe.return_value = "Query that causes timeout"
    agent.dialogue_engine = MagicMock()
    agent.dialogue_engine.classify_and_respond.return_value = ("Safe fallback line", "fallback_unrecognized")
    agent.llm_fallback_service = MagicMock()
    agent.llm_fallback_service.generate_response.side_effect = TimeoutError("Groq API timeout")
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (np.zeros(1600, dtype=np.float32), 22050)
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)

    utterance = np.zeros(16000, dtype=np.float32)
    agent._handle_utterance(utterance)

    # When LLM throws error, degrades gracefully to generic fallback
    agent.tts.synthesize.assert_called_once_with("Safe fallback line", cache=True)
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "fallback_unrecognized"


def test_button_long_hold_shutdown():
    """Verify 10-second button hold invokes long hold callback."""
    led = ConsoleLEDController(verbose=False)
    press_called = False
    hold_called = False
    long_hold_called = False

    def on_p():
        nonlocal press_called
        press_called = True

    def on_h():
        nonlocal hold_called
        hold_called = True

    def on_lh():
        nonlocal long_hold_called
        long_hold_called = True

    led.set_button_callbacks(
        on_press=on_p,
        on_hold=on_h,
        on_long_hold=on_lh,
        hold_time_sec=5.0,
        long_hold_time_sec=10.0,
    )
    assert led.button_long_hold_callback is not None
    assert led.long_hold_time_sec == 10.0

    led.trigger_button_long_hold()
    assert long_hold_called
    assert not press_called
    assert not hold_called


def test_agent_shutdown_on_button_long_hold(monkeypatch):
    """Verify agent announces Powering Off and triggers shutdown on 10s hold."""
    from unittest.mock import MagicMock
    from tests.agent_helpers import make_bare_agent
    import argparse
    import numpy as np

    args = argparse.Namespace(
        engine="intent",
        input_device=None,
        output_device=None,
        volume=0.5,
        mic_gain=1.0,
        silence_timeout=0.6,
        threshold=0.003,
        min_speech_sec=0.2,
        tempo=1.0,
        noise=0.667,
        model_path=None,
        no_greeting=True,
    )

    agent = make_bare_agent()
    agent.args = args
    agent.running = False
    agent.is_disabled = True
    agent.audio = MagicMock()
    agent.audio.volume = 0.0
    agent.vad = MagicMock()
    agent.dialogue_engine = MagicMock()
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (np.zeros(1600, dtype=np.float32), 22050)
    agent.logger = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)
    agent.led.set_state(LEDState.IDLE)

    shutdown_called = False

    def mock_shutdown(dry_run=False):
        nonlocal shutdown_called
        shutdown_called = True
        return True

    monkeypatch.setattr("core.system_utils.trigger_host_shutdown", mock_shutdown)

    agent._on_button_long_hold()

    agent.tts.synthesize.assert_called_with("Powering off.")
    agent.audio.play_audio.assert_called_once()
    assert agent.audio.volume == 0.5  # Restored volume for announcement
    assert agent.led.current_state == LEDState.OFF
    assert shutdown_called is True


def test_create_led_controller_custom_params_and_env(monkeypatch):
    """Verify create_led_controller respects custom parameters and environment variables."""
    # Test explicit params
    ctrl = create_led_controller(
        verbose=False,
        led_pin=18,
        button_pin=24,
        hold_time_sec=3.5,
        long_hold_time_sec=7.0,
    )
    assert ctrl.hold_time_sec == 3.5
    assert ctrl.long_hold_time_sec == 7.0

    # Test environment variable resolution
    monkeypatch.setenv("BUTTON_HOLD_SEC", "4.2")
    monkeypatch.setenv("BUTTON_SHUTDOWN_SEC", "8.5")
    ctrl_env = create_led_controller(verbose=False)
    assert ctrl_env.hold_time_sec == 4.2
    assert ctrl_env.long_hold_time_sec == 8.5





def _tracker_events(hold_sec=0.05, long_hold_sec=0.4):
    """Return a controller, its gesture tracker, and the list collecting callback events."""
    from core.led_controller import ButtonGestureTracker

    led = ConsoleLEDController(verbose=False)
    events = []
    led.set_button_callbacks(
        on_press=lambda: events.append("press"),
        on_hold=lambda: events.append("hold"),
        on_long_hold=lambda: events.append("long_hold"),
        hold_time_sec=hold_sec,
        long_hold_time_sec=long_hold_sec,
    )
    return ButtonGestureTracker(led), events


def test_gesture_tracker_short_press():
    """A quick press and release fires only the press callback."""
    import time
    tracker, events = _tracker_events()
    tracker.pressed()
    tracker.released()
    time.sleep(0.6)  # timers were cancelled
    assert events == ["press"]


def test_gesture_tracker_hold_suppresses_press():
    """Holding past hold_time fires hold once and no press on release."""
    import time
    tracker, events = _tracker_events()
    tracker.pressed()
    time.sleep(0.2)
    tracker.released()
    time.sleep(0.5)  # long-hold timer was cancelled
    assert events == ["hold"]


def test_gesture_tracker_long_hold_suppresses_hold():
    """Holding past long_hold_time fires long_hold on timeout and suppresses hold on release."""
    import time
    tracker, events = _tracker_events()
    tracker.pressed()
    time.sleep(0.8)
    tracker.released()
    assert events == ["long_hold"]


def test_gesture_tracker_reads_updated_callbacks():
    """Callbacks registered after construction are used for later gestures."""
    tracker, events = _tracker_events()
    tracker._controller.button_callback = lambda: events.append("replaced")
    tracker.pressed()
    tracker.released()
    assert events == ["replaced"]


def test_gesture_tracker_callback_error_is_reported(capsys):
    """A failing callback is reported on stderr and does not raise."""
    from core.led_controller import ButtonGestureTracker

    led = ConsoleLEDController(verbose=False)

    def boom():
        raise RuntimeError("boom")

    led.set_button_callbacks(on_press=boom)
    tracker = ButtonGestureTracker(led)
    tracker.pressed()
    tracker.released()
    assert "press callback error: boom" in capsys.readouterr().err


def test_pi_flash_is_deferred_to_pwm_thread():
    """flash() returns at once; the queued pulses run once on the PWM thread."""
    from unittest.mock import patch
    from core.led_controller import RaspberryPiLEDController

    led = RaspberryPiLEDController(led_pin=25, button_pin=23)
    led.set_state(LEDState.LISTENING)
    with patch.object(led, "_set_duty_cycle") as duty:
        led.flash(count=2, on_time=0.001, off_time=0.001)
        assert led.current_state == LEDState.THINKING
        duty.assert_not_called()

        assert led._run_pending_flash() is True
        assert [c.args[0] for c in duty.call_args_list] == [100.0, 0.0, 100.0, 0.0]
        assert led._run_pending_flash() is False


def test_pi_flash_does_not_wake_disabled_led():
    """flash() leaves an OFF LED in the OFF state."""
    from core.led_controller import RaspberryPiLEDController

    led = RaspberryPiLEDController(led_pin=25, button_pin=23)
    led.set_state(LEDState.OFF)
    led.flash(count=1)
    assert led.current_state == LEDState.OFF
