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
    from main import AlsaGeistAgent
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

    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
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
    from main import AlsaGeistAgent
    import numpy as np

    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
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
    agent.tts.synthesize.assert_called_once_with("African or European swallow, meatbag?!")
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "llm_fallback_custom"


def test_agent_llm_fallback_disabled_uses_generic():
    from unittest.mock import MagicMock
    from main import AlsaGeistAgent
    import numpy as np

    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
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
    agent.tts.synthesize.assert_called_once_with("Generic fallback joke")
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "fallback_unrecognized"


def test_agent_llm_fallback_offline_degradation():
    from unittest.mock import MagicMock
    from main import AlsaGeistAgent
    import numpy as np

    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
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
    agent.tts.synthesize.assert_called_once_with("Offline fallback line")


def test_agent_llm_fallback_error_degradation():
    from unittest.mock import MagicMock
    from main import AlsaGeistAgent
    import numpy as np

    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
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
    agent.tts.synthesize.assert_called_once_with("Safe fallback line")
    assert agent.logger.log_interaction.call_args[1]["matched_intent"] == "fallback_unrecognized"

