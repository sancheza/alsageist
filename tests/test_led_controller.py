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
