"""LED and Arcade Button controller with hardware and mock implementations.

Controls status LED states (Idle breathing, Listening solid, Speaking flicker, Error blink)
and reads arcade button trigger events.
"""

from abc import ABC, abstractmethod
import enum
import math
import sys
import threading
import time
from typing import Callable, Optional


class LEDState(enum.Enum):
    """LED operating states."""
    IDLE = "idle"            # Smooth PWM breathing pulse (0.5 Hz)
    LISTENING = "listening"  # Solid 100% brightness
    SPEAKING = "speaking"    # Dynamic fast audio-reactive flicker
    ERROR = "error"          # Rapid alert blink (5 Hz)
    OFF = "off"              # Completely dark


class BaseLEDController(ABC):
    """Abstract base class for LED and button controllers."""

    def __init__(self) -> None:
        self.current_state: LEDState = LEDState.OFF
        self.button_callback: Optional[Callable[[], None]] = None

    @abstractmethod
    def start(self) -> None:
        """Start background LED modulation threads."""

    @abstractmethod
    def stop(self) -> None:
        """Stop modulation and turn off LED."""

    @abstractmethod
    def set_state(self, state: LEDState) -> None:
        """Set the active LED state."""

    def set_button_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for arcade button press events."""
        self.button_callback = callback
        if hasattr(self, "_gpio_btn") and self._gpio_btn is not None:
            self._gpio_btn.when_pressed = callback


class ConsoleLEDController(BaseLEDController):
    """Simulated terminal LED controller with visual status indicators for macOS."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the console LED simulator."""
        self._running = True

    def stop(self) -> None:
        """Stop the console LED simulator."""
        self._running = False
        self.set_state(LEDState.OFF)

    def set_state(self, state: LEDState) -> None:
        """Set simulated LED state and print terminal indicator."""
        with self._lock:
            if self.current_state == state:
                return
            self.current_state = state

            if not self.verbose:
                return

            symbols = {
                LEDState.IDLE: "🟢 [LED: IDLE (Breathing ~ ~ ~)]",
                LEDState.LISTENING: "🟡 [LED: LISTENING (Solid 100%)]",
                LEDState.SPEAKING: "🟣 [LED: SPEAKING (Audio Flicker ⚡)]",
                LEDState.ERROR: "🔴 [LED: ERROR (Fast Blink !)]",
                LEDState.OFF: "⚪ [LED: OFF]",
            }
            msg = symbols.get(state, f"[LED: {state.value}]")
            sys.stdout.write(f"\r{msg}\n")
            sys.stdout.flush()


class RaspberryPiLEDController(BaseLEDController):
    """Hardware LED controller for Raspberry Pi and AIY Voice HAT (V1/V2).

    Supports modern Raspberry Pi OS Bookworm / Pi 5 via gpiozero (lgpio/gpiod),
    with automatic fallback to RPi.GPIO on legacy releases.
    """

    def __init__(self, led_pin: int = 25, button_pin: int = 23) -> None:
        """Initialize AIY Voice HAT GPIO pins.

        Args:
            led_pin: BCM GPIO pin for button LED (default: 25 for Voice HAT V1).
            button_pin: BCM GPIO pin for button switch (default: 23 for Voice HAT V1).
        """
        super().__init__()
        self.led_pin = led_pin
        self.button_pin = button_pin
        self._backend: Optional[str] = None
        self._gpio_led = None
        self._gpio_btn = None
        self._rpi_pwm = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Initialize hardware GPIO and start PWM background thread."""
        # 1. Try modern gpiozero first (Debian 12 Bookworm, Pi 5, Pi 4, Pi 3)
        try:
            from gpiozero import PWMLED, Button  # pylint: disable=import-error
            self._gpio_led = PWMLED(self.led_pin)
            self._gpio_btn = Button(self.button_pin, pull_up=True, bounce_time=0.3)
            if self.button_callback:
                self._gpio_btn.when_pressed = self.button_callback
            self._backend = "gpiozero"
            self._running = True
            self._thread = threading.Thread(target=self._pwm_worker, daemon=True)
            self._thread.start()
            return
        except Exception:
            pass

        # 2. Fallback to legacy RPi.GPIO (Bullseye / Buster)
        try:
            import RPi.GPIO as GPIO  # pylint: disable=import-error
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            self._rpi_pwm = GPIO.PWM(self.led_pin, 100)
            self._rpi_pwm.start(0)

            if self.button_callback:
                GPIO.add_event_detect(
                    self.button_pin,
                    GPIO.FALLING,
                    callback=lambda ch: self.button_callback(),
                    bouncetime=300,
                )

            self._backend = "rpi.gpio"
            self._running = True
            self._thread = threading.Thread(target=self._pwm_worker, daemon=True)
            self._thread.start()
            return
        except Exception:
            sys.stderr.write("Warning: Hardware GPIO unavailable (gpiozero/RPi.GPIO). Running in console mode.\n")

    def _set_duty_cycle(self, duty_percent: float) -> None:
        """Set LED brightness duty cycle (0.0 to 100.0)."""
        clamped = max(0.0, min(100.0, float(duty_percent)))
        if self._backend == "gpiozero" and self._gpio_led:
            self._gpio_led.value = clamped / 100.0
        elif self._backend == "rpi.gpio" and self._rpi_pwm:
            self._rpi_pwm.ChangeDutyCycle(clamped)

    def stop(self) -> None:
        """Stop PWM and clean up GPIO."""
        self._running = False
        if self._backend == "gpiozero":
            if self._gpio_led:
                self._gpio_led.close()
            if self._gpio_btn:
                self._gpio_btn.close()
        elif self._backend == "rpi.gpio":
            if self._rpi_pwm:
                try:
                    self._rpi_pwm.stop()
                    import RPi.GPIO as GPIO  # pylint: disable=import-error
                    GPIO.cleanup([self.led_pin, self.button_pin])
                except Exception:
                    pass

    def set_state(self, state: LEDState) -> None:
        """Update hardware LED operating state."""
        self.current_state = state

    def _pwm_worker(self) -> None:
        """Background thread modulating PWM duty cycle based on current state."""
        step = 0.0
        while self._running:
            if self.current_state == LEDState.IDLE:
                # 0.5 Hz smooth sinusoidal breathing
                duty = (math.sin(step) + 1.0) / 2.0 * 80.0 + 5.0
                self._set_duty_cycle(duty)
                step += 0.08
                time.sleep(0.02)
            elif self.current_state == LEDState.LISTENING:
                self._set_duty_cycle(100.0)
                time.sleep(0.05)
            elif self.current_state == LEDState.SPEAKING:
                # Syllable-rate speech cadence (3-4 Hz organic speaking pulse)
                duty = (math.sin(step) + 1.0) / 2.0 * 85.0 + 15.0
                self._set_duty_cycle(duty)
                step += 0.35
                time.sleep(0.04)
            elif self.current_state == LEDState.ERROR:
                # 5 Hz rapid blink
                self._set_duty_cycle(100.0 if int(step) % 2 == 0 else 0.0)
                step += 1.0
                time.sleep(0.1)
            else:
                self._set_duty_cycle(0.0)
                time.sleep(0.1)


def create_led_controller(verbose: bool = True) -> BaseLEDController:
    """Factory creating platform-appropriate LED controller.

    Returns:
        RaspberryPiLEDController if running on Raspberry Pi hardware,
        otherwise ConsoleLEDController.
    """
    is_pi = False
    # Check Linux device-tree model (standard on Raspberry Pi OS 64-bit Bookworm/Bullseye)
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8") as f:
            if "Raspberry Pi" in f.read():
                is_pi = True
    except Exception:
        pass

    if not is_pi:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                cpuinfo = f.read()
                if "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo:
                    is_pi = True
        except Exception:
            pass

    if is_pi:
        return RaspberryPiLEDController()

    return ConsoleLEDController(verbose=verbose)
