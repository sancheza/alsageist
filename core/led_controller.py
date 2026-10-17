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
from typing import Callable, Optional, Tuple

from core.env_utils import env_float, env_int


class LEDState(enum.Enum):
    """LED operating states."""
    IDLE = "idle"            # Smooth PWM breathing pulse (0.5 Hz)
    LISTENING = "listening"  # Solid 100% brightness
    THINKING = "thinking"    # Heartbeat double-pulse (~1.1 Hz lub-dub)
    SPEAKING = "speaking"    # Dynamic fast audio-reactive flicker
    ERROR = "error"          # Rapid alert blink (5 Hz)
    OFF = "off"              # Completely dark


class BaseLEDController(ABC):
    """Abstract base class for LED and button controllers."""

    def __init__(self) -> None:
        self.current_state: LEDState = LEDState.OFF
        self.button_callback: Optional[Callable[[], None]] = None
        self.button_hold_callback: Optional[Callable[[], None]] = None
        self.button_long_hold_callback: Optional[Callable[[], None]] = None
        self.hold_time_sec: float = 5.0
        self.long_hold_time_sec: float = 10.0

    @abstractmethod
    def start(self) -> None:
        """Start background LED modulation threads."""

    @abstractmethod
    def stop(self) -> None:
        """Stop modulation and turn off LED."""

    @abstractmethod
    def set_state(self, state: LEDState) -> None:
        """Set the active LED state."""

    @abstractmethod
    def flash(self, count: int = 3, on_time: float = 0.09, off_time: float = 0.08) -> None:
        """Flash the LED rapidly a specified number of times."""

    def set_button_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for arcade button press events."""
        self.set_button_callbacks(
            on_press=callback,
            on_hold=self.button_hold_callback,
            on_long_hold=self.button_long_hold_callback,
            hold_time_sec=self.hold_time_sec,
            long_hold_time_sec=self.long_hold_time_sec,
        )

    def set_button_callbacks(
        self,
        on_press: Optional[Callable[[], None]] = None,
        on_hold: Optional[Callable[[], None]] = None,
        on_long_hold: Optional[Callable[[], None]] = None,
        hold_time_sec: float = 5.0,
        long_hold_time_sec: float = 10.0,
    ) -> None:
        """Register callbacks for arcade button events.

        Args:
            on_press: Callback invoked on short press / release (< hold_time_sec).
            on_hold: Callback invoked when button is held continuously for hold_time_sec (5s).
            on_long_hold: Callback invoked when button is held continuously for long_hold_time_sec (10s).
            hold_time_sec: Medium hold duration in seconds (default: 5.0).
            long_hold_time_sec: Long hold duration in seconds (default: 10.0).
        """
        self.button_callback = on_press
        self.button_hold_callback = on_hold
        self.button_long_hold_callback = on_long_hold
        self.hold_time_sec = hold_time_sec
        self.long_hold_time_sec = long_hold_time_sec
        self._wire_hardware_button()

    def _wire_hardware_button(self) -> None:
        """Wire callbacks to underlying hardware if available (overridden by subclasses)."""


class ButtonGestureTracker:
    """Turns raw press and release edges into short-press, hold, and long-hold callbacks.

    Callbacks and hold durations are read from the owning controller when each
    gesture starts, so they can be changed after the hardware is wired.
    """

    def __init__(self, controller: "BaseLEDController") -> None:
        """Create a tracker bound to a controller.

        Args:
            controller: Controller whose button_callback, button_hold_callback,
                button_long_hold_callback, hold_time_sec and long_hold_time_sec are used.
        """
        self._controller = controller
        self._lock = threading.Lock()
        self._hold_timer: Optional[threading.Timer] = None
        self._long_hold_timer: Optional[threading.Timer] = None
        self._hold_fired = False
        self._long_hold_fired = False

    def pressed(self, *_args) -> None:
        """Handle a button press edge: start the hold and long-hold timers."""
        with self._lock:
            self._hold_fired = False
            self._long_hold_fired = False
            self._cancel_timers()
            self._hold_timer = self._start_timer(self._controller.hold_time_sec, self._on_hold)
            self._long_hold_timer = self._start_timer(self._controller.long_hold_time_sec, self._on_long_hold)

    def released(self, *_args) -> None:
        """Handle a button release edge: fire the short-press callback if no hold fired."""
        with self._lock:
            self._cancel_timers()
            fire_short = not self._hold_fired and not self._long_hold_fired
            self._hold_fired = False
            self._long_hold_fired = False

        if fire_short:
            self._invoke(self._controller.button_callback, "press")

    def cancel(self) -> None:
        """Cancel pending timers without firing any callback."""
        with self._lock:
            self._cancel_timers()

    def _cancel_timers(self) -> None:
        """Cancel and clear both timers. Caller holds self._lock."""
        for timer in (self._hold_timer, self._long_hold_timer):
            if timer is not None:
                timer.cancel()
        self._hold_timer = None
        self._long_hold_timer = None

    @staticmethod
    def _start_timer(delay_sec: float, action: Callable[[], None]) -> threading.Timer:
        """Start a daemon timer that runs action after delay_sec."""
        timer = threading.Timer(delay_sec, action)
        timer.daemon = True
        timer.start()
        return timer

    def _on_hold(self) -> None:
        """Timer action: the button has been held for hold_time_sec."""
        with self._lock:
            if not self._long_hold_fired:
                self._hold_fired = True
        self._invoke(self._controller.button_hold_callback, "hold")

    def _on_long_hold(self) -> None:
        """Timer action: the button has been held for long_hold_time_sec."""
        with self._lock:
            self._long_hold_fired = True
        self._invoke(self._controller.button_long_hold_callback, "long hold")

    @staticmethod
    def _invoke(callback: Optional[Callable[[], None]], label: str) -> None:
        """Run a callback, reporting errors instead of raising into a timer thread."""
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            sys.stderr.write(f"Button {label} callback error: {exc}\n")


class ConsoleLEDController(BaseLEDController):
    """Simulated terminal LED controller with visual status indicators for macOS."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self._running = False
        self._lock = threading.Lock()
        self.flash_count: int = 0

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
                LEDState.THINKING: "💓 [LED: THINKING (Heartbeat Double-Pulse)]",
                LEDState.SPEAKING: "🟣 [LED: SPEAKING (Audio Flicker)]",
                LEDState.ERROR: "🔴 [LED: ERROR (Fast Blink !)]",
                LEDState.OFF: "⚪ [LED: OFF]",
            }
            msg = symbols.get(state, f"[LED: {state.value}]")
            sys.stdout.write(f"\r{msg}\n")
            sys.stdout.flush()

    def flash(self, count: int = 3, on_time: float = 0.09, off_time: float = 0.08) -> None:
        """Simulate rapid flashing on the console."""
        with self._lock:
            self.flash_count += count
            if self.current_state != LEDState.OFF:
                self.current_state = LEDState.THINKING
            if not self.verbose:
                return

            sys.stdout.write(f"\r[LED: ACKNOWLEDGED (Flashed {count}x rapidly)]\n")
            sys.stdout.flush()

    def trigger_button_press(self) -> None:
        """Simulate arcade button short press event."""
        if self.button_callback:
            self.button_callback()

    def trigger_button_hold(self) -> None:
        """Simulate arcade button 5-second hold event."""
        if self.button_hold_callback:
            self.button_hold_callback()

    def trigger_button_long_hold(self) -> None:
        """Simulate arcade button 10-second hold event (poweroff)."""
        if self.button_long_hold_callback:
            self.button_long_hold_callback()


class RaspberryPiLEDController(BaseLEDController):
    """Hardware LED controller for Raspberry Pi and AIY Voice HAT (V1/V2).

    Supports modern Raspberry Pi OS Bookworm / Pi 5 via gpiozero (lgpio/gpiod),
    with automatic fallback to RPi.GPIO on legacy releases.
    """

    def __init__(
        self,
        led_pin: int = 25,
        button_pin: int = 23,
        hold_time_sec: float = 5.0,
        long_hold_time_sec: float = 10.0,
    ) -> None:
        """Initialize AIY Voice HAT GPIO pins.

        Args:
            led_pin: BCM GPIO pin for button LED (default: 25 for Voice HAT V1).
            button_pin: BCM GPIO pin for button switch (default: 23 for Voice HAT V1).
            hold_time_sec: Duration in seconds to distinguish hold from short press.
            long_hold_time_sec: Duration in seconds for long-hold shutdown (default: 10.0).
        """
        super().__init__()
        self.led_pin = led_pin
        self.button_pin = button_pin
        self.hold_time_sec = hold_time_sec
        self.long_hold_time_sec = long_hold_time_sec
        self._backend: Optional[str] = None
        self._gpio_led = None
        self._gpio_btn = None
        self._rpi_pwm = None
        self._running = False
        self._flashing = False
        self._flash_request: Optional[Tuple[int, float, float]] = None
        self._flash_lock = threading.Lock()
        self._buttons = ButtonGestureTracker(self)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Initialize hardware GPIO and start PWM background thread."""
        # 1. Try modern gpiozero first (Debian 12 Bookworm, Pi 5, Pi 4, Pi 3)
        try:
            from gpiozero import PWMLED, Button  # pylint: disable=import-error
            self._gpio_led = PWMLED(self.led_pin)
            self._gpio_btn = Button(
                self.button_pin,
                pull_up=True,
                bounce_time=0.05,
            )
            self._wire_gpiozero_button()
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

            self._backend = "rpi.gpio"
            self._wire_rpigpio_button()
            self._running = True
            self._thread = threading.Thread(target=self._pwm_worker, daemon=True)
            self._thread.start()
            return
        except Exception:
            sys.stderr.write("Warning: Hardware GPIO unavailable (gpiozero/RPi.GPIO). Running in console mode.\n")

    def _wire_hardware_button(self) -> None:
        """Wire callbacks to underlying hardware backend."""
        if self._backend == "gpiozero" and self._gpio_btn is not None:
            self._wire_gpiozero_button()
        elif self._backend == "rpi.gpio":
            self._wire_rpigpio_button()

    def _wire_gpiozero_button(self) -> None:
        """Route gpiozero press and release events to the gesture tracker."""
        if self._gpio_btn is None:
            return
        self._gpio_btn.when_pressed = self._buttons.pressed
        self._gpio_btn.when_released = self._buttons.released

    def _wire_rpigpio_button(self) -> None:
        """Route RPi.GPIO edge events to the gesture tracker."""
        try:
            import RPi.GPIO as GPIO  # pylint: disable=import-error
            try:
                GPIO.remove_event_detect(self.button_pin)
            except Exception:
                pass

            def _handle_edge(channel):
                if GPIO.input(self.button_pin) == GPIO.LOW:
                    self._buttons.pressed()
                else:
                    self._buttons.released()

            GPIO.add_event_detect(self.button_pin, GPIO.BOTH, callback=_handle_edge, bouncetime=50)
        except Exception:
            pass

    def _set_duty_cycle(self, duty_percent: float) -> None:
        """Set LED brightness duty cycle (0.0 to 100.0)."""
        clamped = max(0.0, min(100.0, float(duty_percent)))
        try:
            if self._backend == "gpiozero" and self._gpio_led:
                self._gpio_led.value = clamped / 100.0
            elif self._backend == "rpi.gpio" and self._rpi_pwm:
                self._rpi_pwm.ChangeDutyCycle(clamped)
        except Exception:
            pass

    def stop(self) -> None:
        """Stop PWM and clean up GPIO."""
        self._running = False
        self._buttons.cancel()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._backend == "gpiozero":
            if self._gpio_led:
                try:
                    self._gpio_led.close()
                except Exception:
                    pass
                self._gpio_led = None
            if self._gpio_btn:
                try:
                    self._gpio_btn.close()
                except Exception:
                    pass
                self._gpio_btn = None
        elif self._backend == "rpi.gpio":
            if self._rpi_pwm:
                try:
                    self._rpi_pwm.stop()
                    import RPi.GPIO as GPIO  # pylint: disable=import-error
                    GPIO.cleanup([self.led_pin, self.button_pin])
                except Exception:
                    pass
                self._rpi_pwm = None

    def set_state(self, state: LEDState) -> None:
        """Update hardware LED operating state."""
        self.current_state = state

    def flash(self, count: int = 3, on_time: float = 0.09, off_time: float = 0.08) -> None:
        """Request a rapid flash sequence to acknowledge speech input.

        Returns immediately; the PWM thread, the only writer of the LED duty
        cycle, performs the pulses.
        """
        with self._flash_lock:
            self._flash_request = (count, on_time, off_time)
        if self.current_state != LEDState.OFF:
            self.current_state = LEDState.THINKING

    def _run_pending_flash(self) -> bool:
        """Perform a requested flash sequence on the PWM thread.

        Returns:
            True if a sequence ran.
        """
        with self._flash_lock:
            request, self._flash_request = self._flash_request, None
        if request is None:
            return False

        count, on_time, off_time = request
        self._flashing = True
        try:
            for _ in range(count):
                self._set_duty_cycle(100.0)
                time.sleep(on_time)
                self._set_duty_cycle(0.0)
                time.sleep(off_time)
        finally:
            self._flashing = False
        return True

    def _pwm_worker(self) -> None:
        """Background thread modulating PWM duty cycle based on current state."""
        step = 0.0
        while self._running:
            if self._run_pending_flash():
                continue
            if step > 6283.0:  # ~1000 * 2*pi
                step %= (2.0 * math.pi)
            if self.current_state == LEDState.IDLE:
                # 0.5 Hz smooth sinusoidal breathing
                duty = (math.sin(step) + 1.0) / 2.0 * 80.0 + 5.0
                self._set_duty_cycle(duty)
                step += 0.08
                time.sleep(0.02)
            elif self.current_state == LEDState.LISTENING:
                self._set_duty_cycle(100.0)
                time.sleep(0.05)
            elif self.current_state == LEDState.THINKING:
                # Heartbeat double-pulse (~1.1 Hz rhythm: lub-dub... pause)
                # Cycle length ~0.9s = 45 steps at 0.02s sleep
                cycle_step = int(step) % 45
                if cycle_step < 7:
                    # First beat ("lub") ramp up and down
                    duty = math.sin((cycle_step / 7.0) * math.pi) * 85.0 + 10.0
                elif cycle_step < 12:
                    # Dip between beats
                    duty = 10.0
                elif cycle_step < 20:
                    # Second beat ("dub") sharper ramp up and down
                    duty = math.sin(((cycle_step - 12) / 8.0) * math.pi) * 95.0 + 5.0
                else:
                    # Diastolic rest pause
                    duty = 2.0
                self._set_duty_cycle(duty)
                step += 1.0
                time.sleep(0.02)
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


def create_led_controller(
    verbose: bool = True,
    led_pin: Optional[int] = None,
    button_pin: Optional[int] = None,
    hold_time_sec: Optional[float] = None,
    long_hold_time_sec: Optional[float] = None,
) -> BaseLEDController:
    """Factory creating platform-appropriate LED controller.

    Args:
        verbose: Whether console simulator logs state changes.
        led_pin: BCM GPIO pin for LED (default: $LED_PIN or 25).
        button_pin: BCM GPIO pin for button switch (default: $BUTTON_PIN or 23).
        hold_time_sec: Hold duration in seconds to trigger sleep mode (default: $BUTTON_HOLD_SEC or 5.0).
        long_hold_time_sec: Long hold duration in seconds to trigger shutdown (default: $BUTTON_SHUTDOWN_SEC or 10.0).

    Returns:
        RaspberryPiLEDController if running on Raspberry Pi hardware,
        otherwise ConsoleLEDController.
    """
    if led_pin is None:
        led_pin = env_int("LED_PIN", 25)
    if button_pin is None:
        button_pin = env_int("BUTTON_PIN", 23)
    if hold_time_sec is None:
        hold_time_sec = env_float("BUTTON_HOLD_SEC", 5.0)
    if long_hold_time_sec is None:
        long_hold_time_sec = env_float("BUTTON_SHUTDOWN_SEC", 10.0)

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
        return RaspberryPiLEDController(
            led_pin=led_pin,
            button_pin=button_pin,
            hold_time_sec=hold_time_sec,
            long_hold_time_sec=long_hold_time_sec,
        )

    ctrl = ConsoleLEDController(verbose=verbose)
    ctrl.hold_time_sec = hold_time_sec
    ctrl.long_hold_time_sec = long_hold_time_sec
    return ctrl
