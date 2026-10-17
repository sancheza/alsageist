"""Raspberry Pi power and thermal monitor.

Reads the firmware throttle flags (`vcgencmd get_throttled`, or the kernel's
under-voltage alarm as a fallback) and logs a warning while the CPU is
throttled or the supply voltage is low. A throttled CPU runs CPU-bound work such
as loading a voice model at roughly half speed, so the cause (usually an
inadequate power supply) is worth surfacing in the service log.

On machines without these interfaces (macOS, containers) the monitor does nothing.
"""

from dataclasses import dataclass
import glob
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional

DEFAULT_CHECK_SEC = 60.0
DEFAULT_WARN_SEC = 900.0

# Bit meanings from the Raspberry Pi firmware documentation for get_throttled.
FLAGS_NOW = {
    0: "under-voltage detected",
    1: "ARM frequency capped",
    2: "currently throttled",
    3: "soft temperature limit active",
}
FLAGS_SINCE_BOOT = {
    16: "under-voltage has occurred",
    17: "ARM frequency capping has occurred",
    18: "throttling has occurred",
    19: "soft temperature limit has occurred",
}
THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"


@dataclass(frozen=True)
class PowerStatus:
    """Throttle flags and temperature at one moment.

    Attributes:
        raw: The get_throttled bit field.
        temp_c: CPU temperature in degrees Celsius, if readable.
    """

    raw: int
    temp_c: Optional[float] = None

    @property
    def active(self) -> List[str]:
        """Conditions in effect right now."""
        return [text for bit, text in FLAGS_NOW.items() if self.raw & (1 << bit)]

    @property
    def occurred(self) -> List[str]:
        """Conditions that have occurred at any time since boot."""
        return [text for bit, text in FLAGS_SINCE_BOOT.items() if self.raw & (1 << bit)]


def _read_temp_c() -> Optional[float]:
    """Return the CPU temperature in Celsius, or None if unavailable."""
    try:
        with open(THERMAL_ZONE, "r", encoding="utf-8") as handle:
            return int(handle.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _read_vcgencmd() -> Optional[int]:
    """Return the get_throttled bit field from vcgencmd, or None if unavailable."""
    binary = shutil.which("vcgencmd")
    if not binary:
        return None
    try:
        result = subprocess.run([binary, "get_throttled"], capture_output=True, text=True, timeout=2)
        if result.returncode != 0:
            return None
        return int(result.stdout.strip().split("=", 1)[1], 16)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def _read_hwmon_undervoltage() -> Optional[int]:
    """Return bit 0 set when the kernel reports under-voltage, or None if unavailable."""
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(hwmon, "name"), "r", encoding="utf-8") as handle:
                if handle.read().strip() != "rpi_volt":
                    continue
            with open(os.path.join(hwmon, "in0_lcrit_alarm"), "r", encoding="utf-8") as handle:
                return 1 if handle.read().strip() == "1" else 0
        except OSError:
            continue
    return None


def read_status() -> Optional[PowerStatus]:
    """Read the current power status.

    Returns:
        PowerStatus, or None when this machine exposes no throttle information.
    """
    raw = _read_vcgencmd()
    if raw is None:
        raw = _read_hwmon_undervoltage()
    if raw is None:
        return None
    return PowerStatus(raw=raw, temp_c=_read_temp_c())


class PowerMonitor:
    """Logs a warning while the CPU is throttled, and a notice when it clears."""

    def __init__(
        self,
        log: Callable[[str], None] = print,
        reader: Callable[[], Optional[PowerStatus]] = read_status,
        check_interval_sec: float = DEFAULT_CHECK_SEC,
        warn_interval_sec: float = DEFAULT_WARN_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a monitor.

        Args:
            log: Function that receives each message.
            reader: Function returning the current PowerStatus, or None if unavailable.
            check_interval_sec: Seconds between checks; 0 or less disables the monitor.
            warn_interval_sec: Minimum seconds between repeated warnings while throttling persists.
            clock: Monotonic time source in seconds.
        """
        self._log = log
        self._reader = reader
        self.check_interval_sec = check_interval_sec
        self.warn_interval_sec = warn_interval_sec
        self._clock = clock
        self._last_warning: Optional[float] = None
        self._was_active = False
        self._reported_occurred: set = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_env(cls, log: Callable[[str], None] = print) -> "PowerMonitor":
        """Create a monitor configured by THROTTLE_CHECK_SEC and THROTTLE_WARN_SEC.

        Args:
            log: Function that receives each message.

        Returns:
            Configured PowerMonitor.
        """
        return cls(
            log=log,
            check_interval_sec=float(os.environ.get("THROTTLE_CHECK_SEC", DEFAULT_CHECK_SEC)),
            warn_interval_sec=float(os.environ.get("THROTTLE_WARN_SEC", DEFAULT_WARN_SEC)),
        )

    def check(self) -> Optional[PowerStatus]:
        """Evaluate the current status once and log if needed.

        Returns:
            The status read, or None if this machine exposes none.
        """
        status = self._reader()
        if status is None:
            return None

        now = self._clock()
        temp = f", CPU {status.temp_c:.1f}C" if status.temp_c is not None else ""

        if status.active:
            due = self._last_warning is None or now - self._last_warning >= self.warn_interval_sec
            if due:
                self._last_warning = now
                self._log(
                    f"Warning: CPU throttling: {', '.join(status.active)}{temp}. {self._advice(status)}"
                )
            self._was_active = True
            self._reported_occurred.update(status.occurred)
            return status

        if self._was_active:
            self._was_active = False
            self._last_warning = None
            self._log(f"CPU is no longer throttled{temp}.")

        new = [text for text in status.occurred if text not in self._reported_occurred]
        if new:
            self._reported_occurred.update(new)
            self._log(f"Warning: since boot the CPU reported: {', '.join(new)}{temp}. {self._advice(status)}")
        return status

    @staticmethod
    def _advice(status: PowerStatus) -> str:
        """Return what to check, based on which flags are set."""
        parts = []
        if status.raw & ((1 << 0) | (1 << 16)):
            parts.append("Check the power supply and USB cable (the Pi 3 needs a solid 5 V, 2.5 A).")
        if status.raw & ((1 << 3) | (1 << 19)):
            parts.append("Check cooling and airflow.")
        return " ".join(parts) or "Throttling slows CPU-bound work such as loading a voice model."

    def start(self) -> bool:
        """Run one check now, then keep checking on a background thread.

        Returns:
            True if monitoring is running; False when disabled or unsupported on this machine.
        """
        if self.check_interval_sec <= 0 or self._thread is not None:
            return False
        if self.check() is None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AlsaGeist-PowerMonitor")
        self._thread.start()
        return True

    def _run(self) -> None:
        """Background loop: check every check_interval_sec until stopped."""
        while not self._stop.wait(self.check_interval_sec):
            try:
                self.check()
            except Exception as exc:  # isolation point: monitoring must never stop the service
                self._log(f"Warning: power monitor check failed: {exc}")

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
