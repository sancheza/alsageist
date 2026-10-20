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

from core.env_utils import env_float

DEFAULT_CHECK_SEC = 60.0
DEFAULT_WARN_SEC = 900.0
DEFAULT_CLEAR_SEC = 300.0
DEFAULT_ANNOUNCE_SEC = 3600.0
DEFAULT_POLL_SEC = 2.0
UNDER_VOLTAGE_NOW = 1 << 0

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
HWMON_GLOB = "/sys/class/hwmon/hwmon*"


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


_alarm_path: Optional[str] = None  # remembered after the first successful lookup


def _find_hwmon_alarm_path() -> Optional[str]:
    """Locate the rpi_volt under-voltage alarm file, or None if this machine has none."""
    for hwmon in glob.glob(HWMON_GLOB):
        try:
            with open(os.path.join(hwmon, "name"), "r", encoding="utf-8") as handle:
                if handle.read().strip() != "rpi_volt":
                    continue
        except OSError:
            continue
        path = os.path.join(hwmon, "in0_lcrit_alarm")
        if os.path.exists(path):
            return path
    return None


def _read_hwmon_undervoltage() -> Optional[int]:
    """Return 1 when the kernel reports under-voltage, 0 when not, or None if unavailable.

    The alarm file is found once and then read directly, so repeated polling is a single
    small file read. If the file disappears the lookup is repeated on the next call.
    """
    global _alarm_path  # pylint: disable=global-statement
    if _alarm_path is None:
        _alarm_path = _find_hwmon_alarm_path()
        if _alarm_path is None:
            return None
    try:
        with open(_alarm_path, "r", encoding="utf-8") as handle:
            return 1 if handle.read().strip() == "1" else 0
    except OSError:
        _alarm_path = None
        return None


def read_undervoltage_alarm() -> Optional[bool]:
    """Return whether the kernel reports under-voltage right now, or None if it cannot say.

    This is a plain file read, cheap enough to repeat every couple of seconds, which catches
    dips that last only a few seconds.
    """
    raw = _read_hwmon_undervoltage()
    return None if raw is None else bool(raw)


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
    """Logs a warning while the CPU is throttled, and a notice once it has stayed clear.

    A supply that keeps dipping is one ongoing problem, not many: warnings are limited to one
    per warn_interval_sec, and the recovery notice is logged only after the CPU has been clear
    for clear_after_sec.
    """

    def __init__(
        self,
        log: Callable[[str], None] = print,
        reader: Callable[[], Optional[PowerStatus]] = read_status,
        check_interval_sec: float = DEFAULT_CHECK_SEC,
        warn_interval_sec: float = DEFAULT_WARN_SEC,
        clock: Callable[[], float] = time.monotonic,
        clear_after_sec: float = DEFAULT_CLEAR_SEC,
        announce: Optional[Callable[["PowerStatus"], bool]] = None,
        announce_interval_sec: float = DEFAULT_ANNOUNCE_SEC,
        alarm_reader: Optional[Callable[[], Optional[bool]]] = read_undervoltage_alarm,
        poll_interval_sec: float = DEFAULT_POLL_SEC,
    ) -> None:
        """Create a monitor.

        Args:
            log: Function that receives each message.
            reader: Function returning the current PowerStatus, or None if unavailable.
            check_interval_sec: Seconds between checks; 0 or less disables the monitor.
            warn_interval_sec: Minimum seconds between repeated warnings while throttling persists.
            clock: Monotonic time source in seconds.
            clear_after_sec: Seconds the CPU must stay clear before the recovery notice is logged.
            announce: Optional function called while the supply is under-voltage, at most once per
                announce_interval_sec. It returns True if it spoke; returning False (for example
                because a visitor is talking) does not use up the interval.
            announce_interval_sec: Minimum seconds between announcements.
            alarm_reader: Optional cheap reader of the under-voltage alarm. It is polled every
                poll_interval_sec between full checks, so a dip of a few seconds is not missed;
                a dip since the last check counts as under-voltage at the next check, once.
            poll_interval_sec: Seconds between alarm polls.
        """
        self._log = log
        self._reader = reader
        self.check_interval_sec = check_interval_sec
        self.warn_interval_sec = warn_interval_sec
        self._clock = clock
        self.clear_after_sec = clear_after_sec
        self._announce = announce
        self._alarm_reader = alarm_reader
        self.poll_interval_sec = poll_interval_sec
        self._last_undervoltage: Optional[float] = None
        self.announce_interval_sec = announce_interval_sec
        self._last_announce: Optional[float] = None
        self._last_warning: Optional[float] = None
        self._was_active = False
        self._clear_since: Optional[float] = None
        self._reported_occurred: set = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_env(
        cls,
        log: Callable[[str], None] = print,
        announce: Optional[Callable[["PowerStatus"], bool]] = None,
    ) -> "PowerMonitor":
        """Create a monitor configured by THROTTLE_CHECK_SEC, THROTTLE_WARN_SEC and LOW_POWER_ANNOUNCE_SEC.

        Args:
            log: Function that receives each message.
            announce: Optional spoken-announcement function (see __init__).

        Returns:
            Configured PowerMonitor.
        """
        return cls(
            log=log,
            check_interval_sec=env_float("THROTTLE_CHECK_SEC", DEFAULT_CHECK_SEC),
            warn_interval_sec=env_float("THROTTLE_WARN_SEC", DEFAULT_WARN_SEC),
            announce=announce,
            announce_interval_sec=env_float("LOW_POWER_ANNOUNCE_SEC", DEFAULT_ANNOUNCE_SEC),
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
        if self._last_undervoltage is not None:
            # The alarm fired since the last full check. Report it once, here, so a dip that
            # lasted only a few seconds is neither missed nor counted at two checks in a row.
            recent = now - self._last_undervoltage <= self.check_interval_sec * 2
            self._last_undervoltage = None
            if recent and not status.raw & UNDER_VOLTAGE_NOW:
                status = PowerStatus(raw=status.raw | UNDER_VOLTAGE_NOW, temp_c=status.temp_c)
        temp = f", CPU {status.temp_c:.1f}C" if status.temp_c is not None else ""

        if status.active:
            due = self._last_warning is None or now - self._last_warning >= self.warn_interval_sec
            if due:
                self._last_warning = now
                self._log(
                    f"Warning: CPU throttling: {', '.join(status.active)}{temp}. {self._advice(status)}"
                )
            self._was_active = True
            self._clear_since = None
            self._reported_occurred.update(status.occurred)
            self._maybe_announce(status, now)
            return status

        if self._was_active:
            if self._clear_since is None:
                self._clear_since = now
            if now - self._clear_since >= self.clear_after_sec:
                self._was_active = False
                self._clear_since = None
                self._last_warning = None
                self._log(f"CPU is no longer throttled{temp}.")

        new = [text for text in status.occurred if text not in self._reported_occurred]
        if new == ["ARM frequency capping has occurred"] and not (status.raw & ((1 << 16) | (1 << 18) | (1 << 19))):
            # Benign bootloader-time frequency cap latch without under-voltage or throttling
            self._reported_occurred.update(new)
        elif new:
            self._reported_occurred.update(new)
            advice = self._advice(status)
            suffix = f" {advice}" if advice else ""
            self._log(f"Warning: since boot the CPU reported: {', '.join(new)}{temp}.{suffix}")
        return status

    def _maybe_announce(self, status: PowerStatus, now: float) -> None:
        """Call the announcement function while under-voltage, at most once per interval."""
        if self._announce is None or not status.raw & UNDER_VOLTAGE_NOW:
            return
        if self._last_announce is not None and now - self._last_announce < self.announce_interval_sec:
            return
        try:
            spoke = self._announce(status)
        except Exception as exc:  # isolation point: a failed announcement must not stop monitoring
            self._log(f"Warning: low-power announcement failed: {exc}")
            return
        if spoke:
            self._last_announce = now

    @staticmethod
    def _advice(status: PowerStatus) -> str:
        """Return what to check, based on which flags are set."""
        parts = []
        if status.raw & ((1 << 0) | (1 << 16)):
            parts.append("Check the power supply and USB cable (the Pi 3 needs a solid 5 V, 2.5 A).")
        if status.raw & ((1 << 3) | (1 << 19)):
            parts.append("Check cooling and airflow.")
        if status.raw & ((1 << 2) | (1 << 18)):
            parts.append("Throttling slows CPU-bound work such as loading a voice model.")
        return " ".join(parts)

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

    def _poll_alarm(self, now: float) -> None:
        """Remember when the kernel last reported under-voltage."""
        if self._alarm_reader is None:
            return
        try:
            if self._alarm_reader():
                self._last_undervoltage = now
        except Exception:  # isolation point: a failed read must not stop monitoring
            pass

    def _run(self) -> None:
        """Background loop: poll the alarm often, run a full check every check_interval_sec."""
        wait = min(self.poll_interval_sec, self.check_interval_sec)
        next_check = self._clock() + self.check_interval_sec
        while not self._stop.wait(wait):
            now = self._clock()
            self._poll_alarm(now)
            if now < next_check:
                continue
            next_check = now + self.check_interval_sec
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
