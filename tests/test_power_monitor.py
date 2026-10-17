"""Unit tests for core/power_monitor.py."""

import threading
import time
from unittest.mock import MagicMock, patch

from core import power_monitor
from core.power_monitor import PowerMonitor, PowerStatus

UNDER_VOLTAGE_NOW = (1 << 0) | (1 << 2)          # under-voltage detected, currently throttled
SINCE_BOOT_ONLY = (1 << 16) | (1 << 18)          # both have happened, neither is happening now


def _monitor(statuses, warn_interval=900.0):
    """Return a monitor fed from a list of statuses, with a controllable clock and a message list."""
    messages = []
    queue = list(statuses)
    clock = {"now": 0.0}
    monitor = PowerMonitor(
        log=messages.append,
        reader=lambda: queue.pop(0) if queue else None,
        warn_interval_sec=warn_interval,
        clock=lambda: clock["now"],
    )
    return monitor, messages, clock


def test_status_decodes_flags():
    """0x50005 is under-voltage and throttling now and since boot."""
    status = PowerStatus(raw=0x50005)
    assert status.active == ["under-voltage detected", "currently throttled"]
    assert status.occurred == ["under-voltage has occurred", "throttling has occurred"]
    assert PowerStatus(raw=0).active == [] and PowerStatus(raw=0).occurred == []


def test_warning_logged_while_throttled_with_temperature_and_advice():
    """A throttled CPU produces one warning naming the flags, the temperature and what to check."""
    monitor, messages, _ = _monitor([PowerStatus(UNDER_VOLTAGE_NOW, temp_c=61.2)])
    monitor.check()

    assert len(messages) == 1
    assert messages[0].startswith("Warning: CPU throttling: under-voltage detected, currently throttled")
    assert "CPU 61.2C" in messages[0]
    assert "power supply" in messages[0]


def test_warnings_are_rate_limited_while_throttling_continues():
    """Repeated checks inside the warn interval stay quiet; the next one after it warns again."""
    monitor, messages, clock = _monitor([PowerStatus(UNDER_VOLTAGE_NOW)] * 4, warn_interval=900)

    monitor.check()                 # t=0: warns
    clock["now"] = 60
    monitor.check()                 # inside the interval: quiet
    clock["now"] = 600
    monitor.check()                 # still inside: quiet
    assert len(messages) == 1

    clock["now"] = 901
    monitor.check()                 # interval passed: warns again
    assert len(messages) == 2


def test_recovery_is_logged_once_after_staying_clear_and_rearms_the_warning():
    """The recovery notice waits for the CPU to stay clear, then a new episode warns at once."""
    # The firmware sets the "since boot" bits together with the "now" bits, so a real episode reads 0x50005
    episode = PowerStatus(0x50005)
    clear = PowerStatus(0x50000)   # nothing happening now; the since-boot flags were reported with the warning
    monitor, messages, clock = _monitor([episode, clear, clear, clear, episode])
    monitor.clear_after_sec = 300

    monitor.check()                 # t=0: warning
    clock["now"] = 60
    monitor.check()                 # clear for the first time: too early to call it recovered
    clock["now"] = 200
    monitor.check()                 # still under the 300 s debounce
    assert len(messages) == 1

    clock["now"] = 361
    monitor.check()                 # clear for 301 s: recovery notice
    assert messages[1].startswith("CPU is no longer throttled")

    clock["now"] = 420
    monitor.check()                 # a new episode warns immediately, not rate limited
    assert messages[2].startswith("Warning: CPU throttling")
    assert len(messages) == 3


def test_flapping_supply_is_one_ongoing_problem():
    """A supply that dips every minute produces warnings at the warn interval and no recovery notices."""
    episode = PowerStatus(0x50005)
    clear = PowerStatus(0x50000)
    readings = [episode if minute % 2 == 0 else clear for minute in range(40)]   # 40 minutes of flapping
    monitor, messages, clock = _monitor(readings, warn_interval=900)
    monitor.clear_after_sec = 300

    for minute in range(40):
        clock["now"] = minute * 60
        monitor.check()

    warnings = [m for m in messages if m.startswith("Warning: CPU throttling")]
    assert not any("no longer throttled" in m for m in messages)
    assert len(warnings) == 3, messages          # t=0, t=900, t=1800
    assert len(messages) == 3


def test_since_boot_flags_reported_once_when_not_currently_throttled():
    """A past event seen at startup is reported once, not on every check."""
    monitor, messages, _ = _monitor([PowerStatus(SINCE_BOOT_ONLY)] * 3)
    for _ in range(3):
        monitor.check()

    assert len(messages) == 1
    assert "since boot" in messages[0]
    assert "under-voltage has occurred" in messages[0]


def test_new_since_boot_flag_is_reported_when_it_appears():
    """A flag that first appears between checks produces its own notice."""
    monitor, messages, _ = _monitor([PowerStatus(0), PowerStatus(1 << 16), PowerStatus(1 << 16)])
    for _ in range(3):
        monitor.check()
    assert len(messages) == 1
    assert "under-voltage has occurred" in messages[0]


def test_healthy_cpu_logs_nothing():
    """No flags set means no output."""
    monitor, messages, _ = _monitor([PowerStatus(0, temp_c=45.0)] * 3)
    for _ in range(3):
        monitor.check()
    assert messages == []


def test_soft_temperature_limit_advises_cooling():
    """A thermal flag points at cooling rather than the power supply."""
    monitor, messages, _ = _monitor([PowerStatus(1 << 3, temp_c=82.0)])
    monitor.check()
    assert "soft temperature limit active" in messages[0]
    assert "cooling" in messages[0]
    assert "power supply" not in messages[0]


def test_start_does_nothing_on_machines_without_throttle_info():
    """With no readable flags the monitor neither logs nor starts a thread."""
    messages = []
    monitor = PowerMonitor(log=messages.append, reader=lambda: None)
    assert monitor.start() is False
    assert monitor._thread is None
    assert messages == []


def test_start_is_disabled_by_zero_interval():
    """check_interval_sec=0 disables monitoring even when flags are readable."""
    reader = MagicMock(return_value=PowerStatus(UNDER_VOLTAGE_NOW))
    monitor = PowerMonitor(log=lambda m: None, reader=reader, check_interval_sec=0)
    assert monitor.start() is False
    reader.assert_not_called()


def test_background_thread_checks_periodically_and_stops():
    """The thread keeps checking until stopped, and a reader error does not end it."""
    calls = []
    seen = threading.Event()

    def reader():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("vcgencmd hiccup")
        if len(calls) >= 4:
            seen.set()
        return PowerStatus(0)

    messages = []
    monitor = PowerMonitor(log=messages.append, reader=reader, check_interval_sec=0.02)
    assert monitor.start() is True
    assert seen.wait(timeout=3)
    monitor.stop()

    assert monitor._thread is None
    assert any("power monitor check failed" in m for m in messages)
    count = len(calls)
    time.sleep(0.1)
    assert len(calls) == count  # stopped


def test_from_env_reads_intervals(monkeypatch):
    """THROTTLE_CHECK_SEC and THROTTLE_WARN_SEC configure the intervals."""
    monkeypatch.setenv("THROTTLE_CHECK_SEC", "5")
    monkeypatch.setenv("THROTTLE_WARN_SEC", "30")
    monitor = PowerMonitor.from_env()
    assert (monitor.check_interval_sec, monitor.warn_interval_sec) == (5.0, 30.0)

    monkeypatch.delenv("THROTTLE_CHECK_SEC")
    monkeypatch.delenv("THROTTLE_WARN_SEC")
    monitor = PowerMonitor.from_env()
    assert (monitor.check_interval_sec, monitor.warn_interval_sec) == (60.0, 900.0)


def test_read_status_parses_vcgencmd_output():
    """vcgencmd's 'throttled=0x...' output is parsed as hex."""
    with patch("core.power_monitor.shutil.which", return_value="/usr/bin/vcgencmd"), \
         patch("core.power_monitor.subprocess.run") as run, \
         patch("core.power_monitor._read_temp_c", return_value=50.5):
        run.return_value = MagicMock(returncode=0, stdout="throttled=0x50005\n")
        status = power_monitor.read_status()
    assert status == PowerStatus(raw=0x50005, temp_c=50.5)


def test_read_status_falls_back_to_hwmon_then_none():
    """Without vcgencmd the kernel alarm gives the under-voltage bit; with neither the result is None."""
    with patch("core.power_monitor.shutil.which", return_value=None), \
         patch("core.power_monitor._read_hwmon_undervoltage", return_value=1), \
         patch("core.power_monitor._read_temp_c", return_value=None):
        assert power_monitor.read_status() == PowerStatus(raw=1)

    with patch("core.power_monitor.shutil.which", return_value=None), \
         patch("core.power_monitor._read_hwmon_undervoltage", return_value=None):
        assert power_monitor.read_status() is None


def test_read_status_survives_vcgencmd_failures():
    """A failing or garbled vcgencmd is treated as unavailable, not an error."""
    with patch("core.power_monitor.shutil.which", return_value="/usr/bin/vcgencmd"), \
         patch("core.power_monitor._read_hwmon_undervoltage", return_value=None):
        with patch("core.power_monitor.subprocess.run", side_effect=OSError("exec failed")):
            assert power_monitor.read_status() is None
        with patch("core.power_monitor.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="garbage")
            assert power_monitor.read_status() is None


# ---- spoken low-power announcement ----

UNDER_VOLTAGE = PowerStatus(0x50005)       # under-voltage and throttled now
THERMAL_ONLY = PowerStatus(1 << 3)         # soft temperature limit; the supply is fine


def _announcing_monitor(statuses, announce, interval=3600.0):
    messages = []
    queue = list(statuses)
    clock = {"now": 0.0}
    monitor = PowerMonitor(
        log=messages.append,
        reader=lambda: queue.pop(0) if queue else None,
        clock=lambda: clock["now"],
        announce=announce,
        announce_interval_sec=interval,
    )
    return monitor, messages, clock


def test_announces_under_voltage_at_most_once_per_interval():
    """The spoken line is limited to one per hour even though the supply stays low."""
    calls = []
    monitor, _, clock = _announcing_monitor([UNDER_VOLTAGE] * 5, lambda status: calls.append(status) or True)

    monitor.check()                 # t=0: announces
    clock["now"] = 60
    monitor.check()
    clock["now"] = 3000
    monitor.check()
    assert len(calls) == 1 and calls[0].raw == 0x50005

    clock["now"] = 3601
    monitor.check()                 # an hour has passed: announces again
    assert len(calls) == 2


def test_declined_announcement_does_not_use_up_the_interval():
    """If the agent declines (a visitor is talking), the next check tries again."""
    answers = iter([False, False, True, True])
    calls = []
    monitor, _, clock = _announcing_monitor([UNDER_VOLTAGE] * 4, lambda s: calls.append(1) or next(answers))

    for minute in range(4):
        clock["now"] = minute * 60
        monitor.check()

    assert len(calls) == 3          # declined twice, spoke on the third, then the hour limit applies


def test_no_announcement_without_under_voltage():
    """Healthy readings and thermal throttling never trigger the spoken line."""
    calls = []
    monitor, _, _ = _announcing_monitor([PowerStatus(0), THERMAL_ONLY, PowerStatus(0x50000)], lambda s: calls.append(1) or True)
    for _ in range(3):
        monitor.check()
    assert calls == []


def test_announcement_failure_is_logged_and_retried():
    """An error while speaking is logged; monitoring continues and tries again next check."""
    attempts = []

    def flaky(status):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("audio busy")
        return True

    monitor, messages, clock = _announcing_monitor([UNDER_VOLTAGE, UNDER_VOLTAGE], flaky)
    monitor.check()
    clock["now"] = 60
    monitor.check()

    assert len(attempts) == 2
    assert any("low-power announcement failed: audio busy" in m for m in messages)


def test_no_announce_function_means_log_only():
    """Without an announce function the monitor only logs, as before."""
    monitor, messages, _ = _announcing_monitor([UNDER_VOLTAGE], announce=None)
    monitor.check()
    assert len(messages) == 1 and messages[0].startswith("Warning: CPU throttling")


def test_from_env_reads_announce_interval(monkeypatch):
    """LOW_POWER_ANNOUNCE_SEC sets the interval and the announce function is passed through."""
    function = lambda status: True  # noqa: E731
    monkeypatch.setenv("LOW_POWER_ANNOUNCE_SEC", "120")
    monitor = PowerMonitor.from_env(announce=function)
    assert monitor.announce_interval_sec == 120.0 and monitor._announce is function

    monkeypatch.delenv("LOW_POWER_ANNOUNCE_SEC")
    assert PowerMonitor.from_env().announce_interval_sec == 3600.0
