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
        alarm_reader=None,
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


def test_benign_frequency_capping_since_boot_alone_does_not_warn():
    """Bit 17 (0x20000) alone without under-voltage or throttling is benign and logs no warning."""
    monitor, messages, _ = _monitor([PowerStatus(1 << 17, temp_c=59.1)] * 3)
    for _ in range(3):
        monitor.check()
    assert messages == []


def test_advice_requires_actual_throttling_to_claim_throttling():
    """_advice only mentions throttling when bit 2 or bit 18 is set."""
    assert "Throttling slows" not in PowerMonitor._advice(PowerStatus(1 << 17))
    assert "Throttling slows" in PowerMonitor._advice(PowerStatus(1 << 18))
    assert "Throttling slows" in PowerMonitor._advice(PowerStatus(1 << 2))


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
    monitor = PowerMonitor(log=messages.append, reader=lambda: None, alarm_reader=None)
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
    monitor = PowerMonitor(log=messages.append, reader=reader, check_interval_sec=0.02, alarm_reader=None)
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
        alarm_reader=None,
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


# ---- catching dips that last only a few seconds ----

def test_read_undervoltage_alarm_wraps_the_kernel_reading():
    """The alarm reader returns True, False, or None when the machine has no such alarm."""
    for raw, expected in ((1, True), (0, False), (None, None)):
        with patch("core.power_monitor._read_hwmon_undervoltage", return_value=raw):
            assert power_monitor.read_undervoltage_alarm() is expected


def _alarm_monitor(alarm_values):
    """Monitor whose firmware reading is always healthy; only the polled alarm sees the dip."""
    messages, announced = [], []
    clock = {"now": 0.0}
    values = iter(alarm_values)
    monitor = PowerMonitor(
        log=messages.append,
        reader=lambda: PowerStatus(0),
        check_interval_sec=60.0,
        clock=lambda: clock["now"],
        alarm_reader=lambda: next(values, False),
        announce=lambda status: announced.append(status.raw) or True,
    )
    return monitor, messages, announced, clock


def test_a_short_dip_between_checks_is_reported_at_the_next_check():
    """The polled alarm remembers a dip; the full check at the next interval reports and announces it."""
    monitor, messages, announced, clock = _alarm_monitor([False, True, False])

    for t in (2, 4, 6):                # three alarm polls; the second one sees the dip
        clock["now"] = t
        monitor._poll_alarm(t)
    clock["now"] = 60
    monitor.check()                    # the firmware flag is clear by now, but the dip was 56 s ago

    assert len(messages) == 1 and "under-voltage detected" in messages[0]
    assert announced == [1]


def test_an_old_dip_is_not_reported():
    """A dip older than 1.5 check intervals does not make the supply look bad."""
    monitor, messages, announced, clock = _alarm_monitor([True])
    monitor._poll_alarm(0)

    clock["now"] = 200                 # 200 s later; the window is 90 s
    monitor.check()

    assert messages == [] and announced == []


def test_no_alarm_reader_changes_nothing():
    """Without an alarm reader only the firmware flags matter."""
    monitor, messages, _, clock = _alarm_monitor([])
    monitor._alarm_reader = None
    monitor._poll_alarm(1)
    clock["now"] = 60
    monitor.check()
    assert messages == []


def test_alarm_read_errors_are_ignored():
    """A failing alarm read never stops monitoring."""
    monitor, messages, _, clock = _alarm_monitor([])
    monitor._alarm_reader = MagicMock(side_effect=OSError("sysfs gone"))
    monitor._poll_alarm(1)             # must not raise
    clock["now"] = 60
    monitor.check()
    assert messages == []


def test_background_thread_polls_the_alarm_between_full_checks():
    """With a slow full check and a fast poll, the thread still notices a short dip and reports it."""
    messages = []
    seen = threading.Event()
    polls = {"n": 0}

    def alarm():
        polls["n"] += 1
        return polls["n"] == 3         # one brief dip, a few polls in

    def log(message):
        messages.append(message)
        seen.set()

    monitor = PowerMonitor(
        log=log,
        reader=lambda: PowerStatus(0),
        check_interval_sec=0.3,
        alarm_reader=alarm,
        poll_interval_sec=0.02,
    )
    monitor.start()
    assert seen.wait(timeout=3)
    monitor.stop()

    assert any("under-voltage detected" in m for m in messages)
    assert polls["n"] > 5              # polled much more often than the full check ran


def test_a_dip_is_reported_once_not_at_every_following_check():
    """One short dip produces one report, even with no rate limit and several checks afterwards."""
    messages, announced = [], []
    clock = {"now": 0.0}
    monitor = PowerMonitor(
        log=messages.append, reader=lambda: PowerStatus(0), check_interval_sec=60.0,
        warn_interval_sec=0.0, announce_interval_sec=0.0, clear_after_sec=0.0,
        clock=lambda: clock["now"], alarm_reader=lambda: False,
        announce=lambda status: announced.append(1) or True,
    )
    monitor._poll_alarm(0.0)                      # not a dip
    monitor._last_undervoltage = 5.0              # one dip at t=5 s

    for t in (60, 120, 180):
        clock["now"] = t
        monitor.check()

    reports = [m for m in messages if m.startswith("Warning: CPU throttling")]
    assert len(reports) == 1 and len(announced) == 1


def test_two_separate_dips_are_two_reports():
    """Dips in different check intervals are each reported."""
    messages = []
    clock = {"now": 0.0}
    monitor = PowerMonitor(
        log=messages.append, reader=lambda: PowerStatus(0), check_interval_sec=60.0,
        warn_interval_sec=0.0, clear_after_sec=0.0, clock=lambda: clock["now"], alarm_reader=lambda: False,
    )
    monitor._last_undervoltage = 5.0
    clock["now"] = 60
    monitor.check()
    clock["now"] = 120
    monitor.check()                               # nothing new: clear
    monitor._last_undervoltage = 130.0
    clock["now"] = 180
    monitor.check()

    assert len([m for m in messages if m.startswith("Warning: CPU throttling")]) == 2


# ---- the alarm file is found once and then read directly ----

def _fake_hwmon(tmp_path, devices):
    """Create fake hwmon directories: {dirname: (name, alarm_value_or_None)}."""
    for dirname, (name, alarm) in devices.items():
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / "name").write_text(name + "\n")
        if alarm is not None:
            (directory / "in0_lcrit_alarm").write_text(f"{alarm}\n")
    return str(tmp_path / "hwmon*")


def test_alarm_is_found_among_other_hwmon_devices_and_read(tmp_path, monkeypatch):
    """Only the rpi_volt device counts, and its alarm value is returned."""
    monkeypatch.setattr(power_monitor, "HWMON_GLOB", _fake_hwmon(tmp_path, {
        "hwmon0": ("cpu_thermal", None),
        "hwmon1": ("rpi_volt", 1),
    }))
    monkeypatch.setattr(power_monitor, "_alarm_path", None)

    assert power_monitor._read_hwmon_undervoltage() == 1
    (tmp_path / "hwmon1" / "in0_lcrit_alarm").write_text("0\n")
    assert power_monitor._read_hwmon_undervoltage() == 0


def test_alarm_path_is_looked_up_once_not_on_every_poll(tmp_path, monkeypatch):
    """After the first read, polling does not scan the hwmon directories again."""
    monkeypatch.setattr(power_monitor, "HWMON_GLOB", _fake_hwmon(tmp_path, {"hwmon0": ("rpi_volt", 0)}))
    monkeypatch.setattr(power_monitor, "_alarm_path", None)

    with patch("core.power_monitor.glob.glob", wraps=power_monitor.glob.glob) as scan:
        for _ in range(50):
            assert power_monitor._read_hwmon_undervoltage() == 0
    assert scan.call_count == 1


def test_missing_alarm_device_returns_none_and_is_retried(tmp_path, monkeypatch):
    """A machine without rpi_volt reports None; the lookup is repeated if the device appears later."""
    monkeypatch.setattr(power_monitor, "HWMON_GLOB", _fake_hwmon(tmp_path, {"hwmon0": ("cpu_thermal", None)}))
    monkeypatch.setattr(power_monitor, "_alarm_path", None)
    assert power_monitor._read_hwmon_undervoltage() is None

    _fake_hwmon(tmp_path, {"hwmon1": ("rpi_volt", 1)})
    assert power_monitor._read_hwmon_undervoltage() == 1


def test_vanished_alarm_file_is_rediscovered(tmp_path, monkeypatch):
    """If the cached file disappears the read returns None once, then finds the device again."""
    monkeypatch.setattr(power_monitor, "HWMON_GLOB", _fake_hwmon(tmp_path, {"hwmon0": ("rpi_volt", 1)}))
    monkeypatch.setattr(power_monitor, "_alarm_path", None)
    assert power_monitor._read_hwmon_undervoltage() == 1

    (tmp_path / "hwmon0" / "in0_lcrit_alarm").unlink()
    assert power_monitor._read_hwmon_undervoltage() is None       # cache dropped
    (tmp_path / "hwmon0" / "in0_lcrit_alarm").write_text("1\n")
    assert power_monitor._read_hwmon_undervoltage() == 1          # found again
