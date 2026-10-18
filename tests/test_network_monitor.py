"""Unit tests for the online/offline mode monitor."""

import threading

from services.network_monitor import NetworkMonitor


def _monitor(reachable=True, interval=10.0):
    return NetworkMonitor(probe=lambda: reachable, check_interval_sec=interval)


def _finish_probe(monitor):
    monitor._thread.join(timeout=2)


def test_first_result_is_adopted_unless_forced():
    monitor = _monitor(reachable=False)
    monitor.start_probe()
    _finish_probe(monitor)
    monitor.apply_first_result()
    assert monitor.is_online is False

    forced = _monitor(reachable=False)
    forced.set_forced(True)
    forced.start_probe()
    _finish_probe(forced)
    forced.apply_first_result()
    assert forced.is_online is True


def test_poll_reports_a_change_once():
    monitor = _monitor(reachable=False, interval=3600.0)
    monitor.start_probe()
    _finish_probe(monitor)
    assert monitor.poll(now=monitor.last_check_time + 1) is False
    assert monitor.poll(now=monitor.last_check_time + 2) is None


def test_poll_starts_a_probe_only_when_due_and_none_is_running():
    gate = threading.Event()
    calls = []

    def slow_probe():
        calls.append(1)
        gate.wait(timeout=2)
        return True

    monitor = NetworkMonitor(probe=slow_probe, check_interval_sec=10.0)
    monitor.poll(now=100.0)
    first = monitor._thread
    monitor.poll(now=200.0)  # a probe is still running
    assert monitor._thread is first
    gate.set()
    first.join(timeout=2)
    assert len(calls) == 1


def test_forced_mode_ignores_probe_results_and_starts_no_probe():
    monitor = _monitor(reachable=True)
    monitor.set_forced(False)
    monitor._result = True
    assert monitor.poll(now=1e9) is None
    assert monitor.is_online is False
    assert monitor._thread is None
    assert monitor.mode_forced is True


def test_returning_to_automatic_schedules_an_immediate_probe():
    monitor = _monitor()
    monitor.set_forced(False)
    monitor.last_check_time = 12345.0
    monitor.set_forced(None)
    assert monitor.last_check_time == 0.0
    assert monitor.mode_forced is False


def test_request_check_is_rate_limited_and_skipped_when_forced_or_offline():
    monitor = _monitor()
    monitor.last_check_time = 1000.0
    monitor.request_check()
    assert monitor.last_check_time == 0.0

    monitor.last_check_time = 5000.0
    monitor.request_check()
    assert monitor.last_check_time == 5000.0

    monitor._last_request_time -= 31
    monitor.request_check()
    assert monitor.last_check_time == 0.0

    offline = _monitor()
    offline.is_online = False
    offline.last_check_time = 1000.0
    offline.request_check()
    assert offline.last_check_time == 1000.0
