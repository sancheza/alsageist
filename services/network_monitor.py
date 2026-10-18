"""Online and offline mode tracking for the agent.

Probes connectivity on a background thread so the audio loop never waits on the network,
supports a mode forced by a control command, and lets the agent ask for an immediate probe
(for example after every cloud speech backend failed).
"""

import threading
import time
from typing import Callable, Optional

DEFAULT_CHECK_INTERVAL_SEC = 15.0 * 60.0
DEFAULT_REQUEST_INTERVAL_SEC = 30.0


class NetworkMonitor:
    """Tracks whether the agent is online, from periodic probes or a forced mode."""

    def __init__(
        self,
        probe: Callable[[], bool],
        check_interval_sec: float = DEFAULT_CHECK_INTERVAL_SEC,
    ) -> None:
        """Create a monitor; call start_probe() to run the first check.

        Args:
            probe: Function returning True when the internet is reachable. It may block.
            check_interval_sec: Seconds between periodic probes.
        """
        self._probe = probe
        self.check_interval_sec = check_interval_sec
        self.is_online = True
        self.forced: Optional[bool] = None  # a mode fixed by command; None means automatic
        self.last_check_time = 0.0
        self._last_request_time = 0.0
        self._result: Optional[bool] = None  # probe result not yet applied
        self._lock = threading.Lock()  # guards _result
        self._thread: Optional[threading.Thread] = None

    @property
    def mode_forced(self) -> bool:
        """Return True while the mode is fixed by a control command."""
        return self.forced is not None

    def start_probe(self) -> None:
        """Start a background probe and count it as the latest periodic check."""
        self._thread = threading.Thread(target=self._run_probe, daemon=True, name="AlsaGeist-NetCheck")
        self._thread.start()
        self.last_check_time = time.time()

    def _run_probe(self) -> None:
        """Probe connectivity and publish the result for the next poll."""
        reachable = self._probe()
        with self._lock:
            self._result = reachable

    def apply_first_result(self, wait_sec: float = 1.0) -> None:
        """Wait briefly for the first probe and adopt its result unless a mode is forced.

        Args:
            wait_sec: Longest time to wait for the probe thread.
        """
        thread = self._thread
        if self.forced is None and thread is not None and thread.is_alive():
            thread.join(timeout=wait_sec)
        with self._lock:
            if self.forced is None and self._result is not None:
                self.is_online = self._result
                self._result = None

    def set_forced(self, online: Optional[bool]) -> None:
        """Force the online or offline mode, or return to automatic detection.

        Args:
            online: True or False to force a mode, None for automatic detection.
        """
        self.forced = None if online is None else bool(online)
        if online is None:
            self.last_check_time = 0.0  # probe on the next poll
        else:
            self.is_online = bool(online)

    def request_check(self, min_interval_sec: float = DEFAULT_REQUEST_INTERVAL_SEC) -> None:
        """Make the next poll probe connectivity now instead of waiting for the interval.

        Does nothing when the mode is forced or already offline (the periodic check brings
        it back online), and at most one request per min_interval_sec.

        Args:
            min_interval_sec: Minimum seconds between requests.
        """
        if self.forced is not None or not self.is_online:
            return
        now = time.time()
        if now - self._last_request_time < min_interval_sec:
            return
        self._last_request_time = now
        self.last_check_time = 0.0

    def poll(self, now: float) -> Optional[bool]:
        """Apply a finished probe and start the next one when it is due.

        Args:
            now: Current time in seconds since the epoch.

        Returns:
            The new online state when it changed, else None.
        """
        with self._lock:
            result, self._result = self._result, None
        if self.forced is not None:
            return None

        changed: Optional[bool] = None
        if result is not None and result != self.is_online:
            self.is_online = result
            changed = result

        probing = self._thread is not None and self._thread.is_alive()
        if not probing and now - self.last_check_time > self.check_interval_sec:
            self.start_probe()
            self.last_check_time = now
        return changed
