"""core/wifi_manager.py - Wi-Fi connection management and fallback orchestration.

Discovers saved NetworkManager Wi-Fi profiles and parses optional fallback entries
from environment configurations, scans for visible networks in the air, and coordinates
connection attempts via nmcli with progress callbacks for vocal status announcements.
"""

import os
import shutil
import subprocess
import sys
from typing import Callable, List, Optional, Set, Tuple

# Default connection and scan timeouts in seconds
DEFAULT_SCAN_TIMEOUT_SEC = 10.0
DEFAULT_CONNECT_TIMEOUT_SEC = 20.0


def parse_fallback_networks(raw: Optional[str]) -> List[Tuple[str, str]]:
    """Parse a semicolon-separated list of SSID:PASSWORD pairs.

    Supports quotes around SSIDs or passwords (e.g. 'My Network':'pass;123').

    Args:
        raw: Formatted string such as 'SSID1:PASSWORD; SSID2:PASSWORD'.

    Returns:
        List of (ssid, password) tuples.
    """
    if not raw or not raw.strip():
        return []

    tokens: List[str] = []
    current: List[str] = []
    in_quote: Optional[str] = None
    for char in raw:
        if char in ('"', "'"):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            current.append(char)
        elif char == ";" and in_quote is None:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current).strip())

    results: List[Tuple[str, str]] = []
    for token in tokens:
        if not token:
            continue
        parts: List[str] = []
        cur_part: List[str] = []
        in_quote = None
        for char in token:
            if char in ('"', "'"):
                if in_quote == char:
                    in_quote = None
                elif in_quote is None:
                    in_quote = char
                cur_part.append(char)
            elif char == ":" and in_quote is None and not parts:
                parts.append("".join(cur_part).strip())
                cur_part = []
            else:
                cur_part.append(char)
        parts.append("".join(cur_part).strip())

        def clean_val(val: str) -> str:
            val = val.strip()
            if len(val) >= 2 and val[0] in ('"', "'") and val[0] == val[-1]:
                val = val[1:-1]
            return val

        if len(parts) >= 2:
            ssid = clean_val(parts[0])
            password = clean_val(parts[1])
            if ssid:
                results.append((ssid, password))
        elif len(parts) == 1:
            ssid = clean_val(parts[0])
            if ssid:
                results.append((ssid, ""))
    return results


def _run_nmcli(args: List[str], timeout: float = 10.0) -> Tuple[int, str]:
    """Execute an nmcli command with optional sudo fallback.

    Args:
        args: Command arguments after 'nmcli'.
        timeout: Maximum seconds to wait for execution.

    Returns:
        Tuple of (exit_code, stdout_output).
    """
    nmcli_bin = shutil.which("nmcli")
    if not nmcli_bin:
        return 127, ""

    cmd = [nmcli_bin, *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return 0, proc.stdout
        # If permission denied and not running as root, attempt sudo -n
        if proc.returncode != 0 and hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
            sudo_cmd = ["sudo", "-n", nmcli_bin, *args]
            sudo_proc = subprocess.run(sudo_cmd, capture_output=True, text=True, timeout=timeout)
            return sudo_proc.returncode, sudo_proc.stdout
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def get_current_wifi_ssid() -> Optional[str]:
    """Return the currently connected Wi-Fi SSID, or None if not connected.

    Returns:
        Active Wi-Fi SSID string, or None.
    """
    code, out = _run_nmcli(["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev"], timeout=5.0)
    if code != 0 or not out:
        return None

    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) >= 4 and parts[1] == "wifi" and parts[2] == "connected":
            connection_name = parts[3].strip()
            if connection_name and connection_name != "--":
                return connection_name

    # Fallback check using dev wifi list
    code, out = _run_nmcli(["-t", "-f", "active,ssid", "dev", "wifi"], timeout=5.0)
    if code == 0 and out:
        for line in out.splitlines():
            if line.startswith("yes:"):
                ssid = line[4:].strip()
                if ssid and ssid != "--":
                    return ssid

    return None


def get_saved_wifi_connections() -> List[str]:
    """Return a list of saved NetworkManager Wi-Fi connection profile names.

    Returns:
        List of saved Wi-Fi connection names.
    """
    code, out = _run_nmcli(["-t", "-f", "NAME,TYPE", "con", "show"], timeout=5.0)
    if code != 0 or not out:
        return []

    saved: List[str] = []
    for line in out.splitlines():
        if ":" in line:
            name, con_type = line.rsplit(":", 1)
            name = name.strip()
            if con_type.strip() in ("802-11-wireless", "wifi") and name:
                saved.append(name)
    return saved


def scan_visible_ssids(timeout: float = DEFAULT_SCAN_TIMEOUT_SEC) -> List[str]:
    """Scan and return currently visible Wi-Fi SSIDs in the air.

    Args:
        timeout: Seconds to wait for Wi-Fi scanning.

    Returns:
        List of visible SSID strings (deduplicated, order preserved).
    """
    code, out = _run_nmcli(["-t", "-f", "SSID", "dev", "wifi", "list", "--rescan", "yes"], timeout=timeout)
    if code != 0 or not out:
        # Retry without --rescan in case device does not support immediate rescan request
        code, out = _run_nmcli(["-t", "-f", "SSID", "dev", "wifi", "list"], timeout=timeout)
        if code != 0 or not out:
            return []

    visible: List[str] = []
    seen: Set[str] = set()
    for line in out.splitlines():
        ssid = line.strip()
        if ssid and ssid != "--" and ssid not in seen:
            seen.add(ssid)
            visible.append(ssid)
    return visible


def connect_to_wifi(ssid: str, password: Optional[str] = None, timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC) -> bool:
    """Attempt to connect to a Wi-Fi network by SSID.

    Args:
        ssid: Target Wi-Fi SSID.
        password: Optional WPA/WPA2 passphrase.
        timeout: Maximum seconds to allow for connection and DHCP negotiation.

    Returns:
        True if the connection succeeded, False otherwise.
    """
    saved = get_saved_wifi_connections()

    # If connection profile is already saved in NetworkManager and no new password provided
    if ssid in saved and not password:
        code, _ = _run_nmcli(["con", "up", "id", ssid], timeout=timeout)
        if code == 0:
            return True

    # Connect via dev wifi connect
    cmd_args = ["dev", "wifi", "connect", ssid]
    if password:
        cmd_args.extend(["password", password])

    code, _ = _run_nmcli(cmd_args, timeout=timeout)
    if code == 0:
        return True

    # If dev wifi connect failed and it was saved, try con up as fallback
    if ssid in saved:
        code, _ = _run_nmcli(["con", "up", "id", ssid], timeout=timeout)
        return code == 0

    return False


def manage_wifi_startup_fallback(
    speak_fn: Callable[[str], None],
    announcement_fn: Callable[[str], str],
    fallback_env_str: Optional[str] = None,
    is_online_fn: Optional[Callable[[], bool]] = None,
) -> bool:
    """Orchestrate Wi-Fi failover on startup with spoken status announcements.

    If not connected, announces network missing, scans for visible candidate SSIDs,
    and attempts connection to each visible network while announcing progress.

    Args:
        speak_fn: Spoken audio announcement callback.
        announcement_fn: Template lookup function (key -> speech template).
        fallback_env_str: Optional raw fallback configuration string from environment.
        is_online_fn: Connectivity verification callback.

    Returns:
        True if a connection succeeded and network is reachable, False otherwise.
    """
    if is_online_fn and is_online_fn():
        return True

    # Announce initial network not found
    not_found_msg = announcement_fn("wifi_not_found")
    if not_found_msg:
        speak_fn(not_found_msg)

    # Scan visible SSIDs in the air
    visible_ssids = scan_visible_ssids()
    if not visible_ssids:
        sys.stdout.write("[WiFi] No visible Wi-Fi networks found in scan.\n")
        return False

    # Collect candidates: saved NetworkManager profiles first, followed by .env entries
    saved_connections = get_saved_wifi_connections()
    env_candidates = parse_fallback_networks(fallback_env_str)

    candidates: List[Tuple[str, Optional[str]]] = []
    seen_ssids: Set[str] = set()

    for name in saved_connections:
        if name not in seen_ssids:
            seen_ssids.add(name)
            # Check if there is an env password for this network
            env_match = next((pwd for s, pwd in env_candidates if s == name), None)
            candidates.append((name, env_match))

    for ssid, pwd in env_candidates:
        if ssid not in seen_ssids:
            seen_ssids.add(ssid)
            candidates.append((ssid, pwd))

    # Filter candidates to only those visible in the scan
    visible_candidates = [(ssid, pwd) for ssid, pwd in candidates if ssid in visible_ssids]

    if not visible_candidates:
        sys.stdout.write("[WiFi] None of the known fallback Wi-Fi networks are currently visible.\n")
        return False

    connecting_tpl = announcement_fn("wifi_connecting_template")
    success_tpl = announcement_fn("wifi_connect_success_template")
    failed_tpl = announcement_fn("wifi_connect_failed_template")

    for ssid, pwd in visible_candidates:
        if connecting_tpl:
            msg = connecting_tpl.format(ssid=ssid)
            sys.stdout.write(f"[WiFi] {msg}\n")
            speak_fn(msg)

        connected = connect_to_wifi(ssid, password=pwd)

        # Verify connectivity if connection succeeded
        if connected and is_online_fn:
            connected = is_online_fn()

        if connected:
            if success_tpl:
                msg = success_tpl.format(ssid=ssid)
                sys.stdout.write(f"[WiFi] {msg}\n")
                speak_fn(msg)
            return True
        else:
            if failed_tpl:
                msg = failed_tpl.format(ssid=ssid)
                sys.stdout.write(f"[WiFi] {msg}\n")
                speak_fn(msg)

    return False
