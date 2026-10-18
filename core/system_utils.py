"""core/system_utils.py - Operating system lifecycle and shutdown utilities.

Provides reliable cross-platform mechanisms for triggering host system poweroff
across native bare-metal Linux (systemd / logind / shutdown) and containerized
Docker environments (shared D-Bus socket / shared systemd socket / SysRq trigger),
with safe simulation on developer workstations (macOS). Also loads the `.env` file and
probes internet connectivity. Voice model resolution is in core.voice_config.
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import List


def is_running_in_docker() -> bool:
    """Return True if executing inside a Docker or OCI container environment.

    Returns:
        bool: True if container environment flags or control groups are detected.
    """
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as f:
            content = f.read()
            return "docker" in content or "containerd" in content or "kubepods" in content
    except Exception:
        return False


def _command_succeeds(cmd: List[str], timeout: float = 5.0) -> bool:
    """Run a command and report whether it exited with status 0.

    Args:
        cmd: Command and arguments.
        timeout: Seconds to wait before giving up.

    Returns:
        True if the command ran and exited with status 0.
    """
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def trigger_host_shutdown(dry_run: bool = False) -> bool:
    """Trigger an orderly poweroff and shutdown of the host system.

    Attempts methods in sequence:
    1. macOS workstation: Simulates shutdown safely to avoid powering down developer machine.
    2. systemctl poweroff, then shutdown -h now (each direct, then via sudo).
    3. D-Bus systemd-logind method call over shared system_bus_socket.
    4. Linux Magic SysRq trigger (/proc/sysrq-trigger: sync -> remount ro -> power off).

    Args:
        dry_run: If True, log intent and return without executing shutdown commands.

    Returns:
        bool: True if a shutdown sequence was dispatched, False if all methods failed.
    """
    if dry_run:
        sys.stdout.write("[SHUTDOWN] Dry run active: Shutdown command simulated successfully.\n")
        return True

    # 1. Developer Workstation Safety Guard
    if sys.platform == "darwin":
        sys.stdout.write("[SHUTDOWN] macOS host detected: Shutdown simulated safely (workstation will not power off).\n")
        return True

    # 2. Try systemctl poweroff, then shutdown -h now
    for fallback_path, args in (
        ("/bin/systemctl", ["poweroff"]),
        ("/sbin/shutdown", ["-h", "now"]),
    ):
        binary = shutil.which(os.path.basename(fallback_path)) or fallback_path
        if not os.path.isfile(binary):
            continue
        cmds_to_try = [[binary, *args]]
        if hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
            cmds_to_try.append(["sudo", "-n", binary, *args])
        if any(_command_succeeds(cmd) for cmd in cmds_to_try):
            return True

    # 3. Try D-Bus method call to systemd-logind via busctl or dbus-send
    if os.path.exists("/var/run/dbus/system_bus_socket") or os.path.exists("/run/dbus/system_bus_socket"):
        dbus_cmds = []
        busctl_path = shutil.which("busctl")
        if busctl_path:
            dbus_cmds.append([
                busctl_path, "call", "org.freedesktop.login1", "/org/freedesktop/login1",
                "org.freedesktop.login1.Manager", "PowerOff", "b", "true",
            ])
        dbus_send_path = shutil.which("dbus-send")
        if dbus_send_path:
            dbus_cmds.append([
                dbus_send_path, "--system", "--print-reply", "--dest=org.freedesktop.login1",
                "/org/freedesktop/login1", "org.freedesktop.login1.Manager.PowerOff", "boolean:true",
            ])
        if any(_command_succeeds(cmd) for cmd in dbus_cmds):
            return True

    # 4. Fallback: Linux Magic SysRq trigger (sync -> unmount -> poweroff)
    sysrq_path = "/proc/sysrq-trigger"
    sysrq_mask_path = "/proc/sys/kernel/sysrq"
    if os.path.exists(sysrq_path) and os.path.exists(sysrq_mask_path):
        try:
            with open(sysrq_mask_path, "r", encoding="utf-8") as f:
                mask = int(f.read().strip())
            # 1 enables all commands; bit 7 (128) enables poweroff ('o') and bit 5 (32) enables remount-ro ('u')
            can_poweroff = (mask == 1) or bool(mask & 128)
            can_unmount = (mask == 1) or bool(mask & 32)
            if can_poweroff:
                with open(sysrq_path, "w", encoding="utf-8") as f:
                    f.write("s\n")
                time.sleep(1.0)
                if can_unmount:
                    with open(sysrq_path, "w", encoding="utf-8") as f:
                        f.write("u\n")
                    time.sleep(0.5)
                with open(sysrq_path, "w", encoding="utf-8") as f:
                    f.write("o\n")
                time.sleep(1.5)
        except Exception as exc:
            sys.stderr.write(f"SysRq shutdown error: {exc}\n")

    return False


def _parse_env_value(raw: str) -> str:
    """Return the value of a `.env` assignment, without quotes or a trailing comment.

    A quoted value ends at its first unescaped matching quote, so `#` inside quotes is kept.
    An unquoted value ends at the first `#` that starts a comment (at the start of the
    value or after whitespace), so `#` inside a token such as a key is kept.

    Args:
        raw: Text after the `=` sign.

    Returns:
        The parsed value.
    """
    val = raw.strip()
    if val[:1] in ('"', "'"):
        index = 1
        while index < len(val):
            if val[index] == "\\":  # a backslash-escaped quote does not close the value
                index += 2
            elif val[index] == val[0]:
                return val[1:index]
            else:
                index += 1
        return val
    if val.startswith("#"):
        return ""
    return re.split(r"\s#", val, maxsplit=1)[0].strip()


def load_env_file(path: str = ".env") -> None:
    """Load key-value environment variables from an environment file if present.

    An absolute path is used as given. A relative path is looked up in the repository
    root first and then in the current working directory, so a stray `.env` in the
    directory a command runs from cannot take precedence over the project's own.
    Existing environment variables are never overwritten.

    Args:
        path: Relative or absolute path to the environment file.
    """
    if os.path.isabs(path):
        candidates = [path]
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(repo_root, path), path]
    target_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not target_path:
        return

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = _parse_env_value(val)
    except Exception as exc:
        print(f"Warning: Failed to load environment file from {target_path}: {exc}")


def check_internet_connectivity(timeout: float = 2.5) -> bool:
    """Check if external internet connectivity is available via TCP socket probes.

    Tries Cloudflare (1.1.1.1) and Google (8.8.8.8) on port 443, which the cloud
    services use and which networks rarely block, then on port 53. No DNS lookup is
    needed because the probes use IP addresses.

    Args:
        timeout: Socket connection timeout in seconds (default: 2.5).

    Returns:
        bool: True if external host connects successfully, False otherwise.
    """
    test_hosts = [("1.1.1.1", 443), ("8.8.8.8", 443), ("1.1.1.1", 53), ("8.8.8.8", 53)]
    for host, port in test_hosts:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            return True
        except OSError:
            continue
        finally:
            if sock is not None:
                sock.close()
    return False

