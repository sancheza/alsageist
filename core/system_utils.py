"""core/system_utils.py - Operating system lifecycle and shutdown utilities.

Provides reliable cross-platform mechanisms for triggering host system poweroff
across native bare-metal Linux (systemd / logind / shutdown) and containerized
Docker environments (shared D-Bus socket / shared systemd socket / SysRq trigger),
with safe simulation on developer workstations (macOS).
"""

import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional


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


def trigger_host_shutdown(dry_run: bool = False) -> bool:
    """Trigger an orderly poweroff and shutdown of the host system.

    Attempts methods in sequence:
    1. macOS workstation: Simulates shutdown safely to avoid powering down developer machine.
    2. systemctl poweroff (direct or via sudo).
    3. shutdown -h now (direct or via sudo).
    4. D-Bus systemd-logind method call over shared system_bus_socket.
    5. Linux Magic SysRq trigger (/proc/sysrq-trigger: sync -> remount ro -> power off).

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

    # 2. Try systemctl poweroff
    systemctl_path = shutil.which("systemctl") or "/bin/systemctl"
    if os.path.isfile(systemctl_path):
        cmds_to_try = [
            [systemctl_path, "poweroff"],
        ]
        if hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
            cmds_to_try.append(["sudo", "-n", systemctl_path, "poweroff"])

        for cmd in cmds_to_try:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    return True
            except Exception:
                pass

    # 3. Try shutdown -h now
    shutdown_path = shutil.which("shutdown") or "/sbin/shutdown"
    if os.path.isfile(shutdown_path):
        cmds_to_try = [
            [shutdown_path, "-h", "now"],
        ]
        if hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
            cmds_to_try.append(["sudo", "-n", shutdown_path, "-h", "now"])

        for cmd in cmds_to_try:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    return True
            except Exception:
                pass

    # 4. Try D-Bus method call to systemd-logind via busctl or dbus-send
    dbus_socket = "/var/run/dbus/system_bus_socket"
    if os.path.exists(dbus_socket) or os.path.exists("/run/dbus/system_bus_socket"):
        busctl_path = shutil.which("busctl")
        if busctl_path:
            try:
                res = subprocess.run(
                    [
                        busctl_path,
                        "call",
                        "org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager",
                        "PowerOff",
                        "b",
                        "true",
                    ],
                    capture_output=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        dbus_send_path = shutil.which("dbus-send")
        if dbus_send_path:
            try:
                res = subprocess.run(
                    [
                        dbus_send_path,
                        "--system",
                        "--print-reply",
                        "--dest=org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager.PowerOff",
                        "boolean:true",
                    ],
                    capture_output=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

    # 5. Fallback: Linux Magic SysRq trigger (sync -> unmount -> poweroff)
    sysrq_path = "/proc/sysrq-trigger"
    if os.path.exists(sysrq_path):
        try:
            with open(sysrq_path, "w", encoding="utf-8") as f:
                f.write("s\n")
            time.sleep(1.0)
            with open(sysrq_path, "w", encoding="utf-8") as f:
                f.write("u\n")
            time.sleep(0.5)
            with open(sysrq_path, "w", encoding="utf-8") as f:
                f.write("o\n")
            return True
        except Exception as exc:
            sys.stderr.write(f"SysRq shutdown error: {exc}\n")

    return False


def load_env_file(path: str = ".env") -> None:
    """Load key-value environment variables from an environment file if present.

    Searches relative to the current working directory, the package directory,
    or the parent directory. Existing environment variables are never overwritten.

    Args:
        path: Relative or absolute path to the environment file.
    """
    candidates = [
        path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path),
    ]
    target_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not target_path:
        return

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        print(f"Warning: Failed to load environment file from {target_path}: {exc}")


def check_internet_connectivity(timeout: float = 2.5) -> bool:
    """Check if external internet connectivity is available via DNS socket probes.

    Attempts TCP connections to Google (8.8.8.8) and Cloudflare (1.1.1.1) DNS servers
    on port 53.

    Args:
        timeout: Socket connection timeout in seconds (default: 2.5).

    Returns:
        bool: True if external host connects successfully, False otherwise.
    """
    test_hosts = [("8.8.8.8", 53), ("1.1.1.1", 53)]
    for host, port in test_hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            continue
    return False


def resolve_piper_model_path(
    model_path: Optional[str] = None,
    voice_name: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> str:
    """Resolve the file path to a Piper ONNX neural voice model.

    Evaluates explicit model paths, voice names (e.g. 'en_GB-alan-medium' or 'alan'),
    and default locations in `models/piper/`.

    Args:
        model_path: Explicit path from CLI flags or environment variables.
        voice_name: Character voice identifier from personality profile or string.
        repo_root: Root directory of the repository. Auto-resolved if None.

    Returns:
        Resolved absolute or normalized model file path.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Explicit model path from argument or environment
    if model_path:
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path
        candidate = os.path.join(repo_root, model_path)
        if os.path.exists(candidate):
            return candidate
        return model_path

    # 2. Voice name from profile (e.g. 'en_GB-alan-medium' or 'models/piper/en_GB-alan-medium.onnx')
    if voice_name:
        if voice_name.endswith(".onnx"):
            if os.path.isabs(voice_name) and os.path.exists(voice_name):
                return voice_name
            candidate = os.path.join(repo_root, voice_name)
            if os.path.exists(candidate):
                return candidate
            candidate_in_piper = os.path.join(repo_root, "models", "piper", os.path.basename(voice_name))
            if os.path.exists(candidate_in_piper):
                return candidate_in_piper
            return candidate
        else:
            # Short name or identifier without .onnx extension
            candidate = os.path.join(repo_root, "models", "piper", f"{voice_name}.onnx")
            if os.path.exists(candidate):
                return candidate

    # 3. Default fallback model
    return os.path.join(repo_root, "models", "piper", "en_GB-alan-medium.onnx")

