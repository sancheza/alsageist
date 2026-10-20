#!/usr/bin/env python3
"""scripts/alsageist_ctl.py - Command-line control utility for AlsaGeist.

Purpose:
    Provides a fast, zero-restart CLI control interface for a running AlsaGeist
    daemon over a local Unix domain socket. Supports real-time adjustments for
    output volume, microphone muting, sleep/wake toggling, online/offline mode
    switching, personality hot-swapping, and live status monitoring.

Usage:
    # Query live status dashboard:
    ./scripts/alsageist_ctl.py status

    # Query or change volume:
    ./scripts/alsageist_ctl.py volume
    ./scripts/alsageist_ctl.py volume 75
    ./scripts/alsageist_ctl.py volume +10   # +10 percentage points
    ./scripts/alsageist_ctl.py volume -10

    # Power off the host (requires confirmation):
    ./scripts/alsageist_ctl.py shutdown --yes

    # Microphone and display controls:
    ./scripts/alsageist_ctl.py mute
    ./scripts/alsageist_ctl.py unmute
    ./scripts/alsageist_ctl.py disable
    ./scripts/alsageist_ctl.py enable
    ./scripts/alsageist_ctl.py toggle

    # Switch operational mode:
    ./scripts/alsageist_ctl.py mode offline
    ./scripts/alsageist_ctl.py mode online
    ./scripts/alsageist_ctl.py mode auto

    # Hot-swap personality profile:
    ./scripts/alsageist_ctl.py personality snowman
    ./scripts/alsageist_ctl.py personality grimjaw

    # Speak immediate custom text:
    ./scripts/alsageist_ctl.py say "Happy Halloween, mortals!"

    # Preview command without sending (Dry Run):
    ./scripts/alsageist_ctl.py volume 50 --dry-run
"""

import argparse
import json
import os
import socket
import sys
from typing import Any, Dict, Optional, Tuple

VERSION = "1.1.0"
DEFAULT_TIMEOUT_SEC = 3.0
# Commands that wait for speech or a voice model to load need longer than a status query
COMMAND_TIMEOUTS_SEC = {"say": 60.0, "personality": 120.0}


def send_command(
    cmd: str,
    value: Any = None,
    socket_path: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Send command to AlsaGeist Unix domain socket and parse response.

    Args:
        cmd: Action command name.
        value: Optional command argument.
        socket_path: Path to the UNIX domain socket file (default: $ALSAGEIST_SOCKET or
            the per-user path /tmp/alsageist-<uid>/alsageist.sock).
        timeout: Socket communication timeout in seconds (default: 3 s, longer for
            commands that wait for speech or a voice model).

    Returns:
        Tuple of (success_boolean, response_dict).
    """
    if timeout is None:
        timeout = COMMAND_TIMEOUTS_SEC.get(cmd, DEFAULT_TIMEOUT_SEC)
    if socket_path is None:
        from core import control_server
        socket_path = control_server.default_socket_path()

    if not os.path.exists(socket_path):
        return False, {
            "status": "error",
            "message": (
                f"Socket file '{socket_path}' not found.\n"
                "Is the AlsaGeist daemon running?\n"
                "Check service status with: sudo systemctl status alsageist.service"
            ),
        }

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        payload = json.dumps({"cmd": cmd, "value": value}) + "\n"
        sock.sendall(payload.encode("utf-8"))

        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk

        if not data:
            return False, {"status": "error", "message": "Empty response received from daemon."}

        resp_dict = json.loads(data.decode("utf-8").strip())
        is_ok = resp_dict.get("status") == "ok"
        return is_ok, resp_dict

    except socket.timeout:
        return False, {"status": "error", "message": f"Connection to {socket_path} timed out."}
    except ConnectionRefusedError:
        return False, {
            "status": "error",
            "message": (
                f"Connection refused at {socket_path}.\n"
                "The daemon process may have terminated unexpectedly."
            ),
        }
    except Exception as exc:
        return False, {"status": "error", "message": f"Socket communication error: {exc}"}
    finally:
        sock.close()


def format_status(data: Dict[str, Any]) -> str:
    """Format status dictionary into a scannable console dashboard.

    Args:
        data: Status dictionary from daemon.

    Returns:
        Formatted multi-line text dashboard.
    """
    state = data.get("state", "UNKNOWN")
    state_display = "ACTIVE (Listening)" if state == "ACTIVE" else "DISABLED (Sleep Mode)"
    personality = data.get("personality", "unknown")
    display_name = data.get("display_name", "AlsaGeist")
    voice = data.get("voice", "default")
    vol_pct = data.get("volume_pct", 0)
    vol_float = data.get("volume_float", 0.0)
    mic_muted = data.get("mic_muted", False)
    mic_display = data.get("mic_state") or ("MUTED" if mic_muted else "UNMUTED (Capturing)")
    mode = data.get("mode", "UNKNOWN")
    engine = data.get("engine", "INTENT")
    llm_fallback = "Enabled" if data.get("llm_fallback") else "Disabled"
    led_state = data.get("led_state", "UNKNOWN")
    pid = data.get("pid", "Unknown")

    uptime_sec = data.get("uptime_sec", 0)
    hrs = uptime_sec // 3600
    mins = (uptime_sec % 3600) // 60
    secs = uptime_sec % 60
    uptime_str = f"{hrs}h {mins}m {secs}s" if hrs > 0 else f"{mins}m {secs}s"

    lines = [
        "=" * 64,
        " AlsaGeist Runtime Status",
        "=" * 64,
        f"  Display State:     {state_display}",
        f"  Active Persona:    {display_name} ({personality})",
        f"  Voice Model:       {voice} (Local ONNX)",
        f"  Audio Output Vol:  {vol_pct}% ({vol_float})",
        f"  Microphone State:  {mic_display}",
        f"  Network Mode:      {mode}{' (forced)' if data.get('mode_forced') else ' (auto)'} (Cloud STT / LLM: {llm_fallback})",
        f"  Dialogue Engine:   {engine}",
        f"  Hardware LED:      {led_state}",
        f"  Process PID:       {pid} (Uptime: {uptime_str})",
        "=" * 64,
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="""\
================================================================================
AlsaGeist Real-Time Management & Control Utility
================================================================================
PURPOSE:
  Control an actively running AlsaGeist daemon in real time without restarting.
  Sends instantaneous commands over a local Unix domain socket that only the daemon's
  user can open (default: /tmp/alsageist-<uid>/alsageist.sock, or $ALSAGEIST_SOCKET).
""",
        epilog="""\
COMMON USAGE EXAMPLES:

  1. Query live system dashboard:
     ./scripts/alsageist_ctl.py status

  2. Adjust or check audio output volume:
     ./scripts/alsageist_ctl.py volume 80      # Set volume to 80%
     ./scripts/alsageist_ctl.py volume +10     # Increase volume by 10%
     ./scripts/alsageist_ctl.py volume -10     # Decrease volume by 10%
     ./scripts/alsageist_ctl.py volume         # Show current volume

  3. Mute or unmute the microphone:
     ./scripts/alsageist_ctl.py mute
     ./scripts/alsageist_ctl.py unmute

  4. Put display to sleep or wake it up:
     ./scripts/alsageist_ctl.py disable        # Sleep (mute mic/audio, turn off LED)
     ./scripts/alsageist_ctl.py enable         # Wake up
     ./scripts/alsageist_ctl.py toggle         # Toggle between sleep and active

  5. Switch operational network mode:
     ./scripts/alsageist_ctl.py mode offline   # 100% local Vosk STT (no cloud)
     ./scripts/alsageist_ctl.py mode online    # Restore cloud STT & LLM fallback
     ./scripts/alsageist_ctl.py mode auto      # Detect connectivity automatically again

  6. Hot-swap active personality on the fly:
     ./scripts/alsageist_ctl.py personality snowman
     ./scripts/alsageist_ctl.py personality grimjaw

  7. Speak a test line immediately through the speaker:
     ./scripts/alsageist_ctl.py say "Happy Halloween, mortals!"

  8. Toggle or inspect dynamic LLM fallback for indeterminate prompts:
     ./scripts/alsageist_ctl.py fallback on     # Enable LLM fallback (initializes cloud LLM on demand)
     ./scripts/alsageist_ctl.py fallback off    # Disable LLM fallback
     ./scripts/alsageist_ctl.py fallback        # Show current fallback state
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status",
            "volume",
            "mute",
            "unmute",
            "disable",
            "enable",
            "toggle",
            "mode",
            "personality",
            "say",
            "fallback",
            "shutdown",
            "poweroff",
        ],
        help="Action to execute (default: status).",
    )
    parser.add_argument(
        "value",
        nargs="?",
        default=None,
        help="Optional value argument for volume, mode, personality, say, or fallback.",
    )
    parser.add_argument(
        "--socket",
        dest="socket_path",
        default=None,
        help="Path to AlsaGeist UNIX domain socket (default: $ALSAGEIST_SOCKET or /tmp/alsageist-<uid>/alsageist.sock).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for the daemon's reply (default: 3, 60 for say, 120 for personality).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a host shutdown (required for shutdown and poweroff).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response from daemon.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview command without communicating with daemon.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    cmd = args.command
    val = args.value

    socket_path = args.socket_path
    if socket_path is None:
        from core import control_server
        socket_path = control_server.default_socket_path()

    if cmd in ("shutdown", "poweroff"):
        if not args.yes and not args.dry_run:
            print(f"Error: '{cmd}' powers off the host. Re-run with --yes to confirm.", file=sys.stderr)
            return 1
        val = "confirm"

    if args.dry_run:
        print(f"[DRY RUN] Would send command '{cmd}' with value '{val}' to {socket_path}.")
        return 0

    ok, resp = send_command(cmd, val, socket_path=socket_path, timeout=args.timeout)

    if args.json:
        print(json.dumps(resp, indent=2))
        return 0 if ok else 1

    if not ok:
        print(f"Error: {resp.get('message', 'Command failed.')}", file=sys.stderr)
        return 1

    if cmd == "status":
        print(format_status(resp))
    elif cmd == "volume":
        if val is None:
            print(f"Current Output Volume: {resp.get('volume_pct', 0)}% ({resp.get('volume_float', 0.0)})")
        else:
            print(f"✅ {resp.get('message', 'Volume updated.')}")
    elif cmd == "personality" and val is None:
        print(f"Current Personality: {resp.get('personality', 'unknown')}")
    else:
        msg = resp.get("message", "Command executed successfully.")
        print(f"✅ {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
