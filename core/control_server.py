"""Unix domain socket IPC control server for AlsaGeist.

Provides real-time local control over a running AlsaGeist daemon, allowing
CLI clients to adjust volume, mute the microphone, toggle sleep/wake state,
switch online/offline modes, hot-swap personalities, and query status.
"""

import json
import os
import socket
import stat
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional

MAX_REQUEST_BYTES = 64 * 1024
REQUEST_DEADLINE_SEC = 3.0  # total time a client has to send its request
MAX_SAY_CHARS = 500
SHUTDOWN_CONFIRMATION = "confirm"


class ControlSocketError(RuntimeError):
    """Raised when the control socket location cannot be made private."""


def default_socket_dir() -> str:
    """Return the per-user private directory holding the default control socket."""
    return f"/tmp/alsageist-{os.getuid()}"


def default_socket_path() -> str:
    """Return the control socket path: $ALSAGEIST_SOCKET, else a per-user private path.

    Returns:
        Socket file path.
    """
    explicit = os.environ.get("ALSAGEIST_SOCKET")
    if explicit:
        return explicit
    return os.path.join(default_socket_dir(), "alsageist.sock")


def _secure_private_dir(directory: str) -> None:
    """Create a directory readable only by the current user, or verify an existing one.

    Args:
        directory: Directory to create or check.

    Raises:
        ControlSocketError: If the directory is a symlink, is owned by another user,
            or is accessible to group or others.
    """
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass

    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ControlSocketError(f"'{directory}' is not a plain directory.")
    if info.st_uid != os.getuid():
        raise ControlSocketError(f"'{directory}' is owned by another user.")
    if info.st_mode & 0o077:
        raise ControlSocketError(f"'{directory}' must not be accessible to group or others (mode 0700).")


class ControlServer:
    """Unix domain socket server running in a background thread within AlsaGeist."""

    def __init__(
        self,
        agent: Any,
        socket_path: Optional[str] = None,
    ) -> None:
        """Initialize the control server.

        Args:
            agent: The parent AlsaGeistAgent instance.
            socket_path: Path to the UNIX domain socket file.
        """
        self.agent = agent
        self.socket_path = socket_path or default_socket_path()
        self.running = False
        self._bound = False
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._client_semaphore = threading.BoundedSemaphore(8)

    def start(self) -> None:
        """Start the control server background listening thread.

        The socket is created with mode 0600 inside a directory that only the
        current user can open. On failure the server stays stopped and a warning
        is printed; the agent keeps running.
        """
        if self.running:
            return

        try:
            if os.path.dirname(self.socket_path) == default_socket_dir():
                _secure_private_dir(default_socket_dir())
            self._remove_stale_socket()

            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Create the socket file owner-only from the start (no window with wider access)
            previous_umask = os.umask(0o177)
            try:
                self._server_sock.bind(self.socket_path)
            finally:
                os.umask(previous_umask)
            self._bound = True
            os.chmod(self.socket_path, 0o600)
            self._server_sock.listen(5)
            self._server_sock.settimeout(1.0)
            self.running = True

            self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="AlsaGeist-ControlServer")
            self._thread.start()
        except Exception as exc:
            print(f"Warning: Could not start control server on {self.socket_path}: {exc}")
            self.running = False
            if self._server_sock is not None:
                self._server_sock.close()
                self._server_sock = None
            if self._bound:
                self._unlink_socket()
                self._bound = False

    def stop(self) -> None:
        """Stop the server and remove the socket file it created."""
        self.running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        if self._bound:
            self._unlink_socket()
            self._bound = False

    def _unlink_socket(self) -> None:
        """Remove the socket file, ignoring errors."""
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    def _remove_stale_socket(self) -> None:
        """Remove a dead socket file left by an earlier run.

        Raises:
            ControlSocketError: If a live server owns the path, or the path is not a
                socket owned by the current user.
        """
        try:
            info = os.lstat(self.socket_path)
        except FileNotFoundError:
            return

        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise ControlSocketError(f"'{self.socket_path}' exists and is not a socket owned by this user.")
        if self._is_socket_alive():
            raise ControlSocketError(f"another instance is already listening on '{self.socket_path}'.")
        os.unlink(self.socket_path)

    def _is_socket_alive(self) -> bool:
        """Check if an existing socket file has an active listener."""
        test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            test_sock.settimeout(0.5)
            test_sock.connect(self.socket_path)
            return True
        except OSError:
            return False
        finally:
            test_sock.close()

    def _listen_loop(self) -> None:
        """Main connection listening loop running on background thread."""
        while self.running and self._server_sock:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            if not self._peer_allowed(conn):
                conn.close()
                continue

            if not self._client_semaphore.acquire(blocking=False):
                try:
                    conn.settimeout(1.0)
                    busy_msg = (
                        json.dumps({"status": "error", "message": "Server busy: maximum concurrent clients reached."})
                        + "\n"
                    )
                    conn.sendall(busy_msg.encode("utf-8"))
                except Exception:
                    pass
                finally:
                    conn.close()
                continue

            # Handle each connection in a quick worker thread
            client_thread = threading.Thread(
                target=self._handle_client,
                args=(conn,),
                daemon=True,
            )
            client_thread.start()

    @staticmethod
    def _peer_allowed(conn: socket.socket) -> bool:
        """Check that the connecting process runs as this user (or root).

        Where the OS cannot report peer credentials (non-Linux), the 0600 socket
        file and private directory are the only access control.

        Args:
            conn: Accepted client connection.

        Returns:
            True if the connection is allowed.
        """
        if not sys.platform.startswith("linux"):
            return True
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", creds)
        except (OSError, AttributeError, struct.error):
            return False
        return uid in (os.geteuid(), 0)

    def _handle_client(self, conn: socket.socket) -> None:
        """Read command from client, dispatch to agent, and send JSON response."""
        try:
            data = b""
            deadline = time.monotonic() + REQUEST_DEADLINE_SEC
            while not data.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    conn.settimeout(1.0)
                    conn.sendall((json.dumps({"status": "error", "message": "Request timed out."}) + "\n").encode("utf-8"))
                    return
                conn.settimeout(remaining)
                chunk = conn.recv(1024)
                if not chunk:
                    break
                data += chunk
                if len(data) > MAX_REQUEST_BYTES:
                    conn.sendall((json.dumps({"status": "error", "message": "Request too large."}) + "\n").encode("utf-8"))
                    return

            if not data:
                return

            req_str = data.decode("utf-8").strip()
            try:
                payload = json.loads(req_str)
            except json.JSONDecodeError:
                payload = {"cmd": req_str}
            if not isinstance(payload, dict) or not isinstance(payload.get("cmd", ""), str):
                return self._reply(conn, {"status": "error", "message": "Request must be a JSON object with a string 'cmd'."})

            cmd = payload.get("cmd", "").strip().lower()
            val = payload.get("value")

            conn.settimeout(3.0)
            response = self.dispatch_command(cmd, val)
            resp_bytes = (json.dumps(response) + "\n").encode("utf-8")
            conn.sendall(resp_bytes)
        except Exception as exc:
            err_resp = {"status": "error", "message": str(exc)}
            try:
                conn.sendall((json.dumps(err_resp) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self._client_semaphore.release()

    @staticmethod
    def _reply(conn: socket.socket, response: Dict[str, Any]) -> None:
        """Send one JSON response line."""
        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

    def dispatch_command(self, cmd: str, value: Any = None) -> Dict[str, Any]:
        """Dispatch command to the AlsaGeist agent and return response dict.

        Args:
            cmd: Command action name.
            value: Optional command argument.

        Returns:
            Response dictionary.
        """
        agent = self.agent

        if cmd == "status":
            return self._cmd_status()

        elif cmd == "volume":
            return self._cmd_volume(value)

        elif cmd == "mute":
            agent.set_mic_muted(True)
            return {"status": "ok", "message": "Microphone muted.", "mic_muted": True}

        elif cmd == "unmute":
            agent.set_mic_muted(False)
            return {"status": "ok", "message": "Microphone unmuted.", "mic_muted": False}

        elif cmd == "disable":
            agent.disable()
            return {"status": "ok", "message": "Display disabled (sleep mode).", "state": "DISABLED"}

        elif cmd == "enable":
            agent.enable()
            return {"status": "ok", "message": "Display enabled (active).", "state": "ACTIVE"}

        elif cmd == "toggle":
            state_str = "DISABLED" if agent.toggle() else "ACTIVE"
            return {"status": "ok", "message": f"Display toggled to {state_str}.", "state": state_str}

        elif cmd == "mode":
            mode_str = str(value).strip().lower() if value else ""
            if mode_str in ("online", "on"):
                agent.set_online(True)
            elif mode_str in ("offline", "off"):
                agent.set_online(False)
            elif mode_str == "auto":
                agent.set_online(None)
            else:
                return {"status": "error", "message": f"Invalid mode '{value}'. Use 'online', 'offline' or 'auto'."}
            mode_name = "ONLINE" if agent.is_online else "OFFLINE"
            if agent.mode_forced:
                message = f"Operational mode forced to {mode_name}."
            else:
                message = f"Operational mode is automatic (currently {mode_name})."
            return {"status": "ok", "message": message, "mode": mode_name, "mode_forced": agent.mode_forced}

        elif cmd == "personality":
            if not value:
                profile = agent.active_profile
                return {"status": "ok", "personality": profile.personality if profile else "unknown"}

            try:
                profile = agent.switch_personality(str(value).strip())
            except Exception as exc:
                return {"status": "error", "message": f"Failed to switch personality: {exc}"}
            return {
                "status": "ok",
                "message": f"Personality switched to '{profile.display_name}'.",
                "personality": profile.personality,
                "display_name": profile.display_name,
            }

        elif cmd == "say":
            phrase = str(value).strip() if value else ""
            if not phrase:
                return {"status": "error", "message": "No phrase provided to speak."}
            if len(phrase) > MAX_SAY_CHARS:
                return {"status": "error", "message": f"Phrase is too long ({len(phrase)} characters; limit {MAX_SAY_CHARS})."}
            if agent.is_disabled:
                return {"status": "error", "message": "Display is disabled (sleep mode); enable it first."}
            if not agent.speak(phrase):
                return {"status": "error", "message": "Speech failed."}
            return {"status": "ok", "message": f"Spoke: \"{phrase}\"", "spoken": phrase}

        elif cmd in ("shutdown", "poweroff"):
            if str(value) != SHUTDOWN_CONFIRMATION:
                return {
                    "status": "error",
                    "message": f"Shutdown powers off the host. Send value '{SHUTDOWN_CONFIRMATION}' to confirm.",
                }
            agent.request_shutdown()
            return {"status": "ok", "message": "Host system shutdown initiated. If the host stays up, see the daemon log."}

        else:
            return {
                "status": "error",
                "message": f"Unknown command '{cmd}'. Available: status, volume, mute, unmute, disable, enable, toggle, mode, personality, say, shutdown.",
            }

    def _cmd_status(self) -> Dict[str, Any]:
        """Compile comprehensive runtime status."""
        agent = self.agent
        profile = agent.active_profile
        vol_pct = int(round(agent.volume * 100))

        uptime_sec = int(time.time() - agent.start_time)

        return {
            "status": "ok",
            "state": "DISABLED" if agent.is_disabled else "ACTIVE",
            "personality": profile.personality if profile else "unknown",
            "display_name": profile.display_name if profile else "AlsaGeist Persona",
            "voice": profile.voice if profile else "default",
            "volume_pct": vol_pct,
            "volume_float": round(agent.volume, 2),
            "mic_muted": agent.mic_muted,
            "disabled": agent.is_disabled,
            "mode": "ONLINE" if agent.is_online else "OFFLINE",
            "mode_forced": agent.mode_forced,
            "engine": agent.args.engine.upper(),
            "llm_fallback": getattr(agent, "llm_fallback_enabled", False),
            "led_state": agent.led_state_name,
            "uptime_sec": uptime_sec,
            "pid": os.getpid(),
        }

    def _cmd_volume(self, value: Any) -> Dict[str, Any]:
        """Adjust or query audio volume."""
        agent = self.agent
        if value is None:
            pct = int(round(agent.volume * 100))
            return {"status": "ok", "volume_pct": pct, "volume_float": round(agent.volume, 2)}

        val_str = str(value).strip()
        current_vol = agent.volume

        # A leading + or - is a change in percentage points ("+1" is one point, not 100%)
        if val_str.startswith("+"):
            try:
                delta = float(val_str[1:]) / 100.0
                new_vol = min(1.0, current_vol + delta)
            except ValueError:
                return {"status": "error", "message": f"Invalid volume increment '{val_str}'"}
        elif val_str.startswith("-"):
            try:
                delta = float(val_str[1:]) / 100.0
                new_vol = max(0.0, current_vol - delta)
            except ValueError:
                return {"status": "error", "message": f"Invalid volume decrement '{val_str}'"}
        else:
            try:
                num = float(val_str)
                # If given as percentage (e.g. 75), normalize to 0.75
                new_vol = num / 100.0 if num > 1.0 else num
                new_vol = min(1.0, max(0.0, new_vol))
            except ValueError:
                return {"status": "error", "message": f"Invalid volume value '{val_str}'. Use 0-100 or +10/-10."}

        applied = agent.set_volume(new_vol)
        pct = int(round(applied * 100))
        return {
            "status": "ok",
            "message": f"Volume set to {pct}%.",
            "volume_pct": pct,
            "volume_float": applied,
        }
