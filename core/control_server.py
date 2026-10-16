"""Unix domain socket IPC control server for AlsaGeist.

Provides real-time local control over a running AlsaGeist daemon, allowing
CLI clients to adjust volume, mute the microphone, toggle sleep/wake state,
switch online/offline modes, hot-swap personalities, and query status.
"""

import json
import os
import socket
import threading
import time
from typing import Any, Dict, Optional


DEFAULT_SOCKET_PATH = "/tmp/alsageist.sock"


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
        self.socket_path = socket_path or os.environ.get("ALSAGEIST_SOCKET", DEFAULT_SOCKET_PATH)
        self.running = False
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the control server background listening thread."""
        if self.running:
            return

        # Clean up stale socket file if not responding
        if os.path.exists(self.socket_path):
            if not self._is_socket_alive():
                try:
                    os.unlink(self.socket_path)
                except OSError:
                    pass
            else:
                print(f"Warning: Another instance appears to be listening on {self.socket_path}")

        try:
            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(self.socket_path)
            # Allow all users in the same Linux group or local user to communicate
            try:
                os.chmod(self.socket_path, 0o666)
            except OSError:
                pass
            self._server_sock.listen(5)
            self._server_sock.settimeout(1.0)
            self.running = True

            self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="AlsaGeist-ControlServer")
            self._thread.start()
        except Exception as exc:
            print(f"Warning: Could not start control server on {self.socket_path}: {exc}")
            self.running = False

    def stop(self) -> None:
        """Stop the server and remove the socket file."""
        self.running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def _is_socket_alive(self) -> bool:
        """Check if an existing socket file has an active listener."""
        test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            test_sock.settimeout(0.5)
            test_sock.connect(self.socket_path)
            test_sock.close()
            return True
        except Exception:
            return False

    def _listen_loop(self) -> None:
        """Main connection listening loop running on background thread."""
        while self.running and self._server_sock:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            # Handle each connection in a quick worker thread
            client_thread = threading.Thread(
                target=self._handle_client,
                args=(conn,),
                daemon=True,
            )
            client_thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        """Read command from client, dispatch to agent, and send JSON response."""
        try:
            conn.settimeout(3.0)
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(1024)
                if not chunk:
                    break
                data += chunk

            if not data:
                return

            req_str = data.decode("utf-8").strip()
            try:
                payload = json.loads(req_str)
            except json.JSONDecodeError:
                payload = {"cmd": req_str}

            cmd = payload.get("cmd", "").strip().lower()
            val = payload.get("value")

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
            agent.audio.mute_capture()
            return {"status": "ok", "message": "Microphone muted.", "mic_muted": True}

        elif cmd == "unmute":
            agent.audio.unmute_capture()
            return {"status": "ok", "message": "Microphone unmuted.", "mic_muted": False}

        elif cmd == "disable":
            agent.is_disabled = True
            agent.audio.stop_playback()
            agent.audio.mute_capture()
            agent._reset_dialogue()
            agent.vad.reset()
            from core.led_controller import LEDState
            agent.led.set_state(LEDState.OFF)
            return {"status": "ok", "message": "Display disabled (sleep mode).", "state": "DISABLED"}

        elif cmd == "enable":
            agent.is_disabled = False
            agent.audio.unmute_capture()
            agent.vad.reset()
            from core.led_controller import LEDState
            agent.led.set_state(LEDState.IDLE)
            return {"status": "ok", "message": "Display enabled (active).", "state": "ACTIVE"}

        elif cmd == "toggle":
            agent._on_button_hold_toggle()
            state_str = "DISABLED" if agent.is_disabled else "ACTIVE"
            return {"status": "ok", "message": f"Display toggled to {state_str}.", "state": state_str}

        elif cmd == "mode":
            mode_str = str(value).strip().lower() if value else ""
            if mode_str in ("online", "on"):
                agent.is_online = True
            elif mode_str in ("offline", "off"):
                agent.is_online = False
            else:
                return {"status": "error", "message": f"Invalid mode '{value}'. Use 'online' or 'offline'."}
            mode_name = "ONLINE" if agent.is_online else "OFFLINE"
            return {"status": "ok", "message": f"Operational mode set to {mode_name}.", "mode": mode_name}

        elif cmd == "personality":
            if not value:
                p_name = getattr(agent.dialogue_engine, "profile", None) and agent.dialogue_engine.profile.personality or "unknown"
                return {"status": "ok", "personality": p_name}

            try:
                from services.intent_service import PersonaIntentEngine
                new_engine = PersonaIntentEngine(personality_name_or_path=str(value).strip())
                agent.dialogue_engine = new_engine

                # Update LLM prompt if active
                if agent.llm_fallback_enabled:
                    profile = getattr(new_engine, "profile", None)
                    if profile:
                        from services.llm_service import ConversationalLLMService
                        persona_prompt = (
                            f"You are {profile.display_name}. {profile.description}\n"
                            "Rules you must strictly obey:\n"
                            "1. Keep every response under 2 sentences maximum.\n"
                            "2. Be witty, dry, and stay in character.\n"
                            "3. Never output emojis, markdown formatting, asterisks, or sound effect tags.\n"
                            "4. Return plain spoken dialogue only."
                        )
                        agent.llm_fallback_service = ConversationalLLMService(system_prompt=persona_prompt)

                display_title = new_engine.profile.display_name if new_engine.profile else str(value)
                return {
                    "status": "ok",
                    "message": f"Personality switched to '{display_title}'.",
                    "personality": new_engine.profile.personality if new_engine.profile else str(value),
                    "display_name": display_title,
                }
            except Exception as exc:
                return {"status": "error", "message": f"Failed to switch personality: {exc}"}

        elif cmd == "say":
            phrase = str(value).strip() if value else ""
            if not phrase:
                return {"status": "error", "message": "No phrase provided to speak."}

            try:
                from core.led_controller import LEDState
                audio_data, sample_rate = agent.tts.synthesize(phrase)
                agent.led.set_state(LEDState.SPEAKING)
                agent.audio.mute_capture()
                agent.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=True)
                agent.audio.unmute_capture()
                agent.vad.reset()
                agent.led.set_state(LEDState.IDLE)
                return {"status": "ok", "message": f"Spoke: \"{phrase}\"", "spoken": phrase}
            except Exception as exc:
                return {"status": "error", "message": f"Speech failed: {exc}"}

        elif cmd in ("shutdown", "poweroff"):
            if hasattr(agent, "_on_button_long_hold"):
                threading.Thread(target=agent._on_button_long_hold, daemon=True).start()
            else:
                from core.system_utils import trigger_host_shutdown
                trigger_host_shutdown()
            return {"status": "ok", "message": "Host system shutdown initiated."}

        else:
            return {
                "status": "error",
                "message": f"Unknown command '{cmd}'. Available: status, volume, mute, unmute, disable, enable, toggle, mode, personality, say, shutdown.",
            }

    def _cmd_status(self) -> Dict[str, Any]:
        """Compile comprehensive runtime status."""
        agent = self.agent
        profile = getattr(agent.dialogue_engine, "profile", None)
        vol_pct = int(round(agent.audio.volume * 100))

        start_time = getattr(agent, "start_time", None) or time.time()
        uptime_sec = int(time.time() - start_time)

        led_state = "UNKNOWN"
        if hasattr(agent.led, "current_state"):
            led_state = getattr(agent.led.current_state, "name", str(agent.led.current_state))

        return {
            "status": "ok",
            "state": "DISABLED" if agent.is_disabled else "ACTIVE",
            "personality": profile.personality if profile else "unknown",
            "display_name": profile.display_name if profile else "AlsaGeist Persona",
            "voice": profile.voice if profile else "default",
            "volume_pct": vol_pct,
            "volume_float": round(agent.audio.volume, 2),
            "mic_muted": getattr(agent.audio, "_is_muted", False),
            "disabled": agent.is_disabled,
            "mode": "ONLINE" if agent.is_online else "OFFLINE",
            "engine": agent.args.engine.upper(),
            "llm_fallback": getattr(agent, "llm_fallback_enabled", False),
            "led_state": led_state,
            "uptime_sec": uptime_sec,
            "pid": os.getpid(),
        }

    def _cmd_volume(self, value: Any) -> Dict[str, Any]:
        """Adjust or query audio volume."""
        agent = self.agent
        if value is None:
            pct = int(round(agent.audio.volume * 100))
            return {"status": "ok", "volume_pct": pct, "volume_float": round(agent.audio.volume, 2)}

        val_str = str(value).strip()
        current_vol = agent.audio.volume

        if val_str.startswith("+"):
            try:
                delta = float(val_str[1:])
                delta = delta / 100.0 if delta > 1.0 else delta
                new_vol = min(1.0, current_vol + delta)
            except ValueError:
                return {"status": "error", "message": f"Invalid volume increment '{val_str}'"}
        elif val_str.startswith("-"):
            try:
                delta = float(val_str[1:])
                delta = delta / 100.0 if delta > 1.0 else delta
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

        agent.audio.volume = round(new_vol, 2)
        pct = int(round(agent.audio.volume * 100))
        return {
            "status": "ok",
            "message": f"Volume set to {pct}%.",
            "volume_pct": pct,
            "volume_float": agent.audio.volume,
        }
