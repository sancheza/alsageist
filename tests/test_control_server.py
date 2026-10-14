"""Unit tests for Unix domain socket IPC control server and alsageist_ctl CLI."""

import argparse
import json
import os
import socket
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

from core.control_server import ControlServer
from core.led_controller import ConsoleLEDController, LEDState
from services.intent_service import PersonaIntentEngine


class MockAgent:
    """Mock AlsaGeistAgent for ControlServer testing."""

    def __init__(self):
        self.is_disabled = False
        self.is_online = True
        self.start_time = time.time() - 100
        self.args = argparse.Namespace(engine="intent")
        self.dialogue_engine = PersonaIntentEngine("grimjaw")
        self.led = ConsoleLEDController(verbose=False)
        self.vad = MagicMock()
        self.vad.reset = MagicMock()

        self.audio = MagicMock()
        self.audio.volume = 0.8
        self.audio._is_muted = False
        self.audio.mute_capture = MagicMock(side_effect=lambda: setattr(self.audio, "_is_muted", True))
        self.audio.unmute_capture = MagicMock(side_effect=lambda: setattr(self.audio, "_is_muted", False))
        self.audio.stop_playback = MagicMock()
        self.audio.play_audio = MagicMock()

        self.tts = MagicMock()
        self.tts.synthesize = MagicMock(return_value=(b"\x00" * 3200, 16000))

        self.llm_fallback_enabled = False
        self.llm_fallback_service = None

    def _reset_dialogue(self):
        pass

    def _on_button_hold_toggle(self):
        self.is_disabled = not self.is_disabled


import uuid


@pytest.fixture
def test_socket_path():
    """Provide a unique short socket path under /tmp to respect OS socket path length limits."""
    sock_path = f"/tmp/alsageist_t_{uuid.uuid4().hex[:8]}.sock"
    yield sock_path
    if os.path.exists(sock_path):
        try:
            os.unlink(sock_path)
        except OSError:
            pass


@pytest.fixture
def server_and_agent(test_socket_path):
    """Provide a running ControlServer and MockAgent, tearing down cleanly."""
    agent = MockAgent()
    server = ControlServer(agent=agent, socket_path=test_socket_path)
    server.start()
    # Allow background thread a moment to bind and listen
    time.sleep(0.05)
    yield server, agent
    server.stop()


def test_control_server_lifecycle(test_socket_path):
    """Test start and stop lifecycle and socket file creation/removal."""
    agent = MockAgent()
    server = ControlServer(agent=agent, socket_path=test_socket_path)
    assert not os.path.exists(test_socket_path)

    server.start()
    time.sleep(0.05)
    assert server.running
    assert os.path.exists(test_socket_path)

    server.stop()
    assert not server.running
    assert not os.path.exists(test_socket_path)


def test_control_server_stale_socket_cleanup(test_socket_path):
    """Test that dead/stale socket file is cleaned up on start."""
    # Create a dummy stale socket file
    with open(test_socket_path, "w") as f:
        f.write("stale")

    agent = MockAgent()
    server = ControlServer(agent=agent, socket_path=test_socket_path)
    server.start()
    time.sleep(0.05)
    assert server.running
    assert os.path.exists(test_socket_path)
    server.stop()


def test_dispatch_status(server_and_agent):
    """Test status command returns expected fields."""
    server, agent = server_and_agent
    resp = server.dispatch_command("status")
    assert resp["status"] == "ok"
    assert resp["state"] == "ACTIVE"
    assert resp["personality"] == "grimjaw"
    assert resp["volume_pct"] == 80
    assert resp["mic_muted"] is False
    assert resp["mode"] == "ONLINE"
    assert resp["engine"] == "INTENT"
    assert resp["uptime_sec"] >= 100


def test_dispatch_volume_query(server_and_agent):
    """Test volume query without value."""
    server, agent = server_and_agent
    agent.audio.volume = 0.65
    resp = server.dispatch_command("volume")
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 65
    assert resp["volume_float"] == 0.65


def test_dispatch_volume_set(server_and_agent):
    """Test setting absolute volume values."""
    server, agent = server_and_agent

    # Percentage string
    resp = server.dispatch_command("volume", "50")
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 50
    assert agent.audio.volume == 0.5

    # Decimal float
    resp = server.dispatch_command("volume", 0.9)
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 90
    assert agent.audio.volume == 0.9

    # Relative increment
    resp = server.dispatch_command("volume", "+10")
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 100
    assert agent.audio.volume == 1.0

    # Relative decrement
    resp = server.dispatch_command("volume", "-30")
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 70
    assert agent.audio.volume == 0.7

    # Clamping below 0
    resp = server.dispatch_command("volume", "-99")
    assert resp["status"] == "ok"
    assert resp["volume_pct"] == 0
    assert agent.audio.volume == 0.0

    # Invalid volume string
    resp = server.dispatch_command("volume", "invalid")
    assert resp["status"] == "error"


def test_dispatch_mute_and_unmute(server_and_agent):
    """Test mute and unmute commands."""
    server, agent = server_and_agent

    resp = server.dispatch_command("mute")
    assert resp["status"] == "ok"
    assert resp["mic_muted"] is True
    agent.audio.mute_capture.assert_called()

    resp = server.dispatch_command("unmute")
    assert resp["status"] == "ok"
    assert resp["mic_muted"] is False
    agent.audio.unmute_capture.assert_called()


def test_dispatch_disable_enable_toggle(server_and_agent):
    """Test disable, enable, and toggle sleep/wake commands."""
    server, agent = server_and_agent

    # Disable
    resp = server.dispatch_command("disable")
    assert resp["status"] == "ok"
    assert agent.is_disabled is True
    assert resp["state"] == "DISABLED"

    # Enable
    resp = server.dispatch_command("enable")
    assert resp["status"] == "ok"
    assert agent.is_disabled is False
    assert resp["state"] == "ACTIVE"

    # Toggle
    resp = server.dispatch_command("toggle")
    assert resp["status"] == "ok"
    assert agent.is_disabled is True
    assert resp["state"] == "DISABLED"

    resp = server.dispatch_command("toggle")
    assert resp["status"] == "ok"
    assert agent.is_disabled is False
    assert resp["state"] == "ACTIVE"


def test_dispatch_mode(server_and_agent):
    """Test mode switching between online and offline."""
    server, agent = server_and_agent

    resp = server.dispatch_command("mode", "offline")
    assert resp["status"] == "ok"
    assert resp["mode"] == "OFFLINE"
    assert agent.is_online is False

    resp = server.dispatch_command("mode", "online")
    assert resp["status"] == "ok"
    assert resp["mode"] == "ONLINE"
    assert agent.is_online is True

    resp = server.dispatch_command("mode", "unsupported")
    assert resp["status"] == "error"


def test_dispatch_personality(server_and_agent):
    """Test personality query and hot-swapping."""
    server, agent = server_and_agent

    # Query current
    resp = server.dispatch_command("personality")
    assert resp["status"] == "ok"
    assert resp["personality"] == "grimjaw"

    # Switch to grimjaw explicitly
    resp = server.dispatch_command("personality", "grimjaw")
    assert resp["status"] == "ok"
    assert resp["personality"] == "grimjaw"
    assert "Grimjaw" in resp["display_name"]

    # Switch to nonexistent profile
    resp = server.dispatch_command("personality", "nonexistent_persona_xyz")
    assert resp["status"] == "error"


def test_dispatch_say(server_and_agent):
    """Test say command executes TTS synthesis and playback."""
    server, agent = server_and_agent

    resp = server.dispatch_command("say", "Hello there!")
    assert resp["status"] == "ok"
    assert resp["spoken"] == "Hello there!"
    agent.tts.synthesize.assert_called_with("Hello there!")
    agent.audio.play_audio.assert_called()

    # Empty phrase error
    resp = server.dispatch_command("say", "")
    assert resp["status"] == "error"


def test_dispatch_unknown_command(server_and_agent):
    """Test unknown command returns error response."""
    server, _ = server_and_agent
    resp = server.dispatch_command("unknown_xyz")
    assert resp["status"] == "error"
    assert "Unknown command" in resp["message"]


def test_socket_ipc_roundtrip(server_and_agent, test_socket_path):
    """Test end-to-end socket client communication."""
    from scripts.alsageist_ctl import send_command

    ok, resp = send_command("status", socket_path=test_socket_path)
    assert ok is True
    assert resp["status"] == "ok"
    assert resp["state"] == "ACTIVE"

    ok, resp = send_command("volume", 45, socket_path=test_socket_path)
    assert ok is True
    assert resp["volume_pct"] == 45

    ok, resp = send_command("volume", socket_path=test_socket_path)
    assert ok is True
    assert resp["volume_pct"] == 45


def test_cli_dry_run():
    """Test CLI main() with dry-run flag."""
    from scripts.alsageist_ctl import main

    test_args = ["alsageist_ctl.py", "volume", "70", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        ret = main()
        assert ret == 0


def test_cli_missing_socket():
    """Test CLI handles missing socket gracefully."""
    from scripts.alsageist_ctl import send_command

    ok, resp = send_command("status", socket_path="/tmp/nonexistent_socket_12345.sock")
    assert ok is False
    assert resp["status"] == "error"
    assert "Socket file" in resp["message"]
