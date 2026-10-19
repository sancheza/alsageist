"""Unit tests for Unix domain socket IPC control server and alsageist_ctl CLI."""

import argparse
import json
import os
import socket
import stat
import struct
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

from core.control_server import (
    MAX_REQUEST_BYTES,
    ControlServer,
    ControlSocketError,
    _secure_private_dir,
    default_socket_path,
    is_socket_active,
    parse_volume,
)
from core.intent_builder import PersonalityProfile, find_personality_file
from services.intent_service import PersonaIntentEngine
from tests.agent_helpers import make_bare_agent


def build_agent():
    """Return a real AlsaGeistAgent running against mocked hardware subsystems."""
    profile = PersonalityProfile.load_from_file(find_personality_file("grimjaw"))
    agent = make_bare_agent(
        profile=profile,
        dialogue_engine=PersonaIntentEngine(profile),
        start_time=time.time() - 100,
    )
    agent.audio.volume = 0.8
    # Personality switches rebuild the TTS voice; keep the mocked synthesizer.
    agent._build_tts = MagicMock(return_value=agent.tts)
    return agent


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
    """Provide a running ControlServer and agent, tearing down cleanly."""
    agent = build_agent()
    server = ControlServer(agent=agent, socket_path=test_socket_path)
    server.start()
    # Allow background thread a moment to bind and listen
    time.sleep(0.05)
    yield server, agent
    server.stop()


def test_control_server_lifecycle(test_socket_path):
    """Test start and stop lifecycle and socket file creation/removal."""
    agent = build_agent()
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
    """Test that a dead socket left by a crashed run is replaced on start."""
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(test_socket_path)
    stale.close()  # file remains, nothing listens
    assert os.path.exists(test_socket_path)

    agent = build_agent()
    server = ControlServer(agent=agent, socket_path=test_socket_path)
    server.start()
    time.sleep(0.05)
    assert server.running
    assert os.path.exists(test_socket_path)
    server.stop()


def test_control_socket_is_owner_only(server_and_agent, test_socket_path):
    """The socket file is created with mode 0600."""
    assert stat.S_IMODE(os.stat(test_socket_path).st_mode) == 0o600


def test_control_server_refuses_non_socket_path(test_socket_path, capsys):
    """A regular file at the socket path is left alone and the server stays stopped."""
    with open(test_socket_path, "w") as f:
        f.write("important")

    server = ControlServer(agent=build_agent(), socket_path=test_socket_path)
    server.start()

    assert not server.running
    assert open(test_socket_path).read() == "important"
    assert "not a socket owned by this user" in capsys.readouterr().out
    server.stop()
    assert os.path.exists(test_socket_path)


def test_second_instance_does_not_remove_first_instances_socket(server_and_agent, test_socket_path, capsys):
    """A second server on a live path fails to start and leaves the first server working."""
    second = ControlServer(agent=build_agent(), socket_path=test_socket_path)
    second.start()
    assert not second.running
    assert "already listening" in capsys.readouterr().out

    second.stop()
    assert os.path.exists(test_socket_path)

    from scripts.alsageist_ctl import send_command
    ok, _ = send_command("status", socket_path=test_socket_path)
    assert ok is True


def test_default_socket_path_is_per_user_private(monkeypatch):
    """Without $ALSAGEIST_SOCKET the default lives in a per-user directory."""
    monkeypatch.delenv("ALSAGEIST_SOCKET", raising=False)
    assert default_socket_path() == f"/tmp/alsageist-{os.getuid()}/alsageist.sock"

    monkeypatch.setenv("ALSAGEIST_SOCKET", "/run/custom.sock")
    assert default_socket_path() == "/run/custom.sock"


def test_default_socket_dir_is_created_private(monkeypatch):
    """The default socket directory is created with mode 0700 and the server binds inside it."""
    import shutil
    import tempfile

    base = tempfile.mkdtemp(prefix="ag", dir="/tmp")  # short path: AF_UNIX limits path length
    private_dir = os.path.join(base, "priv")
    monkeypatch.setattr("core.control_server.default_socket_dir", lambda: private_dir)

    server = ControlServer(agent=build_agent(), socket_path=os.path.join(private_dir, "a.sock"))
    server.start()
    try:
        assert server.running
        assert stat.S_IMODE(os.stat(private_dir).st_mode) == 0o700
    finally:
        server.stop()
        shutil.rmtree(base, ignore_errors=True)


def test_secure_private_dir_rejects_open_permissions(tmp_path):
    """A pre-existing directory open to group or others is refused."""
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    os.chmod(open_dir, 0o755)
    with pytest.raises(ControlSocketError, match="0700"):
        _secure_private_dir(str(open_dir))


def test_secure_private_dir_rejects_symlink(tmp_path):
    """A symlink in place of the private directory is refused."""
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ControlSocketError, match="plain directory"):
        _secure_private_dir(str(link))


def test_peer_check_rejects_other_users(monkeypatch):
    """On Linux a connection from a different, non-root user is rejected."""
    monkeypatch.setattr("core.control_server.sys.platform", "linux")
    me = os.geteuid()
    other_uid = me + 1 if me != 0 else 1000

    def creds_for(uid):
        conn = MagicMock()
        conn.getsockopt.return_value = struct.pack("3i", 1234, uid, 100)
        return conn

    monkeypatch.setattr("core.control_server.socket.SO_PEERCRED", 17, raising=False)
    assert ControlServer._peer_allowed(creds_for(me)) is True
    assert ControlServer._peer_allowed(creds_for(0)) is True
    assert ControlServer._peer_allowed(creds_for(other_uid)) is False

    failing = MagicMock()
    failing.getsockopt.side_effect = OSError("no creds")
    assert ControlServer._peer_allowed(failing) is False


def test_oversized_request_is_rejected(server_and_agent, test_socket_path):
    """A request larger than the cap gets an error instead of being buffered without limit."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(3)
    client.connect(test_socket_path)
    try:
        # One byte over the cap: the server reads everything before it replies, so no reset races the reply
        client.sendall(b"x" * (MAX_REQUEST_BYTES + 1))
        reply = client.recv(4096)
    finally:
        client.close()
    assert b"Request too large" in reply


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


def test_dispatch_shutdown(server_and_agent, monkeypatch):
    """Test shutdown command initiates system poweroff."""
    server, agent = server_and_agent
    agent._on_button_long_hold = MagicMock()

    resp = server.dispatch_command("shutdown", "confirm")
    assert resp["status"] == "ok"
    assert "shutdown initiated" in resp["message"]


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


def test_dispatch_say_rejected_while_disabled(server_and_agent):
    """Test say is refused in sleep mode instead of playing audio."""
    server, agent = server_and_agent
    server.dispatch_command("disable")
    agent.audio.play_audio.reset_mock()

    resp = server.dispatch_command("say", "Hello there!")
    assert resp["status"] == "error"
    agent.audio.play_audio.assert_not_called()


def test_personality_switch_updates_all_components(server_and_agent):
    """Test a personality switch replaces profile, engine, voice, info service and logger name."""
    server, agent = server_and_agent
    old_engine, old_info = agent.dialogue_engine, agent.basic_info

    resp = server.dispatch_command("personality", "grimjaw")
    assert resp["status"] == "ok"
    assert agent.dialogue_engine is not old_engine
    assert agent.basic_info is not old_info
    assert agent.profile.personality == "grimjaw"
    assert agent.logger.persona_name == agent.profile.display_name
    agent._build_tts.assert_called_once_with(agent.profile)


def test_personality_switch_failure_leaves_agent_unchanged(server_and_agent):
    """Test a failed personality switch keeps the running persona."""
    server, agent = server_and_agent
    engine, profile = agent.dialogue_engine, agent.profile

    resp = server.dispatch_command("personality", "nonexistent_persona_xyz")
    assert resp["status"] == "error"
    assert agent.dialogue_engine is engine
    assert agent.profile is profile


def test_personality_switch_voice_failure_leaves_agent_unchanged(server_and_agent):
    """Test a missing voice model aborts the switch before any component is replaced."""
    server, agent = server_and_agent
    engine = agent.dialogue_engine
    agent._build_tts = MagicMock(side_effect=FileNotFoundError("voice missing"))

    resp = server.dispatch_command("personality", "grimjaw")
    assert resp["status"] == "error"
    assert agent.dialogue_engine is engine


def test_dispatch_mode_auto_and_status_flag(server_and_agent):
    """mode auto clears a forced mode, and status reports whether the mode is forced."""
    server, agent = server_and_agent

    resp = server.dispatch_command("mode", "offline")
    assert resp["mode_forced"] is True
    assert "forced" in resp["message"]
    assert server.dispatch_command("status")["mode_forced"] is True

    resp = server.dispatch_command("mode", "auto")
    assert resp["status"] == "ok"
    assert resp["mode_forced"] is False
    assert "automatic" in resp["message"]
    assert server.dispatch_command("status")["mode_forced"] is False


def test_ctl_uses_longer_timeouts_for_slow_commands():
    """say and personality wait longer than a status query; --timeout overrides."""
    from scripts import alsageist_ctl

    def used_timeout(cmd, **kwargs):
        with patch("scripts.alsageist_ctl.os.path.exists", return_value=True), \
             patch("scripts.alsageist_ctl.socket.socket") as sock_cls:
            sock = sock_cls.return_value
            sock.recv.return_value = b'{"status": "ok"}\n'
            alsageist_ctl.send_command(cmd, socket_path="/tmp/x.sock", **kwargs)
            return sock.settimeout.call_args.args[0]

    assert used_timeout("status") == 3.0
    assert used_timeout("say") == 60.0
    assert used_timeout("personality") == 120.0
    assert used_timeout("status", timeout=9.0) == 9.0


@pytest.mark.parametrize("value,current,expected", [
    ("75", 0.5, 0.75),
    ("+10", 0.5, 0.6),
    ("-10", 0.5, 0.4),
    ("+150", 0.5, 1.0),
    ("1", 0.5, 0.01),
    ("1.0", 0.5, 1.0),
    ("0.9", 0.5, 0.9),
    (0.9, 0.5, 0.9),
    (100, 0.5, 1.0),
    ("0", 0.5, 0.0),
])
def test_parse_volume_accepts(value, current, expected):
    assert parse_volume(value, current) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["+-5", "-+5", "nan", "inf", "abc", "-", "+", "", "--5"])
def test_parse_volume_rejects(value):
    with pytest.raises(ValueError):
        parse_volume(value, 0.5)


def test_is_socket_active(server_and_agent, test_socket_path):
    """is_socket_active returns True while the server listens, and False after stop."""
    server, _ = server_and_agent
    assert is_socket_active(test_socket_path) is True
    assert is_socket_active("/tmp/nonexistent_socket_path.sock") is False
    server.stop()
    assert is_socket_active(test_socket_path) is False
