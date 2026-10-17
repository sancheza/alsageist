"""Unit tests for core/system_utils.py."""

from unittest.mock import MagicMock, patch
import os
import sys
import pytest

from core.system_utils import (
    check_internet_connectivity,
    is_running_in_docker,
    load_env_file,
    resolve_piper_model_path,
    trigger_host_shutdown,
)


def test_is_running_in_docker_env(monkeypatch):
    """Verify docker environment detection when /.dockerenv exists."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.side_effect = lambda p: p == "/.dockerenv"
        assert is_running_in_docker() is True

    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert is_running_in_docker() is False


def test_trigger_host_shutdown_dry_run():
    """Verify dry-run mode returns True without calling host shutdown."""
    assert trigger_host_shutdown(dry_run=True) is True


def test_trigger_host_shutdown_darwin(monkeypatch):
    """Verify shutdown on macOS simulates safely and returns True."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert trigger_host_shutdown(dry_run=False) is True


def test_trigger_host_shutdown_linux_systemctl(monkeypatch):
    """Verify Linux execution attempts systemctl poweroff."""
    monkeypatch.setattr(sys, "platform", "linux")

    with patch("shutil.which") as mock_which, \
         patch("os.path.isfile") as mock_isfile, \
         patch("subprocess.run") as mock_run:
        mock_which.side_effect = lambda cmd: "/bin/" + cmd if cmd == "systemctl" else None
        mock_isfile.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        assert trigger_host_shutdown(dry_run=False) is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["/bin/systemctl", "poweroff"]


def test_trigger_host_shutdown_sysrq_masked_returns_false(monkeypatch):
    """When sysrq mask does not permit poweroff (e.g. mask is 0), SysRq is not run and returns False."""
    from unittest.mock import mock_open

    monkeypatch.setattr(sys, "platform", "linux")
    with patch("shutil.which", return_value=None), \
         patch("os.path.isfile", return_value=False), \
         patch("os.path.exists", side_effect=lambda p: p in ("/proc/sysrq-trigger", "/proc/sys/kernel/sysrq")), \
         patch("builtins.open", mock_open(read_data="0")):
        assert trigger_host_shutdown(dry_run=False) is False


def test_load_env_file(tmp_path, monkeypatch):
    """Verify load_env_file parses key-value pairs into os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_CUSTOM_KEY=hello_world\n# Comment\nINVALID_LINE\n")

    monkeypatch.delenv("TEST_CUSTOM_KEY", raising=False)
    load_env_file(str(env_file))
    assert os.environ.get("TEST_CUSTOM_KEY") == "hello_world"


def test_check_internet_connectivity():
    """Verify check_internet_connectivity socket handling."""
    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock_cls.return_value = mock_sock
        assert check_internet_connectivity(timeout=1.0) is True

    with patch("socket.socket") as mock_sock_cls:
        mock_sock_cls.side_effect = OSError("Network unreachable")
        assert check_internet_connectivity(timeout=1.0) is False


def test_resolve_piper_model_path(tmp_path):
    """Verify resolve_piper_model_path handles explicit paths, voice names, and fallbacks."""
    # 1. Fallback default
    path = resolve_piper_model_path(repo_root=str(tmp_path))
    assert path == str(tmp_path / "models" / "piper" / "en_GB-alan-medium.onnx")

    # 2. Short voice name
    models_dir = tmp_path / "models" / "piper"
    models_dir.mkdir(parents=True)
    custom_model = models_dir / "custom-voice.onnx"
    custom_model.write_text("model_bytes")

    path = resolve_piper_model_path(voice_name="custom-voice", repo_root=str(tmp_path))
    assert path == str(custom_model)

    # 3. Explicit path
    path = resolve_piper_model_path(model_path=str(custom_model), repo_root=str(tmp_path))
    assert path == str(custom_model)



def test_resolve_piper_model_path_bare_voice_id_as_model_path(tmp_path):
    """Verify a bare voice ID passed as model_path resolves under models/piper."""
    models_dir = tmp_path / "models" / "piper"
    models_dir.mkdir(parents=True)
    voice = models_dir / "custom-voice.onnx"
    voice.write_text("model_bytes")

    path = resolve_piper_model_path(model_path="custom-voice", repo_root=str(tmp_path))
    assert path == str(voice)


def test_check_internet_connectivity_closes_sockets():
    """Every probe socket is closed, including after a failed connect."""
    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("refused")
        mock_sock_cls.return_value = mock_sock
        assert check_internet_connectivity(timeout=1.0) is False
        assert mock_sock.close.call_count == 4


def test_check_internet_connectivity_prefers_https_port_and_falls_back():
    """A network that blocks port 53 but allows 443 is still reported online."""
    attempts = []

    def connect(address):
        attempts.append(address)
        if address[1] == 53:
            raise OSError("port 53 blocked")

    with patch("socket.socket") as mock_sock_cls:
        mock_sock_cls.return_value.connect.side_effect = connect
        assert check_internet_connectivity(timeout=1.0) is True
    assert attempts == [("1.1.1.1", 443)]

    attempts.clear()

    def only_dns(address):
        attempts.append(address)
        if address[1] == 443:
            raise OSError("443 blocked")

    with patch("socket.socket") as mock_sock_cls:
        mock_sock_cls.return_value.connect.side_effect = only_dns
        assert check_internet_connectivity(timeout=1.0) is True
    assert attempts[-1] == ("1.1.1.1", 53)


def test_resolve_piper_model_path_missing_voice_is_not_replaced_by_default(tmp_path):
    """A named voice that is not installed resolves to its own path, never to the default voice."""
    path = resolve_piper_model_path(voice_name="en_US-lessac-medium", repo_root=str(tmp_path))
    assert path == str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx")

    path = resolve_piper_model_path(model_path="typo-voice", repo_root=str(tmp_path))
    assert path == str(tmp_path / "models" / "piper" / "typo-voice.onnx")
