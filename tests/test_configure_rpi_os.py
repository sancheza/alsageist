"""Unit tests for scripts/configure_rpi_os.sh."""

import os
import subprocess
import tempfile
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "configure_rpi_os.sh"))


def test_configure_rpi_os_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage: configure_rpi_os.sh [OPTIONS]" in result.stdout


def test_configure_rpi_os_version():
    """Verify -v / --version prints version and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_configure_rpi_os_dry_run():
    """Verify --dry-run exits with 0 and makes no modifications."""
    result = subprocess.run([SCRIPT_PATH, "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Mode: DRY RUN" in result.stdout
    assert "Dry-run complete." in result.stdout


def test_configure_rpi_os_invalid_profile():
    """Verify invalid profile exits with code 1."""
    result = subprocess.run([SCRIPT_PATH, "--profile", "invalid"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error: Invalid profile" in result.stderr


def test_configure_rpi_os_execution_custom_files():
    """Verify actual file creation when passing custom config paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.txt")
        asound_path = os.path.join(tmpdir, "asound.conf")

        # Create initial config.txt with onboard audio enabled
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("dtparam=audio=on\n")

        result = subprocess.run(
            [
                SCRIPT_PATH,
                "--profile", "aiy",
                "--config-file", config_path,
                "--asound-file", asound_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify config.txt updates
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "dtparam=audio=off" in content
        assert "dtoverlay=googlevoicehat-soundcard" in content
        assert "dtparam=i2s=on" in content

        # Verify asound.conf creation
        assert os.path.exists(asound_path)
        with open(asound_path, "r", encoding="utf-8") as f:
            asound_content = f.read()
        assert "sndrpigooglevoi" in asound_content


def test_configure_rpi_os_user_override():
    """Verify --user sets custom target user."""
    result = subprocess.run([SCRIPT_PATH, "--dry-run", "--user", "customuser"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Target user: customuser" in result.stdout
    assert "User Group Configuration for user 'customuser'" in result.stdout

