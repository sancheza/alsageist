"""Unit tests for scripts/setup_systemd_service.sh."""

import os
import subprocess
import tempfile
import pytest

SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "setup_systemd_service.sh")
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_PATH = os.path.join(REPO_ROOT, "alsageist.service")


def test_setup_systemd_service_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage: setup_systemd_service.sh [OPTIONS]" in result.stdout
    assert "-y, --yes" in result.stdout


def test_setup_systemd_service_version():
    """Verify -v / --version prints version and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_setup_systemd_service_dry_run():
    """Verify --dry-run previews rendered unit without writing and displays Docker warning."""
    result = subprocess.run(
        [
            SCRIPT_PATH,
            "--dry-run",
            "--user", "ghost",
            "--dir", "/opt/ghost/alsageist",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Target User:        ghost" in result.stdout
    assert "Working Directory:  /opt/ghost/alsageist" in result.stdout
    assert "WARNING: BARE-METAL SERVICE INSTALLATION" in result.stdout
    assert "This script is NOT required for Docker installations" in result.stdout
    assert "User=ghost" in result.stdout
    assert "WorkingDirectory=/opt/ghost/alsageist" in result.stdout
    assert "Dry-run complete." in result.stdout


def test_setup_systemd_service_abort_on_declined_confirmation():
    """Verify execution aborts cleanly when user declines confirmation prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "alsageist.service")
        result = subprocess.run(
            [
                SCRIPT_PATH,
                "--user", "spook",
                "--dir", "/home/spook/alsageist",
                "--template", TEMPLATE_PATH,
                "--output", output_path,
            ],
            input="n\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "WARNING: BARE-METAL SERVICE INSTALLATION" in result.stdout
        assert "This script is NOT required for Docker installations" in result.stdout
        assert "Installation aborted by user." in result.stdout
        assert not os.path.exists(output_path)


def test_setup_systemd_service_confirmed_via_stdin():
    """Verify service installs when user responds 'y' to confirmation prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "alsageist.service")
        result = subprocess.run(
            [
                SCRIPT_PATH,
                "--user", "spook",
                "--dir", "/home/spook/alsageist",
                "--template", TEMPLATE_PATH,
                "--output", output_path,
            ],
            input="y\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "WARNING: BARE-METAL SERVICE INSTALLATION" in result.stdout
        assert "Successfully installed service unit" in result.stdout
        assert os.path.exists(output_path)


def test_setup_systemd_service_render_to_custom_output_with_yes_flag():
    """Verify writing rendered unit to a custom output path with -y flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "alsageist.service")
        result = subprocess.run(
            [
                SCRIPT_PATH,
                "--yes",
                "--user", "spook",
                "--dir", "/home/spook/alsageist",
                "--template", TEMPLATE_PATH,
                "--output", output_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "User=spook" in content
        assert "WorkingDirectory=/home/spook/alsageist" in content
        assert "ExecStart=/home/spook/alsageist/venv/bin/python3 /home/spook/alsageist/main.py" in content
        assert "{{ALSAGEIST_USER}}" not in content
        assert "{{ALSAGEIST_DIR}}" not in content

