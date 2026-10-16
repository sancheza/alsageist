"""Unit tests for scripts/setup_docker.sh."""

import os
import subprocess
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "setup_docker.sh"))


def test_setup_docker_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage: setup_docker.sh [OPTIONS]" in result.stdout
    assert "--profile <aiy|usb>" in result.stdout


def test_setup_docker_version():
    """Verify -v / --version prints script version and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1.1.0" in result.stdout


def test_setup_docker_invalid_profile():
    """Verify an invalid profile exits with code 1 and error message."""
    result = subprocess.run([SCRIPT_PATH, "--profile", "unsupported"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error: Invalid profile" in result.stderr


def test_setup_docker_dry_run():
    """Verify --dry-run mode executes cleanly without modifying system files."""
    result = subprocess.run(
        [SCRIPT_PATH, "--dry-run", "--non-interactive", "--user", "pi"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Mode: DRY RUN" in result.stdout
    assert "Profile: AIY | Target User: pi" in result.stdout
    assert "Dry-run complete. System was not modified." in result.stdout


def test_setup_docker_prewarm_dry_run():
    """Verify --prewarm flag triggers pre-warming command preview in dry run."""
    result = subprocess.run(
        [SCRIPT_PATH, "--dry-run", "--non-interactive", "--prewarm"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Pre-warming dialogue audio cache..." in result.stdout
    assert "Would run: python3" in result.stdout
    assert "prewarm_cache.py --all" in result.stdout


def test_setup_docker_usb_profile_dry_run():
    """Verify USB audio profile flags in dry-run mode."""
    result = subprocess.run(
        [SCRIPT_PATH, "--dry-run", "--non-interactive", "--profile", "usb"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Profile: USB" in result.stdout
    assert "Dry-run complete. System was not modified." in result.stdout
