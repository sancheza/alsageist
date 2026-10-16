"""Unit tests for scripts/download_model.py."""

import os
import subprocess
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "download_model.py"))


def test_download_model_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Download Piper TTS neural voice and Vosk STT models." in result.stdout
    assert "--vosk" in result.stdout
    assert "--all" in result.stdout


def test_download_model_version():
    """Verify -v / --version prints script version and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1.1.0" in result.stdout


def test_download_model_dry_run_voice():
    """Verify --dry-run prints voice download URLs without downloading."""
    result = subprocess.run([SCRIPT_PATH, "--voice", "alan", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "en_GB-alan-medium.onnx" in result.stdout
    assert "en_GB-alan-medium.onnx.json" in result.stdout
    assert "vosk" not in result.stdout


def test_download_model_dry_run_vosk():
    """Verify --vosk --dry-run prints Vosk download actions."""
    result = subprocess.run([SCRIPT_PATH, "--vosk", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "vosk-model-small-en-us-0.15" in result.stdout
    assert "piper" not in result.stdout


def test_download_model_dry_run_all():
    """Verify --all --dry-run downloads both voice model and Vosk model."""
    result = subprocess.run([SCRIPT_PATH, "--all", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "en_GB-alan-medium.onnx" in result.stdout
    assert "vosk-model-small-en-us-0.15" in result.stdout
