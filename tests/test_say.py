"""Unit tests for scripts/say.py argument parsing and personality resolution."""

import os
import subprocess
import sys
import pytest


def test_say_help():
    """Verify say.py responds to --help with personality and options."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    res = subprocess.run(
        [sys.executable, "scripts/say.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "--personality" in res.stdout
    assert "--model-path" in res.stdout
    assert "--tempo" in res.stdout
    assert "--volume" in res.stdout


def test_say_empty_text():
    """Verify say.py rejects empty text."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo_root}
    res = subprocess.run(
        [sys.executable, "scripts/say.py", "   "],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "Error: Spoken text cannot be empty." in res.stderr
