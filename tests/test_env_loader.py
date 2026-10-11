"""Unit tests for load_env_file in main.py."""

import os
import tempfile
import pytest
from main import load_env_file


def test_load_env_file_basic():
    """Verify key-value pairs are loaded into os.environ."""
    test_key = "ALSAGEIST_TEST_KEY_BASIC"
    if test_key in os.environ:
        del os.environ[test_key]

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(f"# This is a comment\n{test_key}=test_secret_value\n")
        tmp_name = f.name

    try:
        load_env_file(tmp_name)
        assert os.environ.get(test_key) == "test_secret_value"
    finally:
        os.unlink(tmp_name)
        if test_key in os.environ:
            del os.environ[test_key]


def test_load_env_file_strips_quotes_and_ignores_comments():
    """Verify quotes are stripped and comments/whitespace are ignored."""
    key1 = "ALSAGEIST_TEST_QUOTE_SINGLE"
    key2 = "ALSAGEIST_TEST_QUOTE_DOUBLE"
    for k in (key1, key2):
        if k in os.environ:
            del os.environ[k]

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(f"  \n# Comment line\n{key1}='single_quoted'\n{key2}=\"double_quoted\"\ninvalid line without equals\n")
        tmp_name = f.name

    try:
        load_env_file(tmp_name)
        assert os.environ.get(key1) == "single_quoted"
        assert os.environ.get(key2) == "double_quoted"
    finally:
        os.unlink(tmp_name)
        for k in (key1, key2):
            if k in os.environ:
                del os.environ[k]


def test_load_env_file_does_not_overwrite_existing():
    """Verify existing environment variables take precedence."""
    test_key = "ALSAGEIST_EXISTING_KEY"
    os.environ[test_key] = "pre_existing_value"

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(f"{test_key}=new_value\n")
        tmp_name = f.name

    try:
        load_env_file(tmp_name)
        assert os.environ.get(test_key) == "pre_existing_value"
    finally:
        os.unlink(tmp_name)
        del os.environ[test_key]
