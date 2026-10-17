"""Checks that constraints.txt matches pyproject.toml and is used by the Dockerfile."""

import os
import re

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

tomllib = pytest.importorskip("tomllib")  # Python 3.11 or newer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(REPO_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def _pins():
    pins = {}
    for line in _read("constraints.txt").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[canonicalize_name(name)] = Version(version)
    return pins


def _declared():
    project = tomllib.loads(_read("pyproject.toml"))["project"]
    requirements = list(project["dependencies"]) + list(project["optional-dependencies"]["rpi"])
    return [Requirement(text) for text in requirements]


def test_every_declared_dependency_is_pinned_within_its_range():
    """Each runtime and rpi dependency has an exact pin that satisfies the pyproject range."""
    pins = _pins()
    for requirement in _declared():
        name = canonicalize_name(requirement.name)
        assert name in pins, f"{requirement.name} is not pinned in constraints.txt"
        assert requirement.specifier.contains(pins[name]), f"{name}=={pins[name]} violates {requirement}"


def test_constraints_use_exact_versions_and_no_build_tools():
    """Every line is name==version, and build tools are left to the installer."""
    for line in _read("constraints.txt").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-+]+", line), line
    assert not {"pip", "setuptools", "wheel"} & set(_pins())


def test_dev_tools_are_not_pinned_as_runtime_dependencies():
    """pytest belongs to the dev extra, so it must not appear in the runtime pins."""
    assert "pytest" not in _pins()


def test_dockerfile_installs_with_the_constraints():
    """The Dockerfile copies constraints.txt and passes it to pip, and .dockerignore keeps it."""
    dockerfile = _read("Dockerfile")
    assert "constraints.txt" in dockerfile.split("RUN chmod", 1)[0]
    assert re.search(r'pip install -c constraints\.txt -e "\.\[rpi\]"', dockerfile)
    ignored = [l.strip() for l in _read(".dockerignore").splitlines()]
    assert "constraints.txt" not in ignored
