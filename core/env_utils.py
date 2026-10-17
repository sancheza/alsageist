"""core/env_utils.py - Environment variable parsing shared by every module.

A malformed value falls back to the default instead of raising, so a typo in
`.env` cannot stop the daemon from starting.
"""

import os
from typing import Optional

TRUE_VALUES = ("true", "1", "yes", "on")
FALSE_VALUES = ("false", "0", "no", "off")


def env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float environment variable.

    Args:
        name: Variable name.
        default: Value returned when the variable is unset, blank or not a number.

    Returns:
        The parsed float, or default.
    """
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable.

    Args:
        name: Variable name.
        default: Value returned when the variable is unset, blank or not an integer.

    Returns:
        The parsed integer, or default.
    """
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Accepts true/1/yes/on and false/0/no/off, case-insensitive.

    Args:
        name: Variable name.
        default: Value returned when the variable is unset, blank or unrecognized.

    Returns:
        The parsed boolean, or default.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    text = val.strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def groq_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return the Groq API key: the explicit value, else $GROQ_API_KEY."""
    return explicit or os.environ.get("GROQ_API_KEY") or None


def gemini_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return the Gemini API key: the explicit value, else $GEMINI_API_KEY, else $GOOGLE_API_KEY."""
    return explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None
