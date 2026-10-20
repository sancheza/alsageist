"""Voice command parser for system lifecycle and operational mode management.

Provides high-reliability pattern matching for spoken system commands (shutdown,
mode switching between online and offline, and mode status queries). Designed
to operate with 100% reliability with offline speech recognition engines (Vosk)
by accommodating lowercased unpunctuated text, compounding variations ('off line'
vs 'offline'), and optional conversational prefixes.
"""

import re
from typing import Optional

from core.intent_builder import normalize_query

COMMAND_SHUTDOWN = "system_shutdown"
COMMAND_MODE_OFFLINE = "system_mode_offline"
COMMAND_MODE_ONLINE = "system_mode_online"
COMMAND_MODE_AUTO = "system_mode_auto"
COMMAND_MODE_QUERY = "system_mode_query"
COMMAND_VOICE_QUERY = "system_voice_query"
COMMAND_ENGINE_QUERY = "system_engine_query"

# Common polite or wake-word prefixes to strip before command evaluation
_PREFIX_PATTERN = re.compile(
    r"^(?:(?:hey|okay|ok|hello|hi|please|can\s+you|could\s+you|would\s+you|tell\s+me)\s+)*"
    r"(?:(?:grimjaw|alsageist|assistant|creature|skeleton)\s+)*"
    r"(?:(?:please|can\s+you|could\s+you|would\s+you|tell\s+me)\s+)*",
    re.IGNORECASE,
)

# Shutdown commands: "shut down", "power off", "turn off the system", etc.
_SHUTDOWN_PATTERN = re.compile(
    r"^(?:shut\s*down|shutdown|power\s*off|power\s*down)"
    r"(?:\s+(?:the\s+)?(?:system|host|pi|display|alsageist|device|yourself|unit))?$"
    r"|^turn\s*off(?:\s+(?:the\s+)?(?:system|host|pi|display|alsageist|device|yourself|unit))$",
    re.IGNORECASE,
)

# Mode query commands: "what mode are you in", "are you online or offline", "current mode", etc.
_MODE_QUERY_PATTERN = re.compile(
    r"^(?:(?:what(?:'?s|\s+is)?|which)\s+)?(?:your\s+|the\s+|current\s+)?mode(?:\s+(?:are\s+you\s+in|is\s+this|is\s+it))?$"
    r"|^what\s+mode(?:\s+(?:are\s+you\s+in|is\s+this|is\s+it))?$"
    r"|^are\s+you\s+(?:online\s+or\s+offline|offline\s+or\s+online|online|offline|on\s*line|off\s*line)$"
    r"|^(?:current|check|status|system|network)\s+mode$"
    r"|^mode\s+(?:status|check)$",
    re.IGNORECASE,
)

# Voice model query commands: "what is your voice model", "what voice are you using", "voice model", etc.
_VOICE_QUERY_PATTERN = re.compile(
    r"^(?:(?:what(?:'?s|\s+is)?|which)\s+)?(?:your\s+|the\s+|current\s+)?voice(?:\s+model)?(?:\s+(?:are\s+you\s+using|do\s+you\s+use|is\s+this|is\s+it))?$"
    r"|^what\s+voice(?:\s+model)?(?:\s+(?:are\s+you\s+using|do\s+you\s+use|is\s+this|is\s+it))?$"
    r"|^(?:which|what)\s+voice\s+model$"
    r"|^(?:current|check|status|system)\s+voice(?:\s+model)?$"
    r"|^voice\s+model(?:\s+(?:status|check))?$",
    re.IGNORECASE,
)

# Dialogue engine query commands: "what is your dialogue engine", "what engine are you using", etc.
_ENGINE_QUERY_PATTERN = re.compile(
    r"^(?:(?:what(?:'?s|\s+is)?|which)\s+)?(?:your\s+|the\s+|current\s+)?(?:dialog(?:ue)?\s+)?engine(?:\s+(?:are\s+you\s+using|are\s+you\s+running|do\s+you\s+use|is\s+this|is\s+it|is\s+running))?$"
    r"|^what\s+(?:dialog(?:ue)?\s+)?engine(?:\s+(?:are\s+you\s+using|are\s+you\s+running|do\s+you\s+use|is\s+this|is\s+it|is\s+running))?$"
    r"|^are\s+you\s+using\s+(?:intent\s+or\s+llm|llm\s+or\s+intent)$"
    r"|^(?:current|check|status|system)\s+(?:dialog(?:ue)?\s+)?engine$"
    r"|^(?:dialog(?:ue)?\s+engine)(?:\s+(?:status|check))?$"
    r"|^engine\s+(?:status|check)$",
    re.IGNORECASE,
)

# Mode switch to offline: "go offline", "switch to offline mode", "disconnect", etc.
_MODE_OFFLINE_PATTERN = re.compile(
    r"^(?:switch\s+to\s+|change\s+to\s+|set\s+(?:mode\s+)?|go\s+|enter\s+|turn\s+)"
    r"(?:off\s*line|offline)(?:\s+mode)?$"
    r"|^offline\s+mode$|^off\s*line\s+mode$"
    r"|^(?:disconnect|disconnect\s+from\s+internet)$",
    re.IGNORECASE,
)

# Mode switch to online: "go online", "switch to online mode", "connect", "reconnect", etc.
_MODE_ONLINE_PATTERN = re.compile(
    r"^(?:switch\s+to\s+|change\s+to\s+|set\s+(?:mode\s+)?|go\s+|enter\s+|turn\s+)"
    r"(?:on\s*line|online)(?:\s+mode)?$"
    r"|^online\s+mode$|^on\s*line\s+mode$"
    r"|^(?:connect|reconnect)(?:\s+to\s+internet)?$",
    re.IGNORECASE,
)

# Mode switch to automatic: "auto mode", "automatic mode", "set mode auto", etc.
_MODE_AUTO_PATTERN = re.compile(
    r"^(?:switch\s+to\s+|change\s+to\s+|set\s+(?:mode\s+)?|go\s+|enter\s+)?"
    r"(?:auto|automatic)(?:\s+mode)?$",
    re.IGNORECASE,
)


def match_voice_command(user_text: str) -> Optional[str]:
    """Classify transcribed speech as a system voice command.

    Handles offline recognition nuances such as unpunctuated text, phonetic
    word splits ('off line'), and conversational phrasing. Evaluates query
    patterns before mode-setting patterns to avoid ambiguous collisions.

    Args:
        user_text: Raw or normalized transcribed speech string.

    Returns:
        One of the COMMAND_* string constants if a command is recognized, or None.
    """
    if not user_text or not user_text.strip():
        return None

    clean = normalize_query(user_text)
    stripped = _PREFIX_PATTERN.sub("", clean).strip()
    if not stripped:
        return None

    # 1. Shutdown command
    if _SHUTDOWN_PATTERN.match(stripped):
        return COMMAND_SHUTDOWN

    # 2. Mode query command (evaluated before mode-setting to prevent 'are you offline' matching 'offline')
    if _MODE_QUERY_PATTERN.match(stripped):
        return COMMAND_MODE_QUERY

    # 3. Voice model query command
    if _VOICE_QUERY_PATTERN.match(stripped):
        return COMMAND_VOICE_QUERY

    # 4. Dialogue engine query command
    if _ENGINE_QUERY_PATTERN.match(stripped):
        return COMMAND_ENGINE_QUERY

    # 5. Switch to offline mode
    if _MODE_OFFLINE_PATTERN.match(stripped):
        return COMMAND_MODE_OFFLINE

    # 4. Switch to online mode
    if _MODE_ONLINE_PATTERN.match(stripped):
        return COMMAND_MODE_ONLINE

    # 5. Switch to automatic mode
    if _MODE_AUTO_PATTERN.match(stripped):
        return COMMAND_MODE_AUTO

    return None
