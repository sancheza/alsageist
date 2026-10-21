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
COMMAND_FALLBACK_QUERY = "system_fallback_query"
COMMAND_FALLBACK_ENABLE = "system_fallback_enable"
COMMAND_FALLBACK_DISABLE = "system_fallback_disable"
COMMAND_ACCESS_DENIED = "system_access_denied"

PRIVILEGED_COMMANDS = {
    COMMAND_SHUTDOWN,
    COMMAND_MODE_OFFLINE,
    COMMAND_MODE_ONLINE,
    COMMAND_MODE_AUTO,
}

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

# Fallback status query commands: "what is your fallback status", "is fallback enabled", etc.
_FALLBACK_QUERY_PATTERN = re.compile(
    r"^(?:(?:what(?:'?s|\s+is)?)\s+)?(?:your\s+|the\s+|current\s+)?(?:l\s*l\s*m\s+)?fallback(?:\s+status)?(?:\s+(?:is\s+this|is\s+it))?$"
    r"|^(?:what(?:'?s|\s+is)?\s+)?(?:l\s*l\s*m\s+)?fallback\s+status$"
    r"|^is\s+(?:l\s*l\s*m\s+)?fallback(?:\s+(?:on|enabled|active))?$"
    r"|^(?:check|query)\s+(?:l\s*l\s*m\s+)?fallback(?:\s+status)?$"
    r"|^(?:l\s*l\s*m\s+)?fallback\s+(?:status|check)$"
    r"|^are\s+you\s+using\s+(?:l\s*l\s*m\s+)?fallback$",
    re.IGNORECASE,
)

# Fallback enable commands: "enable fallback", "turn on llm fallback", "set fallback to enabled", "fallback on", etc.
_FALLBACK_ENABLE_PATTERN = re.compile(
    r"^(?:enable|turn\s+on|activate|start|switch\s+on)\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback$"
    r"|^turn\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback\s+on$"
    r"|^switch\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback\s+on$"
    r"|^set\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback(?:\s+to)?\s+(?:on|enabled|enable|active|true)$"
    r"|^(?:l\s*l\s*m\s+)?fallback\s+(?:on|enable|enabled)$",
    re.IGNORECASE,
)

# Fallback disable commands: "disable fallback", "turn off llm fallback", "set fallback to disabled", "fallback off", etc.
_FALLBACK_DISABLE_PATTERN = re.compile(
    r"^(?:disable|turn\s+off|deactivate|stop|switch\s+off)\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback$"
    r"|^turn\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback\s+off$"
    r"|^switch\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback\s+off$"
    r"|^set\s+(?:the\s+)?(?:l\s*l\s*m\s+)?fallback(?:\s+to)?\s+(?:off|disabled|disable|inactive|false)$"
    r"|^(?:l\s*l\s*m\s+)?fallback\s+(?:off|disable|disabled)$",
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


def match_voice_command(user_text: str, password: Optional[str] = None) -> Optional[str]:
    """Classify transcribed speech as a system voice command.

    Handles offline recognition nuances such as unpunctuated text, phonetic
    word splits ('off line'), and conversational phrasing. Evaluates query
    patterns before mode-setting patterns to avoid ambiguous collisions.

    If a verbal password is provided and not disabled, checks that privileged
    commands (e.g. shutdown, mode switching, fallback toggles) include the password.

    Args:
        user_text: Raw or normalized transcribed speech string.
        password: Optional verbal password required for privileged operations.

    Returns:
        One of the COMMAND_* string constants if a command is recognized, or None.
    """
    if not user_text or not user_text.strip():
        return None

    clean = normalize_query(user_text)

    has_password = False
    clean_password = normalize_query(password) if password else ""
    if clean_password and clean_password not in ("none", "off", "false"):
        pwd_pattern = re.compile(r"\b" + re.escape(clean_password) + r"\b", re.IGNORECASE)
        if pwd_pattern.search(clean):
            has_password = True
            clean = pwd_pattern.sub(" ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

    stripped = _PREFIX_PATTERN.sub("", clean).strip()
    if not stripped:
        return None

    matched_cmd = None
    # 1. Shutdown command
    if _SHUTDOWN_PATTERN.match(stripped):
        matched_cmd = COMMAND_SHUTDOWN
    # 2. Mode query command (evaluated before mode-setting to prevent 'are you offline' matching 'offline')
    elif _MODE_QUERY_PATTERN.match(stripped):
        matched_cmd = COMMAND_MODE_QUERY
    # 3. Voice model query command
    elif _VOICE_QUERY_PATTERN.match(stripped):
        matched_cmd = COMMAND_VOICE_QUERY
    # 4. Dialogue engine query command
    elif _ENGINE_QUERY_PATTERN.match(stripped):
        matched_cmd = COMMAND_ENGINE_QUERY
    # 5. Fallback query command
    elif _FALLBACK_QUERY_PATTERN.match(stripped):
        matched_cmd = COMMAND_FALLBACK_QUERY
    # 6. Fallback enable command
    elif _FALLBACK_ENABLE_PATTERN.match(stripped):
        matched_cmd = COMMAND_FALLBACK_ENABLE
    # 7. Fallback disable command
    elif _FALLBACK_DISABLE_PATTERN.match(stripped):
        matched_cmd = COMMAND_FALLBACK_DISABLE
    # 8. Switch to offline mode
    elif _MODE_OFFLINE_PATTERN.match(stripped):
        matched_cmd = COMMAND_MODE_OFFLINE
    # 9. Switch to online mode
    elif _MODE_ONLINE_PATTERN.match(stripped):
        matched_cmd = COMMAND_MODE_ONLINE
    # 10. Switch to automatic mode
    elif _MODE_AUTO_PATTERN.match(stripped):
        matched_cmd = COMMAND_MODE_AUTO

    if matched_cmd is None:
        return None

    # Check privilege authorization if password is required
    if clean_password and clean_password not in ("none", "off", "false"):
        if matched_cmd in PRIVILEGED_COMMANDS and not has_password:
            return COMMAND_ACCESS_DENIED

    return matched_cmd
