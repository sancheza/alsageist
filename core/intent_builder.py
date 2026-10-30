"""Reusable intent construction, parsing, and personality management tooling.

Provides models and utilities for parsing natural language prompts into high-precision
regular expression intents and responses, and managing external JSON/YAML personality profiles.
Designed for use across multiple voice assist projects.
"""

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# Spoken-dialogue rules shared by every LLM system prompt
PROMPT_RULES = (
    "Rules you must strictly obey:\n"
    "1. Keep every response under 2 sentences maximum.\n"
    "2. Be witty, dry, and stay in character as a theatrical Halloween porch guardian.\n"
    "3. Never output emojis, markdown formatting, asterisks, stage directions, or sound effect tags "
    "(like *laughs* or [cackles]).\n"
    "4. Return plain spoken dialogue only.\n"
    "5. Everything you say must suit all ages: absolutely no profanity, vulgarity, slurs, or adult topics, even if asked to repeat them.\n"
    "6. Never make demeaning, cruel, or personally insulting remarks about a visitor's appearance, costume, or intelligence. "
    "Keep sarcasm playful, self-deprecating, and focused on skeleton anatomy, candy guarding, or Halloween tropes—never personal attacks."
)

# System announcements used when a profile does not define the key. The runtime and the
# cache pre-warmer both read these, so the pre-warmed audio always matches what is spoken.
DEFAULT_ANNOUNCEMENTS = {
    "online_mode": "NOW IN ONLINE MODE",
    "offline_mode": "NOW IN OFFLINE MODE",
    "powering_off": "Powering off.",
    "volume_template": "Volume {percent} percent.",
    "status_online": "I am in online mode.",
    "status_offline": "I am in offline mode.",
    "engine_intent": "The dialogue engine is intent.",
    "engine_llm": "The dialogue engine is neural L L M.",
    "voice_model_template": "My voice model is {model}.",
    "fallback_enabled": "L L M fallback is enabled.",
    "fallback_disabled": "L L M fallback is disabled.",
    "access_denied": "Access denied: verbal password required.",
    "muted": "Microphone muted.",
    "wifi_not_found": "WiFi network not found.",
    "wifi_connecting_template": "Attempting to connect to WiFi network {ssid}.",
    "wifi_connect_success_template": "Attempting to connect to WiFi network {ssid} succeeded.",
    "wifi_connect_failed_template": "Attempting to connect to WiFi network {ssid} failed.",
}
DEFAULT_GREETING_TEMPLATE = "I am {name}. System online and ready."

# Typographic apostrophes that speech recognizers emit in place of the ASCII one
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u2032": "'", "\u00b4": "'", "`": "'"})


def normalize_query(text: str) -> str:
    """Reduce visitor text to the form intent patterns are written against.

    Lowercases, maps typographic apostrophes to the ASCII one, replaces other punctuation
    with spaces and collapses whitespace.

    Args:
        text: Transcribed or typed phrase.

    Returns:
        Normalized text.
    """
    cleaned = re.sub(r"[^\w\s']", " ", text.translate(_APOSTROPHES).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class CustomIntent:
    """Represents a conversational intent with match patterns and responses."""

    name: str
    patterns: List[str]
    responses: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate intent fields and compile regex patterns."""
        if not self.name or not self.name.strip():
            raise ValueError("Intent name cannot be empty.")
        # Ensure name is snake_case
        self.name = re.sub(r"[^a-zA-Z0-9_]+", "_", self.name.strip().lower()).strip("_")

        validated_patterns: List[str] = []
        for pat in self.patterns:
            if pat and pat.strip():
                try:
                    re.compile(pat, re.IGNORECASE)
                    validated_patterns.append(pat.strip())
                except re.error as err:
                    raise ValueError(f"Invalid regular expression pattern '{pat}': {err}") from err
        self.patterns = validated_patterns

        self.responses = [r.strip() for r in self.responses if r and r.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize intent to dictionary."""
        data: Dict[str, Any] = {
            "name": self.name,
            "patterns": self.patterns,
            "responses": self.responses,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomIntent":
        """Construct intent from dictionary."""
        return cls(
            name=data.get("name", ""),
            patterns=list(data.get("patterns", [])),
            responses=list(data.get("responses", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PersonalityProfile:
    """Represents a complete character personality with voice and intent rules."""

    personality: str
    display_name: str
    description: str
    voice: str = "en_GB-alan-medium"
    costume_prompts: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)
    escalation_lines: List[str] = field(default_factory=list)
    low_power_lines: List[str] = field(default_factory=list)
    intents: List[CustomIntent] = field(default_factory=list)
    startup_greeting: Optional[str] = None
    priority_intents: List[str] = field(default_factory=list)
    system_announcements: Dict[str, str] = field(default_factory=dict)
    voice_settings: Dict[str, float] = field(default_factory=dict)
    time_responses: Dict[str, str] = field(default_factory=dict)
    weather_responses: Dict[str, str] = field(default_factory=dict)
    docstring: Optional[Any] = None

    @property
    def doc(self) -> Optional[str]:
        """Return the documentation string for this personality profile."""
        if isinstance(self.docstring, list):
            return "\n".join(self.docstring)
        return self.docstring

    def build_system_prompt(self) -> str:
        """Build the LLM system prompt that keeps replies in this persona's character.

        Returns:
            System prompt string naming the persona and the spoken-dialogue rules.
        """
        return f"You are {self.display_name}. {self.description}\n{PROMPT_RULES}"

    def announcement(self, key: str, default: Optional[str] = None) -> str:
        """Return a spoken system announcement, falling back to a default.

        Args:
            key: Announcement key (for example 'online_mode' or 'powering_off').
            default: Text used when the profile does not define the key.

        Returns:
            Announcement text.
        """
        return self.system_announcements.get(key) or default or DEFAULT_ANNOUNCEMENTS.get(key, "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize personality profile to JSON-compatible dictionary."""
        data: Dict[str, Any] = {}
        if self.docstring:
            data["_doc"] = self.docstring
        data.update({
            "personality": self.personality,
            "display_name": self.display_name,
            "description": self.description,
            "voice": self.voice,
            "costume_prompts": self.costume_prompts,
            "fallbacks": self.fallbacks,
            "escalation_lines": self.escalation_lines,
            "intents": [intent.to_dict() for intent in self.intents],
        })
        if self.startup_greeting:
            data["startup_greeting"] = self.startup_greeting
        if self.priority_intents:
            data["priority_intents"] = self.priority_intents
        if self.system_announcements:
            data["system_announcements"] = self.system_announcements
        if self.low_power_lines:
            data["low_power_lines"] = self.low_power_lines
        if self.voice_settings:
            data["voice_settings"] = self.voice_settings
        if self.time_responses:
            data["time_responses"] = self.time_responses
        if self.weather_responses:
            data["weather_responses"] = self.weather_responses
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        """Construct personality profile from dictionary."""
        intents_data = data.get("intents", [])
        intents = [
            CustomIntent.from_dict(i) if isinstance(i, dict) else i
            for i in intents_data
        ]
        return cls(
            personality=data.get("personality", "default"),
            display_name=data.get("display_name", "Default Persona"),
            description=data.get("description", ""),
            voice=data.get("voice", "en_GB-alan-medium"),
            costume_prompts=list(data.get("costume_prompts", [])),
            fallbacks=list(data.get("fallbacks", [])),
            escalation_lines=list(data.get("escalation_lines", [])),
            low_power_lines=list(data.get("low_power_lines", [])),
            intents=intents,
            startup_greeting=data.get("startup_greeting"),
            priority_intents=list(data.get("priority_intents", [])),
            system_announcements=dict(data.get("system_announcements", {})),
            voice_settings=dict(data.get("voice_settings", {})),
            time_responses=dict(data.get("time_responses", {})),
            weather_responses=dict(data.get("weather_responses", {})),
            docstring=data.get("_doc") or data.get("__doc__") or data.get("docstring"),
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> "PersonalityProfile":
        """Load personality profile from a JSON file.

        Args:
            file_path: Absolute or relative path to the personality JSON file.

        Returns:
            Instantiated PersonalityProfile.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Personality profile not found at '{file_path}'.")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, file_path: str) -> None:
        """Save personality profile to a JSON file.

        Args:
            file_path: Target JSON file path.
        """
        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)
        # Write to a temporary file and rename, so a power loss mid-save cannot leave a
        # truncated profile behind.
        tmp_path = f"{file_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def add_intent(self, intent: CustomIntent, overwrite: bool = True, prepend: bool = True) -> None:
        """Add or update an intent in the personality profile.

        Args:
            intent: CustomIntent to add.
            overwrite: If True, replaces existing intent with matching name.
            prepend: If True, places the intent at the beginning of the rules for priority matching.
        """
        existing_idx = next(
            (idx for idx, item in enumerate(self.intents) if item.name == intent.name),
            None,
        )
        if existing_idx is not None:
            if overwrite:
                if prepend and existing_idx > 0:
                    self.intents.pop(existing_idx)
                    self.intents.insert(0, intent)
                else:
                    self.intents[existing_idx] = intent
            else:
                # Merge patterns and responses without duplicates
                existing = self.intents[existing_idx]
                merged_patterns = list(dict.fromkeys(existing.patterns + intent.patterns))
                merged_responses = list(dict.fromkeys(existing.responses + intent.responses))
                self.intents[existing_idx] = CustomIntent(
                    name=intent.name,
                    patterns=merged_patterns,
                    responses=merged_responses,
                    metadata={**existing.metadata, **intent.metadata},
                )
        else:
            if prepend:
                self.intents.insert(0, intent)
            else:
                self.intents.append(intent)

    def get_intent(self, name: str) -> Optional[CustomIntent]:
        """Look up an intent by name."""
        return next((i for i in self.intents if i.name == name), None)

    def check_query(self, query: str) -> Optional[Tuple[CustomIntent, str]]:
        """Check if an input query matches any existing intent pattern in this profile.

        Args:
            query: Input question or phrase.

        Returns:
            Tuple of (matched_intent, matched_pattern) or None if no match.
        """
        clean_text = normalize_query(query)
        if not clean_text:
            return None

        for intent in self.intents:
            for pat in intent.patterns:
                try:
                    if re.search(pat, clean_text, re.IGNORECASE):
                        return (intent, pat)
                except re.error:
                    continue
        return None

    def search_intents(self, keyword: str) -> List[CustomIntent]:
        """Search intents by name, pattern, or response text.

        Args:
            keyword: Substring or keyword to search for.

        Returns:
            List of matching CustomIntent objects.
        """
        kw = keyword.lower().strip()
        if not kw:
            return []

        results: List[CustomIntent] = []
        for intent in self.intents:
            name_match = kw in intent.name.lower()
            pattern_match = any(kw in p.lower() for p in intent.patterns)
            response_match = any(kw in r.lower() for r in intent.responses)
            if name_match or pattern_match or response_match:
                results.append(intent)
        return results

    def append_responses(self, intent_name: str, new_responses: List[str]) -> bool:
        """Append responses to an existing intent.

        Args:
            intent_name: Name of the intent to update.
            new_responses: Candidate response strings to append.

        Returns:
            True if updated, False if intent not found.
        """
        existing = self.get_intent(intent_name)
        if not existing:
            return False

        merged_responses = list(dict.fromkeys(existing.responses + [r.strip() for r in new_responses if r.strip()]))
        existing.responses = merged_responses
        return True

    def compile_rules(self) -> List[Tuple[str, List[str], List[str]]]:
        """Compile intents into standard (name, patterns, responses) rule tuples.

        Returns:
            List of 3-element tuples matching engine expectations.
        """
        return [(intent.name, intent.patterns, intent.responses) for intent in self.intents]


def find_personality_file(
    personality_name_or_path: str = "grimjaw",
    search_dirs: Optional[List[str]] = None,
) -> str:
    """Resolve personality name or path to an absolute JSON file path.

    Args:
        personality_name_or_path: Name (e.g. 'grimjaw') or file path.
        search_dirs: Optional directories to search within.

    Returns:
        Resolved file path.
    """
    if os.path.isfile(personality_name_or_path):
        return os.path.abspath(personality_name_or_path)

    dirs = search_dirs or [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "personalities"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "personalities"),
        os.path.join(os.getcwd(), "data", "personalities"),
    ]

    base_name = personality_name_or_path if personality_name_or_path.endswith(".json") else f"{personality_name_or_path}.json"

    for d in dirs:
        candidate = os.path.join(d, base_name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # Return primary candidate even if not created yet (for new personality creation)
    primary_dir = dirs[0]
    return os.path.abspath(os.path.join(primary_dir, base_name))


def load_personality(name_or_path: str) -> PersonalityProfile:
    """Find and load a personality profile by name or path.

    Args:
        name_or_path: Personality name (for example 'grimjaw') or JSON file path.

    Returns:
        The loaded profile.

    Raises:
        FileNotFoundError: If no profile file exists for the name or path.
    """
    profile_path = find_personality_file(name_or_path)
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            f"Personality profile not found: '{name_or_path}' (searched at: {profile_path})"
        )
    return PersonalityProfile.load_from_file(profile_path)
