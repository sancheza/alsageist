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

from core import cloud_api

INTENT_API_TIMEOUT_SEC = 6.0


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
        return (
            f"You are {self.display_name}. {self.description}\n"
            "Rules you must strictly obey:\n"
            "1. Keep every response under 2 sentences maximum.\n"
            "2. Be witty, dry, and stay in character.\n"
            "3. Never output emojis, markdown formatting, asterisks, or sound effect tags.\n"
            "4. Return plain spoken dialogue only."
        )

    def announcement(self, key: str, default: str) -> str:
        """Return a spoken system announcement, falling back to a default.

        Args:
            key: Announcement key (for example 'online_mode' or 'powering_off').
            default: Text used when the profile does not define the key.

        Returns:
            Announcement text.
        """
        return self.system_announcements.get(key) or default

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
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

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
        clean_text = re.sub(r"[^\w\s\']", " ", query.lower()).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
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


class IntentCollisionError(Exception):
    """Raised when an input query conflicts with an existing intent."""

    def __init__(self, query: str, existing_intent: CustomIntent, matched_pattern: str) -> None:
        """Initialize intent collision error with conflict details."""
        super().__init__(
            f"Query '{query}' conflicts with existing intent '{existing_intent.name}' "
            f"via pattern: '{matched_pattern}'"
        )
        self.query = query
        self.existing_intent = existing_intent
        self.matched_pattern = matched_pattern


class IntentParser:
    """Hybrid intent parser combining LLM-assisted pattern expansion and offline regex synthesis."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the intent parser.

        Args:
            gemini_api_key: Optional Gemini API key (defaults to $GEMINI_API_KEY).
            groq_api_key: Optional Groq API key (defaults to $GROQ_API_KEY).
        """
        self.gemini_api_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

    def parse(
        self,
        text: str,
        responses: Optional[List[str]] = None,
        name: Optional[str] = None,
        persona: Optional[PersonalityProfile] = None,
        use_llm: bool = True,
    ) -> CustomIntent:
        """Parse user query text and responses into a validated CustomIntent.

        Uses LLM expansion if available and requested, otherwise falls back to deterministic
        offline rule-based parsing.

        Args:
            text: Trigger phrase, sample question, or intent description.
            responses: Optional candidate response strings.
            name: Optional explicit intent name.
            persona: Optional personality context for tailoring responses.
            use_llm: Whether to attempt LLM-assisted parsing.

        Returns:
            Validated CustomIntent.
        """
        cleaned_input = text.strip()
        if not cleaned_input:
            raise ValueError("Input text cannot be empty.")

        response_list = [r.strip() for r in (responses or []) if r and r.strip()]

        if use_llm and (self.gemini_api_key or self.groq_api_key):
            try:
                llm_intent = self._parse_with_llm(
                    text=cleaned_input,
                    responses=response_list,
                    name=name,
                    persona=persona,
                )
                if llm_intent and llm_intent.patterns and llm_intent.responses:
                    return llm_intent
            except Exception:
                # Graceful fallback to offline deterministic parser on network or API failure
                pass

        return self._parse_rule_based(
            text=cleaned_input,
            responses=response_list,
            name=name,
        )

    def _parse_with_llm(
        self,
        text: str,
        responses: List[str],
        name: Optional[str],
        persona: Optional[PersonalityProfile],
    ) -> Optional[CustomIntent]:
        """Query LLM to generate regex patterns and responses."""
        persona_desc = (
            f"Persona: {persona.display_name} ({persona.description})"
            if persona
            else "Persona: Witty, conversational voice assistant"
        )

        prompt = (
            f"{persona_desc}\n"
            f"User Prompt / Sample Query: \"{text}\"\n"
        )
        if name:
            prompt += f"Required Intent Name: \"{name}\"\n"
        if responses:
            prompt += f"User-Provided Responses:\n" + "\n".join(f"- {r}" for r in responses) + "\n"

        prompt += (
            "\nTask:\n"
            "1. Generate a concise snake_case intent name (if not provided).\n"
            "2. Generate 3 to 6 high-precision Python regular expression patterns matching this intent.\n"
            "   - Use word boundaries (\\b) on all keywords.\n"
            "   - Cover common phrasing variations, contractions (what's/what is), and pronouns (you/he/she).\n"
            "   - Ensure valid regex syntax without syntax errors.\n"
            "3. Generate 2 to 4 witty, conversational responses in persona (or refine/keep the provided responses).\n"
            "Return ONLY a raw JSON object with keys: 'name', 'patterns', and 'responses'."
        )

        raw_json_str = ""
        if self.gemini_api_key:
            raw_json_str = self._call_gemini(prompt)
        elif self.groq_api_key:
            raw_json_str = self._call_groq(prompt)

        if not raw_json_str:
            return None

        # Clean JSON markdown if wrapped in ```json ... ```
        clean_json = re.sub(r"^```(?:json)?\s*", "", raw_json_str.strip(), flags=re.IGNORECASE)
        clean_json = re.sub(r"\s*```$", "", clean_json.strip())

        data = json.loads(clean_json)
        intent_name = name or data.get("name", "")
        intent_patterns = data.get("patterns", [])
        intent_responses = responses if responses else data.get("responses", [])

        # Validate patterns
        valid_patterns = []
        for pat in intent_patterns:
            try:
                re.compile(pat, re.IGNORECASE)
                valid_patterns.append(pat)
            except re.error:
                continue

        if not valid_patterns:
            return None

        return CustomIntent(
            name=intent_name or self._derive_snake_name(text),
            patterns=valid_patterns,
            responses=intent_responses or [f"I heard you ask about {text}!"],
            metadata={"source": "llm_expanded"},
        )

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API for intent expansion."""
        models = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }

        for model in models:
            try:
                parts = cloud_api.gemini_generate_parts(
                    self.gemini_api_key, model, payload, timeout=INTENT_API_TIMEOUT_SEC
                )
                if parts:
                    return parts[0].get("text", "").strip()
            except Exception:
                continue
        return ""

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API for intent expansion."""
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a regex intent parser. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        try:
            return cloud_api.groq_chat(self.groq_api_key, payload, timeout=INTENT_API_TIMEOUT_SEC)
        except Exception:
            return ""

    @staticmethod
    def _derive_snake_name(text: str) -> str:
        """Derive a clean snake_case intent name from sample text."""
        # Strip common leading question words
        clean = re.sub(
            r"^(what('s| is)|who('s| is)|where('s| is)|when('s| is)|how('s| is)|do you|can you|tell me|give me)\s+",
            "",
            text.lower().strip(),
        )
        words = re.findall(r"[a-z0-9]+", clean)
        # Drop trivial stop words
        stop_words = {"a", "an", "the", "about", "your", "my", "some", "of", "to", "in", "for"}
        filtered = [w for w in words if w not in stop_words]
        chosen = filtered[:4] if filtered else words[:3]
        name = "_".join(chosen) if chosen else "custom_intent"
        return name

    def _parse_rule_based(
        self,
        text: str,
        responses: List[str],
        name: Optional[str],
    ) -> CustomIntent:
        """Deterministic rule-based pattern generator when LLM is unavailable."""
        intent_name = name or self._derive_snake_name(text)
        normalized = re.sub(r"[^\w\s\']", " ", text.lower()).strip()
        tokens = [t for t in normalized.split() if t]

        patterns: List[str] = []

        # 1. Exact phrase with word boundaries
        exact_clean = " ".join(tokens)
        if exact_clean:
            patterns.append(rf"\b{re.escape(exact_clean)}\b")

        # 2. Key content words joined with wildcards
        stop_words = {"what", "whats", "who", "where", "when", "how", "is", "are", "do", "you", "a", "an", "the", "your", "my"}
        content_tokens = [t for t in tokens if t not in stop_words]
        if len(content_tokens) >= 2:
            patterns.append(r"\b" + r"\b.*\b".join(re.escape(t) for t in content_tokens) + r"\b")

        # 3. Question phrasing with contractions if applicable
        if any(w in tokens for w in ["favorite", "best", "like"]):
            subject_tokens = [t for t in content_tokens if t not in ("favorite", "best", "like")]
            key_subject = " ".join(subject_tokens) if subject_tokens else (content_tokens[-1] if content_tokens else "it")
            patterns.append(rf"\b(favorite|best|like)\b.*\b{re.escape(key_subject)}\b")
            patterns.append(rf"\bwhat('s| is) (your|his) favorite {re.escape(key_subject)}\b")

        if not patterns:
            patterns.append(rf"\b{re.escape(text.strip().lower())}\b")

        # Unique ordered patterns
        unique_patterns = list(dict.fromkeys(patterns))

        default_responses = responses if responses else [
            f"You asked about {exact_clean}! That's quite a curious question."
        ]

        return CustomIntent(
            name=intent_name,
            patterns=unique_patterns,
            responses=default_responses,
            metadata={"source": "rule_based"},
        )


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
    os.makedirs(primary_dir, exist_ok=True)
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


def resolve_voice_settings(
    profile: Optional[PersonalityProfile],
    tempo: Optional[float] = None,
    noise: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve Piper tempo and noise, preferring explicit values over the profile.

    Args:
        profile: Personality profile supplying voice_settings, or None.
        tempo: Explicit tempo (phoneme length scale), or None to use the profile.
        noise: Explicit noise scale, or None to use the profile.

    Returns:
        Tuple of (tempo, noise), defaulting to (1.15, 0.85).
    """
    settings = (profile.voice_settings if profile else None) or {}
    return (
        tempo if tempo is not None else settings.get("tempo", 1.15),
        noise if noise is not None else settings.get("noise", 0.85),
    )


def create_custom_intent(
    query_or_text: str,
    responses: Optional[List[str]] = None,
    personality_name_or_path: str = "grimjaw",
    name: Optional[str] = None,
    prewarm: bool = False,
    dry_run: bool = False,
    use_llm: bool = True,
    check_collision: bool = True,
    collision_action: str = "raise",
) -> Tuple[CustomIntent, PersonalityProfile]:
    """Universal function to create, parse, and attach a custom intent to a personality.

    Can be reused across voice assist projects.

    Args:
        query_or_text: Sample user query, prompt description, or phrase.
        responses: List of character responses.
        personality_name_or_path: Target personality profile identifier or path.
        name: Optional explicit intent name.
        prewarm: Whether to synthesize response audio into the local cache.
        dry_run: If True, parses intent without saving to disk.
        use_llm: Whether to attempt LLM-assisted pattern and response expansion.
        check_collision: Whether to check if query already matches an existing intent.
        collision_action: Action when collision occurs: 'raise', 'append', 'overwrite', or 'force'.

    Returns:
        Tuple of (CustomIntent, PersonalityProfile).

    Raises:
        IntentCollisionError: If query conflicts with an existing intent and collision_action is 'raise'.
    """
    profile_path = find_personality_file(personality_name_or_path)

    if os.path.isfile(profile_path):
        profile = PersonalityProfile.load_from_file(profile_path)
    else:
        # Create brand new personality profile
        profile = PersonalityProfile(
            personality=os.path.splitext(os.path.basename(profile_path))[0],
            display_name=personality_name_or_path.capitalize(),
            description=f"Custom voice assistant personality for {personality_name_or_path}.",
        )

    # 1. Collision check against existing intent patterns
    if check_collision:
        collision = profile.check_query(query_or_text)
        if collision is not None:
            existing_intent, matched_pat = collision
            if collision_action == "raise":
                raise IntentCollisionError(query_or_text, existing_intent, matched_pat)
            elif collision_action == "append":
                if responses:
                    profile.append_responses(existing_intent.name, responses)
                if not dry_run:
                    profile.save_to_file(profile_path)
                    if prewarm and responses:
                        try:
                            from services.tts_service import PiperTTSService
                            tts = PiperTTSService()
                            for resp in responses:
                                tts.synthesize(resp)
                        except Exception as tts_err:
                            print(f"Warning: Pre-warm audio synthesis skipped: {tts_err}")
                return existing_intent, profile
            elif collision_action == "overwrite":
                name = existing_intent.name  # Overwrite with matching name

    parser = IntentParser()
    intent = parser.parse(
        text=query_or_text,
        responses=responses,
        name=name,
        persona=profile,
        use_llm=use_llm,
    )

    if not dry_run:
        profile.add_intent(intent, overwrite=True, prepend=True)
        profile.save_to_file(profile_path)

        if prewarm and intent.responses:
            try:
                from services.tts_service import PiperTTSService
                tts = PiperTTSService()
                for resp in intent.responses:
                    tts.synthesize(resp)
            except Exception as tts_err:
                print(f"Warning: Pre-warm audio synthesis skipped: {tts_err}")

    return intent, profile
