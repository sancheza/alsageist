"""Custom intent authoring: LLM-assisted regex generation and attaching intents to personalities.

Builds on the profile model in core.intent_builder. Used by scripts/add_intent.py.
"""

import json
import os
import re
from typing import Callable, List, Optional, Tuple

from core import cloud_api, env_utils, intent_samples
from core.intent_builder import (
    CustomIntent,
    PersonalityProfile,
    find_personality_file,
    normalize_query,
)

INTENT_API_TIMEOUT_SEC = 6.0
MAX_PATTERN_CHARS = 300
# A quantified group that itself contains a quantifier, such as (a+)+, can take exponential time
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d*,\d*\})")


def is_backtracking_risk(pattern: str) -> bool:
    """Return True if a regex is long or nests quantifiers, so matching could stall the voice loop.

    Args:
        pattern: Regular expression source.

    Returns:
        True if the pattern should not be added to a profile.
    """
    return len(pattern) > MAX_PATTERN_CHARS or _NESTED_QUANTIFIER.search(pattern) is not None


def find_shadowed_intents(profile: PersonalityProfile, patterns: List[str], ignore_name: str) -> List[str]:
    """Return existing intents whose sample questions a new intent's patterns would capture.

    A new intent is matched before the existing ones, so a broad pattern silently takes over
    their questions.

    Args:
        profile: Profile that holds the existing intents.
        patterns: Regex patterns of the new intent.
        ignore_name: Name of the intent being added or replaced; it is not compared with itself.

    Returns:
        Names of the intents that would lose questions.
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    shadowed: List[str] = []
    for intent in profile.intents:
        if intent.name == ignore_name:
            continue
        samples = intent_samples.extract_sample_questions(intent.patterns, max_questions=3)
        if any(c.search(normalize_query(sample)) for sample in samples for c in compiled):
            shadowed.append(intent.name)
    return shadowed


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


class IntentShadowError(Exception):
    """Raised when a new intent's patterns would capture questions of existing intents."""

    def __init__(self, intent_name: str, shadowed: List[str]) -> None:
        """Initialize with the new intent's name and the intents it would shadow."""
        super().__init__(
            f"Intent '{intent_name}' would capture questions of existing intents: {', '.join(shadowed)}"
        )
        self.intent_name = intent_name
        self.shadowed = shadowed


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
        self.gemini_api_key = env_utils.gemini_api_key(gemini_api_key)
        self.groq_api_key = env_utils.groq_api_key(groq_api_key)

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
            except Exception as exc:  # isolation point: any LLM failure falls back to the offline parser
                print(f"Warning: LLM pattern expansion failed ({type(exc).__name__}: {exc}); using the rule-based parser.")

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
            except re.error:
                continue
            if not is_backtracking_risk(pat):
                valid_patterns.append(pat)

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
        models = cloud_api.gemini_llm_models()
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }

        return cloud_api.gemini_generate(
            self.gemini_api_key, models, payload, INTENT_API_TIMEOUT_SEC,
            extract=lambda parts: parts[0].get("text", "").strip(),
        )

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API for intent expansion."""
        model = cloud_api.groq_llm_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a regex intent parser. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            **cloud_api.groq_model_params(model),
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
    prewarm_fn: Optional[Callable[[List[str], PersonalityProfile], None]] = None,
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
        prewarm_fn: Callback `(responses, profile) -> None` that synthesizes phrases into the cache;
            required for `prewarm` to have any effect (core code does not import services).

    Returns:
        Tuple of (CustomIntent, PersonalityProfile).

    Raises:
        IntentCollisionError: If query conflicts with an existing intent and collision_action is 'raise'.
        IntentShadowError: If the new patterns would capture questions of other intents and
            collision_action is not 'force'.
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

    def _execute_prewarm(phrases: List[str], target_profile: PersonalityProfile) -> None:
        if prewarm and phrases and prewarm_fn is not None:
            prewarm_fn(phrases, target_profile)

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
                    _execute_prewarm(responses or [], profile)
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

    if collision_action != "force":
        shadowed = find_shadowed_intents(profile, intent.patterns, ignore_name=intent.name)
        if shadowed:
            raise IntentShadowError(intent.name, shadowed)

    if not dry_run:
        profile.add_intent(intent, overwrite=True, prepend=True)
        profile.save_to_file(profile_path)
        _execute_prewarm(intent.responses, profile)

    return intent, profile
