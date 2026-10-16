"""Offline Intent and Banter Engine driven by external personality profiles.

Provides instant, zero-latency conversational responses to trick-or-treaters
and visitors, loaded dynamically from external personality JSON profiles.
"""

from collections import OrderedDict
import os
import random
import re
from typing import List, Optional, Tuple, Union

from core.intent_builder import PersonalityProfile, load_personality


class PersonaIntentEngine:
    """Configurable intent and banter engine driven by external personality JSON profiles."""

    def __init__(self, personality_name_or_path: Union[str, PersonalityProfile] = "grimjaw") -> None:
        """Initialize the intent engine with conversational patterns and responses.

        Args:
            personality_name_or_path: Name, file path, or PersonalityProfile object.

        Raises:
            FileNotFoundError: If the requested personality profile JSON file cannot be found.
        """
        self._recently_used: "OrderedDict[str, None]" = OrderedDict()
        self._max_recent: int = 25
        self._turn_count: int = 0
        self._asked_costume: bool = False

        if isinstance(personality_name_or_path, PersonalityProfile):
            self.profile = personality_name_or_path
            self._personality_name = personality_name_or_path.personality
        else:
            self._personality_name = personality_name_or_path
            self.profile = load_personality(personality_name_or_path)
        self._costume_prompts = list(self.profile.costume_prompts)
        self._fallbacks = list(self.profile.fallbacks)
        self._escalation_lines = list(self.profile.escalation_lines)
        self._intent_rules = self.profile.compile_rules()
        self._compiled_intent_rules: List[Tuple[str, List[re.Pattern], List[str]]] = []
        for name, patterns, responses in self._intent_rules:
            compiled = []
            for pat in patterns:
                if pat and pat.strip():
                    try:
                        compiled.append(re.compile(pat, re.IGNORECASE))
                    except re.error:
                        pass
            self._compiled_intent_rules.append((name, compiled, responses))

        # Compile a curated pool of quotes, songs, jokes, and lore (excluding costume feedback)
        self._random_content_pool: List[str] = []
        for name, _, responses in self._intent_rules:
            if name.startswith("costume") or name in ("costume", "favorite_costume"):
                continue
            for r in responses:
                if r and r not in self._random_content_pool:
                    self._random_content_pool.append(r)

    def reset_conversation(self) -> None:
        """Reset turn counter and costume inquiry state for a new visitor."""
        self._turn_count = 0
        self._asked_costume = False

    def classify_and_respond(self, user_text: str) -> Tuple[str, str]:
        """Classify user text and return both the response and matched intent name.

        Args:
            user_text: Transcribed input string from the user.

        Returns:
            Tuple of (response_text, matched_intent_name).
        """
        if not user_text or not user_text.strip():
            return random.choice(self._fallbacks), "fallback_empty"

        self._turn_count += 1
        clean_text = re.sub(r"[^\w\s\']", " ", user_text.lower()).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)

        # 1. Check intent rules first to always answer direct questions and costume statements accurately
        for intent_name, compiled_patterns, responses in self._compiled_intent_rules:
            for pattern in compiled_patterns:
                if pattern.search(clean_text):
                    if intent_name == "random_content":
                        chosen = self._pick_unique(self._random_content_pool)
                    else:
                        chosen = self._pick_unique(responses)
                    if intent_name in ("goodbye", "holidays_and_farewell") and any(
                        w in clean_text for w in ["bye", "see you", "thank", "next year"]
                    ):
                        self.reset_conversation()
                    return chosen, intent_name

        # 2. Once a conversation has been initiated (Turn 2), occasionally (1 in 3 chance) ask who they are dressed as
        if self._turn_count >= 2 and not self._asked_costume and random.random() < 0.33:
            self._asked_costume = True
            prompt_line = random.choice(self._costume_prompts)
            return prompt_line, "costume_inquiry"

        # 3. Lingerer escalation for unrecognized phrases on late turns
        if self._turn_count >= 8 and random.random() < 0.6:
            chosen = self._pick_unique(self._escalation_lines)
            self.reset_conversation()
            return chosen, "lingerer_escalation"

        # 4. Fallback for unrecognized phrases
        return self._pick_unique(self._fallbacks), "fallback_unrecognized"

    def generate_response(self, user_text: str) -> str:
        """Classify user text and return the best conversational response.

        Args:
            user_text: Transcribed input string from the user.

        Returns:
            A sarcastic, relevant persona spoken line.
        """
        reply, _ = self.classify_and_respond(user_text)
        return reply

    def _pick_unique(self, choices: List[str]) -> str:
        """Select a random response that hasn't been used recently.

        Args:
            choices: Candidate response strings.

        Returns:
            Selected response string.
        """
        available = [c for c in choices if c not in self._recently_used]
        if not available:
            available = choices
            self._recently_used.clear()

        chosen = random.choice(available)
        self._recently_used[chosen] = None
        self._recently_used.move_to_end(chosen)

        # Forget the oldest phrase first so a line just chosen is never the one dropped
        while len(self._recently_used) > self._max_recent:
            self._recently_used.popitem(last=False)

        return chosen
