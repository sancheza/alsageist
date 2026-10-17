"""Filters for transcripts that are silence or recognizer artifacts rather than speech."""

import re
from typing import Optional, Tuple

ARTIFACT_PHRASES = frozenset(["you", "thank you", "shh", "the", "a", "subtitles by", "bye", "uh", "um", ""])
ARTIFACT_MAX_RMS = 0.010  # quiet clips that transcribe to a stock phrase are recognizer hallucinations
_TIMECODE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def classify_utterance(user_text: Optional[str], audio_rms: float) -> Tuple[bool, str, str]:
    """Decide whether a transcript should be ignored as silence or an ambient artifact.

    Args:
        user_text: Transcribed string from speech-to-text.
        audio_rms: Root-mean-square energy of the audio segment.

    Returns:
        Tuple of (should_filter, matched_intent, filter_status).
    """
    if not user_text or len(user_text.strip()) < 2:
        return True, "empty_silence", "FILTERED_SILENCE"

    clean = user_text.strip().lower().rstrip(".,!?")
    if (clean in ARTIFACT_PHRASES and audio_rms < ARTIFACT_MAX_RMS) or _TIMECODE.match(clean):
        return True, "artifact_filter", "FILTERED_ARTIFACT"

    return False, "", ""
