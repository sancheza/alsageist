"""Piper voice resolution: model file location, tempo and noise for a persona."""

import os
from typing import Any, Optional, Tuple

DEFAULT_VOICE = "en_GB-alan-medium"
DEFAULT_TEMPO = 1.15
DEFAULT_NOISE = 0.85


def resolve_voice_settings(
    profile: Optional[Any],
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
        tempo if tempo is not None else settings.get("tempo", DEFAULT_TEMPO),
        noise if noise is not None else settings.get("noise", DEFAULT_NOISE),
    )


def resolve_piper_model_path(
    model_path: Optional[str] = None,
    voice_name: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> str:
    """Resolve the file path to a Piper ONNX neural voice model.

    Evaluates explicit model paths and voice names (e.g. 'en_GB-alan-medium'). A voice
    name resolves to `models/piper/<name>.onnx` whether or not the file exists; the default
    model is used only when neither a path nor a voice name is given.

    Args:
        model_path: Explicit path from CLI flags or environment variables.
        voice_name: Character voice identifier from personality profile or string.
        repo_root: Root directory of the repository. Auto-resolved if None.

    Returns:
        Resolved absolute or normalized model file path.
    """
    if repo_root is None:
        explicit_root = os.environ.get("ALSAGEIST_ROOT") or os.environ.get("REPO_ROOT")
        if explicit_root and os.path.isdir(explicit_root):
            repo_root = explicit_root
        elif os.path.isdir(os.path.join(os.getcwd(), "models")):
            repo_root = os.getcwd()
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Explicit model path from argument or environment; a bare voice ID is treated as voice_name
    if model_path and not model_path.endswith(".onnx") and os.sep not in model_path:
        voice_name = voice_name or model_path
        model_path = None

    if model_path:
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path
        candidate = os.path.join(repo_root, model_path)
        if os.path.exists(candidate):
            return candidate
        return model_path

    # 2. Voice name from profile (e.g. 'en_GB-alan-medium' or 'models/piper/en_GB-alan-medium.onnx')
    if voice_name:
        if voice_name.endswith(".onnx"):
            if os.path.isabs(voice_name) and os.path.exists(voice_name):
                return voice_name
            candidate = os.path.join(repo_root, voice_name)
            if os.path.exists(candidate):
                return candidate
            candidate_in_piper = os.path.join(repo_root, "models", "piper", os.path.basename(voice_name))
            if os.path.exists(candidate_in_piper):
                return candidate_in_piper
            return candidate
        # Short name without .onnx; a voice that is not installed resolves to its expected
        # path so the caller reports the missing file
        return os.path.join(repo_root, "models", "piper", f"{voice_name}.onnx")

    # 3. Default model when no model path or voice name is given
    return os.path.join(repo_root, "models", "piper", f"{DEFAULT_VOICE}.onnx")


def resolve_agent_voice_config(
    profile: Optional[Any] = None,
    model_path: Optional[str] = None,
    tempo: Optional[float] = None,
    noise: Optional[float] = None,
    repo_root: Optional[str] = None,
) -> Tuple[str, float, float]:
    """Resolve the voice model path, tempo, and noise for a persona in one place.

    CLI arguments take precedence over environment variables, which take
    precedence over profile settings, falling back to system defaults.

    Args:
        profile: Optional PersonalityProfile supplying voice and voice_settings.
        model_path: Explicit model path CLI flag or environment override.
        tempo: Explicit tempo CLI flag or environment override.
        noise: Explicit noise CLI flag or environment override.
        repo_root: Root directory of repository.

    Returns:
        Tuple of (model_path, tempo, noise).
    """
    tempo_val, noise_val = resolve_voice_settings(profile, tempo=tempo, noise=noise)
    resolved_path = resolve_piper_model_path(
        model_path=model_path or os.environ.get("MODEL_PATH"),
        voice_name=getattr(profile, "voice", None) if profile else None,
        repo_root=repo_root,
    )
    return resolved_path, tempo_val, noise_val
