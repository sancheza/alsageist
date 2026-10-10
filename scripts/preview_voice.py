#!/usr/bin/env python3
"""Preview creature voices for AlsaGeist using Gemini Audio synthesis.

Usage:
    ./scripts/preview_voice.py --persona skeleton --play
    ./scripts/preview_voice.py --persona witch --voice Aoede --play
    ./scripts/preview_voice.py --text "Halt! Who dares approach my crypt?" --voice Fenrir --play
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional
import urllib.error
import urllib.request

VERSION = "0.1.0"

PERSONA_PROMPTS = {
    "skeleton": {
        "voice": "Charon",
        "system": (
            "You are Grimjaw, a sarcastic, ancient skeletal guardian stationed on the porch guarding "
            "the Halloween candy bowl. Speak in a dry, gravelly, raspy voice with witty, teasing remarks. "
            "Never use markdown, asterisks, or sound effects tags. Return spoken dialogue only."
        ),
        "default_spoken": (
            "Halt, little mortal! Reach into my candy bowl if you dare, but watch your fingers—I might just keep them!"
        ),
    },
    "witch": {
        "voice": "Aoede",
        "system": (
            "You are Madame Hexcrow, a wicked, theatrical witch styled after the Wicked Witch of the West. "
            "Speak in a shrill, raspy, high-pitched theatrical cackle with sharp wit and menacing glee. "
            "Never use markdown, asterisks, or sound effects tags. Return spoken dialogue only."
        ),
        "default_spoken": (
            "I'll get you, my pretty, and your little candy bag too! Hehehehehe! Scram before I turn you into a wart-covered toad!"
        ),
    },
}


def generate_creature_audio(
    prompt: str,
    system_instruction: str,
    voice_name: str,
    api_key: str,
    model_name: str = "gemini-3.8-flash-tts",
) -> bytes:
    """Generate audio bytes from Gemini using specified voice configuration.

    Args:
        prompt: User input or scenario prompt.
        system_instruction: Persona and behavioral prompt.
        voice_name: Prebuilt voice name (e.g., Fenrir, Puck, Charon, Aoede, Kore).
        api_key: Gemini API key.
        model_name: Gemini model ID.

    Returns:
        Raw audio bytes (WAV/PCM/MP3).

    Raises:
        RuntimeError: If API call fails or no audio data is returned.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            },
        },
    }
    if system_instruction and not model_name.endswith("-tts"):
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8")
        raise RuntimeError(f"Gemini API request failed ({exc.code}): {err_body}") from exc

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates returned from Gemini API.")

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        inline_data = part.get("inlineData", {})
        if inline_data.get("data"):
            return base64.b64decode(inline_data["data"])

    raise RuntimeError("No audio data found in Gemini API response.")


def play_audio(audio_data: bytes) -> None:
    """Play audio bytes on macOS using afplay.

    Args:
        audio_data: Raw audio data bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name

    try:
        subprocess.run(["afplay", temp_path], check=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


VOICE_CATALOG = {
    "Puck": {"gender": "Male", "pitch": "Medium", "timbre": "Upbeat, playful, energetic"},
    "Charon": {"gender": "Male", "pitch": "Low", "timbre": "Deep, hollow, grave, resonant"},
    "Kore": {"gender": "Female", "pitch": "Medium-High", "timbre": "Firm, crisp, sharp"},
    "Fenrir": {"gender": "Male", "pitch": "Low", "timbre": "Gruff, aggressive, gravelly"},
    "Aoede": {"gender": "Female", "pitch": "High", "timbre": "Theatrical, expressive, breezy, high-pitch"},
    "Leda": {"gender": "Female", "pitch": "High", "timbre": "Calm, soft, smooth"},
    "Orus": {"gender": "Male", "pitch": "Low", "timbre": "Confident, direct, authoritative"},
    "Zephyr": {"gender": "Female / Neutral", "pitch": "High", "timbre": "Bright, light, airy"},
}


def list_voices() -> None:
    """Print formatted catalog of prebuilt Gemini voices."""
    print(f"{'Voice Name':<12} | {'Gender':<18} | {'Pitch':<12} | {'Timbre / Character'}")
    print("-" * 75)
    for name, info in VOICE_CATALOG.items():
        print(f"{name:<12} | {info['gender']:<18} | {info['pitch']:<12} | {info['timbre']}")
    print("\nUsage example:")
    print("  ./scripts/preview_voice.py --voice Aoede --text 'Your custom line here' --play")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Preview creature voices for AlsaGeist using Gemini Audio synthesis."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--list-voices",
        "-l",
        action="store_true",
        help="List all available prebuilt Gemini voice models and exit.",
    )
    parser.add_argument(
        "--persona",
        choices=["skeleton", "witch"],
        default="skeleton",
        help="Target creature persona (default: skeleton)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Prebuilt voice name (e.g. Fenrir, Puck, Charon, Aoede, Kore). Overrides persona default.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Custom input text to trigger dialogue response. Defaults to persona standard test line.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash-tts",
        help="Gemini model ID (default: gemini-3.8-flash-tts)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to save generated audio file (WAV).",
    )
    parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Automatically play audio via afplay on macOS.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    if args.list_voices:
        list_voices()
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.stderr.write("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.\n")
        return 1

    persona_cfg = PERSONA_PROMPTS[args.persona]
    voice_name = args.voice if args.voice else persona_cfg["voice"]
    prompt_text = args.text if args.text else persona_cfg["default_spoken"]
    system_inst = persona_cfg["system"]

    print(f"Synthesizing [{args.persona.upper()}] using {args.model} with voice '{voice_name}'...")
    print(f"Input: \"{prompt_text}\"")

    try:
        audio_data = generate_creature_audio(
            prompt=prompt_text,
            system_instruction=system_inst,
            voice_name=voice_name,
            api_key=api_key,
            model_name=args.model,
        )
    except RuntimeError as err:
        sys.stderr.write(f"Synthesis failed: {err}\n")
        return 1

    out_file = args.output or f"preview_{args.persona}_{voice_name.lower()}.wav"
    with open(out_file, "wb") as f:
        f.write(audio_data)
    print(f"Saved audio sample to: {out_file}")

    if args.play:
        print("Playing audio sample...")
        play_audio(audio_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
