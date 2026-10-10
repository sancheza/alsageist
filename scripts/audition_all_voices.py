#!/usr/bin/env python3
"""Audition all available Gemini voices sequentially on macOS.

This script displays each voice's details and immediately plays a sample
phrase through the system speakers using afplay.

Usage:
    ./scripts/audition_all_voices.py
    ./scripts/audition_all_voices.py --phrase "Happy Halloween, mortal!"
    ./scripts/audition_all_voices.py --gender Female
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional
import urllib.error
import urllib.request

VERSION = "0.1.0"

VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "Charon": {
        "gender": "Male",
        "pitch": "Low",
        "timbre": "Deep, hollow, grave, resonant",
        "sample": "Halt, mortal! Who dares disturb my eternal slumber at this crypt?",
    },
    "Fenrir": {
        "gender": "Male",
        "pitch": "Low",
        "timbre": "Gruff, aggressive, gravelly rasp",
        "sample": "More candy? You'll have to answer my riddle first, or leave empty-handed!",
    },
    "Orus": {
        "gender": "Male",
        "pitch": "Low",
        "timbre": "Authoritative, confident, commanding",
        "sample": "Step forward and present your Halloween tribute to the porch guardian.",
    },
    "Puck": {
        "gender": "Male",
        "pitch": "Medium",
        "timbre": "Upbeat, playful, energetic",
        "sample": "Trick or treat! Look at that fantastic costume you've got on tonight!",
    },
    "Aoede": {
        "gender": "Female",
        "pitch": "High",
        "timbre": "Theatrical, expressive, sharp cackle (Wicked Witch)",
        "sample": "I'll get you, my pretty, and your candy bag too! Hehehehehe! Scram!",
    },
    "Kore": {
        "gender": "Female",
        "pitch": "Medium-High",
        "timbre": "Firm, crisp, sharp",
        "sample": "Double, double toil and trouble, fire burn and cauldron bubble!",
    },
    "Leda": {
        "gender": "Female",
        "pitch": "High",
        "timbre": "Calm, soft, smooth, eerie whisper",
        "sample": "Beware the shadows tonight, little traveler... things are not what they seem.",
    },
    "Zephyr": {
        "gender": "Female / Neutral",
        "pitch": "High",
        "timbre": "Bright, light, airy, ghostly",
        "sample": "Woooo! I am the spirit trapped inside this pumpkin! Feed me candy!",
    },
}


def get_cache_path(voice_name: str, text: str, cache_dir: str) -> str:
    """Generate deterministic file path for cached voice audio.

    Args:
        voice_name: Voice model name.
        text: Synthesized text.
        cache_dir: Target directory for cache files.

    Returns:
        Absolute path to cached WAV file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = voice_name.lower().replace(" ", "_")
    return os.path.join(cache_dir, f"{safe_name}.wav")


def synthesize_voice_sample(
    voice_name: str,
    text: str,
    api_key: str,
    cache_dir: Optional[str] = "samples/voices",
    model_name: str = "gemini-3.8-flash-tts",
    max_retries: int = 4,
) -> bytes:
    """Synthesize voice sample audio bytes from Gemini API with disk caching.

    Args:
        voice_name: Prebuilt voice name.
        text: Text phrase to synthesize.
        api_key: Gemini API key.
        cache_dir: Optional directory to load/save cached WAV audio.
        model_name: Gemini model ID.
        max_retries: Maximum retry attempts on rate-limit (429).

    Returns:
        Raw audio bytes.

    Raises:
        RuntimeError: If synthesis fails after retries.
    """
    if cache_dir:
        cached_file = get_cache_path(voice_name, text, cache_dir)
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 0:
            with open(cached_file, "rb") as f:
                return f.read()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
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

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates returned from Gemini API.")
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData", {})
                if inline_data.get("data"):
                    audio_bytes = base64.b64decode(inline_data["data"])
                    if cache_dir:
                        cached_file = get_cache_path(voice_name, text, cache_dir)
                        with open(cached_file, "wb") as f:
                            f.write(audio_bytes)
                    return audio_bytes
            raise RuntimeError("No audio data found in response parts.")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            if exc.code == 429 and attempt < max_retries:
                retry_wait = 20
                try:
                    err_json = json.loads(err_body)
                    for detail in err_json.get("error", {}).get("details", []):
                        if "retryDelay" in detail:
                            delay_str = detail["retryDelay"].rstrip("s")
                            retry_wait = int(float(delay_str)) + 1
                            break
                except Exception:
                    pass

                if retry_wait > 25:
                    raise RuntimeError(
                        f"Daily API free-tier quota exhausted ({err_json.get('error', {}).get('message', 'Quota exceeded')[:120]}...)"
                    ) from exc

                print(f"\n  [Quota limit (3 RPM) reached. Pausing {retry_wait}s for token refresh...]", end="", flush=True)
                for _ in range(retry_wait):
                    time.sleep(1)
                    print(".", end="", flush=True)
                print(" Resuming.")
                continue
            raise RuntimeError(f"Gemini API request failed ({exc.code}): {err_body}") from exc

    raise RuntimeError("Synthesis failed after maximum retries.")


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Audition all available Gemini voices sequentially on macOS."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--phrase",
        "-p",
        default=None,
        help="Optional custom phrase to use for all voices instead of their character sample.",
    )
    parser.add_argument(
        "--gender",
        "-g",
        choices=["Male", "Female", "All"],
        default="All",
        help="Filter voices by gender (default: All).",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=2.0,
        help="Delay in seconds between voices (default: 2.0).",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.stderr.write("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.\n")
        return 1

    selected_voices: List[str] = []
    for name, info in VOICE_CATALOG.items():
        if args.gender == "All" or args.gender in info["gender"]:
            selected_voices.append(name)

    print("=" * 70)
    print(f" ALSA-GEIST VOICE AUDITION ({len(selected_voices)} Voices)")
    print("=" * 70)

    for idx, name in enumerate(selected_voices, start=1):
        info = VOICE_CATALOG[name]
        sample_phrase = args.phrase if args.phrase else info["sample"]

        print(f"\n[{idx}/{len(selected_voices)}] Voice: {name}")
        print(f"       Gender:  {info['gender']}")
        print(f"       Pitch:   {info['pitch']}")
        print(f"       Profile: {info['timbre']}")
        print(f"       Phrase:  \"{sample_phrase}\"")
        print("       Status:  Synthesizing & Playing... ", end="", flush=True)

        try:
            audio_data = synthesize_voice_sample(
                voice_name=name,
                text=sample_phrase,
                api_key=api_key,
            )
            print("▶ Playing audio")
            play_audio(audio_data)
        except RuntimeError as err:
            print(f"FAILED ({err})")

        if idx < len(selected_voices):
            time.sleep(args.delay)

    print("\n" + "=" * 70)
    print(" Audition complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
