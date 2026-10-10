#!/usr/bin/env python3
"""Generate audio mockups across all free TTS engines for Skeleton and Witch personas.

Engines tested:
1. Edge-TTS (Microsoft Neural Cloud - Free & Unlimited, No API Key)
2. Piper TTS (100% Local On-Device Neural - Free & Offline)
3. Gemini Live Samples (Pre-cached from Developer API)

Usage:
    ./scripts/generate_all_mockups.py
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

VERSION = "0.1.0"

SKELETON_PHRASE = (
    "Halt, little mortal! Reach into my candy bowl if you dare, but watch your fingers—I might just keep them!"
)

WITCH_PHRASE = (
    "I'll get you, my pretty, and your little candy bag too! Hehehehehe! Scram before I turn you into a wart-covered toad!"
)


def generate_edge_tts(voice: str, pitch: str, rate: str, text: str, output_file: str) -> None:
    """Generate audio clip using edge-tts.

    Args:
        voice: Edge-TTS voice identifier.
        pitch: Pitch adjustment (e.g. '+40Hz', '-25Hz').
        rate: Rate adjustment (e.g. '+15%', '-10%').
        text: Text to synthesize.
        output_file: Destination MP3 file path.
    """
    cmd = [
        "edge-tts",
        "--voice", voice,
        "--pitch", pitch,
        "--rate", rate,
        "--text", text,
        "--write-media", output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_piper_tts(model_path: str, length_scale: float, noise_scale: float, text: str, output_file: str) -> None:
    """Generate audio clip using local Piper ONNX model.

    Args:
        model_path: Path to ONNX model.
        length_scale: Phoneme length scaling factor (speed/tempo).
        noise_scale: Generator noise factor.
        text: Text to synthesize.
        output_file: Destination WAV file path.
    """
    cmd = [
        "piper",
        "-m", model_path,
        "-f", output_file,
        "--length-scale", str(length_scale),
        "--noise-scale", str(noise_scale),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.communicate(input=text.encode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(f"Piper execution failed with code {process.returncode}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate audio mockups across all free TTS engines."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="samples/mockups",
        help="Directory to save generated mockup audio files (default: samples/mockups).",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" GENERATING VOICE MOCKUPS ACROSS ALL FREE ENGINES")
    print("=" * 70)

    # 1. Edge-TTS Mockups
    edge_configs = [
        ("edge_skeleton_1_christopher.mp3", "en-US-ChristopherNeural", "-25Hz", "-10%", SKELETON_PHRASE, "Skeleton: Deep US Pitch Shift"),
        ("edge_skeleton_2_ryan.mp3", "en-GB-RyanNeural", "-30Hz", "-12%", SKELETON_PHRASE, "Skeleton: British Tomb Gravel"),
        ("edge_skeleton_3_connor.mp3", "en-IE-ConnorNeural", "-20Hz", "-15%", SKELETON_PHRASE, "Skeleton: Irish Crypt Resonant"),
        ("edge_witch_1_ana.mp3", "en-US-AnaNeural", "+45Hz", "+12%", WITCH_PHRASE, "Wicked Witch: Shrill Oz Cackle"),
        ("edge_witch_2_sonia.mp3", "en-GB-SoniaNeural", "+40Hz", "+15%", WITCH_PHRASE, "Wicked Witch: Theatrical British Hag"),
        ("edge_witch_3_ava.mp3", "en-US-AvaNeural", "+35Hz", "+10%", WITCH_PHRASE, "Wicked Witch: Fast Raspy Cackle"),
    ]

    print("\n[1/3] Generating Edge-TTS Neural Mockups (Free Cloud Streaming)...")
    for filename, voice, pitch, rate, phrase, desc in edge_configs:
        path = os.path.join(out_dir, filename)
        print(f"  • {desc:<40} -> {filename}")
        generate_edge_tts(voice, pitch, rate, phrase, path)

    # 2. Piper TTS Mockups (Local On-Device)
    print("\n[2/3] Generating Piper TTS Mockups (100% Local Offline Neural Engine)...")
    piper_alan = "models/piper/en_GB-alan-medium.onnx"
    piper_lessac = "models/piper/en_US-lessac-medium.onnx"

    if os.path.exists(piper_alan):
        path = os.path.join(out_dir, "piper_skeleton_alan.wav")
        print(f"  • {'Skeleton: Piper Alan (Local ONNX)':<40} -> piper_skeleton_alan.wav")
        generate_piper_tts(piper_alan, length_scale=1.2, noise_scale=0.8, text=SKELETON_PHRASE, output_file=path)

    if os.path.exists(piper_lessac):
        path = os.path.join(out_dir, "piper_witch_lessac.wav")
        print(f"  • {'Wicked Witch: Piper Lessac (Local ONNX)':<40} -> piper_witch_lessac.wav")
        generate_piper_tts(piper_lessac, length_scale=0.85, noise_scale=0.9, text=WITCH_PHRASE, output_file=path)

    # 3. Copy Gemini Prebuilt Samples for direct comparison
    print("\n[3/3] Copying Gemini Prebuilt References...")
    gemini_samples = [
        ("samples/voices/charon.wav", "gemini_skeleton_charon.wav", "Skeleton: Gemini Charon (Cloud Generative)"),
        ("samples/voices/aoede.wav", "gemini_witch_aoede.wav", "Wicked Witch: Gemini Aoede (Cloud Generative)"),
    ]
    for src, dst_name, desc in gemini_samples:
        if os.path.exists(src):
            dst = os.path.join(out_dir, dst_name)
            subprocess.run(["cp", src, dst], check=True)
            print(f"  • {desc:<40} -> {dst_name}")

    print("\n" + "=" * 70)
    print(f" All mockups generated successfully in '{out_dir}/'")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
