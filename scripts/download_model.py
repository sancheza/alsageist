#!/usr/bin/env python3
"""scripts/download_model.py - Download Piper TTS and Vosk STT models.

Downloads Piper ONNX neural voice models and Vosk offline speech recognition
models into the local models/ directory without requiring git-lfs.

Usage:
    python3 scripts/download_model.py
    python3 scripts/download_model.py --voice alan
    python3 scripts/download_model.py --vosk
    python3 scripts/download_model.py --all
    python3 scripts/download_model.py --dry-run
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

VERSION = "1.1.0"

VOICE_URLS = {
    "alan": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        "target_onnx": "models/piper/en_GB-alan-medium.onnx",
        "target_json": "models/piper/en_GB-alan-medium.onnx.json",
    },
    "lessac": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "target_onnx": "models/piper/en_US-lessac-medium.onnx",
        "target_json": "models/piper/en_US-lessac-medium.onnx.json",
    },
}

VOSK_CONFIG = {
    "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "target_dir": "models/vosk-model-small-en-us-0.15",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line options.
    """
    parser = argparse.ArgumentParser(
        description="Download Piper TTS neural voice and Vosk STT models."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--voice",
        choices=["alan", "lessac"],
        default="alan",
        help="Voice model to download (default: alan).",
    )
    parser.add_argument(
        "--vosk",
        action="store_true",
        help="Download and extract offline Vosk speech recognition model.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both Piper neural voice model and Vosk STT model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display actions without downloading files.",
    )
    return parser.parse_args()


def download_file(url: str, target_path: str, dry_run: bool = False) -> None:
    """Download a remote file to a local destination.

    Args:
        url: Remote file URL.
        target_path: Local path where the file should be saved.
        dry_run: When True, prints action without downloading.
    """
    if os.path.exists(target_path):
        print(f"File already exists: {target_path}")
        return

    print(f"Downloading {url} -> {target_path}")
    if dry_run:
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(url, target_path)
    print(f"Successfully saved {target_path}")


def download_and_extract_vosk(dry_run: bool = False) -> None:
    """Download and extract offline Vosk Kaldi model archive.

    Args:
        dry_run: When True, prints action without downloading.
    """
    target_dir = VOSK_CONFIG["target_dir"]
    url = VOSK_CONFIG["url"]

    if os.path.isdir(target_dir):
        print(f"Vosk model directory already exists: {target_dir}")
        return

    print(f"Downloading {url} -> {target_dir}")
    if dry_run:
        return

    os.makedirs("models", exist_ok=True)
    temp_zip = os.path.join("models", "vosk_temp.zip")
    try:
        urllib.request.urlretrieve(url, temp_zip)
        print(f"Extracting {temp_zip} into models/...")
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall("models")
        print(f"Successfully installed Vosk model to {target_dir}")
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)


def main() -> int:
    """Execute model download workflow.

    Returns:
        int: Process exit code (0 for success).
    """
    args = parse_args()

    # Determine what to download
    download_voice_model = True
    download_vosk_model = args.vosk or args.all

    if args.vosk and not args.all:
        download_voice_model = False

    if download_voice_model:
        config = VOICE_URLS[args.voice]
        download_file(config["onnx"], config["target_onnx"], dry_run=args.dry_run)
        download_file(config["json"], config["target_json"], dry_run=args.dry_run)

    if download_vosk_model:
        download_and_extract_vosk(dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())

