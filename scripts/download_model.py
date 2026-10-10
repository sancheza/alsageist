#!/usr/bin/env python3
"""scripts/download_model.py - Download Piper TTS neural voice models.

Downloads the Piper ONNX neural voice model and configuration file into
the local models/ directory without requiring git-lfs.

Usage:
    python3 scripts/download_model.py
    python3 scripts/download_model.py --voice alan
    python3 scripts/download_model.py --dry-run
"""

import argparse
import os
import sys
import urllib.request

VERSION = "1.0.0"

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line options.
    """
    parser = argparse.ArgumentParser(
        description="Download Piper TTS neural voice models."
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


def main() -> int:
    """Execute model download workflow.

    Returns:
        int: Process exit code (0 for success).
    """
    args = parse_args()
    config = VOICE_URLS[args.voice]

    download_file(config["onnx"], config["target_onnx"], dry_run=args.dry_run)
    download_file(config["json"], config["target_json"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
