#!/usr/bin/env python3
"""scripts/download_model.py - Download Piper TTS and Vosk STT models.

Downloads Piper ONNX neural voice models and Vosk offline speech recognition
models into the models/ directory without requiring git-lfs. Every file is
checked against a pinned SHA-256 digest before it is installed.

Usage:
    python3 scripts/download_model.py
    python3 scripts/download_model.py --voice alan
    python3 scripts/download_model.py --vosk
    python3 scripts/download_model.py --all
    python3 scripts/download_model.py --models-dir /srv/alsageist/models
    python3 scripts/download_model.py --dry-run
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile

VERSION = "1.2.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "models")
REQUEST_TIMEOUT_SEC = 60
CHUNK_BYTES = 1024 * 1024

# Pinned to a commit of the piper-voices repository, so the digests below stay valid
# when the repository's main branch changes.
_PIPER_COMMIT = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
_PIPER_BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/{_PIPER_COMMIT}/en"

# Paths are relative to the models directory. SHA-256 digests of the .onnx files are the
# Hugging Face LFS digests; the .json digests match the published git blob hashes.
VOICE_URLS = {
    "alan": {
        "onnx": f"{_PIPER_BASE}/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "json": f"{_PIPER_BASE}/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        "target_onnx": "piper/en_GB-alan-medium.onnx",
        "target_json": "piper/en_GB-alan-medium.onnx.json",
        "sha256_onnx": "0a309668932205e762801f1efc2736cd4b0120329622adf62be09e56339d3330",
        "sha256_json": "c0f0d124e5895c00e7c03b35dcc8287f319a6998a365b182deb5c8e752ee8c1e",
    },
    "lessac": {
        "onnx": f"{_PIPER_BASE}/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": f"{_PIPER_BASE}/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "target_onnx": "piper/en_US-lessac-medium.onnx",
        "target_json": "piper/en_US-lessac-medium.onnx.json",
        "sha256_onnx": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        "sha256_json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    },
}

VOSK_CONFIG = {
    "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "target_dir": "vosk-model-small-en-us-0.15",
    "sha256": "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498",
}


class ModelDownloadError(RuntimeError):
    """Raised when a model file cannot be downloaded or fails verification."""


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
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory that receives the models (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display actions without downloading files.",
    )
    return parser.parse_args()


def sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(url: str, expected_sha256: str, dest_path: str) -> None:
    """Download a URL to dest_path, keeping the file only if its SHA-256 matches.

    Args:
        url: Remote file URL.
        expected_sha256: Expected lowercase hex digest.
        dest_path: Path written only after verification succeeds.

    Raises:
        ModelDownloadError: If the download fails or the digest differs.
    """
    part_path = f"{dest_path}.part"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SEC) as response, open(part_path, "wb") as out:
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ModelDownloadError(
                f"Checksum mismatch for {url}\n  expected {expected_sha256}\n  actual   {actual}"
            )
        os.replace(part_path, dest_path)
    except OSError as exc:
        raise ModelDownloadError(f"Download failed for {url}: {exc}") from exc
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)


def download_file(url: str, target_path: str, expected_sha256: str, dry_run: bool = False) -> None:
    """Download a remote file to a local destination and verify it.

    An existing file is verified instead of downloaded again.

    Args:
        url: Remote file URL.
        target_path: Local path where the file should be saved.
        expected_sha256: Expected lowercase hex digest.
        dry_run: When True, prints action without downloading.

    Raises:
        ModelDownloadError: If an existing file fails verification, or the download fails.
    """
    if os.path.exists(target_path):
        if dry_run:
            print(f"File already exists: {target_path}")
            return
        if sha256_file(target_path) != expected_sha256:
            raise ModelDownloadError(
                f"{target_path} does not match the expected SHA-256 ({expected_sha256}).\n"
                "  Delete the file and run this script again."
            )
        print(f"File already exists and is verified: {target_path}")
        return

    print(f"Downloading {url} -> {target_path}")
    if dry_run:
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fetch_verified(url, expected_sha256, target_path)
    print(f"Successfully saved {target_path} (SHA-256 verified)")


def extract_zip_safely(archive_path: str, dest_dir: str) -> None:
    """Extract a zip archive, refusing members that would land outside dest_dir.

    Args:
        archive_path: Zip file to extract.
        dest_dir: Directory that receives the contents.

    Raises:
        ModelDownloadError: If any member path escapes dest_dir.
    """
    root = os.path.realpath(dest_dir)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            target = os.path.realpath(os.path.join(root, member))
            if target != root and not target.startswith(root + os.sep):
                raise ModelDownloadError(f"Unsafe path in archive: {member}")
        archive.extractall(root)


def download_and_extract_vosk(models_dir: str, dry_run: bool = False) -> None:
    """Download, verify and extract the offline Vosk Kaldi model archive.

    Args:
        models_dir: Directory that receives the extracted model.
        dry_run: When True, prints action without downloading.

    Raises:
        ModelDownloadError: If the download, verification or extraction fails.
    """
    target_dir = os.path.join(models_dir, VOSK_CONFIG["target_dir"])
    url = VOSK_CONFIG["url"]

    if os.path.isdir(target_dir):
        print(f"Vosk model directory already exists: {target_dir}")
        return

    print(f"Downloading {url} -> {target_dir}")
    if dry_run:
        return

    os.makedirs(models_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir, prefix=".vosk_") as scratch:
        archive_path = os.path.join(scratch, "vosk.zip")
        fetch_verified(url, VOSK_CONFIG["sha256"], archive_path)
        print(f"Extracting verified archive into {models_dir}...")
        extract_zip_safely(archive_path, models_dir)

    if not os.path.isdir(target_dir):
        raise ModelDownloadError(f"Archive did not contain the expected directory {target_dir}")
    print(f"Successfully installed Vosk model to {target_dir}")


def main() -> int:
    """Execute model download workflow.

    Returns:
        int: Process exit code (0 for success, 1 for a download or verification failure).
    """
    args = parse_args()
    models_dir = os.path.abspath(args.models_dir)

    # Determine what to download
    download_voice_model = True
    download_vosk_model = args.vosk or args.all

    if args.vosk and not args.all:
        download_voice_model = False

    try:
        if download_voice_model:
            config = VOICE_URLS[args.voice]
            download_file(
                config["onnx"],
                os.path.join(models_dir, config["target_onnx"]),
                config["sha256_onnx"],
                dry_run=args.dry_run,
            )
            download_file(
                config["json"],
                os.path.join(models_dir, config["target_json"]),
                config["sha256_json"],
                dry_run=args.dry_run,
            )

        if download_vosk_model:
            download_and_extract_vosk(models_dir, dry_run=args.dry_run)
    except ModelDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
