"""Unit tests for scripts/download_model.py."""

import os
import subprocess
import sys
import pytest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "download_model.py"))


def test_download_model_help():
    """Verify -h / --help prints usage and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Download Piper TTS neural voice and Vosk Speech-to-Text" in result.stdout
    assert "--vosk" in result.stdout
    assert "--all" in result.stdout


def test_download_model_version():
    """Verify -v / --version prints script version and exits with 0."""
    result = subprocess.run([SCRIPT_PATH, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1.2.0" in result.stdout


def test_download_model_dry_run_voice():
    """Verify --dry-run prints voice download URLs without downloading."""
    result = subprocess.run([SCRIPT_PATH, "--voice", "alan", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "en_GB-alan-medium.onnx" in result.stdout
    assert "en_GB-alan-medium.onnx.json" in result.stdout
    assert "vosk" not in result.stdout


def test_download_model_dry_run_vosk():
    """Verify --vosk --dry-run prints Vosk download actions."""
    result = subprocess.run([SCRIPT_PATH, "--vosk", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "vosk-model-small-en-us-0.15" in result.stdout
    assert "piper" not in result.stdout


def test_download_model_dry_run_all():
    """Verify --all --dry-run downloads both voice model and Vosk model."""
    result = subprocess.run([SCRIPT_PATH, "--all", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "en_GB-alan-medium.onnx" in result.stdout
    assert "vosk-model-small-en-us-0.15" in result.stdout


# ---- Integrity verification (in-process, served from a local HTTP server) ----

import hashlib
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts import download_model as dm


@pytest.fixture
def file_server():
    """Serve a dict of {path: bytes} from a local HTTP server."""
    files = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = files.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield files, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_pinned_digests_are_well_formed():
    """Every pinned digest is a 64-character lowercase hex SHA-256."""
    digests = [dm.VOSK_CONFIG["sha256"]]
    for config in dm.VOICE_URLS.values():
        digests += [config["sha256_onnx"], config["sha256_json"]]
    for digest in digests:
        assert len(digest) == 64 and digest == digest.lower()
        int(digest, 16)


def test_download_file_installs_verified_content(file_server, tmp_path):
    """A download whose digest matches is installed and no .part file remains."""
    files, base = file_server
    files["/voice.onnx"] = b"voice-bytes"
    target = tmp_path / "piper" / "voice.onnx"

    dm.download_file(f"{base}/voice.onnx", str(target), _sha(b"voice-bytes"))

    assert target.read_bytes() == b"voice-bytes"
    assert not (tmp_path / "piper" / "voice.onnx.part").exists()


def test_download_file_rejects_tampered_content(file_server, tmp_path):
    """A digest mismatch raises and leaves nothing installed."""
    files, base = file_server
    files["/voice.onnx"] = b"tampered"
    target = tmp_path / "voice.onnx"

    with pytest.raises(dm.ModelDownloadError, match="Checksum mismatch"):
        dm.download_file(f"{base}/voice.onnx", str(target), _sha(b"original"))

    assert not target.exists()
    assert not (tmp_path / "voice.onnx.part").exists()


def test_download_file_reports_http_errors(file_server, tmp_path):
    """A missing remote file raises ModelDownloadError instead of leaking urllib errors."""
    _, base = file_server
    with pytest.raises(dm.ModelDownloadError, match="Download failed"):
        dm.download_file(f"{base}/missing.onnx", str(tmp_path / "x.onnx"), _sha(b"x"))


def test_existing_file_is_verified_not_trusted(tmp_path):
    """An existing file with the wrong digest is an error; the right digest is accepted."""
    target = tmp_path / "voice.onnx"
    target.write_bytes(b"corrupt")

    with pytest.raises(dm.ModelDownloadError, match="does not match"):
        dm.download_file("http://unused.invalid/x", str(target), _sha(b"good"))
    assert target.read_bytes() == b"corrupt"  # never overwritten silently

    dm.download_file("http://unused.invalid/x", str(target), _sha(b"corrupt"))


def test_zip_extraction_rejects_path_traversal(tmp_path):
    """An archive member that escapes the destination is refused before anything is written."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ok/file.txt", "fine")
        zf.writestr("../escaped.txt", "bad")
    dest = tmp_path / "dest"
    dest.mkdir()

    with pytest.raises(dm.ModelDownloadError, match="Unsafe path"):
        dm.extract_zip_safely(str(archive), str(dest))

    assert not (tmp_path / "escaped.txt").exists()
    assert list(dest.iterdir()) == []


def test_vosk_download_verifies_and_extracts(file_server, tmp_path, monkeypatch):
    """The Vosk archive is verified, extracted into the models dir, and temp files are removed."""
    files, base = file_server
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("vosk-test-model/am/final.mdl", "model")
    files["/vosk.zip"] = buffer.getvalue()
    monkeypatch.setattr(dm, "VOSK_CONFIG", {
        "url": f"{base}/vosk.zip",
        "target_dir": "vosk-test-model",
        "sha256": _sha(buffer.getvalue()),
    })

    dm.download_and_extract_vosk(str(tmp_path))

    assert (tmp_path / "vosk-test-model" / "am" / "final.mdl").read_text() == "model"
    assert [p.name for p in tmp_path.iterdir()] == ["vosk-test-model"]


def test_vosk_download_rejects_wrong_digest(file_server, tmp_path, monkeypatch):
    """A Vosk archive with the wrong digest is never extracted."""
    files, base = file_server
    files["/vosk.zip"] = b"not the pinned archive"
    monkeypatch.setattr(dm, "VOSK_CONFIG", {
        "url": f"{base}/vosk.zip",
        "target_dir": "vosk-test-model",
        "sha256": _sha(b"the pinned archive"),
    })

    with pytest.raises(dm.ModelDownloadError, match="Checksum mismatch"):
        dm.download_and_extract_vosk(str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_main_returns_nonzero_on_integrity_failure(tmp_path, monkeypatch, capsys):
    """main() prints the error to stderr and exits 1 when verification fails."""
    bad = tmp_path / "piper" / "en_GB-alan-medium.onnx"
    bad.parent.mkdir()
    bad.write_bytes(b"corrupt")
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--voice", "alan", "--models-dir", str(tmp_path)])

    assert dm.main() == 1
    assert "does not match" in capsys.readouterr().err


def test_piper_urls_are_pinned_to_a_commit():
    """Voice URLs name a repository commit, so the pinned digests do not drift with main."""
    for config in dm.VOICE_URLS.values():
        for key in ("onnx", "json"):
            assert f"/resolve/{dm._PIPER_COMMIT}/" in config[key]
    assert len(dm._PIPER_COMMIT) == 40
    int(dm._PIPER_COMMIT, 16)
