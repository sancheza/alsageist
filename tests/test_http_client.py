"""Unit tests for core/http_client.py using a local HTTP server."""

import json
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core import http_client


class _Handler(BaseHTTPRequestHandler):
    """Echo handler recording the last request and answering with JSON."""

    last = {}

    def _respond(self, status=200, body=None):
        data = json.dumps(body if body is not None else {"ok": True}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/slow":
            time.sleep(1.0)
        _Handler.last = {"method": "GET", "path": self.path, "headers": dict(self.headers)}
        self._respond(500 if self.path == "/error" else 200)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Handler.last = {"method": "POST", "path": self.path, "headers": dict(self.headers), "body": body}
        self._respond()

    def log_message(self, *args):
        pass


@pytest.fixture
def server_url():
    """Run a local HTTP server on a free port for the duration of a test."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_get_json_sends_headers(server_url):
    """GET returns decoded JSON and forwards custom headers."""
    result = http_client.get_json(f"{server_url}/x", headers={"User-Agent": "test-agent"})
    assert result == {"ok": True}
    assert _Handler.last["headers"]["User-Agent"] == "test-agent"


def test_post_json_encodes_payload_and_content_type(server_url):
    """POST sends a JSON body with a JSON content type plus caller headers."""
    result = http_client.post_json(f"{server_url}/x", {"a": 1}, headers={"Authorization": "Bearer k"})
    assert result == {"ok": True}
    assert json.loads(_Handler.last["body"]) == {"a": 1}
    assert _Handler.last["headers"]["Content-Type"] == "application/json"
    assert _Handler.last["headers"]["Authorization"] == "Bearer k"


def test_post_multipart_contains_fields_and_file(server_url):
    """Multipart bodies carry the plain fields and the file bytes."""
    http_client.post_multipart(
        f"{server_url}/x",
        fields={"model": "whisper"},
        files={"file": ("audio.wav", "audio/wav", b"RIFFDATA")},
    )
    body = _Handler.last["body"]
    content_type = _Handler.last["headers"]["Content-Type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body and b"whisper" in body
    assert b'filename="audio.wav"' in body and b"RIFFDATA" in body


def test_http_error_status_propagates(server_url):
    """HTTP error statuses raise urllib.error.HTTPError."""
    with pytest.raises(urllib.error.HTTPError):
        http_client.get_json(f"{server_url}/error")


def test_timeout_is_enforced(server_url):
    """A response slower than the timeout raises instead of blocking."""
    start = time.time()
    with pytest.raises((socket.timeout, TimeoutError, urllib.error.URLError)):
        http_client.get_json(f"{server_url}/slow", timeout=0.2)
    assert time.time() - start < 0.9


def test_requests_identify_as_alsageist_by_default(server_url):
    """Without a caller-supplied User-Agent every request sends alsageist/<version>."""
    http_client.get_json(f"{server_url}/x")
    assert _Handler.last["headers"]["User-Agent"] == http_client.USER_AGENT
    assert http_client.USER_AGENT.startswith("alsageist/")
    assert "urllib" not in _Handler.last["headers"]["User-Agent"]

    http_client.post_json(f"{server_url}/x", {})
    assert _Handler.last["headers"]["User-Agent"] == http_client.USER_AGENT

    http_client.post_multipart(f"{server_url}/x", fields={}, files={"file": ("a.wav", "audio/wav", b"x")})
    assert _Handler.last["headers"]["User-Agent"] == http_client.USER_AGENT


def test_caller_user_agent_is_kept(server_url):
    """A User-Agent passed by the caller is not replaced."""
    http_client.get_json(f"{server_url}/x", headers={"User-Agent": "curl/7.88.1"})
    assert _Handler.last["headers"]["User-Agent"] == "curl/7.88.1"


def test_sensitive_headers_marked_unredirected():
    """Sensitive authentication headers are set as unredirected headers to avoid cross-host leakage."""
    req = http_client._build_request(
        "http://example.com/api",
        headers={"Authorization": "Bearer secret", "x-goog-api-key": "xyz", "Custom": "ok"},
    )
    # req.headers holds headers that urllib forwards across redirects. Sensitive ones must NOT be there.
    assert "Authorization" not in req.headers
    assert "X-goog-api-key" not in req.headers and "x-goog-api-key" not in req.headers
    # Custom non-sensitive header should be in req.headers
    assert "Custom" in req.headers
    # Sensitive tokens are safely in unredirected_hdrs
    assert "Authorization" in req.unredirected_hdrs
    assert "x-goog-api-key" in req.unredirected_hdrs or "X-goog-api-key" in req.unredirected_hdrs
