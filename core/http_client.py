"""Minimal JSON and multipart HTTP client built on urllib.

Every request has a timeout, so a stalled connection cannot block the caller
indefinitely, and every request identifies itself with an AlsaGeist User-Agent
(some API providers reject urllib's default one). Network and HTTP errors
propagate as urllib exceptions.
"""

import json
from typing import Any, Dict, Mapping, Optional, Tuple
import urllib.request
import uuid

from core.config import VERSION

DEFAULT_TIMEOUT_SEC = 10.0
USER_AGENT = f"alsageist/{VERSION}"

# (filename, content type, raw bytes)
FileField = Tuple[str, str, bytes]


def _send(request: urllib.request.Request, timeout: float) -> Any:
    """Send a prepared request and decode the JSON response body."""
    if not request.has_header("User-agent"):
        request.add_unredirected_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(
    url: str,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Any:
    """Send a GET request and decode the JSON response.

    Args:
        url: Request URL.
        headers: Optional request headers.
        timeout: Seconds to wait for the connection and each read.

    Returns:
        Decoded JSON response body.
    """
    request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    return _send(request, timeout)


def post_json(
    url: str,
    payload: Any,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Any:
    """Send a JSON POST request and decode the JSON response.

    Args:
        url: Request URL.
        payload: JSON-serializable request body.
        headers: Optional request headers.
        timeout: Seconds to wait for the connection and each read.

    Returns:
        Decoded JSON response body.
    """
    all_headers = {"Content-Type": "application/json", **dict(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=all_headers,
        method="POST",
    )
    return _send(request, timeout)


def post_multipart(
    url: str,
    fields: Mapping[str, str],
    files: Mapping[str, FileField],
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Any:
    """Send a multipart/form-data POST request and decode the JSON response.

    Args:
        url: Request URL.
        fields: Plain form fields.
        files: File fields mapped to (filename, content type, bytes).
        headers: Optional request headers.
        timeout: Seconds to wait for the connection and each read.

    Returns:
        Decoded JSON response body.
    """
    boundary = f"----AlsaGeistBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode("utf-8")
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        body += f"{value}\r\n".encode("utf-8")
    for name, (filename, content_type, data) in files.items():
        body += f"--{boundary}\r\n".encode("utf-8")
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        body += f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        body += data
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")

    all_headers: Dict[str, str] = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **dict(headers or {}),
    }
    request = urllib.request.Request(url, data=bytes(body), headers=all_headers, method="POST")
    return _send(request, timeout)
