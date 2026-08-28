"""Credential-safe transport for paid Harbor agent trials.

Harbor launches the coding agent inside a container with ``docker compose exec
-e KEY=value``, which places the provider credential on the host process argv
(visible through ``ps`` / ``/proc/*/cmdline``).  Redacting stdout and receipts
does not close that hole.  This module keeps the real provider key out of
Harbor entirely:

* a host-side :class:`CredentialRelay` holds the real Volc key and forwards
  only requests on the fixed Volc ``/api/coding`` route, authenticated with a
  short-lived, single-run, revocable relay token, under a hard request cap and
  with no open-proxy / SSRF surface;
* Harbor and the container therefore receive only the relay token — the real
  key never enters Harbor's environment, argv, container filesystem, receipts
  or trajectories;
* :func:`scan_proc_for_secret` and :func:`scan_tree_for_secret` actively look
  for the real credential in host process argv and exported artifacts, so a
  leak trips a machine-readable security barrier instead of shipping.

Error messages never echo a secret value.
"""

from __future__ import annotations

import http.server
import os
import secrets
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .core.errors import ORBenchError


class CredentialRelayError(ORBenchError):
    exit_code = 9


class CredentialSecurityBarrier(ORBenchError):
    """A real provider credential was detected where it must never appear."""

    exit_code = 9


_VOLC_HOST_SUFFIXES = ("volces.com", "volcengine.com")
_ALLOWED_ROUTE = "/api/coding"


def _validate_volc_base(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url.rstrip("/"))
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or not (host.endswith(_VOLC_HOST_SUFFIXES))
        or not path.endswith(_ALLOWED_ROUTE)
    ):
        raise CredentialRelayError("credential relay requires the Volc HTTPS /api/coding route")
    return f"{parsed.scheme}://{parsed.netloc}", path


class CredentialRelay:
    """A host-only relay that swaps a single-run token for the real Volc key."""

    def __init__(
        self,
        *,
        real_base_url: str,
        real_token: str,
        host: str = "127.0.0.1",
        max_requests: int = 10_000,
        upstream_timeout_sec: float = 120.0,
    ) -> None:
        if not real_token:
            raise CredentialRelayError("credential relay requires a real provider token")
        self._real_origin, self._route_path = _validate_volc_base(real_base_url)
        self._real_token = str(real_token)
        self._host = host
        self._max_requests = int(max_requests)
        self._upstream_timeout_sec = float(upstream_timeout_sec)
        self._relay_token = secrets.token_hex(32)
        self._request_count = 0
        self._lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._revoked = False

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "CredentialRelay":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._server is not None:
            return
        relay = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:  # never log paths/secrets
                return

            def _reject(self, code: int, reason: str) -> None:
                body = reason.encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _forward(self) -> None:
                try:
                    relay._forward_request(self)
                except CredentialRelayError as exc:
                    self._reject(403, str(exc))
                except Exception:  # noqa: BLE001 - never leak internals
                    self._reject(502, "relay upstream error")

            do_POST = _forward
            do_GET = _forward

        self._server = http.server.ThreadingHTTPServer((self._host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._revoked = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- public accessors --------------------------------------------------
    @property
    def port(self) -> int:
        if self._server is None:
            raise CredentialRelayError("credential relay is not running")
        return int(self._server.server_address[1])

    @property
    def agent_host(self) -> str:
        return self._host

    @property
    def relay_token(self) -> str:
        return self._relay_token

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def relay_env(self) -> dict[str, str]:
        """Return the container-facing env: relay route + relay token only."""

        return {
            "ANTHROPIC_BASE_URL": f"http://{self._host}:{self.port}{self._route_path}",
            "ANTHROPIC_AUTH_TOKEN": self._relay_token,
        }

    # -- request handling --------------------------------------------------
    def _authorized(self, handler: http.server.BaseHTTPRequestHandler) -> bool:
        supplied = (
            handler.headers.get("x-api-key")
            or handler.headers.get("X-Api-Key")
            or ""
        )
        auth = handler.headers.get("authorization") or handler.headers.get("Authorization") or ""
        bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
        return secrets.compare_digest(supplied or bearer, self._relay_token)

    def _forward_request(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        if self._revoked:
            raise CredentialRelayError("relay token has been revoked")
        if not self._authorized(handler):
            raise CredentialRelayError("relay token is invalid")
        path = handler.path
        # SSRF guard: only the fixed Volc /api/coding route, no host override.
        if not path.startswith(self._route_path) or "://" in path or ".." in path:
            raise CredentialRelayError("relay refuses a non-Volc route")
        with self._lock:
            if self._request_count >= self._max_requests:
                raise CredentialRelayError("relay request cap reached")
            self._request_count += 1
        length = int(handler.headers.get("content-length") or 0)
        payload = handler.rfile.read(length) if length else b""
        target = self._real_origin + path
        request = urllib.request.Request(target, data=payload or None, method=handler.command)
        # Only a minimal, fixed header set crosses to the real provider; the
        # real token is attached here and never seen by the container.
        request.add_header("authorization", f"Bearer {self._real_token}")
        request.add_header("x-api-key", self._real_token)
        content_type = handler.headers.get("content-type")
        if content_type:
            request.add_header("content-type", content_type)
        anthropic_version = handler.headers.get("anthropic-version")
        if anthropic_version:
            request.add_header("anthropic-version", anthropic_version)
        try:
            with urllib.request.urlopen(request, timeout=self._upstream_timeout_sec) as response:
                body = response.read()
                status = response.status
                out_type = response.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status = exc.code
            out_type = exc.headers.get("content-type", "application/json")
        handler.send_response(status)
        handler.send_header("content-type", out_type)
        handler.send_header("content-length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


def free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _secret_needles(secret_values: Sequence[str]) -> list[bytes]:
    return [str(value).encode() for value in secret_values if len(str(value)) >= 8]


def scan_proc_for_secret(secret_values: Sequence[str]) -> list[int]:
    """Return host PIDs whose argv contains a real credential (best effort)."""

    needles = _secret_needles(secret_values)
    if not needles or not Path("/proc").is_dir():
        return []
    offenders: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = entry / "cmdline"
        try:
            data = cmdline.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            offenders.append(int(entry.name))
    return offenders


def scan_tree_for_secret(root: str | Path, secret_values: Sequence[str]) -> list[str]:
    """Return artifact paths under ``root`` that contain a real credential."""

    needles = _secret_needles(secret_values)
    base = Path(root)
    if not needles or not base.exists():
        return []
    offenders: list[str] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            offenders.append(path.relative_to(base).as_posix())
    return offenders


def assert_no_credential_leak(
    *,
    secret_values: Sequence[str],
    artifact_roots: Sequence[str | Path] = (),
    scan_proc: bool = True,
) -> dict[str, Any]:
    """Raise :class:`CredentialSecurityBarrier` if a real credential appears.

    The raised message says only that a leak was detected and where (path or
    pid), never the credential value.
    """

    leaked_files: list[str] = []
    for root in artifact_roots:
        for relative in scan_tree_for_secret(root, secret_values):
            leaked_files.append(f"{root}:{relative}")
    leaked_pids = scan_proc_for_secret(secret_values) if scan_proc else []
    if leaked_files or leaked_pids:
        where = []
        if leaked_files:
            where.append(f"{len(leaked_files)} artifact(s)")
        if leaked_pids:
            where.append(f"{len(leaked_pids)} host process argv")
        raise CredentialSecurityBarrier(
            "provider credential leak detected in " + ", ".join(where)
        )
    return {"scanned_artifact_roots": [str(r) for r in artifact_roots], "leak": False}


__all__ = [
    "CredentialRelay",
    "CredentialRelayError",
    "CredentialSecurityBarrier",
    "assert_no_credential_leak",
    "free_tcp_port",
    "scan_proc_for_secret",
    "scan_tree_for_secret",
]
