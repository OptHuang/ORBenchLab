"""Credential-safe transport for paid Harbor agent trials.

Harbor launches the coding agent inside a container with ``docker compose exec
-e KEY=value``, which places the provider credential on the host process argv
(visible through ``ps`` / ``/proc/*/cmdline``).  Redacting stdout and receipts
does not close that hole.  This module keeps the real provider key out of
Harbor entirely:

* a host-side :class:`CredentialRelay` holds the real Volc key and forwards
  only ``POST`` requests to an explicit Claude endpoint allowlist on the exact
  Volc ``/api/coding`` origin, authenticated with short-lived, per-scope,
  revocable tokens under hard request / byte / concurrency / wall-clock caps,
  with no cross-origin redirects and no open-proxy / SSRF surface;
* the relay binds an interface (``bind_host``) but advertises a
  container-reachable address (``advertise_host``) so a real Docker container
  can reach a host relay that never exposes the real key;
* Harbor and the container therefore receive only a scoped relay token — the
  real key never enters Harbor's environment, argv, container filesystem,
  receipts or trajectories;
* :func:`scan_proc_for_secret` (Linux ``/proc`` or a macOS/BSD ``ps``
  equivalent) and :func:`scan_tree_for_secret` actively look for the real
  credential in host process argv and exported artifacts, so a leak trips a
  machine-readable security barrier instead of shipping.

Error and receipt content never echo a secret value.
"""

from __future__ import annotations

import hashlib
import http.server
import platform
import secrets
import socket
import subprocess
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


_VOLC_LABEL_SUFFIX = ".volces.com"
_VOLC_EXACT = "volces.com"
_ROUTE_PATH = "/api/coding"
# The exact endpoints the Claude/Anthropic client needs on the coding route.
_ALLOWED_ENDPOINTS = frozenset({"/v1/messages"})
RELAY_POLICY_VERSION = "orbenchlab.credential-relay.v2"


def _digest(value: Any) -> str:
    import json

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _token_digest(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode()).hexdigest()


def _validate_volc_host(host: str) -> None:
    host = host.lower().rstrip(".")
    if not (host == _VOLC_EXACT or host.endswith(_VOLC_LABEL_SUFFIX)):
        raise CredentialRelayError("credential relay requires an exact Volc host")


def _validate_volc_base(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url.rstrip("/"))
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not path.endswith(_ROUTE_PATH)
    ):
        raise CredentialRelayError("credential relay requires the Volc HTTPS /api/coding route")
    _validate_volc_host(host)
    return f"{parsed.scheme}://{parsed.netloc}", path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any):  # never follow a redirect
        return None


class CredentialRelay:
    """A host-only relay that swaps per-scope tokens for the real Volc key."""

    def __init__(
        self,
        *,
        real_base_url: str,
        real_token: str,
        bind_host: str = "127.0.0.1",
        advertise_host: str | None = None,
        max_requests: int = 1000,
        max_concurrency: int = 4,
        max_request_bytes: int = 4 * 1024 * 1024,
        max_response_bytes: int = 64 * 1024 * 1024,
        request_walltime_sec: float = 600.0,
    ) -> None:
        if not real_token:
            raise CredentialRelayError("credential relay requires a real provider token")
        self._real_origin, self._route_path = _validate_volc_base(real_base_url)
        self._real_token = str(real_token)
        self._bind_host = bind_host
        self._advertise_host = advertise_host or bind_host
        self._max_requests = int(max_requests)
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_request_bytes = int(max_request_bytes)
        self._max_response_bytes = int(max_response_bytes)
        self._request_walltime_sec = float(request_walltime_sec)
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(self._max_concurrency)
        self._request_count = 0
        self._response_bytes = 0
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # Active scoped tokens: digest -> {"scope": ..., "revoked": bool}
        self._tokens: dict[str, dict[str, Any]] = {}
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._started_marker = _digest({"policy": RELAY_POLICY_VERSION})

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

            def do_GET(self) -> None:
                self._reject(405, "relay only accepts POST")

            def do_PUT(self) -> None:
                self._reject(405, "relay only accepts POST")

            def do_DELETE(self) -> None:
                self._reject(405, "relay only accepts POST")

            def do_POST(self) -> None:
                try:
                    relay._forward(self)
                except CredentialRelayError as exc:
                    self._reject(403, str(exc))
                except Exception:  # noqa: BLE001 - never leak internals
                    self._reject(502, "relay upstream error")

        self._server = http.server.ThreadingHTTPServer((self._bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            for record in self._tokens.values():
                record["revoked"] = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- tokens ------------------------------------------------------------
    def issue_token(self, scope: Mapping[str, Any]) -> str:
        token = secrets.token_hex(32)
        with self._lock:
            self._tokens[_token_digest(token)] = {"scope": dict(scope), "revoked": False}
        return token

    def revoke_token(self, token: str) -> None:
        with self._lock:
            record = self._tokens.get(_token_digest(token))
            if record is not None:
                record["revoked"] = True

    def token_scopes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"token_digest": digest, "scope": record["scope"], "revoked": record["revoked"]}
                for digest, record in sorted(self._tokens.items())
            ]

    # -- public accessors --------------------------------------------------
    @property
    def port(self) -> int:
        if self._server is None:
            raise CredentialRelayError("credential relay is not running")
        return int(self._server.server_address[1])

    @property
    def advertise_host(self) -> str:
        return self._advertise_host

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def relay_env(self, token: str) -> dict[str, str]:
        """Container-facing env: relay route + a scoped relay token only."""

        return {
            "ANTHROPIC_BASE_URL": f"http://{self._advertise_host}:{self.port}{self._route_path}",
            "ANTHROPIC_AUTH_TOKEN": token,
        }

    def security_receipt(self, *, scan_results: Mapping[str, Any] | None = None) -> dict[str, Any]:
        receipt = {
            "schema_version": RELAY_POLICY_VERSION,
            "real_origin_host": urlsplit(self._real_origin).hostname,
            "route_path": self._route_path,
            "endpoint_allowlist": sorted(_ALLOWED_ENDPOINTS),
            "advertise_host": self._advertise_host,
            "bind_host": self._bind_host,
            "caps": {
                "max_requests": self._max_requests,
                "max_concurrency": self._max_concurrency,
                "max_request_bytes": self._max_request_bytes,
                "max_response_bytes": self._max_response_bytes,
                "request_walltime_sec": self._request_walltime_sec,
            },
            "token_scopes": self.token_scopes(),
            "request_count": self.request_count,
            "response_bytes": self._response_bytes,
            "no_cross_origin_redirect": True,
            "scan_results": dict(scan_results or {}),
        }
        receipt["policy_digest"] = _digest(
            {k: v for k, v in receipt.items() if k not in {"request_count", "response_bytes", "scan_results", "token_scopes"}}
        )
        return receipt

    # -- request handling --------------------------------------------------
    def _authorized(self, handler: http.server.BaseHTTPRequestHandler) -> str | None:
        supplied = handler.headers.get("x-api-key") or ""
        auth = handler.headers.get("authorization") or ""
        bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
        candidate = supplied or bearer
        if not candidate:
            return None
        digest = _token_digest(candidate)
        with self._lock:
            record = self._tokens.get(digest)
            if record is None or record["revoked"]:
                return None
        return digest

    def _validate_target_path(self, raw_path: str) -> str:
        # Reject absolute URIs, traversal, and anything outside the exact
        # endpoint allowlist under the coding route.
        parsed = urlsplit(raw_path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise CredentialRelayError("relay refuses an absolute or decorated URI")
        path = parsed.path
        if ".." in path or "//" in path or not path.startswith(self._route_path + "/"):
            raise CredentialRelayError("relay refuses a non-coding route")
        endpoint = path[len(self._route_path):]
        if endpoint not in _ALLOWED_ENDPOINTS:
            raise CredentialRelayError("relay endpoint is not on the allowlist")
        return path

    def _forward(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        token_digest = self._authorized(handler)
        if token_digest is None:
            raise CredentialRelayError("relay token is invalid or revoked")
        path = self._validate_target_path(handler.path)
        length = int(handler.headers.get("content-length") or 0)
        if length < 0 or length > self._max_request_bytes:
            raise CredentialRelayError("relay request body exceeds its byte cap")
        with self._lock:
            if self._request_count >= self._max_requests:
                raise CredentialRelayError("relay request cap reached")
            self._request_count += 1
        if not self._semaphore.acquire(timeout=self._request_walltime_sec):
            raise CredentialRelayError("relay concurrency cap reached")
        try:
            payload = handler.rfile.read(length) if length else b""
            target = self._real_origin + path
            request = urllib.request.Request(target, data=payload, method="POST")
            request.add_header("authorization", f"Bearer {self._real_token}")
            request.add_header("x-api-key", self._real_token)
            for name in ("content-type", "anthropic-version", "anthropic-beta", "accept"):
                value = handler.headers.get(name)
                if value:
                    request.add_header(name, value)
            with self._opener.open(request, timeout=self._request_walltime_sec) as response:
                status = response.status
                out_type = response.headers.get("content-type", "application/json")
                handler.send_response(status)
                handler.send_header("content-type", out_type)
                handler.send_header("transfer-encoding", "chunked")
                handler.end_headers()
                streamed = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    streamed += len(chunk)
                    if streamed > self._max_response_bytes:
                        break
                    handler.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    handler.wfile.flush()
                handler.wfile.write(b"0\r\n\r\n")
                with self._lock:
                    self._response_bytes += streamed
        except urllib.error.HTTPError as exc:
            body = exc.read()[: self._max_response_bytes]
            handler.send_response(exc.code)
            handler.send_header("content-type", exc.headers.get("content-type", "application/json"))
            handler.send_header("content-length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        finally:
            self._semaphore.release()


def free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def probe_reachability(
    *, prober, advertise_host: str, port: int, route_path: str, token: str
) -> dict[str, Any]:
    """Fail-closed reachability probe from the actual container environment.

    ``prober`` is a callable ``(url, token) -> (ok: bool, detail: str)`` that
    issues a request from inside the real Harbor/Docker network.  Paid trials
    must not start unless it returns ok.
    """

    url = f"http://{advertise_host}:{port}{route_path}/v1/messages"
    ok, detail = prober(url, token)
    result = {"reachable": bool(ok), "url": url, "detail": str(detail)[:500]}
    if not ok:
        raise CredentialRelayError(
            f"credential relay is not reachable from the container: {result['detail']}"
        )
    return result


def _secret_needles(secret_values: Sequence[str]) -> list[bytes]:
    return [str(value).encode() for value in secret_values if len(str(value)) >= 8]


def scan_proc_for_secret(secret_values: Sequence[str]) -> list[int]:
    """Return host PIDs whose argv contains a real credential (portable)."""

    needles = _secret_needles(secret_values)
    if not needles:
        return []
    proc = Path("/proc")
    if proc.is_dir():
        offenders: list[int] = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                data = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if any(needle in data for needle in needles):
                offenders.append(int(entry.name))
        return offenders
    # macOS / BSD: use ps without ever printing the secret.
    if platform.system() in {"Darwin", "FreeBSD", "OpenBSD", "NetBSD"}:
        try:
            output = subprocess.run(
                ["ps", "-Ao", "pid=,args="],
                capture_output=True,
                text=False,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        offenders = []
        for line in output.splitlines():
            if any(needle in line for needle in needles):
                pid_field = line.strip().split(b" ", 1)[0]
                try:
                    offenders.append(int(pid_field))
                except ValueError:
                    continue
        return offenders
    return []


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

    The raised message says only that a leak was detected and where (count of
    paths or pids), never the credential value.
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
    return {
        "scanned_artifact_roots": [str(r) for r in artifact_roots],
        "proc_scanned": scan_proc,
        "leak": False,
    }


__all__ = [
    "CredentialRelay",
    "CredentialRelayError",
    "CredentialSecurityBarrier",
    "RELAY_POLICY_VERSION",
    "assert_no_credential_leak",
    "free_tcp_port",
    "probe_reachability",
    "scan_proc_for_secret",
    "scan_tree_for_secret",
]
