from __future__ import annotations

import http.server
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from orbenchlab import harbor_credentials

VOLC = "https://ark.cn-beijing.volces.com/api/coding"
REAL_TOKEN = "REAL-volc-secret-do-not-leak-abcdef0123456789"


class _FakeVolc(http.server.BaseHTTPRequestHandler):
    seen_auth: list[str] = []

    def log_message(self, *_):
        return

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        _FakeVolc.seen_auth.append(self.headers.get("authorization", ""))
        body = json.dumps({"type": "result", "ok": True}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _relay_to_fake(monkeypatch, upstream_port: int) -> harbor_credentials.CredentialRelay:
    # Point the relay's "real" origin at a local fake upstream instead of Volc,
    # bypassing only the origin used for forwarding (route validation stays).
    relay = harbor_credentials.CredentialRelay(
        real_base_url=VOLC, real_token=REAL_TOKEN
    )
    relay._real_origin = f"http://127.0.0.1:{upstream_port}"
    return relay


def test_relay_forwards_with_real_token_and_hides_it_from_client(monkeypatch):
    _FakeVolc.seen_auth = []
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeVolc)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        relay = _relay_to_fake(monkeypatch, upstream.server_address[1])
        with relay:
            env = relay.relay_env()
            # The container-facing env never contains the real token.
            assert env["ANTHROPIC_AUTH_TOKEN"] == relay.relay_token
            assert REAL_TOKEN not in json.dumps(env)
            assert env["ANTHROPIC_BASE_URL"].endswith("/api/coding")
            request = urllib.request.Request(
                env["ANTHROPIC_BASE_URL"] + "/v1/messages",
                data=b"{}",
                method="POST",
                headers={"x-api-key": relay.relay_token, "content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert json.loads(response.read())["ok"] is True
            assert relay.request_count == 1
        # The real token reached the upstream but never the client env.
        assert any(REAL_TOKEN in auth for auth in _FakeVolc.seen_auth)
    finally:
        upstream.shutdown()


def test_relay_rejects_wrong_token_and_non_volc_route(monkeypatch):
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeVolc)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        relay = _relay_to_fake(monkeypatch, upstream.server_address[1])
        with relay:
            base = f"http://127.0.0.1:{relay.port}"
            bad_token = urllib.request.Request(
                base + "/api/coding/v1/messages",
                data=b"{}",
                method="POST",
                headers={"x-api-key": "wrong-token"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(bad_token, timeout=5)
            assert exc.value.code == 403
            bad_route = urllib.request.Request(
                base + "/not-coding/v1/messages",
                data=b"{}",
                method="POST",
                headers={"x-api-key": relay.relay_token},
            )
            with pytest.raises(urllib.error.HTTPError) as exc2:
                urllib.request.urlopen(bad_route, timeout=5)
            assert exc2.value.code == 403
    finally:
        upstream.shutdown()


def test_relay_enforces_request_cap(monkeypatch):
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeVolc)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        relay = harbor_credentials.CredentialRelay(
            real_base_url=VOLC, real_token=REAL_TOKEN, max_requests=1
        )
        relay._real_origin = f"http://127.0.0.1:{upstream.server_address[1]}"
        with relay:
            url = relay.relay_env()["ANTHROPIC_BASE_URL"] + "/v1/messages"
            headers = {"x-api-key": relay.relay_token, "content-type": "application/json"}
            urllib.request.urlopen(
                urllib.request.Request(url, data=b"{}", method="POST", headers=headers), timeout=5
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(
                    urllib.request.Request(url, data=b"{}", method="POST", headers=headers),
                    timeout=5,
                )
            assert exc.value.code == 403
    finally:
        upstream.shutdown()


def test_relay_requires_a_volc_https_route():
    with pytest.raises(harbor_credentials.CredentialRelayError):
        harbor_credentials.CredentialRelay(
            real_base_url="https://evil.example/api/coding", real_token=REAL_TOKEN
        )
    with pytest.raises(harbor_credentials.CredentialRelayError):
        harbor_credentials.CredentialRelay(
            real_base_url="http://ark.cn-beijing.volces.com/api/coding", real_token=REAL_TOKEN
        )


def test_scan_tree_and_barrier_detect_leak_without_echoing_value(tmp_path: Path):
    (tmp_path / "clean.json").write_text('{"token": "relay-only"}', encoding="utf-8")
    assert harbor_credentials.scan_tree_for_secret(tmp_path, [REAL_TOKEN]) == []
    result = harbor_credentials.assert_no_credential_leak(
        secret_values=[REAL_TOKEN], artifact_roots=[tmp_path], scan_proc=False
    )
    assert result["leak"] is False
    (tmp_path / "leaked.json").write_text(
        json.dumps({"key": REAL_TOKEN}), encoding="utf-8"
    )
    offenders = harbor_credentials.scan_tree_for_secret(tmp_path, [REAL_TOKEN])
    assert offenders == ["leaked.json"]
    with pytest.raises(harbor_credentials.CredentialSecurityBarrier) as exc:
        harbor_credentials.assert_no_credential_leak(
            secret_values=[REAL_TOKEN], artifact_roots=[tmp_path], scan_proc=False
        )
    # The barrier message names where, never the secret value.
    assert REAL_TOKEN not in str(exc.value)
    assert "leak detected" in str(exc.value)


def test_scan_proc_finds_a_credential_in_argv(tmp_path: Path):
    import subprocess
    import sys

    # A short-lived process whose argv contains the fake secret must be found.
    proc = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(2)  # {REAL_TOKEN}"])
    try:
        offenders = harbor_credentials.scan_proc_for_secret([REAL_TOKEN])
        assert proc.pid in offenders
    finally:
        proc.kill()
        proc.wait()
