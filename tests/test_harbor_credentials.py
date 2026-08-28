from __future__ import annotations

import http.server
import json
import threading
import urllib.error
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
        # Emit a short SSE-like streamed body.
        chunks = [b'data: {"type":"message_start"}\n\n', b'data: {"type":"message_stop"}\n\n']
        body = b"".join(chunks)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _fake_upstream():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeVolc)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _relay_to(server) -> harbor_credentials.CredentialRelay:
    relay = harbor_credentials.CredentialRelay(real_base_url=VOLC, real_token=REAL_TOKEN)
    relay._real_origin = f"http://127.0.0.1:{server.server_address[1]}"
    return relay


def test_relay_forwards_with_real_token_and_hides_it_from_client():
    _FakeVolc.seen_auth = []
    upstream = _fake_upstream()
    try:
        with _relay_to(upstream) as relay:
            token = relay.issue_token({"job": "j1"})
            env = relay.relay_env(token)
            assert env["ANTHROPIC_AUTH_TOKEN"] == token
            assert REAL_TOKEN not in json.dumps(env)
            assert env["ANTHROPIC_BASE_URL"].endswith("/api/coding")
            request = urllib.request.Request(
                env["ANTHROPIC_BASE_URL"] + "/v1/messages",
                data=b"{}", method="POST",
                headers={"x-api-key": token, "content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert b"message_stop" in response.read()
            assert relay.request_count == 1
        assert any(REAL_TOKEN in auth for auth in _FakeVolc.seen_auth)
    finally:
        upstream.shutdown()


def test_relay_rejects_wrong_token_revoked_token_and_bad_routes():
    upstream = _fake_upstream()
    try:
        with _relay_to(upstream) as relay:
            token = relay.issue_token({"job": "j1"})
            base = f"http://127.0.0.1:{relay.port}"
            def post(path, tok):
                return urllib.request.urlopen(
                    urllib.request.Request(base + path, data=b"{}", method="POST",
                                           headers={"x-api-key": tok}), timeout=5)
            for path, tok, code in [
                ("/api/coding/v1/messages", "wrong", 403),
                ("/api/codingevil/v1/messages", token, 403),  # not on allowlist
                ("/api/coding/v1/complete", token, 403),      # endpoint not allowed
                ("/not-coding/v1/messages", token, 403),
            ]:
                with pytest.raises(urllib.error.HTTPError) as exc:
                    post(path, tok)
                assert exc.value.code == code
            # GET is refused.
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(base + "/api/coding/v1/messages", timeout=5)
            assert exc.value.code in (403, 405)
            # A revoked token is rejected.
            relay.revoke_token(token)
            with pytest.raises(urllib.error.HTTPError) as exc:
                post("/api/coding/v1/messages", token)
            assert exc.value.code == 403
    finally:
        upstream.shutdown()


def test_relay_enforces_request_cap():
    upstream = _fake_upstream()
    try:
        relay = harbor_credentials.CredentialRelay(
            real_base_url=VOLC, real_token=REAL_TOKEN, max_requests=1
        )
        relay._real_origin = f"http://127.0.0.1:{upstream.server_address[1]}"
        with relay:
            token = relay.issue_token({"job": "j1"})
            url = relay.relay_env(token)["ANTHROPIC_BASE_URL"] + "/v1/messages"
            headers = {"x-api-key": token, "content-type": "application/json"}
            urllib.request.urlopen(
                urllib.request.Request(url, data=b"{}", method="POST", headers=headers), timeout=5)
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(
                    urllib.request.Request(url, data=b"{}", method="POST", headers=headers), timeout=5)
            assert exc.value.code == 403
    finally:
        upstream.shutdown()


def test_relay_rejects_lookalike_and_non_https_hosts():
    for bad in (
        "https://evilvolces.com/api/coding",   # suffix lookalike
        "https://volces.com.evil.net/api/coding",
        "http://ark.cn-beijing.volces.com/api/coding",  # not https
        "https://ark.cn-beijing.volces.com/api/coding?x=1",  # query
    ):
        with pytest.raises(harbor_credentials.CredentialRelayError):
            harbor_credentials.CredentialRelay(real_base_url=bad, real_token=REAL_TOKEN)


def test_bind_and_advertise_hosts_are_separable():
    relay = harbor_credentials.CredentialRelay(
        real_base_url=VOLC, real_token=REAL_TOKEN,
        bind_host="127.0.0.1", advertise_host="172.17.0.1",
    )
    with relay:
        env = relay.relay_env(relay.issue_token({"job": "j"}))
        assert env["ANTHROPIC_BASE_URL"].startswith("http://172.17.0.1:")
        assert relay.advertise_host == "172.17.0.1"


def test_reachability_probe_fails_closed():
    relay = harbor_credentials.CredentialRelay(real_base_url=VOLC, real_token=REAL_TOKEN)
    with relay:
        token = relay.issue_token({"job": "j"})
        with pytest.raises(harbor_credentials.CredentialRelayError, match="not reachable"):
            harbor_credentials.probe_reachability(
                prober=lambda url, tok: (False, "connection refused"),
                advertise_host=relay.advertise_host, port=relay.port,
                route_path="/api/coding", token=token,
            )
        result = harbor_credentials.probe_reachability(
            prober=lambda url, tok: (True, "ok"),
            advertise_host=relay.advertise_host, port=relay.port,
            route_path="/api/coding", token=token,
        )
        assert result["reachable"] is True


def test_security_receipt_has_no_secret():
    relay = harbor_credentials.CredentialRelay(real_base_url=VOLC, real_token=REAL_TOKEN)
    with relay:
        relay.issue_token({"job": "j1", "model": "m"})
        receipt = relay.security_receipt()
    dumped = json.dumps(receipt)
    assert REAL_TOKEN not in dumped
    assert receipt["endpoint_allowlist"] == ["/v1/messages"]
    assert receipt["token_scopes"][0]["scope"]["model"] == "m"
    assert receipt["policy_digest"].startswith("sha256:")
    assert "token_digest" in receipt["token_scopes"][0]


def test_scan_tree_and_barrier_detect_leak_without_echoing_value(tmp_path: Path):
    (tmp_path / "clean.json").write_text('{"token": "relay-only"}', encoding="utf-8")
    assert harbor_credentials.scan_tree_for_secret(tmp_path, [REAL_TOKEN]) == []
    (tmp_path / "leaked.json").write_text(json.dumps({"key": REAL_TOKEN}), encoding="utf-8")
    assert harbor_credentials.scan_tree_for_secret(tmp_path, [REAL_TOKEN]) == ["leaked.json"]
    with pytest.raises(harbor_credentials.CredentialSecurityBarrier) as exc:
        harbor_credentials.assert_no_credential_leak(
            secret_values=[REAL_TOKEN], artifact_roots=[tmp_path], scan_proc=False)
    assert REAL_TOKEN not in str(exc.value)
    assert "leak detected" in str(exc.value)


def test_scan_proc_finds_a_credential_in_argv_portably():
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(3)  # {REAL_TOKEN}"])
    try:
        # Portable across Linux (/proc) and macOS/BSD (ps).
        import time as _t
        found = []
        for _ in range(20):
            found = harbor_credentials.scan_proc_for_secret([REAL_TOKEN])
            if proc.pid in found:
                break
            _t.sleep(0.1)
        assert proc.pid in found
    finally:
        proc.kill()
        proc.wait()
