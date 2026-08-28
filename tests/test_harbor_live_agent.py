from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from orbenchlab import harbor_live_agent as hla


class FakeBackend:
    def __init__(self):
        self.built = []
        self.runs = []
        self.execs = []
        self.stopped = []
        self.reward = "1.0"
        self.ctrf = json.dumps({"summary": {"passed": 1}})

    def build_image(self, *, context_dir: Path, tag: str) -> str:
        self.built.append((context_dir, tag))
        return f"img:{tag}"

    def run_no_network(self, *, image: str, name: str, env):
        # A live container must never carry the provider credential.
        assert env == {}
        cid = f"cid-{name}"
        self.runs.append({"image": image, "name": name, "env": dict(env), "cid": cid})
        return cid

    def exec(self, *, container_id: str, argv, stdin: bytes = b""):
        self.execs.append((container_id, list(argv)))
        return hla.ExecOutcome(rc=0, stdout="ok", stderr="")

    def read_text(self, *, container_id: str, path: str):
        if path.endswith("reward.txt"):
            return self.reward
        if path.endswith("ctrf.json"):
            return self.ctrf
        return None

    def stop(self, *, container_id: str):
        self.stopped.append(container_id)


class FakeRelay:
    def __init__(self):
        self.tokens = []

    def issue_token(self, scope):
        self.tokens.append(scope)
        return "scoped-relay-token"

    def relay_env(self, token):
        return {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/api/coding", "ANTHROPIC_AUTH_TOKEN": token}


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "instruction.md").write_text("solve it")
    return task


def _make_executor(tmp_path, backend, *, secret="REAL-secret-xyz", session_runner):
    relay = FakeRelay()

    @contextmanager
    def relay_factory():
        yield relay

    return hla.build_arm_executor(
        task_dir=_task(tmp_path),
        backend=backend,
        relay_factory=relay_factory,
        claude_executable="/opt/claude",
        proxy_argv_prefix=["python", "-m", "orbenchlab.harbor_container_proxy"],
        verifier_argv=["/bin/sh", "/tests/test.sh"],
        model="doubao",
        max_turns=6,
        max_budget_usd=0.2,
        secret_values=[secret],
        session_runner=session_runner,
    ), relay


def test_arm_executor_is_credential_safe_and_joins_verifier(tmp_path: Path):
    backend = FakeBackend()
    secret = "REAL-secret-xyz"
    captured = {}

    def fake_session(**kwargs):
        # The credential must never reach the agent env; only relay creds do.
        env = kwargs["env"]
        captured["env"] = env
        captured["argv"] = kwargs["argv"]
        captured["identity"] = kwargs["harbor_identity"]
        assert secret not in json.dumps(env)
        assert env["ANTHROPIC_AUTH_TOKEN"] == "scoped-relay-token"
        return {
            "protocol_satisfied": True,
            "single_session": True,
            "harbor_identity": kwargs["harbor_identity"],
            "journal_digest": "sha256:" + "a" * 64,
            "error": None,
        }

    executor, relay = _make_executor(tmp_path, backend, secret=secret, session_runner=fake_session)
    out = executor("L1", 1, "iv-abc", tmp_path / "arm")

    # No-network container built and started with an empty env.
    assert backend.runs[0]["env"] == {}
    # Verifier ran in the SAME container and reward/CTRF were joined.
    assert out["reward"] == 1.0
    assert out["ctrf"]["summary"]["passed"] == 1
    assert out["verifier_container_id"] == captured["identity"]["container_id"]
    assert ("cid-orbench-live-iv-abc", ["/bin/sh", "/tests/test.sh"]) in backend.execs
    # Container is always torn down.
    assert backend.stopped == [out["verifier_container_id"]]
    # MCP config points the agent's tools at the container proxy for this cid.
    mcp = json.loads((tmp_path / "arm" / "mcp.json").read_text())
    args = mcp["mcpServers"]["orbench"]["args"]
    assert args[-1] == out["verifier_container_id"]
    # Built-in tools disabled; only proxy tools allowed.
    joined = " ".join(captured["argv"])
    assert "mcp__orbench__bash" in joined and "--disallowedTools" in joined


def test_arm_executor_tears_down_container_on_session_error(tmp_path: Path):
    backend = FakeBackend()

    def exploding_session(**kwargs):
        raise RuntimeError("session blew up")

    executor, _ = _make_executor(tmp_path, backend, session_runner=exploding_session)
    with pytest.raises(RuntimeError):
        executor("L1", 1, "iv-err", tmp_path / "arm")
    # Even on failure the container must be stopped (no leaked containers).
    assert backend.stopped == ["cid-orbench-live-iv-err"]
