from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orbenchlab.agent_sessions import AgentSessionError, run_session
from orbenchlab.cli import main


VOLC = "https://ark.cn-beijing.volces.com/api/coding"


def _fixture_cli(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agent-fixture"
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _env(profile: str) -> dict[str, str]:
    if profile == "claude-code":
        return {
            "ANTHROPIC_BASE_URL": VOLC,
            "ANTHROPIC_AUTH_TOKEN": "fixture-secret",
        }
    return {"OPENAI_BASE_URL": VOLC, "OPENAI_API_KEY": "fixture-secret"}


@pytest.mark.parametrize("profile", ["codex", "claude-code"])
def test_session_writes_bound_receipt_and_reuses_success(tmp_path: Path, profile: str):
    executable = _fixture_cli(tmp_path, 'printf \'fixture-output\'')
    out = tmp_path / "sessions"
    first = run_session(
        profile=profile,
        stage="paper-derive",
        model="fixture-model",
        prompt="derive this task",
        workdir=tmp_path,
        out=out,
        timeout_sec=5,
        environ=_env(profile),
        executable=executable,
    )
    second = run_session(
        profile=profile,
        stage="paper-derive",
        model="fixture-model",
        prompt="derive this task",
        workdir=tmp_path,
        out=out,
        timeout_sec=5,
        environ=_env(profile),
        executable=executable,
    )
    receipt = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert first["status"] == "completed"
    assert second["reused"] is True
    assert receipt["session_id"] == first["session_id"]
    assert receipt["stdout_digest"].startswith("sha256:")
    assert receipt["stderr_digest"].startswith("sha256:")
    assert receipt["trace_digest"].startswith("sha256:")
    assert receipt["usage"] == {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    assert "fixture-secret" not in json.dumps(receipt)


def test_timeout_is_hard_failure_with_atomic_receipt(tmp_path: Path):
    executable = _fixture_cli(tmp_path, "sleep 5")
    result = run_session(
        profile="claude-code",
        stage="repair",
        model="fixture-model",
        prompt="repair",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=0.05,
        environ=_env("claude-code"),
        executable=executable,
    )
    assert result["status"] == "failed"
    assert result["failure_class"] == "wall_clock_timeout"
    assert Path(result["receipt_path"]).is_file()


def test_route_and_environment_fail_closed(tmp_path: Path):
    executable = _fixture_cli(tmp_path, "exit 0")
    with pytest.raises(AgentSessionError, match="Volc"):
        run_session(
            profile="codex",
            stage="reviewer",
            model="fixture-model",
            prompt="review",
            workdir=tmp_path,
            out=tmp_path / "sessions",
            timeout_sec=1,
            environ={"OPENAI_BASE_URL": "https://example.test/v1", "OPENAI_API_KEY": "x"},
            executable=executable,
        )
    poisoned = _env("codex") | {"LD_PRELOAD": "/tmp/not-allowed"}
    with pytest.raises(AgentSessionError, match="environment"):
        run_session(
            profile="codex",
            stage="reviewer",
            model="fixture-model",
            prompt="review",
            workdir=tmp_path,
            out=tmp_path / "sessions",
            timeout_sec=1,
            environ=poisoned,
            executable=executable,
        )


def test_cli_runs_fixture_profile_without_provider_call(tmp_path: Path, monkeypatch, capsys):
    executable = _fixture_cli(tmp_path, 'printf \'cli-output\'')
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("scaffold", encoding="utf-8")
    monkeypatch.setenv("OPENAI_BASE_URL", VOLC)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    code = main(
        [
            "agent-session", "run", "--profile", "codex", "--stage", "scaffold",
            "--model", "fixture-model", "--prompt-file", str(prompt), "--workdir",
            str(tmp_path), "--out", str(tmp_path / "sessions"), "--timeout-sec", "2",
            "--executable", str(executable),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "completed"
    assert "fixture-secret" not in json.dumps(payload)
