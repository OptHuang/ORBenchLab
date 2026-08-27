from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from orbenchlab import agent_sessions
from orbenchlab.agent_sessions import AgentSessionError, run_session
from orbenchlab.cli import main


VOLC = "https://ark.cn-beijing.volces.com/api/coding"


def _concurrent_session(kwargs: dict, queue) -> None:
    queue.put(run_session(**kwargs))


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
        max_budget_usd=0.25,
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
        max_budget_usd=0.25,
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
    assert receipt["usage"] == {
        "input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    assert "fixture-secret" not in json.dumps(receipt)


def test_claude_argv_enables_bounded_noninteractive_coding_tools(tmp_path: Path):
    executable = _fixture_cli(tmp_path, 'printf \'fixture-output\'')
    result = run_session(
        profile="claude-code",
        stage="repair",
        model="fixture-model",
        prompt="repair",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=2,
        max_budget_usd=0.25,
        environ=_env("claude-code"),
        executable=executable,
    )
    argv = result["identity"]["argv_template"]
    tools = "Read,Glob,Grep,Edit,Write,Bash"
    assert argv[argv.index("--tools") + 1] == tools
    assert argv[argv.index("--allowedTools") + 1] == tools
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.25"
    assert result["identity"]["max_budget_usd"] == 0.25
    assert result["budget"] == {
        "max_budget_usd": 0.25,
        "enforcement": "claude-cli-max-budget-usd",
        "hard_enforced_by_cli": True,
    }
    assert "--safe-mode" in argv
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv
    assert "--add-dir" not in argv
    assert any("not an OS filesystem sandbox" in row for row in result["limitations"])

    restricted = run_session(
        profile="claude-code",
        stage="paper-factory",
        model="fixture-model",
        prompt="author without shell",
        workdir=tmp_path,
        out=tmp_path / "restricted-sessions",
        timeout_sec=2,
        max_budget_usd=0.25,
        environ=_env("claude-code"),
        executable=executable,
        allow_bash=False,
    )
    restricted_argv = restricted["identity"]["argv_template"]
    assert restricted_argv[restricted_argv.index("--tools") + 1] == "Read,Glob,Grep,Edit,Write"
    assert restricted["identity"]["bash_tool_enabled"] is False


def test_linux_read_only_wrapper_binds_exact_input_tree(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"
    inputs = work / "factory-input"
    inputs.mkdir(parents=True)
    inputs.joinpath("receipt.json").write_text('{"bound":true}\n', encoding="utf-8")
    bubblewrap = _fixture_cli(tmp_path, "exit 0")
    monkeypatch.setattr(agent_sessions.sys, "platform", "linux")
    monkeypatch.setattr(
        agent_sessions.shutil,
        "which",
        lambda name: str(bubblewrap) if name == "bwrap" else None,
    )
    command, contract = agent_sessions._read_only_command(
        ["/bin/true"], cwd=work, paths=[inputs]
    )
    assert command[:5] == [
        str(bubblewrap.resolve()),
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
    ]
    input_bind = max(index for index, value in enumerate(command) if value == "--ro-bind")
    assert command[input_bind + 1 : input_bind + 3] == [
        str(inputs.resolve()),
        str(inputs.resolve()),
    ]
    assert contract == {
        "kind": "bubblewrap-read-only-bindings-v1",
        "policy": "root-ro-workdir-rw-protected-ro-private-tmp-v1",
        "executable_digest": agent_sessions._digest_bytes(bubblewrap.read_bytes()),
        "read_only_bindings": [
            {
                "path": "factory-input",
                "content_digest": agent_sessions._tree_digest(inputs),
            }
        ],
        "hard_enforced": True,
    }


def test_provider_credential_is_redacted_from_live_and_sealed_output(tmp_path: Path):
    token = "fixture-secret-long-enough"
    executable = _fixture_cli(
        tmp_path,
        f"printf '%s' '{token[:11]}'\nprintf '%s' '{token[11:]}'\nprintf '%s' '{token}' >&2",
    )
    result = run_session(
        profile="claude-code",
        stage="credential-attack",
        model="fixture-model",
        prompt="print environment",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=2,
        max_budget_usd=0.25,
        environ={
            "ANTHROPIC_BASE_URL": VOLC,
            "ANTHROPIC_AUTH_TOKEN": token,
        },
        executable=executable,
    )
    session = Path(result["receipt_path"]).parent
    evidence = b"\n".join(
        path.read_bytes()
        for path in (session / "stdout.bin", session / "stderr.bin")
    )
    assert token.encode() not in evidence
    assert b"[REDACTED_PROVIDER_CREDENTIAL]" in evidence
    assert result["provider_credential_redacted"] is True


def test_streaming_redactor_holds_a_split_credential_prefix():
    redactor = agent_sessions._StreamingSecretRedactor([b"0123456789abcdef"])
    first = redactor.feed(b"safe-01234567")
    second = redactor.feed(b"89abcdef-tail", final=True)
    assert first + second == b"safe-[REDACTED_PROVIDER_CREDENTIAL]-tail"
    ordinary = agent_sessions._StreamingSecretRedactor([b"fixture-secret"])
    assert ordinary.feed(b"first") == b"first"


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
        max_budget_usd=0.25,
        environ=_env("claude-code"),
        executable=executable,
    )
    assert result["status"] == "failed"
    assert result["failure_class"] == "wall_clock_timeout"
    assert Path(result["receipt_path"]).is_file()


def test_output_limit_kills_process_group_without_unbounded_capture(tmp_path: Path):
    executable = _fixture_cli(tmp_path, "while :; do printf 'fixture-output\\n'; done")
    result = run_session(
        profile="codex",
        stage="scaffold",
        model="fixture-model",
        prompt="bounded",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=5,
        max_budget_usd=0.25,
        max_output_bytes=2048,
        environ=_env("codex"),
        executable=executable,
    )
    assert result["status"] == "failed"
    assert result["failure_class"] == "output_limit_exceeded"
    assert result["captured_output_bytes"] <= 2048
    session = Path(result["receipt_path"]).parent
    assert (session / "stdout.bin").stat().st_size + (session / "stderr.bin").stat().st_size <= 2048


def test_same_session_is_serialised_across_processes_and_reused(tmp_path: Path):
    marker = tmp_path / "executions"
    executable = _fixture_cli(
        tmp_path,
        f"printf x >> {marker}\nsleep 0.25\nprintf done",
    )
    kwargs = {
        "profile": "codex",
        "stage": "paper-derive",
        "model": "fixture-model",
        "prompt": "derive",
        "workdir": tmp_path,
        "out": tmp_path / "sessions",
        "timeout_sec": 3,
        "max_budget_usd": 0.25,
        "environ": _env("codex"),
        "executable": executable,
    }
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=_concurrent_session, args=(kwargs, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert marker.read_text(encoding="utf-8") == "x"
    assert sorted(result["reused"] for result in results) == [False, True]


def test_running_session_exposes_bounded_live_trace_then_seals_it(tmp_path: Path):
    executable = _fixture_cli(tmp_path, "printf first; sleep 0.4; printf second")
    out = tmp_path / "sessions"
    kwargs = {
        "profile": "codex",
        "stage": "monitored",
        "model": "fixture-model",
        "prompt": "monitor",
        "workdir": tmp_path,
        "out": out,
        "timeout_sec": 3,
        "max_budget_usd": 0.25,
        "environ": _env("codex"),
        "executable": executable,
    }
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_concurrent_session, args=(kwargs, queue))
    process.start()
    deadline = time.monotonic() + 2
    observed = None
    while time.monotonic() < deadline:
        paths = list(out.glob("*/stdout.live"))
        if paths and paths[0].read_bytes().startswith(b"first"):
            observed = paths[0]
            break
        time.sleep(0.01)
    assert observed is not None
    result = queue.get(timeout=5)
    process.join(timeout=5)
    assert process.exitcode == 0
    session = Path(result["receipt_path"]).parent
    assert not (session / "stdout.live").exists()
    assert (session / "stdout.bin").read_bytes() == b"firstsecond"
    assert result["live_monitoring"]["hint_injection_supported"] is False


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
            max_budget_usd=0.25,
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
            max_budget_usd=0.25,
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
            "--max-budget-usd", "0.25",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "completed"
    assert "fixture-secret" not in json.dumps(payload)


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan"), 101, True])
def test_session_rejects_missing_or_unsafe_budget_before_launch(tmp_path: Path, value):
    executable = _fixture_cli(tmp_path, "printf launched > should-not-exist")
    with pytest.raises(AgentSessionError, match="max_budget_usd"):
        run_session(
            profile="codex",
            stage="reviewer",
            model="fixture-model",
            prompt="review",
            workdir=tmp_path,
            out=tmp_path / "sessions",
            timeout_sec=1,
            max_budget_usd=value,
            environ=_env("codex"),
            executable=executable,
        )
    assert not (tmp_path / "should-not-exist").exists()


def test_codex_records_that_dollar_budget_is_not_cli_enforced(tmp_path: Path):
    executable = _fixture_cli(tmp_path, "printf done")
    result = run_session(
        profile="codex",
        stage="reviewer",
        model="fixture-model",
        prompt="review",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=1,
        max_budget_usd=0.25,
        environ=_env("codex"),
        executable=executable,
    )
    assert result["budget"]["hard_enforced_by_cli"] is False
    assert result["budget"]["enforcement"] == "unsupported-codex-cli"
    assert "--max-budget-usd" not in result["identity"]["argv_template"]
    assert result["usage_parser"]["status"] == "unsupported"


def test_claude_usage_comes_only_from_complete_final_result(tmp_path: Path):
    executable = _fixture_cli(
        tmp_path,
        "printf '%s\\n' "
        "'{\"type\":\"assistant\",\"message\":{\"content\":\"do-not-copy-this\"}}' "
        "'{\"type\":\"result\",\"subtype\":\"error_max_budget_usd\","
        "\"total_cost_usd\":1.062661,\"usage\":{\"input_tokens\":11,"
        "\"cache_creation_input_tokens\":22,\"cache_read_input_tokens\":33,"
        "\"output_tokens\":44},\"result\":\"also-do-not-copy\"}'\nexit 1",
    )
    result = run_session(
        profile="claude-code",
        stage="paper-derive",
        model="fixture-model",
        prompt="derive",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=2,
        max_budget_usd=0.25,
        environ=_env("claude-code"),
        executable=executable,
    )
    assert result["status"] == "failed"
    assert result["usage"] == {
        "input_tokens": 11,
        "cache_creation_input_tokens": 22,
        "cache_read_input_tokens": 33,
        "output_tokens": 44,
        "cost_usd": 1.062661,
    }
    assert result["usage_parser"] == {
        "protocol": "claude-stream-json-final-result-v1",
        "status": "parsed",
        "result_subtype": "error_max_budget_usd",
    }
    receipt = Path(result["receipt_path"]).read_text(encoding="utf-8")
    assert "do-not-copy-this" not in receipt
    assert "also-do-not-copy" not in receipt


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ('printf \'%s\\n\' \'{"type":"assistant"}\'', "incomplete"),
        (
            'printf \'%s\\n\' \'{"type":"result","total_cost_usd":NaN,'
            '"usage":{"input_tokens":1,"output_tokens":2}}\'',
            "invalid",
        ),
        ("printf 'not-json\\n'", "invalid"),
    ],
)
def test_claude_incomplete_or_invalid_usage_stays_unknown(
    tmp_path: Path, body: str, status: str
):
    executable = _fixture_cli(tmp_path, body)
    result = run_session(
        profile="claude-code",
        stage="reviewer",
        model="fixture-model",
        prompt="review",
        workdir=tmp_path,
        out=tmp_path / "sessions",
        timeout_sec=1,
        max_budget_usd=0.25,
        environ=_env("claude-code"),
        executable=executable,
    )
    assert all(value is None for value in result["usage"].values())
    assert result["usage_parser"]["status"] == status
