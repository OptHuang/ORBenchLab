from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import harbor_model_matrix
from orbenchlab.cli import main


PROVIDER = {
    "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding",
    "ANTHROPIC_AUTH_TOKEN": "fixture-provider-secret",
}


def _task(root: Path) -> Path:
    task = root / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        '[task]\nname = "terminal-bench-science/demo-task"\n',
        encoding="utf-8",
    )
    return task


def _claude(root: Path) -> Path:
    executable = root / "claude"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _harbor(root: Path, *, trajectory: bool = True) -> Path:
    executable = root / "harbor"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
model = value('--model')
repetitions = int(value('--n-attempts'))
job = pathlib.Path(value('--jobs-dir')) / value('--job-name')
print(os.environ.get('ANTHROPIC_AUTH_TOKEN', 'missing'))
for attempt in range(1, repetitions + 1):
    trial = job / ('demo-task__' + str(attempt))
    trial.joinpath('verifier').mkdir(parents=True)
    trial.joinpath('agent').mkdir(parents=True)
    reward = 1.0 if model == 'frontier' else 0.0
    exception = None if reward else {{'exception_type': 'NonZeroAgentExitCodeError'}}
    passed = 2 if reward else 0
    failed = 0 if reward else 2
    trial.joinpath('result.json').write_text(json.dumps({{
        'task_name': 'terminal-bench-science/demo-task',
        'exception_info': exception,
        'verifier_result': {{'rewards': {{'reward': reward}}}},
    }}))
    trial.joinpath('verifier/reward.txt').write_text(str(reward) + '\\n')
    trial.joinpath('verifier/ctrf.json').write_text(json.dumps({{
        'results': {{'summary': {{'tests': 2, 'passed': passed, 'failed': failed,
        'skipped': 0, 'pending': 0, 'other': 0}}}}
    }}))
    if {trajectory!r}:
        trial.joinpath('agent/trajectory.json').write_text(json.dumps({{
            'schema_version': 'ATIF-v1.0', 'steps': [{{'step_id': 1}}]
        }}))
job.mkdir(parents=True, exist_ok=True)
job.joinpath('result.json').write_text(json.dumps({{
    'id': model + '-job', 'n_total_trials': repetitions,
    'stats': {{'n_completed_trials': repetitions,
    'n_errored_trials': 0 if model == 'frontier' else repetitions}}
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_launches_reuses_and_validates_rectangular_atif_matrix(tmp_path: Path):
    out = tmp_path / "out"
    first = harbor_model_matrix.launch_matrix(
        _task(tmp_path),
        harbor_executable=_harbor(tmp_path),
        claude_executable=_claude(tmp_path),
        out=out,
        models=["frontier", "weak"],
        repetitions=2,
        provider_env=PROVIDER,
        max_budget_usd=0.5,
        max_turns=10,
        timeout_sec=5,
    )
    second = harbor_model_matrix.launch_matrix(
        tmp_path / "task",
        harbor_executable=tmp_path / "harbor",
        claude_executable=tmp_path / "claude",
        out=out,
        models=["frontier", "weak"],
        repetitions=2,
        provider_env=PROVIDER,
        max_budget_usd=0.5,
        max_turns=10,
        timeout_sec=5,
    )
    assert first["receipt_digest"] == second["receipt_digest"]
    assert first["rectangular"] is True
    assert len(first["trials"]) == 4
    assert {row["status"] for row in first["trials"]} == {"pass", "fail"}
    assert {
        row["agent_failure_class"] for row in first["trials"] if row["model_id"] == "weak"
    } == {"NonZeroAgentExitCodeError"}
    assert all(row["trajectories"][0]["steps"] == 1 for row in first["trials"])
    assert len(list((out / "jobs").glob("*/result.json"))) == 2
    evidence = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in out.rglob("*")
        if path.is_file()
    )
    assert "fixture-provider-secret" not in evidence
    assert "[REDACTED_PROVIDER_CREDENTIAL]" in evidence


def test_missing_atif_trajectory_fails_closed(tmp_path: Path):
    with pytest.raises(harbor_model_matrix.HarborModelMatrixError, match="trajectory"):
        harbor_model_matrix.launch_matrix(
            _task(tmp_path),
            harbor_executable=_harbor(tmp_path, trajectory=False),
            claude_executable=_claude(tmp_path),
            out=tmp_path / "out",
            models=["frontier"],
            repetitions=1,
            provider_env=PROVIDER,
            timeout_sec=5,
        )


def test_cli_runs_real_harbor_matrix_contract(tmp_path: Path, monkeypatch, capsys):
    task = _task(tmp_path)
    harbor = _harbor(tmp_path)
    claude = _claude(tmp_path)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", PROVIDER["ANTHROPIC_BASE_URL"])
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", PROVIDER["ANTHROPIC_AUTH_TOKEN"])
    code = main(
        [
            "harbor-model-matrix",
            "--task-dir",
            str(task),
            "--harbor-executable",
            str(harbor),
            "--claude-executable",
            str(claude),
            "--model",
            "frontier",
            "--repetitions",
            "1",
            "--max-budget-usd",
            "0.5",
            "--max-turns",
            "10",
            "--timeout-sec",
            "5",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["rectangular"] is True
    assert payload["evidence_level"] == "E3"
    assert payload["checkpoint_capability"] is False
    assert "fixture-provider-secret" not in json.dumps(payload)
