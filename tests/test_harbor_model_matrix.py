from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import harbor_launcher, harbor_model_matrix
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
        'agent_result': {{'n_input_tokens': 10, 'n_cache_tokens': 2,
        'n_output_tokens': 3, 'cost_usd': 0.25}},
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
    assert all(row["usage"]["cost_usd"] == 0.25 for row in first["trials"])
    assert len(list((out / "jobs").glob("*/result.json"))) == 2
    assert len(list((out / "reservations").glob("*.json"))) == 2
    assert first["agent"]["max_job_attempts_per_model"] == 2
    assert first["agent"]["maximum_model_liability_usd"] == 4.0
    evidence = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in out.rglob("*")
        if path.is_file()
    )
    assert "fixture-provider-secret" not in evidence
    assert "[REDACTED_PROVIDER_CREDENTIAL]" in evidence

    bundle = harbor_model_matrix.write_trace_bundle(
        first,
        matrix_root=out,
        out=tmp_path / "trace-bundle",
        secret_values=[PROVIDER["ANTHROPIC_AUTH_TOKEN"]],
    )
    assert len(bundle["trajectories"]) == 4
    assert harbor_model_matrix.write_trace_bundle(
        first,
        matrix_root=out,
        out=tmp_path / "trace-bundle",
    )["manifest_digest"] == bundle["manifest_digest"]

    controls = {
        "schema_version": "orbenchlab.screening-report.v1",
        "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
        "task_tree_digest": first["task_tree_digest"],
        "tasks": [
            {
                "task": first["task"],
                "control_gates": {
                    "oracle": {"gate": "pass"},
                    "nop": {"gate": "pass"},
                },
            }
        ],
    }
    controls["report_digest"] = harbor_model_matrix._digest(controls)
    screening = harbor_model_matrix.build_screening_report(
        first,
        harbor_controls=controls,
        out=tmp_path / "screening",
    )
    assert screening["harbor_model_matrix_digest"] == first["receipt_digest"]
    assert screening["tasks"][0]["evidence_level"] == "E3"
    assert len(screening["trials"]) == 4
    assert screening["tasks"][0]["decision"] == "collect-more-evidence"


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


def test_crashed_harbor_jobs_consume_the_persisted_attempt_cap(tmp_path: Path):
    harbor = tmp_path / "harbor"
    harbor.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    harbor.chmod(0o755)
    task = _task(tmp_path)
    claude = _claude(tmp_path)
    kwargs = dict(
        task_dir=task,
        harbor_executable=harbor,
        claude_executable=claude,
        out=tmp_path / "out",
        models=["frontier"],
        repetitions=5,
        provider_env=PROVIDER,
        max_budget_usd=0.5,
        max_job_attempts=2,
        timeout_sec=5,
    )
    for _ in range(2):
        with pytest.raises(harbor_launcher.HarborLauncherError, match="nonzero_exit"):
            harbor_model_matrix.launch_matrix(**kwargs)
    reservations = list((tmp_path / "out/reservations").glob("*.json"))
    assert len(reservations) == 2
    assert sum(json.loads(path.read_text())["reserved_liability_usd"] for path in reservations) == 5.0
    with pytest.raises(harbor_model_matrix.HarborModelMatrixError, match="attempt cap"):
        harbor_model_matrix.launch_matrix(**kwargs)
    assert len(list((tmp_path / "out/reservations").glob("*.json"))) == 2


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


def _flaky_harbor(root: Path) -> Path:
    """First frontier job leaves one trial without verifier evidence."""

    executable = root / "flaky-harbor"
    marker = root / "flaky-marker"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
model = value('--model')
repetitions = int(value('--n-attempts'))
job = pathlib.Path(value('--jobs-dir')) / value('--job-name')
marker = pathlib.Path({str(marker)!r})
first_flaky = model == 'frontier' and not marker.exists()
if first_flaky:
    marker.write_text('used')
for attempt in range(1, repetitions + 1):
    trial = job / ('demo-task__' + value('--job-name')[-3:] + '-' + str(attempt))
    trial.joinpath('verifier').mkdir(parents=True)
    trial.joinpath('agent').mkdir(parents=True)
    incomplete = first_flaky and attempt == repetitions
    reward = 1.0 if model == 'frontier' else 0.0
    exception = None if reward else {{'exception_type': 'NonZeroAgentExitCodeError'}}
    passed = 2 if reward else 0
    failed = 0 if reward else 2
    trial.joinpath('result.json').write_text(json.dumps({{
        'task_name': 'terminal-bench-science/demo-task',
        'exception_info': {{'exception_type': 'VerifierTimeoutError'}} if incomplete else exception,
        'agent_result': {{'n_input_tokens': 10, 'n_cache_tokens': 2,
        'n_output_tokens': 3, 'cost_usd': 0.25}},
        'verifier_result': None if incomplete else {{'rewards': {{'reward': reward}}}},
    }}))
    if incomplete:
        continue
    trial.joinpath('verifier/reward.txt').write_text(str(reward) + '\\n')
    trial.joinpath('verifier/ctrf.json').write_text(json.dumps({{
        'results': {{'summary': {{'tests': 2, 'passed': passed, 'failed': failed,
        'skipped': 0, 'pending': 0, 'other': 0}}}}
    }}))
    trial.joinpath('agent/trajectory.json').write_text(json.dumps({{
        'schema_version': 'ATIF-v1.0', 'steps': [{{'step_id': 1}}]
    }}))
job.mkdir(parents=True, exist_ok=True)
job.joinpath('result.json').write_text(json.dumps({{
    'id': model + '-job', 'n_total_trials': repetitions,
    'stats': {{'n_completed_trials': repetitions - (1 if first_flaky else 0),
    'n_errored_trials': 0 if model == 'frontier' else repetitions}}
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_inconclusive_trial_is_topped_up_without_rebuying_confirmed_trials(
    tmp_path: Path,
):
    out = tmp_path / "out"
    receipt = harbor_model_matrix.launch_matrix(
        _task(tmp_path),
        harbor_executable=_flaky_harbor(tmp_path),
        claude_executable=_claude(tmp_path),
        out=out,
        models=["frontier", "weak"],
        repetitions=3,
        provider_env=PROVIDER,
        max_budget_usd=0.5,
        max_turns=10,
        timeout_sec=10,
        max_job_attempts=3,
    )
    assert receipt["rectangular"] is True
    assert len(receipt["trials"]) == 6
    assert len(receipt["excluded_trials"]) == 1
    assert (
        receipt["excluded_trials"][0]["reason"] == "inconclusive-no-verifier-evidence"
    )
    frontier_jobs = [
        row for row in receipt["jobs"] if row["model_id"] == "frontier"
    ]
    # One partial job (2/3 valid) plus one top-up job of exactly the missing 1.
    assert [row["requested_trials"] for row in frontier_jobs] == [3, 1]
    assert [row["valid_trials"] for row in frontier_jobs] == [2, 1]
    assert any("excluded from the rectangle" in row for row in receipt["limitations"])
    # A resumed launch re-consumes the same jobs byte-for-byte: same receipt,
    # no third frontier job, confirmed trials never re-bought.
    resumed = harbor_model_matrix.launch_matrix(
        tmp_path / "task",
        harbor_executable=tmp_path / "flaky-harbor",
        claude_executable=tmp_path / "claude",
        out=out,
        models=["frontier", "weak"],
        repetitions=3,
        provider_env=PROVIDER,
        max_budget_usd=0.5,
        max_turns=10,
        timeout_sec=10,
        max_job_attempts=3,
    )
    assert resumed["receipt_digest"] == receipt["receipt_digest"]
    assert len(list((out / "jobs").glob("demo_task-frontier-*"))) == 2
    assert len(list((out / "reservations").glob("*.json"))) == 3
