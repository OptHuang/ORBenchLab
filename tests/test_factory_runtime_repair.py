from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import agent_sessions, factory_autopilot, factory_runtime_repair

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"
VOLC = "https://ark.cn-beijing.volces.com/api/coding"
PROVIDER = {"ANTHROPIC_BASE_URL": VOLC, "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}


def test_classify_failure_distinguishes_infra_from_task():
    # Descriptive control-validation messages are task defects.
    assert factory_runtime_repair.classify_failure(
        "Harbor oracle attempt did not produce a valid control job: bad reward"
    ) == "task"
    # Launcher-level transients are infra irrespective of stderr.
    assert factory_runtime_repair.classify_failure("Harbor command failed: timeout") == "infra"
    # Classification is grounded in the real stderr, not the coarse class:
    # a nonzero exit with a Docker-daemon signature is a transient...
    assert factory_runtime_repair.classify_failure(
        "Harbor command failed: nonzero_exit",
        stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
    ) == "infra"
    # ...one with a Docker-build / verifier signature is a repairable task defect...
    assert factory_runtime_repair.classify_failure(
        "Harbor command failed: nonzero_exit",
        stderr="TASK_DOCKERFILE_BUILD_FAILED: missing dependency",
    ) == "task"
    # ...and a bare nonzero exit with no infra signature is a task defect to
    # repair, not a transient to resume forever.
    assert factory_runtime_repair.classify_failure("Harbor command failed: nonzero_exit") == "task"


def _repairing_harbor(root: Path) -> Path:
    """Oracle fails (reward 0) until the task carries a repair marker."""

    executable = root / "harbor"
    executable.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
agent = value('--agent')
snapshot = pathlib.Path(value('--path'))
repaired = (snapshot / 'data' / 'repair-marker.json').is_file()
name = snapshot.name
task_name = 'terminal-bench-science/' + name
job = pathlib.Path(value('--jobs-dir')) / value('--job-name')
trial = job / (name + '__' + agent)
trial.joinpath('verifier').mkdir(parents=True)
trial.joinpath('artifacts').mkdir(parents=True)
if agent == 'oracle':
    reward = 1.0 if repaired else 0.0
else:
    reward = 0.0
passed = 3 if reward == 1.0 else 0
failed = 0 if reward == 1.0 else 3
job.joinpath('result.json').write_text(json.dumps({'id': agent + '-id', 'n_total_trials': 1,
    'stats': {'n_completed_trials': 1, 'n_errored_trials': 0}}))
trial.joinpath('result.json').write_text(json.dumps({'task_name': task_name,
    'verifier_result': {'rewards': {'reward': reward}}, 'exception_info': None}))
trial.joinpath('verifier/ctrf.json').write_text(json.dumps({'results': {'summary':
    {'tests': 3, 'passed': passed, 'failed': failed, 'skipped': 0, 'pending': 0, 'other': 0}}}))
trial.joinpath('verifier/reward.txt').write_text(str(reward) + '\\n')
trial.joinpath('artifacts/manifest.json').write_text(json.dumps([
    {'source': '/root/submission/solver.py', 'status': 'ok' if reward == 1.0 else 'failed'}]))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _infra_harbor(root: Path) -> Path:
    executable = root / "harbor-infra"
    # A genuine infrastructure failure is identified by its real stderr
    # signature, not merely by a nonzero exit.
    executable.write_text(
        "#!/bin/sh\n"
        "echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _repair_agent(root: Path) -> Path:
    """Copies the task to task-vnext/<slug> and adds the repair marker."""

    executable = root / "repair-claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json, shutil, sys
from pathlib import Path
sys.stdin.readline()
task_src = next(p for p in Path("repair-input/task").iterdir() if (p / "task.toml").is_file())
dest = Path("task-vnext") / task_src.name
shutil.copytree(task_src, dest)
(dest / "data").mkdir(exist_ok=True)
(dest / "data" / "repair-marker.json").write_text(json.dumps({"repaired": True}))
print(json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 0.01}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.skipif(not shutil.which("bwrap"), reason="repair sessions need bubblewrap")
def test_task_failure_is_repaired_and_controls_pass(tmp_path: Path):
    workdir = tmp_path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    (workdir / "factory-input" / "paper-provenance.json").write_bytes(
        (GOOD_TASK / "paper-provenance.json").read_bytes()
    )
    task = tmp_path / "task" / "alphaevolve-scheduling"
    shutil.copytree(GOOD_TASK, task)
    harbor = _repairing_harbor(tmp_path)
    repair_cli = _repair_agent(tmp_path)
    final_task, controls, repair_state = factory_autopilot._controls_with_repair(
        task=task,
        workdir=workdir,
        root=tmp_path / "baseline",
        harbor_executable=harbor,
        claude_executable=repair_cli,
        model="fixture-model",
        provider_env=PROVIDER,
        harbor_timeout_sec=60,
        max_repair_rounds=1,
        repair_max_budget_usd=0.5,
        scope="baseline",
    )
    assert repair_state["repaired"] is True
    assert repair_state["rounds"] == 1
    assert (final_task / "data" / "repair-marker.json").is_file()
    assert controls["tasks"][0]["control_gates"]["oracle"]["gate"] == "pass"
    bundle = json.loads((tmp_path / "baseline" / "failure" / "failure-bundle.json").read_text())
    assert bundle["failure_class"] == "task"
    assert bundle["control_jobs"]


@pytest.mark.skipif(not shutil.which("bwrap"), reason="repair sessions need bubblewrap")
def test_task_failure_quarantines_after_repair_cap(tmp_path: Path):
    workdir = tmp_path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    (workdir / "factory-input" / "paper-provenance.json").write_bytes(
        (GOOD_TASK / "paper-provenance.json").read_bytes()
    )
    task = tmp_path / "task" / "alphaevolve-scheduling"
    shutil.copytree(GOOD_TASK, task)
    harbor = _repairing_harbor(tmp_path)
    # A no-op repair agent never adds the marker, so Oracle keeps failing.
    noop = tmp_path / "noop-claude"
    noop.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nsys.stdin.readline()\n"
        'print(json.dumps({"type":"result","subtype":"success","total_cost_usd":0.01}))\n',
        encoding="utf-8",
    )
    noop.chmod(0o755)
    with pytest.raises(factory_autopilot.FactoryRuntimeQuarantine) as excinfo:
        factory_autopilot._controls_with_repair(
            task=task,
            workdir=workdir,
            root=tmp_path / "baseline",
            harbor_executable=harbor,
            claude_executable=noop,
            model="fixture-model",
            provider_env=PROVIDER,
            harbor_timeout_sec=60,
            max_repair_rounds=1,
            repair_max_budget_usd=0.5,
            scope="baseline",
        )
    assert excinfo.value.quarantine["reason"] in {
        "runtime-repair-failed",
        "runtime-control-unrepaired",
    }


def test_infra_failure_raises_resumable_without_mutation(tmp_path: Path):
    workdir = tmp_path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    task = tmp_path / "task" / "alphaevolve-scheduling"
    shutil.copytree(GOOD_TASK, task)
    before = factory_autopilot.volc_rollout._task_tree_digest(task)
    harbor = _infra_harbor(tmp_path)
    with pytest.raises(factory_autopilot.FactoryInfraRetry) as excinfo:
        factory_autopilot._controls_with_repair(
            task=task,
            workdir=workdir,
            root=tmp_path / "baseline",
            harbor_executable=harbor,
            claude_executable=tmp_path / "unused",
            model="fixture-model",
            provider_env=PROVIDER,
            harbor_timeout_sec=60,
            max_repair_rounds=1,
            repair_max_budget_usd=0.5,
            scope="baseline",
        )
    assert excinfo.value.bundle["failure_class"] == "infra"
    # The task tree is untouched by an infrastructure failure.
    assert factory_autopilot.volc_rollout._task_tree_digest(task) == before


def _task_build_harbor(root: Path) -> Path:
    executable = root / "harbor-build-fail"
    executable.write_text(
        "#!/bin/sh\n"
        "echo 'TASK_DOCKERFILE_BUILD_FAILED missing dependency' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_task_build_failure_is_repaired_not_resumed(tmp_path: Path):
    # A Docker-build failure surfaces as nonzero_exit but its real stderr marks a
    # task defect: classification must be 'task' (repair), never 'infra' (resume
    # forever), and the failure bundle must record the real stderr.
    workdir = tmp_path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    task = tmp_path / "task" / "alphaevolve-scheduling"
    shutil.copytree(GOOD_TASK, task)
    harbor = _task_build_harbor(tmp_path)
    with pytest.raises(agent_sessions.AgentSessionError):
        # Repair is attempted (and only fails because the repair CLI is absent),
        # proving the failure was routed to repair rather than infra-resume.
        factory_autopilot._controls_with_repair(
            task=task,
            workdir=workdir,
            root=tmp_path / "baseline",
            harbor_executable=harbor,
            claude_executable=tmp_path / "absent-cli",
            model="fixture-model",
            provider_env=PROVIDER,
            harbor_timeout_sec=60,
            max_repair_rounds=1,
            repair_max_budget_usd=0.5,
            scope="baseline",
        )
    bundle = json.loads((tmp_path / "baseline" / "failure" / "failure-bundle.json").read_text())
    assert bundle["failure_class"] == "task"
    assert "TASK_DOCKERFILE_BUILD_FAILED" in bundle["failure_message"]


def test_failed_repair_session_adopts_no_task_even_with_planted_output(tmp_path: Path):
    # A failed repair session must adopt nothing, even if a task-vnext tree was
    # planted in the round directory before the session ran.
    root = tmp_path / "repair-round-1"
    task = tmp_path / "task" / "alphaevolve-scheduling"
    shutil.copytree(GOOD_TASK, task)
    failure = tmp_path / "failure-bundle.json"
    failure.write_text(json.dumps({"failure": "fixture"}), encoding="utf-8")
    planted = root / "task-vnext" / task.name
    shutil.copytree(task, planted)
    (planted / "planted-marker.json").write_text("{}", encoding="utf-8")
    failing_cli = tmp_path / "failing-claude"
    failing_cli.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    failing_cli.chmod(0o755)

    receipt = factory_runtime_repair.repair_task_once(
        task=task,
        failure_bundle_path=failure,
        paper_ancestors=[task / "paper-provenance.json"],
        out=root,
        claude_executable=failing_cli,
        model="fixture-model",
        provider_env=PROVIDER,
        max_budget_usd=0.1,
        timeout_sec=30,
        round_number=1,
        parent_task_digest=factory_runtime_repair.volc_rollout._task_tree_digest(task),
        failure_bundle_digest="sha256:" + "f" * 64,
    )
    assert receipt["session_status"] == "failed"
    assert receipt["status"] == "session-failed"
    assert receipt["repaired_task_path"] is None
    assert receipt["static_decision"] is None
    # The planted stale tree was cleared before the (failed) session ran.
    assert not planted.joinpath("planted-marker.json").is_file()
