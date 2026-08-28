from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import factory_autopilot, factory_runtime_repair

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"
VOLC = "https://ark.cn-beijing.volces.com/api/coding"
PROVIDER = {"ANTHROPIC_BASE_URL": VOLC, "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}


def test_classify_failure_distinguishes_infra_from_task():
    assert factory_runtime_repair.classify_failure(
        "Harbor oracle attempt did not produce a valid control job: bad reward"
    ) == "task"
    assert factory_runtime_repair.classify_failure("Harbor command failed: timeout") == "infra"
    assert factory_runtime_repair.classify_failure("Harbor command failed: nonzero_exit") == "infra"


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
    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
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
