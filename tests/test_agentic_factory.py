from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from orbenchlab import agentic_factory
from orbenchlab.cli import main


DIGEST = "sha256:" + "a" * 64
VOLC = "https://ark.cn-beijing.volces.com/api/coding"


def _factory_process(queue, plan, workdir, out, executable):
    try:
        result = agentic_factory.run_factory(
            plan,
            workdir=workdir,
            out=out,
            environments=_environments(),
            executables={"codex": executable},
        )
        queue.put(("ok", result["status"]))
    except Exception as exc:  # pragma: no cover - delivered to the parent assertion
        queue.put(("error", repr(exc)))


def _stage(
    stage_id: str,
    *,
    depends_on: list[str] | None = None,
    profile: str = "codex",
    output: str | None = None,
    max_attempts: int = 1,
) -> dict:
    return {
        "id": stage_id,
        "role": f"autonomous {stage_id} agent",
        "profile": profile,
        "model": "fixture-model",
        "prompt": f"Complete the {stage_id} stage.",
        "depends_on": depends_on or [],
        "timeout_sec": 5,
        "max_attempts": max_attempts,
        "max_budget_usd": 0.25,
        "max_output_bytes": 1024 * 1024,
        "required_outputs": [
            {"path": output or f"factory/{stage_id}.json", "kind": "json"}
        ],
    }


def _fixture_cli(path: Path) -> Path:
    executable = path / "fixture-agent"
    executable.write_text(
        """#!/bin/sh
payload=$(cat)
mkdir -p factory
case "$payload" in
  *paper-critic*) printf '{"decision":"checked"}\n' > factory/paper-critic.json ;;
  *paper-derive*) printf '{"claims":["bounded"]}\n' > factory/paper-derive.json ;;
esac
printf 'done'
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _environments() -> dict[str, dict[str, str]]:
    return {
        "codex": {"OPENAI_BASE_URL": VOLC, "OPENAI_API_KEY": "fixture-secret"},
        "claude-code": {
            "ANTHROPIC_BASE_URL": VOLC,
            "ANTHROPIC_AUTH_TOKEN": "fixture-secret",
        },
    }


def test_compile_plan_rejects_cycles_unknown_dependencies_and_shared_outputs():
    with pytest.raises(agentic_factory.AgenticFactoryError, match="cycle"):
        agentic_factory.compile_plan(
            name="cycle",
            source_binding_digest=DIGEST,
            stages=[
                _stage("one", depends_on=["two"]),
                _stage("two", depends_on=["one"]),
            ],
        )
    with pytest.raises(agentic_factory.AgenticFactoryError, match="unknown dependencies"):
        agentic_factory.compile_plan(
            name="unknown",
            source_binding_digest=DIGEST,
            stages=[_stage("one", depends_on=["missing"])],
        )
    with pytest.raises(agentic_factory.AgenticFactoryError, match="owned by both"):
        agentic_factory.compile_plan(
            name="shared",
            source_binding_digest=DIGEST,
            stages=[_stage("one", output="factory/shared.json"), _stage("two", output="factory/shared.json")],
        )


def test_factory_runs_agent_dag_and_resumes_completed_state(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="paper to strict task",
        source_binding_digest=DIGEST,
        stages=[
            _stage("paper-derive"),
            _stage("paper-critic", depends_on=["paper-derive"], profile="codex"),
        ],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = _fixture_cli(tmp_path)
    out = tmp_path / "run"
    first = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable, "claude-code": executable},
    )
    assert first["status"] == "semantic-complete-e1"
    assert first["new_stage_attempts"] == 2
    assert first["completion_digest"].startswith("sha256:")
    attempt = json.loads((out / "stages/paper-derive/attempt-001.json").read_text())
    assert attempt["status"] == "completed"
    assert attempt["session_receipt_digest"].startswith("sha256:")
    assert attempt["output_artifacts"][0]["path"] == "factory/paper-derive.json"

    second = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable, "claude-code": executable},
    )
    assert second["status"] == "semantic-complete-e1"
    assert second["resumed"] is True
    assert second["stages"] == first["stages"]


def test_factory_quarantines_missing_required_output(tmp_path: Path):
    executable = tmp_path / "no-output-agent"
    executable.write_text("#!/bin/sh\ncat >/dev/null\n", encoding="utf-8")
    executable.chmod(0o755)
    plan = agentic_factory.compile_plan(
        name="fail closed",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=tmp_path / "run",
        environments=_environments(),
        executables={"codex": executable},
    )
    assert result["status"] == "quarantined"
    assert result["quarantine"]["stage_id"] == "paper-derive"
    receipt = json.loads((tmp_path / "run/stages/paper-derive/attempt-001.json").read_text())
    assert receipt["failure_class"] == "session_contract_failure"


def test_factory_checkpoint_runs_only_one_ready_stage(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="leased worker",
        source_binding_digest=DIGEST,
        stages=[
            _stage("paper-derive"),
            _stage("paper-critic", depends_on=["paper-derive"], profile="codex"),
        ],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = _fixture_cli(tmp_path)
    first = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=tmp_path / "run",
        environments=_environments(),
        executables={"codex": executable, "claude-code": executable},
        max_new_stages=1,
    )
    assert first["status"] == "active"
    assert first["stages"]["paper-derive"]["status"] == "completed"
    assert first["stages"]["paper-critic"]["status"] == "pending"

    final = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=tmp_path / "run",
        environments=_environments(),
        executables={"codex": executable, "claude-code": executable},
    )
    assert final["status"] == "semantic-complete-e1"
    assert final["new_stage_attempts"] == 1


def test_factory_resume_rejects_tampered_run_and_artifact(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="tamper proof",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = _fixture_cli(tmp_path)
    out = tmp_path / "run"
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable},
    )
    assert result["status"] == "semantic-complete-e1"
    (workdir / "factory/paper-derive.json").write_text('{"claims":["forged"]}\n')
    with pytest.raises(agentic_factory.AgenticFactoryError, match="output changed"):
        agentic_factory.run_factory(
            plan,
            workdir=workdir,
            out=out,
            environments=_environments(),
            executables={"codex": executable},
        )

    run_path = out / "factory-run.json"
    run = json.loads(run_path.read_text())
    run["status"] = "active"
    run_path.write_text(json.dumps(run))
    with pytest.raises(agentic_factory.AgenticFactoryError, match="run state digest"):
        agentic_factory.run_factory(
            plan,
            workdir=workdir,
            out=out,
            environments=_environments(),
            executables={"codex": executable},
        )


def test_factory_rejects_required_output_through_parent_symlink(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "paper-derive.json").write_text('{"outside":true}\n')
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "factory").symlink_to(outside, target_is_directory=True)
    plan = agentic_factory.compile_plan(
        name="no symlink escape",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    executable = _fixture_cli(tmp_path)
    with pytest.raises(agentic_factory.AgenticFactoryError, match="symlink"):
        agentic_factory.run_factory(
            plan,
            workdir=workdir,
            out=tmp_path / "run",
            environments=_environments(),
            executables={"codex": executable},
        )


def test_agent_factory_cli_compiles_blueprint(capsys, tmp_path: Path):
    blueprint = {
        "name": "cli factory",
        "source_binding_digest": DIGEST,
        "stages": [_stage("paper-derive")],
    }
    source = tmp_path / "blueprint.json"
    source.write_text(json.dumps(blueprint), encoding="utf-8")
    plan = tmp_path / "factory-plan.json"
    assert main(
        ["agent-factory", "compile", "--blueprint", str(source), "--out", str(plan)]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["stage_count"] == 1
    assert agentic_factory.load_plan(plan)["factory_id"] == output["factory_id"]


def test_factory_rejects_rehashed_terminal_status_with_pending_stage(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="terminal invariant",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = tmp_path / "run"
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        max_new_stages=0,
    )
    assert result["status"] == "active"
    path = out / "factory-run.json"
    forged = json.loads(path.read_text())
    forged["status"] = "semantic-complete-e1"
    forged["completion_digest"] = agentic_factory._value_digest(
        {
            "factory_id": plan["factory_id"],
            "plan_digest": plan["plan_digest"],
            "stages": forged["stages"],
            "evidence_level": "E1-agent-session-process",
        }
    )
    unsigned = {key: value for key, value in forged.items() if key != "run_digest"}
    forged["run_digest"] = agentic_factory._value_digest(unsigned)
    path.write_text(json.dumps(forged))
    with pytest.raises(agentic_factory.AgenticFactoryError, match="terminal invariants"):
        agentic_factory.run_factory(plan, workdir=workdir, out=out, max_new_stages=0)


def test_completed_attempt_requires_exact_agent_session_binding(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="session chain",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = tmp_path / "run"
    executable = _fixture_cli(tmp_path)
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable},
    )
    assert result["status"] == "semantic-complete-e1"

    attempt_path = out / "stages/paper-derive/attempt-001.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["session_id"] = None
    attempt["session_receipt_digest"] = None
    unsigned_attempt = {
        key: value for key, value in attempt.items() if key != "attempt_digest"
    }
    attempt["attempt_digest"] = agentic_factory._value_digest(unsigned_attempt)
    attempt_path.write_text(json.dumps(attempt))

    run_path = out / "factory-run.json"
    run = json.loads(run_path.read_text())
    run["stages"]["paper-derive"]["attempts"][0]["attempt_digest"] = attempt[
        "attempt_digest"
    ]
    completion = {
        "factory_id": plan["factory_id"],
        "plan_digest": plan["plan_digest"],
        "stages": run["stages"],
        "evidence_level": "E1-agent-session-process",
    }
    run["completion_digest"] = agentic_factory._value_digest(completion)
    unsigned_run = {key: value for key, value in run.items() if key != "run_digest"}
    run["run_digest"] = agentic_factory._value_digest(unsigned_run)
    run_path.write_text(json.dumps(run))
    with pytest.raises(agentic_factory.AgenticFactoryError, match="no agent-session receipt"):
        agentic_factory.run_factory(plan, workdir=workdir, out=out)


def test_factory_lock_serializes_two_processes_for_one_run(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="factory lock",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = tmp_path / "slow-agent"
    executable.write_text(
        """#!/bin/sh
cat >/dev/null
printf x >> executions.log
sleep 0.3
mkdir -p factory
printf '{"claims":["bounded"]}\n' > factory/paper-derive.json
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    out = tmp_path / "run"
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_factory_process,
            args=(queue, plan, str(workdir), str(out), str(executable)),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=2) for _ in processes)
    assert results == [
        ("ok", "semantic-complete-e1"),
        ("ok", "semantic-complete-e1"),
    ]
    assert (workdir / "executions.log").read_text() == "x"


def test_factory_recovers_receipt_written_before_run_state_commit(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="orphan recovery",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = tmp_path / "run"
    executable = _fixture_cli(tmp_path)
    completed = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable},
    )
    assert completed["status"] == "semantic-complete-e1"

    run_path = out / "factory-run.json"
    interrupted = json.loads(run_path.read_text())
    interrupted["status"] = "active"
    interrupted.pop("completion_digest")
    interrupted["stages"]["paper-derive"] = {
        "status": "running",
        "attempts": [],
        "output_artifacts": [],
    }
    unsigned = {key: value for key, value in interrupted.items() if key != "run_digest"}
    interrupted["run_digest"] = agentic_factory._value_digest(unsigned)
    run_path.write_text(json.dumps(interrupted))

    recovered = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"codex": executable},
    )
    assert recovered["status"] == "semantic-complete-e1"
    assert len(recovered["stages"]["paper-derive"]["attempts"]) == 1


def test_factory_quarantines_workspace_secret_leak(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="secret scan",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = tmp_path / "leaking-agent"
    executable.write_text(
        "#!/bin/sh\ncat >/dev/null\nmkdir -p factory\nprintf '{\"leak\":\"fixture-secret\"}\\n' > factory/paper-derive.json\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=tmp_path / "run",
        environments=_environments(),
        executables={"codex": executable},
    )
    assert result["status"] == "quarantined"
    attempt = json.loads((tmp_path / "run/stages/paper-derive/attempt-001.json").read_text())
    assert "provider credential" in attempt["failure_detail"]


def test_factory_requires_disjoint_work_and_evidence_roots(tmp_path: Path):
    plan = agentic_factory.compile_plan(
        name="root separation",
        source_binding_digest=DIGEST,
        stages=[_stage("paper-derive")],
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(agentic_factory.AgenticFactoryError, match="outside the agent workdir"):
        agentic_factory.run_factory(
            plan,
            workdir=workdir,
            out=workdir / "receipts",
            max_new_stages=0,
        )
