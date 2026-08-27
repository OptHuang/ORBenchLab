from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from orbenchlab import agentic_factory, factory_supervisor

PROVIDER = {"ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding", "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}


def _fixture(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    provenance = tmp_path / "paper.json"
    provenance.write_text(json.dumps({"title": "fixture", "url": "https://example.test/paper", "digest": "sha256:" + "b" * 64, "license_status": "reviewed"}), encoding="utf-8")
    paper_digest = "sha256:" + hashlib.sha256(provenance.read_bytes()).hexdigest()
    selected = "factory/tasks/task-v2"
    genome_document = {
        "family": "supervised",
        "title": "fixture",
        "design_goal": "Exercise a bounded fixture task.",
        "selected_task": selected,
        "source": {"paper_provenance_digest": paper_digest},
        "difficulty_axes": {
            "size": {
                "levels": [1, 2],
                "meaning": "fixture scale",
                "expected_direction": "larger is harder",
            }
        },
    }
    agent = tmp_path / "agent"
    agent.write_text(
        "#!/bin/sh\npayload=$(cat)\ncase \"$payload\" in\n"
        "  *final-synthesis*) mkdir -p factory/final; "
        f"printf '%s\\n' '{json.dumps(genome_document, separators=(',', ':'))}' > factory/final/task-genome.json; "
        f"printf '%s\\n' '{{\"selected_task\":\"{selected}\"}}' > factory/final/task-review-summary.json ;;\n"
        "  *task-repair-v2*) mkdir -p factory/tasks/task-v2; "
        "printf '[task]\\nname=\"supervised\"\\n' > factory/tasks/task-v2/task.toml; "
        "printf '# task\\n' > factory/tasks/task-v2/README.md ;;\n"
        "esac\nprintf done\n",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    plan = agentic_factory.compile_plan(
        name="supervisor", source_binding_digest="sha256:" + "a" * 64,
        stages=[
            {"id": "task-repair-v2", "role": "author", "profile": "codex", "model": "fixture", "prompt": "write task-repair-v2", "depends_on": [], "timeout_sec": 5, "max_attempts": 1, "max_budget_usd": .1, "required_outputs": [{"path": "factory/tasks/task-v2", "kind": "directory"}]},
            {"id": "final-synthesis", "role": "summarizer", "profile": "codex", "model": "fixture", "prompt": "write final-synthesis", "depends_on": ["task-repair-v2"], "timeout_sec": 5, "max_attempts": 1, "max_budget_usd": .1, "required_outputs": [{"path": "factory/final/task-review-summary.json", "kind": "json"}, {"path": "factory/final/task-genome.json", "kind": "json"}]},
        ],
    )
    out = tmp_path / "factory"
    plan_path = agentic_factory.write_plan(plan, out / "plan.json")
    result = agentic_factory.run_factory(plan, workdir=work, out=out, environments={"codex": {"OPENAI_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding", "OPENAI_API_KEY": "fixture"}}, executables={"codex": agent})
    assert result["status"] == "semantic-complete-e1"
    genome = work / "factory/final/task-genome.json"
    return plan_path, out / "factory-run.json", work, work / "factory/tasks/task-v2", provenance, genome


def test_missing_external_adapters_quarantine_and_resume(tmp_path: Path):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    kwargs = dict(plan_path=plan, factory_run_path=run, workdir=work, task_dir=task, task_genome=genome,
                  paper_provenance=paper, out=tmp_path / "supervised", harbor_executable=None,
                  semantic_review_executable=None, semantic_review_models=["review-a", "review-b"],
                  harbor_inputs={}, calibration_executable=None,
                  calibration_models=["frontier", "weak"], test_image="fixture@sha256:abc",
                  repetitions=5, timeout_sec=2, provider_env=PROVIDER)
    first = factory_supervisor.run(**kwargs)
    second = factory_supervisor.run(**kwargs)
    assert first["status"] == second["status"] == "quarantined"
    assert first["promoted"] is False
    assert first["stages"]["harbor"]["failure_class"] == "upstream_semantic_blocked"
    assert first["stages"]["calibration"]["failure_class"] == "upstream_semantic_blocked"
    assert first["stages"]["finalize"]["status"] == "blocked"
    assert "fixture-secret" not in (Path(kwargs["out"]) / "supervisor-state.json").read_text()


def test_successful_command_without_receipt_fails_closed(tmp_path: Path, monkeypatch):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    monkeypatch.setattr(
        factory_supervisor.task_authoring,
        "validate_task",
        lambda *args, **kwargs: {"decision": "ready-for-human-review"},
    )
    def write_static(receipt, out):
        path = Path(out) / "authoring-receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return {"json": path}
    monkeypatch.setattr(factory_supervisor.task_authoring, "write_receipt", write_static)
    command = tmp_path / "success"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    state = factory_supervisor.run(
        plan_path=plan, factory_run_path=run, workdir=work, task_dir=task, task_genome=genome,
        paper_provenance=paper, out=tmp_path / "supervised", harbor_executable=command,
        semantic_review_executable=command, semantic_review_models=["review-a", "review-b"],
        harbor_inputs={"executed_task_dir": "x", "oracle_job": "y", "nop_job": "z"},
        calibration_executable=command, calibration_models=["frontier", "weak"],
        test_image="fixture@sha256:abc", repetitions=5, timeout_sec=2,
        provider_env=PROVIDER,
    )
    assert state["promoted"] is False
    assert state["stages"]["semantic_review"]["failure_class"] == "expected_receipt_missing"
    assert state["stages"]["harbor"]["failure_class"] == "upstream_semantic_blocked"
    assert state["stages"]["calibration"]["failure_class"] == "upstream_semantic_blocked"


def test_rejects_weak_or_nonrectangular_calibration_contract(tmp_path: Path):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    try:
        factory_supervisor.run(plan_path=plan, factory_run_path=run, workdir=work, task_dir=task, task_genome=genome,
            paper_provenance=paper, out=tmp_path / "out", harbor_executable=None,
            semantic_review_executable=None, semantic_review_models=["review-a", "review-b"],
            harbor_inputs={}, calibration_executable=None, calibration_models=["same", "same"],
            test_image="image", repetitions=4, provider_env=PROVIDER)
    except factory_supervisor.FactorySupervisorError:
        pass
    else:
        raise AssertionError("unsafe calibration contract accepted")


def test_resume_rejects_modified_completed_output(tmp_path: Path):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    kwargs = dict(plan_path=plan, factory_run_path=run, workdir=work, task_dir=task, task_genome=genome,
                  paper_provenance=paper, out=tmp_path / "supervised", harbor_executable=None,
                  semantic_review_executable=None, semantic_review_models=["review-a", "review-b"],
                  harbor_inputs={}, calibration_executable=None,
                  calibration_models=["frontier", "weak"], test_image="image", repetitions=5)
    kwargs["provider_env"] = PROVIDER
    factory_supervisor.run(**kwargs)
    receipt = Path(kwargs["out"]) / "static" / "authoring-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    try:
        factory_supervisor.run(**kwargs)
    except factory_supervisor.FactorySupervisorError:
        pass
    else:
        raise AssertionError("mutated stage output was reused")


def test_rejects_same_family_task_or_genome_substitution(tmp_path: Path):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    substituted_task = tmp_path / "substituted-task"
    shutil.copytree(task, substituted_task)
    (substituted_task / "README.md").write_text("same family, different task\n")
    kwargs = dict(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=substituted_task,
        task_genome=genome,
        paper_provenance=paper,
        out=tmp_path / "supervised",
        harbor_executable=None,
        semantic_review_executable=None,
        semantic_review_models=["review-a", "review-b"],
        harbor_inputs={},
        calibration_executable=None,
        calibration_models=["frontier", "weak"],
        test_image="image",
        repetitions=5,
        provider_env=PROVIDER,
    )
    try:
        factory_supervisor.run(**kwargs)
    except factory_supervisor.FactorySupervisorError as exc:
        assert "completed factory stage" in str(exc)
    else:
        raise AssertionError("same-family task substitution was accepted")

    substituted_genome = tmp_path / "substituted-genome.json"
    substituted_genome.write_bytes(genome.read_bytes())
    kwargs["task_dir"] = task
    kwargs["task_genome"] = substituted_genome
    try:
        factory_supervisor.run(**kwargs)
    except factory_supervisor.FactorySupervisorError as exc:
        assert "final-synthesis artifact" in str(exc)
    else:
        raise AssertionError("copied genome substitution was accepted")


def test_external_adapter_output_is_hard_bounded(tmp_path: Path):
    command = tmp_path / "spew"
    command.write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.stdout.write('x' * 10000)\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    result = factory_supervisor._command(
        str(command),
        [],
        timeout_sec=2,
        cwd=tmp_path,
        max_output_bytes=128,
    )
    assert result["status"] == "blocked"
    assert result["failure_class"] == "output_limit_exceeded"


def test_failed_external_stage_stops_after_attempt_cap(tmp_path: Path, monkeypatch):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
    monkeypatch.setattr(
        factory_supervisor.task_authoring,
        "validate_task",
        lambda *args, **kwargs: {"decision": "ready-for-human-review"},
    )
    def write_static(receipt, out):
        path = Path(out) / "authoring-receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return {"json": path}
    monkeypatch.setattr(factory_supervisor.task_authoring, "write_receipt", write_static)
    counter = tmp_path / "counter"
    command = tmp_path / "no-receipt"
    command.write_text(
        f"#!/bin/sh\nprintf x >> '{counter}'\nexit 0\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    kwargs = dict(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        task_genome=genome,
        paper_provenance=paper,
        out=tmp_path / "supervised",
        harbor_executable=None,
        semantic_review_executable=command,
        semantic_review_models=["review-a", "review-b"],
        harbor_inputs={},
        calibration_executable=None,
        calibration_models=["frontier", "weak"],
        test_image="image",
        repetitions=5,
        provider_env=PROVIDER,
        max_external_attempts=2,
    )
    factory_supervisor.run(**kwargs)
    factory_supervisor.run(**kwargs)
    third = factory_supervisor.run(**kwargs)
    assert counter.read_text() == "xx"
    assert third["stages"]["semantic_review"]["failure_class"] == "attempts_exhausted"
    assert third["stages"]["semantic_review"]["attempt_count"] == 2
