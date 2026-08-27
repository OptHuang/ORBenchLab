from __future__ import annotations

import json
from pathlib import Path

from orbenchlab import agentic_factory, factory_supervisor

PROVIDER = {"ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding", "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}


def _fixture(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    agent = tmp_path / "agent"
    agent.write_text(
        "#!/bin/sh\ncat >/dev/null\nmkdir -p factory/tasks/task-v2\n"
        "printf '[task]\\nname=\"supervised\"\\n' > factory/tasks/task-v2/task.toml\n"
        "printf '# task\\n' > factory/tasks/task-v2/README.md\nprintf done\n",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    plan = agentic_factory.compile_plan(
        name="supervisor", source_binding_digest="sha256:" + "a" * 64,
        stages=[{"id": "author", "role": "author", "profile": "codex", "model": "fixture", "prompt": "write", "depends_on": [], "timeout_sec": 5, "max_attempts": 1, "max_budget_usd": .1, "required_outputs": [{"path": "factory/tasks/task-v2", "kind": "directory"}]}],
    )
    out = tmp_path / "factory"
    plan_path = agentic_factory.write_plan(plan, out / "plan.json")
    result = agentic_factory.run_factory(plan, workdir=work, out=out, environments={"codex": {"OPENAI_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding", "OPENAI_API_KEY": "fixture"}}, executables={"codex": agent})
    assert result["status"] == "semantic-complete-e1"
    provenance = tmp_path / "paper.json"
    provenance.write_text(json.dumps({"title": "fixture", "url": "https://example.test/paper", "digest": "sha256:" + "b" * 64, "license_status": "reviewed"}), encoding="utf-8")
    genome = tmp_path / "genome.json"
    genome.write_text(json.dumps({"family": "supervised", "title": "fixture", "difficulty_axes": {"size": {"levels": [1, 2]}}}), encoding="utf-8")
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
    assert first["stages"]["harbor"]["failure_class"] == "harbor_inputs_missing"
    assert first["stages"]["calibration"]["failure_class"] == "external_dependency_missing"
    assert first["stages"]["finalize"]["status"] == "blocked"
    assert "fixture-secret" not in (Path(kwargs["out"]) / "supervisor-state.json").read_text()


def test_successful_command_without_receipt_fails_closed(tmp_path: Path):
    plan, run, work, task, paper, genome = _fixture(tmp_path)
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
    assert state["stages"]["harbor"]["failure_class"] == "expected_receipt_missing"
    assert state["stages"]["calibration"]["failure_class"] == "expected_receipt_missing"


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
