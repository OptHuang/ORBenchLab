from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import agentic_factory, factory_blueprints
from orbenchlab.cli import main


ROOT = Path(__file__).parents[1]


def _bound_paper(tmp_path: Path) -> tuple[Path, Path, dict]:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\nfixture paper bytes\n")
    provenance = {
        "paper_provenance_schema_version": "orbenchlab.paper-provenance.v1",
        "title": "Fixture paper",
        "url": "https://example.test/paper",
        "source_content_digest": factory_blueprints._digest_bytes(paper.read_bytes()),
        "source_path": str(paper),
        "license_status": "pending_human",
        "license_assertion": "caller-supplied",
        "intake_id": "fixture",
        "intake_snapshot_digest": "sha256:" + "b" * 64,
        "intake_item_uid": "sha256:" + "c" * 64,
        "intake_metadata_digest": "sha256:" + "d" * 64,
    }
    provenance["binding_digest"] = factory_blueprints._digest_bytes(
        factory_blueprints._canonical(provenance)
    )
    path = tmp_path / "paper-provenance.json"
    path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    return paper, path, provenance


def test_prepare_workspace_is_bound_idempotent_and_tamper_evident(tmp_path: Path):
    paper, provenance_path, provenance = _bound_paper(tmp_path)
    seed = ROOT / "examples/tasks/alphaevolve-scheduling"
    workdir = tmp_path / "workspace"
    first = factory_blueprints.prepare_workspace(
        paper_file=paper,
        paper_provenance=provenance_path,
        seed_task=seed,
        workdir=workdir,
    )
    second = factory_blueprints.prepare_workspace(
        paper_file=paper,
        paper_provenance=provenance_path,
        seed_task=seed,
        workdir=workdir,
    )
    assert second == first
    assert first["source_binding_digest"] == provenance["binding_digest"]
    assert (workdir / "factory-input/seed-task/task.toml").is_file()

    (workdir / "factory-input/paper-provenance.json").write_text("{}")
    with pytest.raises(agentic_factory.AgenticFactoryError, match="digest validation"):
        factory_blueprints.prepare_workspace(
            paper_file=paper,
            paper_provenance=provenance_path,
            seed_task=seed,
            workdir=workdir,
        )


def test_default_plan_assigns_all_semantic_stages_to_agent_sessions():
    plan = factory_blueprints.paper_to_benchmark_plan(
        source_binding_digest="sha256:" + "a" * 64,
        author_model="ark-code-latest",
        reviewer_models=["ark-reviewer-a", "ark-reviewer-b"],
        frontier_model="frontier-model",
        weak_model="weak-model",
    )
    ids = [stage["id"] for stage in plan["stages"]]
    assert ids == [
        "paper-derive-primary",
        "paper-derive-critic",
        "task-design-a",
        "task-design-b",
        "task-design-synthesis",
        "task-author-v1",
        "task-review-science",
        "task-review-verifier",
        "task-repair-v2",
        "runtime-controls",
        "pilot-frontier",
        "pilot-weak",
        "trajectory-diagnosis",
        "intervention-study",
        "difficulty-design",
        "variant-author",
        "calibration",
        "final-synthesis",
    ]
    assert all(stage["profile"] == "claude-code" for stage in plan["stages"])
    assert all(stage["required_outputs"] for stage in plan["stages"])
    assert plan["maximum_model_liability_usd"] == 41.0
    assert agentic_factory.validate_plan(plan) == plan


def test_default_plan_requires_independent_reviewers():
    with pytest.raises(agentic_factory.AgenticFactoryError, match="distinct reviewer"):
        factory_blueprints.paper_to_benchmark_plan(
            source_binding_digest="sha256:" + "a" * 64,
            author_model="author",
            reviewer_models=["same", "same"],
            frontier_model="frontier",
            weak_model="weak",
        )


def test_prepare_paper_cli_writes_eighteen_stage_plan(capsys, tmp_path: Path):
    paper, provenance_path, _ = _bound_paper(tmp_path)
    plan_path = tmp_path / "plan.json"
    workdir = tmp_path / "workspace"
    assert main(
        [
            "agent-factory",
            "prepare-paper",
            "--paper-file",
            str(paper),
            "--paper-provenance",
            str(provenance_path),
            "--seed-task",
            str(ROOT / "examples/tasks/alphaevolve-scheduling"),
            "--workdir",
            str(workdir),
            "--plan-out",
            str(plan_path),
            "--author-model",
            "author",
            "--reviewer-model",
            "reviewer-a",
            "--reviewer-model",
            "reviewer-b",
            "--frontier-model",
            "frontier",
            "--weak-model",
            "weak",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["stage_count"] == 18
    assert agentic_factory.load_plan(plan_path)["factory_id"] == output["factory_id"]


def test_plan_identity_binds_seed_and_rejects_another_workspace(tmp_path: Path):
    paper, provenance_path, _ = _bound_paper(tmp_path)
    seed_a = ROOT / "examples/tasks/alphaevolve-scheduling"
    seed_b = tmp_path / "different-seed"
    shutil.copytree(seed_a, seed_b)
    (seed_b / "README.md").write_text("different strict seed\n")
    work_a = tmp_path / "work-a"
    work_b = tmp_path / "work-b"
    manifest_a = factory_blueprints.prepare_workspace(
        paper_file=paper,
        paper_provenance=provenance_path,
        seed_task=seed_a,
        workdir=work_a,
    )
    manifest_b = factory_blueprints.prepare_workspace(
        paper_file=paper,
        paper_provenance=provenance_path,
        seed_task=seed_b,
        workdir=work_b,
    )
    assert manifest_a["workspace_binding_digest"] != manifest_b["workspace_binding_digest"]
    kwargs = {
        "author_model": "author",
        "reviewer_models": ["reviewer-a", "reviewer-b"],
        "frontier_model": "frontier",
        "weak_model": "weak",
    }
    plan_a = factory_blueprints.paper_to_benchmark_plan(
        source_binding_digest=manifest_a["workspace_binding_digest"], **kwargs
    )
    plan_b = factory_blueprints.paper_to_benchmark_plan(
        source_binding_digest=manifest_b["workspace_binding_digest"], **kwargs
    )
    assert plan_a["factory_id"] != plan_b["factory_id"]
    with pytest.raises(agentic_factory.AgenticFactoryError, match="binding does not match"):
        agentic_factory.run_factory(
            plan_a,
            workdir=work_b,
            out=tmp_path / "run",
            max_new_stages=0,
        )
