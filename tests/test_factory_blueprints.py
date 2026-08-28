from __future__ import annotations

import json
import shutil
import stat
import textwrap
from pathlib import Path

import pytest

from orbenchlab import agentic_factory, factory_blueprints
from orbenchlab.cli import main


ROOT = Path(__file__).parents[1]
_REAL_EXTRACT_PAPER_TEXT = factory_blueprints._extract_paper_text


@pytest.fixture(autouse=True)
def _deterministic_paper_text_fixture(monkeypatch: pytest.MonkeyPatch):
    def fake_extract(paper: Path, **_: object):
        rendered = b"=== PDF PAGE 1 ===\nFixture paper text\n\n"
        receipt = {
            "schema_version": "orbenchlab.paper-text-extraction.v1",
            "source_content_digest": factory_blueprints._digest_bytes(paper.read_bytes()),
            "extractor": "test-fixture",
            "extractor_version": "test-fixture 1",
            "executable_digest": "sha256:" + "e" * 64,
            "argv_template": ["<TEST>"],
            "timeout_sec": 120.0,
            "max_output_bytes": 64 * 1024 * 1024,
            "page_count": 1,
            "text_content_digest": factory_blueprints._digest_bytes(rendered),
        }
        receipt["receipt_digest"] = factory_blueprints._digest_bytes(
            factory_blueprints._canonical(receipt)
        )
        return rendered, receipt

    monkeypatch.setattr(factory_blueprints, "_extract_paper_text", fake_extract)


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
    assert first["paper_text_extraction"]["page_count"] == 1
    assert (workdir / "factory-input/seed-task/task.toml").is_file()
    assert not (workdir / "factory-input/paper.pdf").stat().st_mode & stat.S_IWUSR
    assert not (workdir / "factory-input/paper.txt").stat().st_mode & stat.S_IWUSR
    assert not (workdir / "factory-input/seed-task").stat().st_mode & stat.S_IWUSR

    (workdir / "factory-input/paper-provenance.json").chmod(0o644)
    (workdir / "factory-input/paper-provenance.json").write_text("{}")
    with pytest.raises(agentic_factory.AgenticFactoryError, match="digest validation"):
        factory_blueprints.prepare_workspace(
            paper_file=paper,
            paper_provenance=provenance_path,
            seed_task=seed,
            workdir=workdir,
        )


def test_bounded_pdf_extraction_adds_page_markers_and_receipt(tmp_path: Path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"bound paper")
    executable = tmp_path / "fake-pdftotext"
    executable.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys
            if "-v" in sys.argv:
                print("pdftotext version fixture")
            else:
                sys.stdout.write("first page\\fsecond page\\f")
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    text, receipt = _REAL_EXTRACT_PAPER_TEXT(paper, executable=executable)
    assert text == (
        b"=== PDF PAGE 1 ===\nfirst page\n\n"
        b"=== PDF PAGE 2 ===\nsecond page\n\n"
    )
    assert receipt["page_count"] == 2
    assert receipt["text_content_digest"] == factory_blueprints._digest_bytes(text)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    assert receipt["receipt_digest"] == factory_blueprints._digest_bytes(
        factory_blueprints._canonical(unsigned)
    )


def test_bounded_pdf_extraction_fails_closed_on_output_limit(tmp_path: Path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"bound paper")
    executable = tmp_path / "fake-pdftotext"
    executable.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "print('fixture') if '-v' in sys.argv else sys.stdout.write('too much text')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    with pytest.raises(agentic_factory.AgenticFactoryError, match="output_limit_exceeded"):
        _REAL_EXTRACT_PAPER_TEXT(paper, executable=executable, max_output_bytes=4)


def test_default_plan_assigns_all_semantic_stages_to_agent_sessions(tmp_path: Path):
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
        "paper-derive-normalize",
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
        "intervention-policy",
        "intervention-study",
        "difficulty-design",
        "variant-author",
        "calibration",
        "final-synthesis",
    ]
    assert all(stage["profile"] == "claude-code" for stage in plan["stages"])
    assert all(stage["required_outputs"] for stage in plan["stages"])
    text_outputs = {
        "paper-derive-primary": "factory/evidence/paper-derivation-raw.md",
        "paper-derive-critic": "factory/evidence/paper-derivation-critic.md",
        "task-design-a": "factory/design/task-design-a.md",
        "task-design-b": "factory/design/task-design-b.md",
        "task-design-synthesis": "factory/design/task-design-selected.md",
        "task-review-science": "factory/reviews/task-review-science.md",
        "task-review-verifier": "factory/reviews/task-review-verifier.md",
        "runtime-controls": "factory/runtime/control-index.md",
        "pilot-frontier": "factory/runtime/pilot-frontier.md",
        "pilot-weak": "factory/runtime/pilot-weak.md",
        "trajectory-diagnosis": "factory/analysis/trajectory-diagnosis.md",
        "intervention-study": "factory/analysis/intervention-study.md",
        "difficulty-design": "factory/difficulty/difficulty-lattice.md",
        "calibration": "factory/calibration/calibration-index.md",
    }
    by_id = {stage["id"]: stage for stage in plan["stages"]}
    for stage_id, path in text_outputs.items():
        assert by_id[stage_id]["required_outputs"] == [
            {
                "path": path,
                "kind": "text",
                "max_bytes": by_id[stage_id]["required_outputs"][0]["max_bytes"],
                "json_required_keys": [],
                "json_key_types": {},
                "json_nonempty_keys": [],
                "json_digest_bindings": {},
            }
        ]
    primary = next(stage for stage in plan["stages"] if stage["id"] == "paper-derive-primary")
    assert primary["required_outputs"][0]["path"] == "factory/evidence/paper-derivation-raw.md"
    assert primary["required_outputs"][0]["kind"] == "text"
    assert primary["required_outputs"][0]["max_bytes"] == 64_000
    assert "clear Markdown" in primary["prompt"]
    assert "raw JSON" not in primary["prompt"]
    normalizer = next(stage for stage in plan["stages"] if stage["id"] == "paper-derive-normalize")
    assert normalizer["required_outputs"][0]["max_bytes"] == 32_000
    assert set(normalizer["required_outputs"][0]["json_required_keys"]) == {
        "paper",
        "executable_scientific_core",
        "assumptions",
        "available_artifacts",
        "candidate_terminal_interactions",
        "non_derivable_claims",
        "blockers",
        "explicit_unknowns",
        "raw_evidence_digest",
        "paper_provenance_digest",
    }
    assert normalizer["required_outputs"][0]["json_key_types"]["paper"] == "object"
    assert normalizer["required_outputs"][0]["json_key_types"]["assumptions"] == "array"
    assert normalizer["required_outputs"][0]["json_digest_bindings"] == {
        "paper_provenance_digest": "factory-input/paper-provenance.json",
        "raw_evidence_digest": "factory/evidence/paper-derivation-raw.md",
    }
    assert "paper-derivation-raw.md" in normalizer["prompt"]
    assert "copy those strings" in normalizer["prompt"]
    raw = tmp_path / "factory/evidence/paper-derivation-raw.md"
    provenance = tmp_path / "factory-input/paper-provenance.json"
    raw.parent.mkdir(parents=True)
    provenance.parent.mkdir(parents=True)
    raw.write_text("# Raw evidence\n\n- page 1: bounded claim\n", encoding="utf-8")
    provenance.write_text('{"title":"Fixture"}\n', encoding="utf-8")
    runtime_prompt = agentic_factory._stage_prompt(plan, normalizer, workspace=tmp_path)
    assert agentic_factory._file_digest(raw) in runtime_prompt
    assert agentic_factory._file_digest(provenance) in runtime_prompt
    assert "trusted_json_digest_values" in runtime_prompt
    assert "do not compute, alter, or omit" in runtime_prompt
    final = next(stage for stage in plan["stages"] if stage["id"] == "final-synthesis")
    assert [output["path"] for output in final["required_outputs"]] == [
        "factory/final/task-review-summary.json",
        "factory/final/task-genome.json",
    ]
    assert final["required_outputs"][0]["json_required_keys"] == [
        "selected_task",
        "task_summary",
        "evidence_level",
        "limitations",
    ]
    assert final["required_outputs"][1]["json_required_keys"] == [
        "family",
        "title",
        "design_goal",
        "selected_task",
        "source",
        "difficulty_axes",
    ]
    assert plan["maximum_model_liability_usd"] == 73.5
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


def test_prepare_paper_cli_writes_nineteen_stage_plan(capsys, tmp_path: Path):
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
    assert output["stage_count"] == 20
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
