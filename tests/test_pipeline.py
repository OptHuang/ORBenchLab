from __future__ import annotations

import json
import hashlib
from pathlib import Path

import yaml

from orbenchlab import pipeline


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    return path


def _harbor_receipt(task: str, task_digest: str):
    digest = "sha256:" + "b" * 64
    receipt = {
        "schema_version": "orbenchlab.screening-report.v1",
        "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
        "task_tree_digest": task_digest,
        "authoring_task_tree_digest": task_digest,
        "executed_task_tree_digest": task_digest,
        "tasks": [
            {
                "task": task,
                "family": task,
                "arms": {},
                "control_gates": {
                    "oracle": {
                        "gate": "pass", "control": "oracle", "reward": 1.0,
                        "ctrf_summary": {"tests": 2, "passed": 2, "failed": 0, "skipped": 0, "pending": 0, "other": 0},
                        "task_name": f"terminal-bench-science/{task.replace('_', '-')}",
                        "job_result_digest": digest, "trial_result_digest": digest,
                        "ctrf_digest": digest, "reward_digest": digest,
                        "artifact_manifest_digest": digest,
                    },
                    "nop": {
                        "gate": "pass", "control": "nop", "reward": 0.0,
                        "ctrf_summary": {"tests": 2, "passed": 0, "failed": 2, "skipped": 0, "pending": 0, "other": 0},
                        "task_name": f"terminal-bench-science/{task.replace('_', '-')}",
                        "job_result_digest": digest, "trial_result_digest": digest,
                        "ctrf_digest": digest, "reward_digest": digest,
                        "artifact_manifest_digest": digest,
                    },
                },
                "decision": "collect-more-evidence",
                "evidence_level": "E3",
                "discrimination_index_observed_gap": None,
                "limitations": [],
            }
        ],
    }
    receipt["report_digest"] = pipeline._value_digest(receipt)
    return receipt


def test_pipeline_builds_one_card_from_genome_and_two_screenings(tmp_path):
    tasks = tmp_path / "tasks"
    reports = tmp_path / "reports"
    genome = _write(
        tasks / "demo.yaml",
        {
            "family": "demo",
            "title": "Coupled constraint audit",
            "design_goal": "Check cross-file operational constraints and produce an independent audit.",
            "source": {"url": "https://arxiv.org/abs/2602.02029", "source_content_digest": "sha256:" + "a" * 64},
            "difficulty_axes": {
                "join_depth": {"levels": [0, 1, 2], "meaning": "number of entity hops"},
                "scale": {"levels": ["tiny", "small"], "expected_direction": "harder_with_larger_value"},
            },
            "interventions": {"hint_levels": [0, 1, 2]},
        },
    )
    report_a = _write(
        reports / "first.json",
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "tasks": [
                {
                    "task": "demo",
                    "arms": {
                        "ark": {"n": 2, "complete": 2, "metric_n": 2, "solve_rate": 0.5, "quality_pass_rate": 0.5, "mean_feasibility": 0.75, "infra_exceptions": [], "failure_modes": ["verifier_failed"]}
                    },
                    "discrimination_index_observed_gap": 0.5,
                    "decision": "review-promising",
                    "evidence_level": "E3",
                    "limitations": ["single run"],
                }
            ],
        },
    )
    report_b = _write(
        reports / "replication.json",
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "tasks": [
                {
                    "task": "demo",
                    "arms": {
                        "ark": {"n": 2, "complete": 2, "metric_n": 2, "solve_rate": 1.0, "quality_pass_rate": 1.0, "mean_feasibility": 1.0, "infra_exceptions": []}
                    },
                    "discrimination_index_observed_gap": 0.0,
                    "decision": "revise-or-drop",
                    "evidence_level": "E3",
                    "limitations": ["replication gap zero"],
                }
            ],
        },
    )

    cards, paths = pipeline.build_cards(task_inputs=[tasks], screening_inputs=[reports])
    assert [path.name for path in paths] == ["first.json", "replication.json"]
    assert len(cards) == 1
    card = cards[0]
    assert card["task_id"] == "demo"
    assert card["decision"] == "revise-or-drop"
    model = card["performance"]["models"][0]
    assert model["model"] == "ark"
    assert model["metric_n"] == 4
    assert model["solve_n"] == 4
    assert model["quality_n"] == 4
    assert model["feasibility_n"] == 4
    assert model["solve_rate"] == 0.75
    assert model["quality_pass_rate"] == 0.75
    assert model["failure_modes"] == ["verifier_failed"]
    assert "Coupled constraint audit" in card["summary_markdown"]
    assert len(card["difficulty"]["axes"]) == 2


def test_pipeline_uses_separate_metric_denominators(tmp_path):
    tasks = tmp_path / "tasks"
    reports = tmp_path / "reports"
    _write(tasks / "demo.yaml", {"family": "demo", "difficulty_axes": {"size": [1, 2]}})
    _write(
        reports / "screening.json",
        {
            "tasks": [
                {
                    "task": "demo",
                    "decision": "collect-more-evidence",
                    "evidence_level": "E3",
                    "arms": {
                        "ark": {
                            "n": 4,
                            "complete": 3,
                            "metric_n": 3,
                            "solve_n": 3,
                            "quality_n": 1,
                            "feasibility_n": 0,
                            "solve_rate": 2 / 3,
                            "quality_pass_rate": 1.0,
                            "mean_feasibility": None,
                        }
                    },
                }
            ]
        },
    )

    card = pipeline.build_cards(task_inputs=[tasks], screening_inputs=[reports])[0][0]
    model = card["performance"]["models"][0]
    assert model["solve_n"] == 3
    assert model["quality_n"] == 1
    assert model["feasibility_n"] == 0
    assert model["solve_rate"] == 0.666667
    assert model["quality_pass_rate"] == 1.0
    assert model["mean_feasibility"] is None


def test_pipeline_promotes_only_with_valid_harbor_and_repeated_two_model_gap(tmp_path):
    task_digest = "sha256:" + "a" * 64
    _write(
        tmp_path / "tasks/demo.yaml",
        {"family": "demo", "difficulty_axes": {"size": [1, 2]}},
    )
    _write(tmp_path / "reports/harbor.json", _harbor_receipt("demo", task_digest))
    _write(
        tmp_path / "reports/models.json",
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "task_tree_digest": task_digest,
            "tasks": [
                {
                    "task": "demo",
                    "decision": "review-promising",
                    "evidence_level": "E3",
                    "discrimination_index_observed_gap": 0.5,
                    "arms": {
                        "frontier@hint-0": {"n": 5, "complete": 5, "metric_n": 5, "solve_rate": 1.0},
                        "open@hint-0": {"n": 5, "complete": 5, "metric_n": 5, "solve_rate": 0.0},
                    },
                }
            ],
        },
    )

    card = pipeline.build_cards(
        task_inputs=[tmp_path / "tasks"], screening_inputs=[tmp_path / "reports"]
    )[0][0]
    assert card["decision"] == "review-promising"
    assert "Harbor 打包控制" in card["summary_markdown"]
    assert "2/2 tests passed" in card["summary_markdown"]


def test_pipeline_rejects_forged_harbor_receipt(tmp_path):
    task_digest = "sha256:" + "a" * 64
    receipt = _harbor_receipt("demo", task_digest)
    receipt["tasks"][0]["control_gates"]["oracle"]["reward"] = 0.0
    _write(tmp_path / "reports/harbor.json", receipt)

    try:
        pipeline.build_cards(screening_inputs=[tmp_path / "reports"])
    except pipeline.PipelineError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("forged Harbor receipt must fail closed")


def test_pipeline_quarantines_model_report_without_shared_task_digest(tmp_path):
    task_digest = "sha256:" + "a" * 64
    _write(
        tmp_path / "tasks/demo.yaml",
        {"family": "demo", "difficulty_axes": {"size": [1, 2]}},
    )
    _write(tmp_path / "reports/harbor.json", _harbor_receipt("demo", task_digest))
    _write(
        tmp_path / "reports/models.json",
        {
            "tasks": [
                {
                    "task": "demo", "decision": "review-promising", "evidence_level": "E3",
                    "discrimination_index_observed_gap": 0.5,
                    "arms": {
                        "frontier": {"n": 3, "complete": 3, "metric_n": 3, "solve_rate": 1.0},
                        "open": {"n": 3, "complete": 3, "metric_n": 3, "solve_rate": 0.0},
                    },
                }
            ]
        },
    )

    card = pipeline.build_cards(
        task_inputs=[tmp_path / "tasks"], screening_inputs=[tmp_path / "reports"]
    )[0][0]
    assert card["decision"] == "quarantine"
    assert any("unique shared task-tree digest" in value for value in card["limitations"])


def test_pipeline_run_writes_deterministic_manifest_and_quarantines_unknown(tmp_path):
    _write(
        tmp_path / "tasks" / "unbound.yaml",
        {"family": "unbound", "title": "Unbound task"},
    )
    report = _write(
        tmp_path / "reports" / "unknown.json",
        {"schema_version": "orbenchlab.screening-report.v1", "tasks": []},
    )
    out = tmp_path / "out"
    result = pipeline.run(out=out, task_inputs=[tmp_path / "tasks"], screening_inputs=[report])
    assert result["task_count"] == 1
    assert json.loads((out / "pipeline-summary.json").read_text())["quarantined_count"] == 1
    assert (out / "task-cards.md").read_text().startswith("# ORBenchLab 自动任务总览")
    first = {path.name: path.read_bytes() for path in out.iterdir()}
    pipeline.run(out=out, task_inputs=[tmp_path / "tasks"], screening_inputs=[report])
    second = {path.name: path.read_bytes() for path in out.iterdir()}
    assert first == second
    manifest = json.loads((out / "pipeline-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        actual = "sha256:" + hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == actual
    summary = json.loads((out / "pipeline-summary.json").read_text())
    assert summary["task_cards_digest"] == manifest["files"]["task-cards.json"]
