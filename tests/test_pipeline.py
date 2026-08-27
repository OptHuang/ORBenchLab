from __future__ import annotations

import json
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
                        "ark": {"n": 2, "complete": 2, "metric_n": 2, "solve_rate": 0.5, "quality_pass_rate": 0.5, "mean_feasibility": 0.75, "infra_exceptions": []}
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
    assert model["solve_rate"] == 0.75
    assert model["quality_pass_rate"] == 0.75
    assert "Coupled constraint audit" in card["summary_markdown"]
    assert len(card["difficulty"]["axes"]) == 2


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
