from __future__ import annotations

import json
from pathlib import Path

from tools.screening_report import build_report, render_markdown


def _write_trial(
    job: Path,
    name: str,
    model: str,
    feasibility: float,
    quality: float,
    *,
    trajectory: dict | None = None,
) -> None:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "oragentbench/demo",
                "trial_name": name,
                "finished_at": "2026-08-27T00:01:00Z",
                "config": {"agent": {"model_name": model}},
                "agent_result": {"n_input_tokens": 10, "n_output_tokens": 4, "cost_usd": 0.01},
                "verifier_result": {"rewards": {"feasibility": feasibility, "quality": quality}},
            }
        )
    )
    if trajectory is not None:
        (trial / "agent" / "trajectory.json").write_text(
            json.dumps(trajectory), encoding="utf-8"
        )


def test_report_keeps_observed_gap_conservative(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "j", "stats": {"n_completed_trials": 2}}))
    _write_trial(job, "a__1", "strong", 1.0, 2.0)
    _write_trial(job, "b__1", "weak", 0.0, 0.0)

    report = build_report(job)
    task = report["tasks"][0]
    assert task["discrimination_index_observed_gap"] == 1.0
    assert task["decision"] == "review-promising"
    assert task["evidence_level"] == "E3"
    text = render_markdown(report)
    assert "strong" in text and "weak" in text
    assert "Prompt-free trajectory structure" in text
    assert "capability" not in text.lower()


def test_report_adds_prompt_free_trajectory_structure(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "j"}))
    _write_trial(
        job,
        "a__1",
        "model",
        1.0,
        2.0,
        trajectory={
            "schema_version": "1.0",
            "steps": [
                {
                    "timestamp": "2026-08-27T00:00:00Z",
                    "message": "secret prompt must not appear",
                    "tool_calls": [{"function_name": "Bash"}],
                    "reasoning_content": "private reasoning",
                    "observation": "private output",
                    "metrics": {"prompt_tokens": 3, "completion_tokens": 2},
                },
                {
                    "timestamp": "2026-08-27T00:00:02Z",
                    "tool_calls": [{"function_name": "Bash"}, {"function_name": "Read"}],
                    "metrics": {"prompt_tokens": 4, "completion_tokens": 5},
                },
            ],
        },
    )
    report = build_report(job)
    summary = report["raw_rows"][0]["trajectory"]
    assert summary["present"] is True
    assert summary["evidence_level"] == "E2"
    assert summary["step_count"] == 2
    assert summary["tool_call_count"] == 3
    assert summary["tool_name_counts"] == {"Bash": 2, "Read": 1}
    assert summary["elapsed_seconds"] == 2.0
    assert summary["step_prompt_tokens"] == 7
    assert "secret prompt" not in json.dumps(summary)


def test_timeout_keeps_verifier_observation_but_is_not_complete(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "j"}))
    trial = job / "a__1"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "oragentbench/demo",
                "trial_name": "a__1",
                "finished_at": "2026-08-27T00:01:00Z",
                "config": {"agent": {"model_name": "model"}},
                "exception_info": {"exception_type": "AgentTimeoutError"},
                "verifier_result": {
                    "rewards": {"feasibility": 1.0, "quality": 0.0}
                },
            }
        )
    )
    report = build_report(job)
    row = report["raw_rows"][0]
    assert row["verifier_observed"] is True
    assert row["complete"] is False
    assert row["exception"] == "AgentTimeoutError"
    assert report["tasks"][0]["arms"]["model"]["infra_exceptions"] == [
        "AgentTimeoutError"
    ]
    assert report["tasks"][0]["arms"]["model"]["metric_n"] == 0
    assert report["tasks"][0]["arms"]["model"]["solve_rate"] is None
    assert report["tasks"][0]["arms"]["model"]["mean_feasibility"] is None
    assert report["tasks"][0]["decision"] == "collect-more-evidence"


def test_timeout_verifier_fact_is_excluded_from_rate_denominator(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "j"}))
    _write_trial(job, "complete__1", "model", 1.0, 2.0)
    trial = job / "timeout__1"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "oragentbench/demo",
                "trial_name": "timeout__1",
                "finished_at": "2026-08-27T00:01:00Z",
                "config": {"agent": {"model_name": "model"}},
                "exception_info": {"exception_type": "AgentTimeoutError"},
                "verifier_result": {
                    "rewards": {"feasibility": 0.0, "quality": 0.0}
                },
            }
        )
    )
    report = build_report(job)
    arm = report["tasks"][0]["arms"]["model"]
    assert arm["n"] == 2
    assert arm["complete"] == 1
    assert arm["metric_n"] == 1
    assert arm["solve_rate"] == 1.0
    assert arm["quality_pass_rate"] == 1.0
