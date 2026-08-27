from __future__ import annotations

import json
from pathlib import Path

from tools.monitor_harbor_campaign import snapshot


def test_snapshot_is_observe_only_and_redacts_stream_contents(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "task__abc" / "agent"
    trial.mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps(
            {
                "id": "job-id",
                "started_at": "2026-08-27T00:00:00Z",
                "stats": {"n_running_trials": 1, "n_pending_trials": 0},
            }
        )
    )
    (trial / "claude-code.txt").write_text(
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-08-27T00:00:01Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "private prompt"},
                        {"type": "tool_use", "name": "Bash"},
                    ]
                },
            }
        )
        + "\n"
    )

    observed = snapshot(job)
    assert observed["evidence_level"] == "E1"
    assert observed["trials"][0]["status"] == "running"
    stream = observed["trials"][0]["stream"]
    assert stream["assistant_turns"] == 1
    assert stream["tool_uses"] == 1
    assert "private prompt" not in json.dumps(observed)


def test_snapshot_copies_completed_verifier_fact_without_comparison(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "task__abc"
    (trial / "agent").mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps(
            {
                "id": "job-id",
                "started_at": "2026-08-27T00:00:00Z",
                "finished_at": "2026-08-27T00:01:00Z",
                "stats": {"n_completed_trials": 1},
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "oragentbench/demo",
                "finished_at": "2026-08-27T00:01:00Z",
                "config": {"agent": {"model_name": "example-model"}},
                "verifier_result": {"rewards": {"feasibility": 1.0}},
            }
        )
    )
    observed = snapshot(job)
    assert observed["evidence_level"] == "E3"
    assert observed["trials"][0]["model"] == "example-model"
    assert observed["trials"][0]["rewards"] == {"feasibility": 1.0}
