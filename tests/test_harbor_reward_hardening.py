"""Regression tests for verifier-result integrity at the Harbor ingest seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orbenchlab.core.errors import EvidenceError
from orbenchlab.ingest.harbor import ingest_harbor_bundle
from orbenchlab.report.model import NormalizedRollout, compute_metrics


def _write_bundle(
    tmp_path: Path,
    *,
    rewards: Any,
    exception_type: str | None = None,
    exception_message: str | None = None,
    step_rewards: list[Any] | None = None,
) -> Path:
    run_root = tmp_path / "run"
    plan_dir = run_root / "plan"
    trial = run_root / "jobs" / "job" / "task__fixture"
    (plan_dir / "jobs").mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)

    entry = {
        "run_id": "0123456789abcdef",
        "task_name": "task",
        "agent_id": "oracle",
        "seed": 1,
        "attempt": 1,
        "job_name": "job",
        "match_key": "fixture",
    }
    (plan_dir / "plan.json").write_text(
        json.dumps(
            {
                "campaign_id": "fixture-campaign",
                "integration": "oragentbench",
                "site": "local-docker",
                "evidence_intent": "exploratory",
                "runs": [entry],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (plan_dir / "plan_ledger.json").write_text(
        json.dumps({"campaign_id": "fixture-campaign", "entries": [entry]}) + "\n",
        encoding="utf-8",
    )
    (plan_dir / "jobs" / "job.yaml").write_text(
        "agents:\n  - name: oracle\n", encoding="utf-8"
    )
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.0",
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "trial_name": "task__fixture",
        "task_name": "oragentbench/task",
        "agent_result": {"cost_usd": 0.0},
        "verifier_result": {"rewards": rewards},
        "exception_info": (
            {
                "exception_type": exception_type,
                "exception_message": exception_message,
            }
            if exception_type is not None
            else None
        ),
    }
    if step_rewards is not None:
        result["step_results"] = []
        for index, step_reward in enumerate(step_rewards, start=1):
            step_name = f"step{index}"
            trajectory = trial / "steps" / step_name / "agent" / "trajectory.json"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text(
                json.dumps(
                    {
                        "schema_version": "ATIF-v1.0",
                        "steps": [
                            {"step_id": 1, "source": "agent", "message": "done"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result["step_results"].append(
                {
                    "step_name": step_name,
                    "verifier_result": {"rewards": step_reward},
                    "exception_info": None,
                }
            )
    (trial / "result.json").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    return run_root


def _normalized_trial(run_root: Path) -> dict[str, Any]:
    return json.loads(
        (run_root / "normalized" / "rollout.json").read_text(encoding="utf-8")
    )["trials"][0]


def test_rewards_plus_permission_error_are_preserved_but_quarantined(tmp_path: Path):
    run_root = _write_bundle(
        tmp_path,
        rewards={"feasibility": 1.0, "quality": 1.25},
        exception_type="PermissionError",
    )

    ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")

    trial = _normalized_trial(run_root)
    assert trial["scores"] == {"feasibility": 1.0, "quality": 1.25}
    assert trial["attribution"] == "unknown"
    assert trial["counts_toward_capability"] is False
    assert trial["infra_suspect"] is True
    assert trial["exclusion_basis"] == "ambiguous_reward_with_exception"

    rollout = NormalizedRollout.load(run_root / "normalized" / "rollout.json")
    metrics = {metric.name: metric for metric in compute_metrics(rollout)}
    assert metrics["oracle_pass_rate"].value is None
    assert "excluded" in (metrics["oracle_pass_rate"].unmet_requirement or "")
    assert metrics["excluded_trial_share"].value == 1.0


@pytest.mark.parametrize(
    "bad_rewards",
    [
        {},
        {"feasiblity": 1.0, "quality": 1.0},
        {"feasibility": 1.0},
        {"feasibility": 1.0, "quality": 1.0, "typo": 0.0},
        {"feasibility": True, "quality": 1.0},
        {"feasibility": 1.0, "quality": "1.0"},
        {"feasibility": 1.0, "quality": float("nan")},
        {"feasibility": 1.0, "quality": 1.0, "reward": 2.0 / 3.0},
    ],
    ids=[
        "empty",
        "misspelled-feasibility",
        "missing-quality",
        "unknown-key",
        "boolean",
        "string",
        "non-finite",
        "single-step-scalar-extension",
    ],
)
def test_single_step_reward_schema_rejects_drift(tmp_path: Path, bad_rewards: Any):
    run_root = _write_bundle(tmp_path, rewards=bad_rewards)

    with pytest.raises(EvidenceError, match="reward schema"):
        ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")


def test_real_multistep_scalar_reward_extension_is_versioned(tmp_path: Path):
    step = {"feasibility": 1.0, "quality": 1.0, "reward": 2.0 / 3.0}
    run_root = _write_bundle(tmp_path, rewards=step, step_rewards=[step, step])

    ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")

    trial = _normalized_trial(run_root)
    assert trial["reward_schema_version"] == "oragentbench-multistep-rewards-v1"
    assert trial["scores"] == step


def test_multistep_nested_reward_schema_is_not_silently_ignored(tmp_path: Path):
    valid = {"feasibility": 1.0, "quality": 1.0, "reward": 2.0 / 3.0}
    invalid = {"feasibility": 1.0, "quality": 1.0, "qualitty": 1.0}
    run_root = _write_bundle(
        tmp_path,
        rewards=valid,
        step_rewards=[valid, invalid],
    )

    with pytest.raises(EvidenceError, match=r"step_results\[1\].*reward schema"):
        ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")


@pytest.mark.parametrize(
    ("exception_type", "attribution", "counts", "infra_suspect", "exclusion"),
    [
        ("AgentTimeoutError", "agent", True, False, None),
        (
            "VerifierTimeoutError",
            "verifier",
            False,
            False,
            "provider_or_verifier_error",
        ),
    ],
)
def test_pinned_wrapper_preserved_timeout_zero_reward_is_ingested_with_provenance(
    tmp_path: Path,
    exception_type: str,
    attribution: str,
    counts: bool,
    infra_suspect: bool,
    exclusion: str | None,
):
    message = (
        f"{exception_type} preserved from the previous top-level job summary "
        "during cleanup-before-resume."
    )
    run_root = _write_bundle(
        tmp_path,
        rewards={"quality": 0.0, "feasibility": 0.0, "reward": 0.0},
        exception_type=exception_type,
        exception_message=message,
        step_rewards=[],
    )

    ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")

    trial = _normalized_trial(run_root)
    assert trial["scores"] == {
        "feasibility": 0.0,
        "quality": 0.0,
        "reward": 0.0,
    }
    assert (
        trial["reward_schema_version"]
        == "oragentbench-preserved-timeout-rewards-v1"
    )
    assert (
        trial["reward_provenance"]
        == "oragentbench-wrapper-preserved-timeout-fallback"
    )
    assert trial["attribution"] == attribution
    assert trial["counts_toward_capability"] is counts
    assert trial["infra_suspect"] is infra_suspect
    assert trial["exclusion_basis"] == exclusion


@pytest.mark.parametrize(
    ("exception_type", "exception_message", "step_rewards", "rewards"),
    [
        (
            "AgentTimeoutError",
            "AgentTimeoutError preserved from the previous top-level job summary "
            "during cleanup-before-resume.",
            None,
            {"quality": 0.0, "feasibility": 0.0, "reward": 0.0},
        ),
        (
            "AgentTimeoutError",
            "ordinary Harbor agent timeout",
            [],
            {"quality": 0.0, "feasibility": 0.0, "reward": 0.0},
        ),
        (
            "PermissionError",
            "PermissionError preserved from the previous top-level job summary "
            "during cleanup-before-resume.",
            [],
            {"quality": 0.0, "feasibility": 0.0, "reward": 0.0},
        ),
        (
            "AgentTimeoutError",
            "AgentTimeoutError preserved from the previous top-level job summary "
            "during cleanup-before-resume.",
            [],
            {"quality": 0.0, "feasibility": 0.0, "reward": 0.01},
        ),
    ],
    ids=[
        "missing-empty-step-results-marker",
        "ordinary-timeout-message",
        "wrong-exception-type",
        "nonzero-synthetic-reward",
    ],
)
def test_three_key_single_result_is_rejected_unless_exact_pinned_wrapper_fallback(
    tmp_path: Path,
    exception_type: str,
    exception_message: str,
    step_rewards: list[Any] | None,
    rewards: dict[str, float],
):
    run_root = _write_bundle(
        tmp_path,
        rewards=rewards,
        exception_type=exception_type,
        exception_message=exception_message,
        step_rewards=step_rewards,
    )

    with pytest.raises(EvidenceError, match="reward schema"):
        ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")
