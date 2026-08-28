from __future__ import annotations

from pathlib import Path

import pytest

from orbenchlab import harbor_intervention_study as his


def _journal(level: str, *, satisfied: bool = True, cid: str = "cid-1") -> dict:
    return {
        "protocol_satisfied": satisfied,
        "single_session": True,
        "harbor_identity": {"container_id": cid},
        "journal_digest": "sha256:" + "a" * 64,
        "error": None,
    }


def _make_executor(*, fail_repeat: tuple[str, int] | None = None, calls=None):
    def executor(level, repeat, intervention_id, journal_dir: Path):
        if calls is not None:
            calls.append((level, repeat))
        # Baseline mostly fails; interventions mostly recover.
        reward = 0.0 if level == "baseline" else 1.0
        outcome = {
            "journal": _journal(level),
            "reward": reward,
            "ctrf": {"summary": {"passed": int(reward)}},
            "reward_digest": "sha256:" + "b" * 64,
            "ctrf_digest": "sha256:" + "c" * 64,
            "verifier_container_id": "cid-1",
            "budget_usd": 0.1,
        }
        if fail_repeat == (level, repeat):
            outcome["reward"] = None  # missing verifier reward -> infra_error
            outcome["ctrf"] = None
        return outcome

    return executor


def test_study_estimates_causal_recovery_and_joins_verifier(tmp_path: Path):
    # Acceptance: baseline vs L1/L2/L3 with repeats, verifier-grounded rewards
    # joined per arm, positive recovery gap for interventions.
    receipt = his.run_live_intervention_study(
        task_id="task-1",
        model="doubao",
        levels=["L1", "L2", "L3"],
        repeats=5,
        out=tmp_path,
        arm_executor=_make_executor(),
    )
    assert receipt["evidence_level"] == "E4-controlled-same-session-intervention"
    assert receipt["levels_summary"]["baseline"]["solve_rate"] == 0.0
    assert receipt["levels_summary"]["L1"]["solve_rate"] == 1.0
    gaps = {c["level"]: c for c in receipt["causal_estimates"]}
    assert gaps["L1"]["observed_recovery_gap"] == 1.0
    assert gaps["L1"]["positive_recovery"] is True
    assert receipt["valid_arm_count"] == 20  # 4 levels x 5 repeats
    assert receipt["infra_error_arm_count"] == 0
    assert receipt["total_liability_usd"] == pytest.approx(2.0, abs=1e-6)


def test_missing_verifier_evidence_is_infra_error_and_excluded(tmp_path: Path):
    receipt = his.run_live_intervention_study(
        task_id="task-2",
        model="doubao",
        levels=["L1"],
        repeats=5,
        out=tmp_path,
        arm_executor=_make_executor(fail_repeat=("L1", 3)),
    )
    assert receipt["infra_error_arm_count"] == 1
    # The excluded arm does not count toward L1's valid n.
    assert receipt["levels_summary"]["L1"]["n"] == 4
    bad = [a for a in receipt["arms"] if a["status"] == "infra_error"][0]
    assert bad["level"] == "L1" and bad["repeat"] == 3
    assert "missing_or_nonfinite_reward" in bad["infra_reason"]


def test_protocol_unsatisfied_arm_is_infra_error(tmp_path: Path):
    def executor(level, repeat, intervention_id, journal_dir):
        return {
            "journal": _journal(level, satisfied=(level == "baseline")),
            "reward": 1.0,
            "ctrf": {"summary": {}},
            "verifier_container_id": "cid-1",
            "budget_usd": 0.1,
        }

    receipt = his.run_live_intervention_study(
        task_id="task-3", model="m", levels=["L1"], repeats=5, out=tmp_path, arm_executor=executor
    )
    # Every L1 arm failed the protocol -> all infra_error; baseline is valid.
    assert receipt["levels_summary"]["L1"]["n"] == 0
    assert receipt["levels_summary"]["baseline"]["n"] == 5


def test_study_resume_reuses_arms_without_repaying(tmp_path: Path):
    calls: list = []
    kwargs = dict(task_id="task-4", model="m", levels=["L1"], repeats=5, out=tmp_path)
    first = his.run_live_intervention_study(arm_executor=_make_executor(calls=calls), **kwargs)
    assert len(calls) == 10  # baseline+L1, 5 each
    calls2: list = []

    def exploding(level, repeat, intervention_id, journal_dir):
        calls2.append(1)
        raise AssertionError("must not re-run a completed arm")

    second = his.run_live_intervention_study(arm_executor=exploding, **kwargs)
    assert calls2 == []
    assert second["receipt_digest"] == first["receipt_digest"]
    assert all(a["reused"] for a in second["arms"])


def test_container_mismatch_is_infra_error(tmp_path: Path):
    def executor(level, repeat, intervention_id, journal_dir):
        return {
            "journal": _journal(level, cid="session-cid"),
            "reward": 1.0,
            "ctrf": {"summary": {}},
            "verifier_container_id": "DIFFERENT-cid",
            "budget_usd": 0.1,
        }

    receipt = his.run_live_intervention_study(
        task_id="task-5", model="m", levels=["L1"], repeats=5, out=tmp_path, arm_executor=executor
    )
    assert receipt["valid_arm_count"] == 0
    bad = [a for a in receipt["arms"] if a["status"] == "infra_error"]
    assert bad and all(a["infra_reason"] == "session_verifier_container_mismatch" for a in bad)
