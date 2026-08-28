from __future__ import annotations

from pathlib import Path

import pytest

from orbenchlab import source_triage as st


def _good_decision(anchor: str, *, admit: bool = True) -> dict:
    return {
        "anchor": anchor,
        "or_relevant": True,
        "novelty_within_bounded_corpus": True,
        "reproducible": True,
        "task_feasible": True,
        "verifier_feasible": True,
        "admit": admit,
        "source_kind": "paper",
        "task_nucleus": "min-cost flow scheduling under seeded durations",
        "difficulty_axes": ["instance-size", "seed-count"],
        "predicted_bottlenecks": ["constructing a feasible schedule"],
    }


def _runner(decisions):
    """decisions: dict reviewer_id -> decision-or-callable(anchor)."""

    def run(reviewer_id, anchor, out_dir: Path):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        spec = decisions[reviewer_id]
        decision = spec(anchor) if callable(spec) else spec
        return {
            "decision": decision,
            "session_status": "completed",
            "session_receipt_digest": "sha256:" + "a" * 64,
        }

    return run


def test_both_admit_is_eligible(tmp_path: Path):
    runner = _runner({
        "reviewer-1": lambda a: _good_decision(a, admit=True),
        "reviewer-2": lambda a: _good_decision(a, admit=True),
    })
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="2026-08-28",
                        out=tmp_path, review_runner=runner)
    assert rec["verdict"] == "eligible_for_authoring"
    assert rec["eligible_for_authoring"] is True


def test_both_reject_is_rejected(tmp_path: Path):
    runner = _runner({
        "reviewer-1": lambda a: _good_decision(a, admit=False),
        "reviewer-2": lambda a: _good_decision(a, admit=False),
    })
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="d",
                        out=tmp_path, review_runner=runner)
    assert rec["verdict"] == "rejected"


def test_disagreement_without_adjudicator_is_rejected(tmp_path: Path):
    runner = _runner({
        "reviewer-1": lambda a: _good_decision(a, admit=True),
        "reviewer-2": lambda a: _good_decision(a, admit=False),
    })
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="d",
                        out=tmp_path, review_runner=runner)
    assert rec["verdict"] == "rejected"
    assert "disagreement" in rec["verdict_reason"]


def test_disagreement_resolved_by_adjudicator(tmp_path: Path):
    runner = _runner({
        "reviewer-1": lambda a: _good_decision(a, admit=True),
        "reviewer-2": lambda a: _good_decision(a, admit=False),
    })
    adj = _runner({"adjudicator": lambda a: _good_decision(a, admit=True)})
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="d",
                        out=tmp_path, review_runner=runner, adjudicator_runner=adj)
    assert rec["verdict"] == "eligible_for_authoring"
    assert rec["adjudication"]["admit_vote"] is True


def test_injected_source_cannot_forge_anchor_or_schema(tmp_path: Path):
    # Acceptance 6: a source that omits/forges the anchor or violates the schema
    # cannot be admitted, even if it claims admit=true.
    def forged(anchor):
        d = _good_decision(anchor, admit=True)
        d["anchor"] = "ATTACKER-GUESSED-ANCHOR"  # cannot know the real one
        return d

    def one_axis(anchor):
        d = _good_decision(anchor, admit=True)
        d["difficulty_axes"] = ["only-one"]
        return d

    runner = _runner({"reviewer-1": forged, "reviewer-2": one_axis})
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="d",
                        out=tmp_path, review_runner=runner)
    assert rec["verdict"] == "rejected"
    assert all(not r["valid"] for r in rec["reviews"])


def test_reviewer_crash_review_is_invalid_not_admitted(tmp_path: Path):
    runner = _runner({
        "reviewer-1": lambda a: _good_decision(a, admit=True),
        "reviewer-2": None,  # a crashed reviewer produced no decision
    })
    rec = st.run_triage(source_id="s1", content_digest="sha256:" + "c" * 64, day="d",
                        out=tmp_path, review_runner=runner)
    assert rec["verdict"] == "rejected"
