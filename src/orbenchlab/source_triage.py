"""Dual-reviewer semantic triage for the daily source pipeline (P0-E).

The agent decides semantics; the harness decides eligibility.  Two INDEPENDENT
Volc Claude no-Bash sessions each read only the sandboxed frozen source and emit
an echo-anchored JSON verdict on: OR relevance, source kind, bounded-corpus
novelty, reproducibility, task/verifier feasibility, a task nucleus, >=2
difficulty axes and predicted bottlenecks.  A bounded adjudicator breaks a
disagreement; an unresolved disagreement, a missing/echo-mismatched anchor, or
a schema violation is rejected — never admitted — so a prompt-injected source
cannot force admission or bypass the schema.

The harness then re-verifies the aggregate against the individual reviewer
receipts and emits a signed triage decision whose entry gate is
``eligible_for_authoring`` (``promising`` is reserved for post-Harbor
discrimination evidence).  The reviewer runner is injected so the aggregation,
anchoring, adjudication and re-verification logic is unit-tested with fakes and
the identical driver runs real sandboxed sessions on the host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core.errors import ORBenchError

TRIAGE_SCHEMA = "orbenchlab.source-triage.v1"
REVIEW_SCHEMA = "orbenchlab.source-review.v1"


class SourceTriageError(ORBenchError):
    exit_code = 8


# review_runner(reviewer_id, anchor, out_dir) -> {
#   "decision": {...}, "session_receipt_digest": str, "session_status": str}
ReviewRunner = Callable[[str, str, Path], Mapping[str, Any]]


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def compute_anchor(*, source_id: str, content_digest: str, day: str) -> str:
    """Harness-generated echo anchor the reviewer must reproduce verbatim.

    Because the anchor is derived from harness-held values, a prompt-injected
    source document cannot know it, so it cannot forge a schema-valid verdict.
    """

    return "ORBENCH-TRIAGE-" + _digest({"s": source_id, "c": content_digest, "d": day}).removeprefix("sha256:")[:20]


_BOOL_FIELDS = (
    "or_relevant",
    "novelty_within_bounded_corpus",
    "reproducible",
    "task_feasible",
    "verifier_feasible",
    "admit",
)


def validate_review(decision: Any, *, anchor: str) -> tuple[bool, str | None]:
    """Return (valid, reason). A review must echo the anchor and match schema."""

    if not isinstance(decision, Mapping):
        return False, "review is not an object"
    if decision.get("anchor") != anchor:
        return False, "anchor missing or mismatched"
    for field in _BOOL_FIELDS:
        if not isinstance(decision.get(field), bool):
            return False, f"field {field} must be a boolean"
    if not str(decision.get("source_kind") or "").strip():
        return False, "source_kind is required"
    if not str(decision.get("task_nucleus") or "").strip():
        return False, "task_nucleus is required"
    axes = decision.get("difficulty_axes")
    if not isinstance(axes, list) or len([a for a in axes if str(a).strip()]) < 2:
        return False, "at least two difficulty_axes are required"
    bottlenecks = decision.get("predicted_bottlenecks")
    if not isinstance(bottlenecks, list) or not bottlenecks:
        return False, "predicted_bottlenecks are required"
    return True, None


def _admit_vote(decision: Mapping[str, Any]) -> bool:
    # A reviewer admits only if OR-relevant, novel, reproducible, feasible AND
    # explicitly sets admit; the harness recomputes rather than trusting admit.
    return bool(
        decision.get("or_relevant")
        and decision.get("novelty_within_bounded_corpus")
        and decision.get("reproducible")
        and decision.get("task_feasible")
        and decision.get("verifier_feasible")
        and decision.get("admit")
    )


def run_triage(
    *,
    source_id: str,
    content_digest: str,
    day: str,
    out: str | Path,
    review_runner: ReviewRunner,
    adjudicator_runner: ReviewRunner | None = None,
) -> dict[str, Any]:
    """Run two independent reviews (+ bounded adjudication) and re-verify.

    The verdict ``eligible_for_authoring`` is true only when the harness's own
    recomputation over valid, anchored reviewer receipts agrees on admission.
    """

    root = Path(out)
    anchor = compute_anchor(source_id=source_id, content_digest=content_digest, day=day)
    reviews: list[dict[str, Any]] = []
    for idx in range(2):
        reviewer_id = f"reviewer-{idx + 1}"
        raw = dict(review_runner(reviewer_id, anchor, root / reviewer_id))
        decision = raw.get("decision")
        valid, reason = validate_review(decision, anchor=anchor)
        reviews.append(
            {
                "reviewer_id": reviewer_id,
                "valid": valid,
                "invalid_reason": reason,
                "session_status": raw.get("session_status"),
                "session_receipt_digest": raw.get("session_receipt_digest"),
                "admit_vote": bool(valid and _admit_vote(decision)),
                "decision_digest": _digest(decision) if isinstance(decision, Mapping) else None,
            }
        )

    valid_reviews = [r for r in reviews if r["valid"]]
    verdict = "rejected"
    verdict_reason = None
    adjudication: dict[str, Any] | None = None

    if len(valid_reviews) < 2:
        verdict_reason = "a review was invalid or failed the anchored schema"
    else:
        votes = {r["admit_vote"] for r in valid_reviews}
        if votes == {True}:
            verdict = "eligible_for_authoring"
            verdict_reason = "both independent reviews admit"
        elif votes == {False}:
            verdict_reason = "both independent reviews reject"
        else:
            # Disagreement: a bounded adjudicator must break the tie with a
            # valid anchored decision, else the source is rejected.
            if adjudicator_runner is None:
                verdict_reason = "reviewer disagreement with no adjudicator"
            else:
                raw = dict(adjudicator_runner("adjudicator", anchor, root / "adjudicator"))
                decision = raw.get("decision")
                valid, reason = validate_review(decision, anchor=anchor)
                adjudication = {
                    "valid": valid,
                    "invalid_reason": reason,
                    "session_receipt_digest": raw.get("session_receipt_digest"),
                    "admit_vote": bool(valid and _admit_vote(decision)),
                }
                if not valid:
                    verdict_reason = "adjudicator produced no valid anchored decision"
                elif adjudication["admit_vote"]:
                    verdict = "eligible_for_authoring"
                    verdict_reason = "adjudicator resolved disagreement to admit"
                else:
                    verdict_reason = "adjudicator resolved disagreement to reject"

    receipt = {
        "schema_version": TRIAGE_SCHEMA,
        "source_id": source_id,
        "content_digest": content_digest,
        "day": day,
        "anchor": anchor,
        "reviews": reviews,
        "adjudication": adjudication,
        "verdict": verdict,
        "eligible_for_authoring": verdict == "eligible_for_authoring",
        "verdict_reason": verdict_reason,
    }
    # Aggregate re-verification: the verdict must follow deterministically from
    # the recorded reviewer votes (defeats a forged top-level verdict).
    recomputed = _recompute_verdict(reviews, adjudication)
    if recomputed != verdict:
        raise SourceTriageError("triage aggregate re-verification failed")
    receipt["receipt_digest"] = _digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    _atomic_json(root / "triage-decision.json", receipt)
    return receipt


def _recompute_verdict(reviews: Sequence[Mapping[str, Any]], adjudication: Mapping[str, Any] | None) -> str:
    valid = [r for r in reviews if r["valid"]]
    if len(valid) < 2:
        return "rejected"
    votes = {r["admit_vote"] for r in valid}
    if votes == {True}:
        return "eligible_for_authoring"
    if votes == {False}:
        return "rejected"
    if adjudication and adjudication.get("valid") and adjudication.get("admit_vote"):
        return "eligible_for_authoring"
    return "rejected"


__all__ = [
    "REVIEW_SCHEMA",
    "ReviewRunner",
    "SourceTriageError",
    "TRIAGE_SCHEMA",
    "compute_anchor",
    "run_triage",
    "validate_review",
]
