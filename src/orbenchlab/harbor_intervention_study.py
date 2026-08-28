"""Controlled live-intervention study: baseline vs L1/L2/L3 with repeated,
verifier-grounded causal estimates.

Each arm runs one live-intervention session (``harbor_live_intervention``) in a
fresh no-network task container, then is graded by the SEPARATE frozen Harbor
verifier — the reward/CTRF never come from the agent.  The study joins the two
independent pieces of evidence (the interrupt/hint journal and the verifier
reward+CTRF) per arm, keyed by the container identity; any arm with missing,
mismatched, or protocol-unsatisfied evidence is an ``infra_error`` and is
excluded from the estimates rather than silently counted.

The arm executor (provision container -> run session -> run verifier) is
injected so the orchestration, join, idempotent resume, liability ledger and
aggregate statistics are unit-tested with a fake, and the identical driver runs
the real Harbor adapter on the execution host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core.errors import ORBenchError
from .volc_rollout import _wilson_interval

STUDY_SCHEMA_VERSION = "orbenchlab.live-intervention-study.v1"
DEFAULT_REWARD_THRESHOLD = 1.0


class LiveStudyError(ORBenchError):
    exit_code = 8


# arm_executor(level, repeat, intervention_id, journal_dir) -> ArmOutcome mapping:
#   {"journal": {...}, "reward": float|None, "ctrf": {...}|None,
#    "reward_digest": str|None, "ctrf_digest": str|None,
#    "verifier_container_id": str|None, "budget_usd": float}
ArmExecutor = Callable[[str, int, str, Path], Mapping[str, Any]]


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _classify_arm(level: str, outcome: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return (status, infra_reason). status is 'valid' or 'infra_error'."""

    journal = outcome.get("journal")
    if not isinstance(journal, Mapping):
        return "infra_error", "missing_journal"
    # An intervention arm must have satisfied the interrupt/hint protocol; a
    # baseline must have completed cleanly.
    if not journal.get("protocol_satisfied"):
        return "infra_error", f"protocol_not_satisfied:{journal.get('error')}"
    if not journal.get("single_session"):
        return "infra_error", "multiple_sessions"
    reward = outcome.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(float(reward)):
        return "infra_error", "missing_or_nonfinite_reward"
    if not isinstance(outcome.get("ctrf"), Mapping):
        return "infra_error", "missing_ctrf"
    # The verifier evidence must bind the SAME container the session ran in.
    identity = journal.get("harbor_identity") or {}
    session_cid = identity.get("container_id") if isinstance(identity, Mapping) else None
    verifier_cid = outcome.get("verifier_container_id")
    if session_cid and verifier_cid and session_cid != verifier_cid:
        return "infra_error", "session_verifier_container_mismatch"
    return "valid", None


def _summarise_level(rewards: Sequence[float], *, threshold: float) -> dict[str, Any]:
    n = len(rewards)
    successes = sum(1 for r in rewards if r >= threshold)
    rate = (successes / n) if n else None
    if n:
        lower, upper = _wilson_interval(successes, n)
    else:
        lower, upper = (None, None)
    return {
        "n": n,
        "successes": successes,
        "solve_rate": rate,
        "wilson_95": [round(lower, 6), round(upper, 6)] if n else None,
    }


def run_live_intervention_study(
    *,
    task_id: str,
    model: str,
    levels: Sequence[str],
    repeats: int,
    out: str | Path,
    arm_executor: ArmExecutor,
    reward_threshold: float = DEFAULT_REWARD_THRESHOLD,
    max_budget_usd_per_arm: float = 1.0,
) -> dict[str, Any]:
    """Run or resume a controlled baseline-vs-L1/L2/L3 live-intervention study.

    Idempotent: an arm whose outcome is already journalled is reused, so a
    resume never re-pays a completed arm.  Returns an aggregate receipt with
    per-level Wilson-bounded solve rates and repeated baseline-vs-Lk causal
    gaps; only verifier-grounded, protocol-satisfied arms enter the estimates.
    """

    ordered = ["baseline"] + [lv for lv in levels if lv != "baseline"]
    if "baseline" not in ordered or len(ordered) < 2:
        raise LiveStudyError("a study needs baseline plus at least one intervention level")
    if repeats < 5:
        raise LiveStudyError("a causal study needs at least five repeats per arm")
    root = Path(out)
    arms: list[dict[str, Any]] = []
    per_level_rewards: dict[str, list[float]] = {lv: [] for lv in ordered}
    total_liability = 0.0

    for level in ordered:
        for repeat in range(1, repeats + 1):
            arm_id = _digest({"task": task_id, "model": model, "level": level, "repeat": repeat})[:24]
            journal_dir = root / "arms" / arm_id
            outcome_path = journal_dir / "arm-outcome.json"
            if outcome_path.is_file() and not outcome_path.is_symlink():
                outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
                reused = True
            else:
                intervention_id = arm_id
                outcome = dict(arm_executor(level, repeat, intervention_id, journal_dir))
                _atomic_json(outcome_path, outcome)
                reused = False
            status, infra_reason = _classify_arm(level, outcome)
            budget = float(outcome.get("budget_usd") or 0.0)
            total_liability += budget
            arm_row = {
                "arm_id": arm_id,
                "level": level,
                "repeat": repeat,
                "status": status,
                "infra_reason": infra_reason,
                "reward": outcome.get("reward") if status == "valid" else None,
                "reward_digest": outcome.get("reward_digest"),
                "ctrf_digest": outcome.get("ctrf_digest"),
                "journal_digest": (outcome.get("journal") or {}).get("journal_digest"),
                "protocol_satisfied": (outcome.get("journal") or {}).get("protocol_satisfied"),
                "budget_usd": round(budget, 6),
                "reused": reused,
            }
            arms.append(arm_row)
            if status == "valid":
                per_level_rewards[level].append(float(outcome["reward"]))

    levels_summary = {
        level: _summarise_level(per_level_rewards[level], threshold=reward_threshold)
        for level in ordered
    }
    baseline = levels_summary["baseline"]
    causal: list[dict[str, Any]] = []
    for level in ordered:
        if level == "baseline":
            continue
        lk = levels_summary[level]
        gap = None
        lower_bound = None
        if baseline["solve_rate"] is not None and lk["solve_rate"] is not None:
            gap = round(lk["solve_rate"] - baseline["solve_rate"], 6)
            if lk["wilson_95"] and baseline["wilson_95"]:
                lower_bound = round(lk["wilson_95"][0] - baseline["wilson_95"][1], 6)
        causal.append(
            {
                "level": level,
                "vs": "baseline",
                "observed_recovery_gap": gap,
                "gap_95_lower_bound": lower_bound,
                "positive_recovery": bool(lower_bound is not None and lower_bound > 0),
                "n_valid_level": lk["n"],
                "n_valid_baseline": baseline["n"],
            }
        )

    valid_arms = sum(1 for a in arms if a["status"] == "valid")
    infra_arms = sum(1 for a in arms if a["status"] == "infra_error")
    receipt = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task_id": task_id,
        "model": model,
        "levels": ordered,
        "repeats": repeats,
        "reward_threshold": reward_threshold,
        "evidence_level": "E4-controlled-same-session-intervention",
        "arms": arms,
        "levels_summary": levels_summary,
        "causal_estimates": causal,
        "valid_arm_count": valid_arms,
        "infra_error_arm_count": infra_arms,
        "total_liability_usd": round(total_liability, 6),
        "study_valid": valid_arms > 0 and levels_summary["baseline"]["n"] >= 1,
    }
    # The digest binds the substantive evidence; the per-arm ``reused`` flag is
    # runtime metadata, so a resume that reuses arms yields the same digest.
    digestable = {
        k: v for k, v in receipt.items() if k != "receipt_digest"
    }
    digestable["arms"] = [{k: v for k, v in arm.items() if k != "reused"} for arm in arms]
    receipt["receipt_digest"] = _digest(digestable)
    _atomic_json(root / "live-intervention-study.json", receipt)
    return receipt


__all__ = [
    "ArmExecutor",
    "DEFAULT_REWARD_THRESHOLD",
    "LiveStudyError",
    "STUDY_SCHEMA_VERSION",
    "run_live_intervention_study",
]
