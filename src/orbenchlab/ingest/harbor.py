"""Verifier-grounded ingest for Harbor job directories.

Raw trajectories and verifier logs stay in the run workspace.  This module
reconciles each Harbor trial with the plan ledger, extracts only the small set
of outcome fields the report schema permits, and writes an integrity manifest
covering both raw evidence and derived artefacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core.errors import EvidenceError
from ..report import render as render_mod
from ..report.model import NormalizedRollout, validate_normalized


_ORAGENTBENCH_REQUIRED_REWARD_KEYS = frozenset({"feasibility", "quality"})
_ORAGENTBENCH_MULTISTEP_REWARD_KEYS = frozenset(
    {"feasibility", "quality", "reward"}
)
_ORAGENTBENCH_SINGLE_REWARD_SCHEMA = "oragentbench-rewards-v1"
_ORAGENTBENCH_MULTISTEP_REWARD_SCHEMA = "oragentbench-multistep-rewards-v1"
_ORAGENTBENCH_PRESERVED_TIMEOUT_REWARD_SCHEMA = (
    "oragentbench-preserved-timeout-rewards-v1"
)
_ORAGENTBENCH_UPSTREAM_REWARD_PROVENANCE = "upstream-verifier"
_ORAGENTBENCH_PRESERVED_TIMEOUT_REWARD_PROVENANCE = (
    "oragentbench-wrapper-preserved-timeout-fallback"
)
_ORAGENTBENCH_PRESERVED_TIMEOUT_EXCEPTION_TYPES = frozenset(
    {"AgentTimeoutError", "VerifierTimeoutError"}
)


@dataclass(frozen=True)
class HarborIngestResult:
    trials: int
    orphans: int
    normalized_path: Path
    report_dir: Path
    integrity_path: Path


def _load_json(path: Path, *, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {what} {path}: {type(exc).__name__}: {exc}") from None
    if not isinstance(value, dict):
        raise EvidenceError(f"{what} {path} must contain a JSON object")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _task_name(result: Mapping[str, Any]) -> str:
    raw = str(result.get("task_name") or "")
    return raw.rsplit("/", 1)[-1]


def _is_job_aggregate(result: Mapping[str, Any]) -> bool:
    """Harbor writes ``<job>/result.json`` in addition to trial results."""
    return (
        "n_total_trials" in result
        and "stats" in result
        and not result.get("task_name")
        and not result.get("trial_name")
    )


def _known_job_name(
    result_path: Path,
    jobs_root: Path,
    result: Mapping[str, Any],
    known: set[str],
) -> str | None:
    explicit = result.get("job_name")
    if isinstance(explicit, str) and explicit in known:
        return explicit
    try:
        parts = result_path.relative_to(jobs_root).parts[:-1]
    except ValueError:
        return None
    hits = [part for part in parts if part in known]
    if len(hits) > 1:
        raise EvidenceError(
            f"{result_path}: path contains several planned job names {hits}; refusing to guess"
        )
    return hits[0] if hits else None


def _trace_status(result_path: Path, result: Mapping[str, Any]) -> str:
    trial = result_path.parent
    step_results = result.get("step_results")
    paths: list[Path]
    if isinstance(step_results, list) and step_results:
        paths = []
        for index, step in enumerate(step_results, start=1):
            name = str(step.get("step_name") or f"step{index}") if isinstance(step, dict) else f"step{index}"
            paths.append(trial / "steps" / name / "agent" / "trajectory.json")
    else:
        paths = [trial / "agent" / "trajectory.json"]

    valid = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        steps = payload.get("steps") if isinstance(payload, dict) else None
        schema = payload.get("schema_version") if isinstance(payload, dict) else None
        if str(schema or "").startswith("ATIF-") and isinstance(steps, list) and steps:
            valid += 1
    if valid == len(paths) and paths:
        return "complete"
    return "partial" if valid else "missing"


def _exception_types(result: Mapping[str, Any]) -> list[str]:
    containers: list[Any] = [result.get("exception_info")]
    for step in result.get("step_results") or []:
        if isinstance(step, dict):
            containers.append(step.get("exception_info"))
    values = []
    for info in containers:
        if isinstance(info, dict) and info.get("exception_type"):
            values.append(str(info["exception_type"]))
    return values


def _validated_reward_mapping(
    raw: Any,
    *,
    where: str,
    multistep: bool,
) -> dict[str, float]:
    """Validate the pinned ORAgentBench verifier contract without coercion.

    The pinned dataset emits the two official dimensions on every task.  Its
    multi-step tasks additionally emit Harbor's derived scalar ``reward`` on
    some (but not all fallback) verifier paths.  That extension is accepted
    only for a real Harbor multi-step result; all other keys are drift.
    """

    allowed = (
        _ORAGENTBENCH_MULTISTEP_REWARD_KEYS
        if multistep
        else _ORAGENTBENCH_REQUIRED_REWARD_KEYS
    )
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise EvidenceError(
            f"{where} reward schema violation: expected an object, got "
            f"{type(raw).__name__}"
        )

    actual = set(raw)
    missing = sorted(_ORAGENTBENCH_REQUIRED_REWARD_KEYS - actual)
    unknown = sorted(actual - allowed)
    if missing:
        problems.append(f"missing required key(s) {missing}")
    if unknown:
        problems.append(f"unknown key(s) {unknown}")
    for key in sorted(actual & allowed):
        value = raw[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            problems.append(
                f"{key!r} must be a finite non-boolean number, got "
                f"{type(value).__name__}"
            )
    if problems:
        raise EvidenceError(f"{where} reward schema violation: {'; '.join(problems)}")
    return {key: float(raw[key]) for key in sorted(actual)}


def _preserved_timeout_fallback_exception(
    result: Mapping[str, Any], raw_rewards: Any
) -> str | None:
    """Recognize only the pinned wrapper's synthetic resume-cleanup result.

    ORAgentBench commit ``c9eb952`` synthesizes a zero vector when
    ``--cleanup-before-resume`` recovers an agent or verifier timeout from the
    previous job summary.  The scalar ``reward`` is not part of the ordinary
    single-step verifier contract, so the complete wrapper fingerprint is
    required before accepting it.
    """

    info = result.get("exception_info")
    if not isinstance(info, dict):
        return None
    exception_type = info.get("exception_type")
    if exception_type not in _ORAGENTBENCH_PRESERVED_TIMEOUT_EXCEPTION_TYPES:
        return None
    expected_message = (
        f"{exception_type} preserved from the previous top-level job summary "
        "during cleanup-before-resume."
    )
    if info.get("exception_message") != expected_message:
        return None
    if result.get("step_results") != []:
        return None
    if not isinstance(raw_rewards, dict):
        return None
    if set(raw_rewards) != _ORAGENTBENCH_MULTISTEP_REWARD_KEYS:
        return None
    for value in raw_rewards.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
        ):
            return None
    return str(exception_type)


def _validated_rewards(
    result: Mapping[str, Any],
) -> tuple[dict[str, float] | None, str | None, str | None, str | None]:
    """Validate top-level and per-step rewards, returning the report vector."""

    raw_steps = result.get("step_results")
    if raw_steps is not None and not isinstance(raw_steps, list):
        raise EvidenceError(
            f"trial {result.get('trial_name')!r} step_results must be a list or null"
        )
    multistep = bool(raw_steps)
    trial_name = result.get("trial_name")

    verifier = result.get("verifier_result")
    if verifier is not None and not isinstance(verifier, dict):
        raise EvidenceError(
            f"trial {trial_name!r} verifier_result must be an object or null"
        )
    raw_rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    preserved_timeout_exception = _preserved_timeout_fallback_exception(
        result, raw_rewards
    )
    rewards = (
        _validated_reward_mapping(
            raw_rewards,
            where=f"trial {trial_name!r} verifier_result",
            multistep=multistep or preserved_timeout_exception is not None,
        )
        if raw_rewards is not None
        else None
    )

    for index, step in enumerate(raw_steps or []):
        if not isinstance(step, dict):
            raise EvidenceError(
                f"trial {trial_name!r} step_results[{index}] must be an object"
            )
        step_verifier = step.get("verifier_result")
        if step_verifier is None:
            continue
        if not isinstance(step_verifier, dict):
            raise EvidenceError(
                f"trial {trial_name!r} step_results[{index}].verifier_result "
                "must be an object or null"
            )
        step_raw = step_verifier.get("rewards")
        if step_raw is not None:
            _validated_reward_mapping(
                step_raw,
                where=f"trial {trial_name!r} step_results[{index}] verifier_result",
                multistep=True,
            )

    if rewards is None:
        return None, None, None, None
    if preserved_timeout_exception is not None:
        return (
            rewards,
            _ORAGENTBENCH_PRESERVED_TIMEOUT_REWARD_SCHEMA,
            _ORAGENTBENCH_PRESERVED_TIMEOUT_REWARD_PROVENANCE,
            preserved_timeout_exception,
        )
    version = (
        _ORAGENTBENCH_MULTISTEP_REWARD_SCHEMA
        if multistep
        else _ORAGENTBENCH_SINGLE_REWARD_SCHEMA
    )
    return rewards, version, _ORAGENTBENCH_UPSTREAM_REWARD_PROVENANCE, None


def _attribution(
    result: Mapping[str, Any],
    rewards: Mapping[str, Any] | None,
    preserved_timeout_exception: str | None,
) -> tuple[str, bool, bool, str | None]:
    if preserved_timeout_exception == "AgentTimeoutError":
        # The wrapper deliberately preserves this as a scored zero capability
        # outcome after resume cleanup. It is not contradictory verifier data.
        return "agent", True, False, None
    if preserved_timeout_exception == "VerifierTimeoutError":
        # Preserve the diagnostic zero vector, but a verifier timeout cannot
        # measure agent capability. Existing ingest semantics classify this as
        # a verifier exclusion rather than hard infrastructure evidence.
        return "verifier", False, False, "provider_or_verifier_error"
    if rewards is not None:
        if _exception_types(result):
            # A verifier score and a recorded exception are contradictory
            # outcome signals. Preserve the score for diagnosis, but quarantine
            # it from capability metrics instead of silently declaring success.
            return "unknown", False, True, "ambiguous_reward_with_exception"
        return "agent", True, False, None
    joined = " ".join(_exception_types(result)).lower()
    if not joined:
        raise EvidenceError(
            f"trial {result.get('trial_name')!r} has neither verifier rewards nor an exception; "
            "the runner output is incomplete"
        )
    if any(token in joined for token in ("provider", "ratelimit", "apiusage", "connection")):
        return "provider", False, False, "provider_or_verifier_error"
    if "verifier" in joined:
        return "verifier", False, False, "provider_or_verifier_error"
    if any(
        token in joined
        for token in (
            "environment",
            "docker",
            "setup",
            "artifact",
            "rewardfilenotfound",
            "permission",
        )
    ):
        return "infra", False, True, "hard_infra_evidence"
    # Agent timeouts and solver failures are capability outcomes.  They stay in
    # the denominator, but have no verifier score and therefore remain visible
    # as an empty score vector rather than being turned into a fabricated zero.
    return "agent", True, False, None


def _cost(result: Mapping[str, Any]) -> float:
    calls = result.get("step_results")
    if not isinstance(calls, list) or not calls:
        calls = [result]
    total = 0.0
    for call in calls:
        if isinstance(call, dict):
            value = (call.get("agent_result") or {}).get("cost_usd")
            if isinstance(value, (int, float)) and value >= 0:
                total += float(value)
    return total


def _job_scaffold(plan_dir: Path, job_name: str, fallback: str) -> str:
    path = plan_dir / "jobs" / f"{job_name}.yaml"
    if not path.is_file():
        return fallback
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agents = payload.get("agents") or []
        if agents and isinstance(agents[0], dict):
            return str(agents[0].get("name") or fallback)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        pass
    return fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity(run_root: Path) -> Path:
    target = run_root / "evidence" / "integrity.sha256"
    included_roots = [run_root / name for name in ("plan", "jobs", "normalized", "report")]
    rows: list[str] = []
    for root in included_roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path == target or ".tmp-" in path.name:
                continue
            rows.append(f"{_sha256(path)}  {path.relative_to(run_root).as_posix()}")
    _atomic_text(target, "\n".join(rows) + ("\n" if rows else ""))
    return target


def ingest_harbor_bundle(*, run_root: str | Path, jobs_root: str | Path) -> HarborIngestResult:
    run_root = Path(run_root).resolve()
    jobs_root = Path(jobs_root).resolve()
    plan_dir = run_root / "plan"
    plan = _load_json(plan_dir / "plan.json", what="campaign plan")
    ledger = _load_json(plan_dir / "plan_ledger.json", what="plan ledger")
    entries = ledger.get("entries")
    runs = plan.get("runs")
    if not isinstance(entries, list) or not isinstance(runs, list):
        raise EvidenceError("plan and ledger must each contain a list of runs/entries")

    plan_by_id: dict[str, dict[str, Any]] = {}
    for item in runs:
        if not isinstance(item, dict) or not item.get("run_id"):
            raise EvidenceError("plan contains a non-object run or a run without run_id")
        run_id = str(item["run_id"])
        if run_id in plan_by_id:
            raise EvidenceError(f"plan contains duplicate run_id {run_id}")
        plan_by_id[run_id] = item
    known_jobs = {str(item.get("job_name")) for item in entries if isinstance(item, dict)}
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceError("plan ledger contains a non-object entry")
        key = (str(entry.get("job_name")), str(entry.get("task_name")))
        by_key.setdefault(key, []).append(entry)
    ambiguous = sorted(key for key, values in by_key.items() if len(values) != 1)
    if ambiguous:
        raise EvidenceError(f"plan ledger has ambiguous (job_name, task_name) keys: {ambiguous}")

    trials: list[dict[str, Any]] = []
    orphans: list[dict[str, str]] = []
    consumed: set[str] = set()
    for result_path in sorted(jobs_root.rglob("result.json")):
        result = _load_json(result_path, what="Harbor result")
        if _is_job_aggregate(result):
            continue
        task_name = _task_name(result)
        job_name = _known_job_name(result_path, jobs_root, result, known_jobs)
        entry = by_key.get((job_name or "", task_name), [])
        trial_name = str(result.get("trial_name") or result_path.parent.name)
        if not job_name or len(entry) != 1:
            orphans.append(
                {
                    "trial_name": trial_name,
                    "reason": "no exact (job_name, task_name) plan-ledger match; excluded",
                }
            )
            continue
        ledger_entry = entry[0]
        run_id = str(ledger_entry.get("run_id"))
        if run_id in consumed:
            raise EvidenceError(f"several Harbor results map to planned run {run_id}; refusing to guess")
        consumed.add(run_id)
        planned = plan_by_id.get(run_id)
        if not isinstance(planned, dict):
            raise EvidenceError(f"ledger run {run_id} has no matching entry in plan.json")

        (
            rewards,
            reward_schema_version,
            reward_provenance,
            preserved_timeout_exception,
        ) = _validated_rewards(result)
        attribution, counts, infra_suspect, exclusion = _attribution(
            result, rewards, preserved_timeout_exception
        )
        numeric_scores = dict(rewards or {})
        agent_id = str(planned.get("agent_id") or "unknown")
        trials.append(
            {
                "run_id": run_id,
                "task_name": task_name,
                "agent_id": agent_id,
                "scaffold": _job_scaffold(plan_dir, job_name, agent_id),
                "seed": int(planned.get("seed", ledger_entry.get("seed", 1))),
                "attempt": int(planned.get("attempt", ledger_entry.get("attempt", 1))),
                "attribution": attribution,
                "counts_toward_capability": counts,
                "infra_suspect": infra_suspect,
                "exclusion_basis": exclusion,
                "reward_schema_version": reward_schema_version,
                "reward_provenance": reward_provenance,
                "trace_status": _trace_status(result_path, result),
                "replica_count": 1,
                "cost_usd": _cost(result),
                "scores": numeric_scores,
            }
        )

    if not trials:
        raise EvidenceError(
            f"no planned Harbor trial could be reconciled under {jobs_root}; "
            f"orphans={len(orphans)}"
        )
    missing = sorted(set(plan_by_id) - consumed)
    if missing:
        raise EvidenceError(
            f"Harbor bundle is missing planned run result(s): {missing}; "
            "refusing to render a partial campaign as complete"
        )

    normalized = {
        "normalized_schema_version": "1.0",
        "campaign_id": str(plan.get("campaign_id") or ledger.get("campaign_id")),
        "integration": str(plan.get("integration") or "oragentbench"),
        "evidence_intent": str(plan.get("evidence_intent") or "exploratory"),
        "site": {
            "name": str(plan.get("site") or "unknown"),
            "perf_isolated": False,
            "load_source": "none",
            "cpus": None,
        },
        "scoring": {
            "reward_keys": ["feasibility", "quality"],
            "strict_pass_rule": {
                "description": "strict pass = feasibility >= 1 and quality >= 1.0 using upstream ORAgentBench reward keys",
                "feasibility_key": "feasibility",
                "quality_key": "quality",
                "quality_threshold": 1.0,
            },
        },
        "durability": {"min_replica_count": 1, "verified": False},
        "orphan_trials": orphans,
        "trials": sorted(trials, key=lambda item: item["run_id"]),
    }
    validate_normalized(normalized, name="ingested Harbor rollout")
    normalized_path = run_root / "normalized" / "rollout.json"
    _atomic_text(
        normalized_path,
        json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )

    rollout = NormalizedRollout.load(normalized_path)
    report_dir = run_root / "report"
    render_mod.write_report(render_mod.build_report(rollout), report_dir)
    integrity_path = _write_integrity(run_root)
    return HarborIngestResult(
        trials=len(trials),
        orphans=len(orphans),
        normalized_path=normalized_path,
        report_dir=report_dir,
        integrity_path=integrity_path,
    )
