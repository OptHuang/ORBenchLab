"""Unattended multi-candidate factory orchestration.

One batch invocation consumes a queue of paper candidates from a pluggable
provider, screens each candidate deterministically (provenance binding, paper
bytes, seed task — no model calls), refuses to start when the worst-case
provider liability of the admitted set exceeds the configured cap, and then
drives one isolated autopilot per candidate with its own workspace, budget
chain, receipts and resume state.  A crashed or quarantined candidate never
blocks the others; re-invoking the same batch resumes every non-terminal
candidate through the autopilot's own crash-safe state.

Daily scheduling stays outside this harness: external automation (cron, CI)
re-runs ``orbench agent-factory batch`` against a refreshed intake; the batch
itself is idempotent.
"""

from __future__ import annotations

import re
import concurrent.futures
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from . import agentic_factory, factory_autopilot, factory_blueprints
from .core.errors import ORBenchError


class FactoryBatchError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.factory-batch.v1"
_CANDIDATE_ID = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_candidate_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > 64 or any(char not in _CANDIDATE_ID for char in raw):
        raise FactoryBatchError(
            f"candidate id must be 1..64 chars of [a-z0-9-]: {raw!r}"
        )
    return raw


def _provider_explicit_list(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = spec.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise FactoryBatchError("explicit-list provider requires a candidates list")
    candidates = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise FactoryBatchError("candidate rows must be objects")
        candidates.append(
            {
                "id": _safe_candidate_id(row.get("id")),
                "paper_file": str(row.get("paper_file") or ""),
                "paper_provenance": str(row.get("paper_provenance") or ""),
                "seed_task": str(row.get("seed_task") or spec.get("seed_task") or ""),
            }
        )
    return candidates


def _provider_paper_binding_dir(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(spec.get("root") or ""))
    seed = str(spec.get("seed_task") or "")
    if not root.is_dir() or root.is_symlink():
        raise FactoryBatchError("paper-binding-dir provider requires a real root directory")
    candidates = []
    for child in sorted(root.iterdir()):
        provenance = child / "paper-provenance.json"
        if not child.is_dir() or child.is_symlink() or not provenance.is_file():
            continue
        paper_file = ""
        try:
            document = json.loads(provenance.read_text(encoding="utf-8"))
            source_path = str(document.get("source_path") or "")
            if source_path:
                candidate_paper = Path(source_path)
                if not candidate_paper.is_absolute():
                    candidate_paper = child / candidate_paper
                paper_file = str(candidate_paper)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            paper_file = ""
        if not paper_file:
            for candidate_paper in sorted(child.glob("*.pdf")):
                paper_file = str(candidate_paper)
                break
        candidates.append(
            {
                "id": _safe_candidate_id(child.name),
                "paper_file": paper_file,
                "paper_provenance": str(provenance),
                "seed_task": seed,
            }
        )
    if not candidates:
        raise FactoryBatchError("paper-binding-dir provider found no candidates")
    return candidates


def _provider_triaged_intake(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Candidates from a digest-bound daily candidate manifest (P0-E).

    The manifest and every candidate provenance are re-verified before any
    candidate is yielded, so a tampered receipt is rejected before paid work.
    """

    from . import source_manifest

    manifest_path = str(spec.get("manifest") or "")
    if not manifest_path:
        raise FactoryBatchError("triaged-intake provider requires a manifest path")
    try:
        manifest = source_manifest.load_verified_manifest(manifest_path)
    except source_manifest.CandidateManifestError as exc:
        raise FactoryBatchError(f"triaged-intake manifest failed verification: {exc}") from None
    seed = str(spec.get("seed_task") or "")
    candidates = []
    for entry in manifest["entries"]:
        # Source ids (e.g. arXiv 2401.01234) can carry '.'/'_'; normalise to the
        # candidate-id alphabet [a-z0-9-] deterministically.
        safe = re.sub(r"[^a-z0-9-]+", "-", str(entry["source_id"]).lower()).strip("-")[:64]
        candidates.append(
            {
                "id": _safe_candidate_id(safe),
                "paper_file": str(entry["paper_file"]),
                "paper_provenance": str(entry["paper_provenance"]),
                "seed_task": seed,
            }
        )
    if not candidates:
        raise FactoryBatchError("triaged-intake manifest has no candidates")
    return candidates


_PROVIDERS: dict[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]] = {
    "explicit-list": _provider_explicit_list,
    "paper-binding-dir": _provider_paper_binding_dir,
    "triaged-intake": _provider_triaged_intake,
}


def _reference_plan_liability(spec: Mapping[str, Any]) -> float:
    """Compile a blueprint plan with a dummy binding to read its exact bound.

    The paper-to-benchmark blueprint's per-stage budgets and attempts are
    independent of the specific paper, so one representative compilation gives
    the exact worst-case semantic session liability every candidate plan will
    carry (compile_plan itself enforces <= 100 USD).
    """

    models = spec["models"]
    plan = factory_blueprints.paper_to_benchmark_plan(
        source_binding_digest="sha256:" + "0" * 64,
        author_model=models["author_model"],
        reviewer_models=models["reviewer_models"],
        frontier_model=models["frontier_model"],
        weak_model=models["weak_model"],
    )
    return float(plan["maximum_model_liability_usd"])


def _reviewer_count(spec: Mapping[str, Any]) -> int:
    reviewers = spec["models"]["reviewer_models"]
    return len({str(model) for model in reviewers})


def _candidate_liability(
    *,
    reference_semantic_usd: float,
    per_candidate_harbor_usd: float,
    promotion_review_usd: float,
    promote: bool,
) -> dict[str, Any]:
    # promotion_review_usd is already the exact reviewers x per-session budget
    # (each promotion reviewer runs one CLI session with a hard --max-budget-usd).
    promotion = promotion_review_usd if promote else 0.0
    total = round(reference_semantic_usd + per_candidate_harbor_usd + promotion, 6)
    return {
        "semantic_plan_usd": reference_semantic_usd,
        "harbor_usd": per_candidate_harbor_usd,
        "promotion_review_usd": promotion,
        "intervention_usd": 0.0,
        "total_usd": total,
    }


def load_batch_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    try:
        raw = spec_path.read_text(encoding="utf-8")
        document = (
            json.loads(raw) if spec_path.suffix.lower() == ".json" else yaml.safe_load(raw)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
        raise FactoryBatchError("batch spec is not valid UTF-8 JSON/YAML") from None
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise FactoryBatchError(f"batch spec must declare schema_version {SCHEMA_VERSION}")
    provider = document.get("provider")
    models = document.get("models")
    if not isinstance(provider, Mapping) or provider.get("kind") not in _PROVIDERS:
        raise FactoryBatchError(
            f"batch provider kind must be one of {sorted(_PROVIDERS)}"
        )
    required_models = (
        "author_model",
        "reviewer_models",
        "frontier_model",
        "weak_model",
    )
    if not isinstance(models, Mapping) or any(not models.get(key) for key in required_models):
        raise FactoryBatchError(
            "batch spec models require author_model, reviewer_models, frontier_model, weak_model"
        )
    reviewers = models["reviewer_models"]
    if not isinstance(reviewers, list) or len({str(m) for m in reviewers}) < 2:
        raise FactoryBatchError("batch spec needs at least two distinct reviewer models")
    return {
        "schema_version": SCHEMA_VERSION,
        "provider": dict(provider),
        "models": {
            "author_model": str(models["author_model"]),
            "reviewer_models": [str(m) for m in reviewers],
            "frontier_model": str(models["frontier_model"]),
            "weak_model": str(models["weak_model"]),
        },
    }


def discover_candidates(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    provider = spec["provider"]
    return _PROVIDERS[str(provider["kind"])](provider)


def screen_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic metadata screen; no model call, no file mutation."""

    reasons: list[str] = []
    paper = Path(candidate.get("paper_file") or "")
    provenance_path = Path(candidate.get("paper_provenance") or "")
    seed = Path(candidate.get("seed_task") or "")
    if not paper.is_file() or paper.is_symlink():
        reasons.append("paper file is missing")
    if not provenance_path.is_file() or provenance_path.is_symlink():
        reasons.append("paper provenance is missing")
    if not seed.is_dir() or seed.is_symlink() or not (seed / "task.toml").is_file():
        reasons.append("seed task is not a strict task directory")
    provenance: Mapping[str, Any] | None = None
    if not reasons:
        try:
            provenance = factory_blueprints._load_provenance(provenance_path)
        except agentic_factory.AgenticFactoryError as exc:
            reasons.append(f"provenance binding invalid: {exc}")
    if provenance is not None:
        actual = "sha256:" + hashlib.sha256(paper.read_bytes()).hexdigest()
        if actual != provenance.get("source_content_digest"):
            reasons.append("paper bytes do not match bound provenance digest")
    return {
        "id": candidate["id"],
        "promising": not reasons,
        "reasons": reasons,
        "license_status": (
            str(provenance.get("license_status"))
            if isinstance(provenance, Mapping)
            else None
        ),
    }


def _candidate_worker(
    candidate: Mapping[str, Any],
    spec_models: Mapping[str, Any],
    candidate_root: str,
    autopilot_kwargs: Mapping[str, Any],
    provider_env: Mapping[str, str],
    held_out: bool,
) -> dict[str, Any]:
    root = Path(candidate_root)
    workdir = root / "work"
    try:
        manifest = factory_blueprints.prepare_workspace(
            paper_file=candidate["paper_file"],
            paper_provenance=candidate["paper_provenance"],
            seed_task=candidate["seed_task"],
            workdir=workdir,
        )
        plan = factory_blueprints.paper_to_benchmark_plan(
            source_binding_digest=manifest["workspace_binding_digest"],
            author_model=spec_models["author_model"],
            reviewer_models=spec_models["reviewer_models"],
            frontier_model=spec_models["frontier_model"],
            weak_model=spec_models["weak_model"],
            held_out_confirmation=held_out,
        )
        agentic_factory.write_plan(plan, root / "plan.json")
        state = factory_autopilot.run(
            plan,
            workdir=workdir,
            factory_out=root / "factory-run",
            out=root / "autopilot",
            provider_env=dict(provider_env),
            held_out=held_out,
            **dict(autopilot_kwargs),
        )
        promotion = state.get("promotion") if isinstance(state.get("promotion"), Mapping) else {}
        return {
            "id": candidate["id"],
            "status": state.get("status"),
            "factory_id": plan["factory_id"],
            "plan_digest": plan["plan_digest"],
            "quarantine": state.get("quarantine"),
            "promotion_decision": promotion.get("decision"),
            "final_report": (promotion.get("final_report") or {}).get("markdown"),
            "roots": {
                "workdir": str(workdir),
                "factory_out": str(root / "factory-run"),
                "autopilot_out": str(root / "autopilot"),
            },
        }
    except BaseException as exc:  # noqa: BLE001 - isolate every candidate crash
        # A single candidate's failure (ORBench error, OS error, JSON error,
        # even an unexpected crash) is archived and never aborts the batch or
        # its sibling candidates.
        return {
            "id": candidate["id"],
            "status": "error",
            "error_class": type(exc).__name__,
            "error": str(exc)[:2000],
            "roots": {"workdir": str(workdir)},
        }


def run_batch(
    *,
    spec: Mapping[str, Any],
    out: str | Path,
    provider_env: Mapping[str, str],
    harbor_executable: str | Path,
    claude_executable: str | Path,
    repetitions: int = 5,
    max_budget_usd: float = 0.5,
    max_turns: int = 40,
    harbor_timeout_sec: float = 10_800,
    max_variants: int = 3,
    max_job_attempts: int = 2,
    max_harbor_liability_usd_per_candidate: float = 40.0,
    max_total_liability_usd: float = 200.0,
    max_promotion_review_usd: float | None = None,
    max_candidates: int | None = None,
    max_parallel: int = 1,
    held_out: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    """Screen, admit and drive one autopilot per candidate; idempotent."""

    if max_parallel < 1 or max_parallel > 8:
        raise FactoryBatchError("max_parallel must be in 1..8")
    if max_promotion_review_usd is not None and max_promotion_review_usd < 0:
        raise FactoryBatchError("max_promotion_review_usd must be non-negative")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A single batch state chain is serialized; two writers on the same root
    # cannot race the ledger.
    batch_lock = (root / ".batch.lock").open("a+b")
    fcntl.flock(batch_lock.fileno(), fcntl.LOCK_EX)
    try:
        return _run_batch_locked(
            spec=spec,
            root=root,
            provider_env=provider_env,
            harbor_executable=harbor_executable,
            claude_executable=claude_executable,
            repetitions=repetitions,
            max_budget_usd=max_budget_usd,
            max_turns=max_turns,
            harbor_timeout_sec=harbor_timeout_sec,
            max_variants=max_variants,
            max_job_attempts=max_job_attempts,
            max_harbor_liability_usd_per_candidate=max_harbor_liability_usd_per_candidate,
            max_total_liability_usd=max_total_liability_usd,
            max_promotion_review_usd=max_promotion_review_usd,
            max_candidates=max_candidates,
            max_parallel=max_parallel,
            held_out=held_out,
            promote=promote,
        )
    finally:
        fcntl.flock(batch_lock.fileno(), fcntl.LOCK_UN)
        batch_lock.close()


def _run_batch_locked(
    *,
    spec: Mapping[str, Any],
    root: Path,
    provider_env: Mapping[str, str],
    harbor_executable: str | Path,
    claude_executable: str | Path,
    repetitions: int,
    max_budget_usd: float,
    max_turns: int,
    harbor_timeout_sec: float,
    max_variants: int,
    max_job_attempts: int,
    max_harbor_liability_usd_per_candidate: float,
    max_total_liability_usd: float,
    max_promotion_review_usd: float,
    max_candidates: int | None,
    max_parallel: int,
    held_out: bool,
    promote: bool,
) -> dict[str, Any]:
    candidates = discover_candidates(spec)
    ids = [row["id"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise FactoryBatchError("candidate ids must be unique")
    spec_digest = _digest(dict(spec))
    ledger_path = root / "batch-ledger.json"
    if ledger_path.is_file() and not ledger_path.is_symlink():
        prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        if prior.get("spec_digest") != spec_digest:
            raise FactoryBatchError("batch output binds a different immutable spec")
    screening = [screen_candidate(row) for row in candidates]
    promising = [
        row
        for row, screen in zip(candidates, screening)
        if screen["promising"]
    ]
    if max_candidates is not None:
        promising = promising[: max(0, int(max_candidates))]
    per_candidate_harbor = (
        (1 + max_variants) * 2 * repetitions * max_budget_usd * max_job_attempts
    )
    if per_candidate_harbor > max_harbor_liability_usd_per_candidate:
        raise FactoryBatchError(
            "per-candidate Harbor liability exceeds its configured cap"
        )
    reference_semantic = _reference_plan_liability(spec)
    # Exact promotion-review liability: one CLI session per distinct reviewer,
    # each carrying the same hard --max-budget-usd the autopilot passes to the
    # review session (equal to the per-trial budget).
    review_budget = (
        max_promotion_review_usd
        if max_promotion_review_usd is not None
        else max_budget_usd
    )
    promotion_review_usd = round(_reviewer_count(spec) * float(review_budget), 6)
    per_candidate = _candidate_liability(
        reference_semantic_usd=reference_semantic,
        per_candidate_harbor_usd=per_candidate_harbor,
        promotion_review_usd=promotion_review_usd,
        promote=promote,
    )
    total_liability = round(len(promising) * per_candidate["total_usd"], 6)
    if total_liability > max_total_liability_usd:
        raise FactoryBatchError(
            f"worst-case batch liability {total_liability:.2f} USD exceeds the "
            f"configured cap {max_total_liability_usd:.2f}; admit fewer candidates "
            "or raise the cap explicitly"
        )
    # Persist an immutable liability ledger before any candidate can produce a
    # provider or Harbor marker.  A resumed batch revalidates the same digest.
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "spec_digest": spec_digest,
        "per_candidate": per_candidate,
        "admitted": [row["id"] for row in promising],
        "worst_case_total_usd": total_liability,
        "cap_usd": max_total_liability_usd,
        "promote": promote,
    }
    ledger["ledger_digest"] = _digest(
        {key: value for key, value in ledger.items() if key != "ledger_digest"}
    )
    if ledger_path.is_file() and not ledger_path.is_symlink():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        if existing != ledger:
            raise FactoryBatchError("batch ledger differs from the admitted contract")
    else:
        _atomic_json(ledger_path, ledger)
    autopilot_kwargs = {
        "harbor_executable": str(harbor_executable),
        "claude_executable": str(claude_executable),
        "frontier_model": spec["models"]["frontier_model"],
        "weak_model": spec["models"]["weak_model"],
        "repetitions": repetitions,
        "max_budget_usd": max_budget_usd,
        "max_turns": max_turns,
        "harbor_timeout_sec": harbor_timeout_sec,
        "max_variants": max_variants,
        "max_job_attempts": max_job_attempts,
        "max_harbor_liability_usd": max_harbor_liability_usd_per_candidate,
        "promote": promote,
    }
    results: list[dict[str, Any]] = []
    jobs = [
        (
            candidate,
            spec["models"],
            str(root / candidate["id"]),
            autopilot_kwargs,
            dict(provider_env),
            held_out,
        )
        for candidate in promising
    ]
    if max_parallel == 1 or len(jobs) <= 1:
        for job in jobs:
            results.append(_candidate_worker(*job))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_parallel,
            mp_context=__import__("multiprocessing").get_context("fork"),
        ) as pool:
            futures = [pool.submit(_candidate_worker, *job) for job in jobs]
            results = [future.result() for future in futures]
    results.sort(key=lambda row: row["id"])
    state = {
        "schema_version": SCHEMA_VERSION,
        "spec_digest": spec_digest,
        "ledger_digest": ledger["ledger_digest"],
        "screening": screening,
        "admitted": [row["id"] for row in promising],
        "skipped": [row["id"] for row in screening if not row["promising"]],
        "liability": {
            "per_candidate": per_candidate,
            "worst_case_total_usd": total_liability,
            "cap_usd": max_total_liability_usd,
        },
        "candidates": results,
        "status_counts": {
            status: sum(row.get("status") == status for row in results)
            for status in sorted({str(row.get("status")) for row in results})
        }
        if results
        else {},
    }
    state["batch_digest"] = _digest(
        {key: value for key, value in state.items() if key != "batch_digest"}
    )
    _atomic_json(root / "batch-state.json", state)
    return state


__all__ = [
    "FactoryBatchError",
    "SCHEMA_VERSION",
    "discover_candidates",
    "load_batch_spec",
    "run_batch",
    "screen_candidate",
]
