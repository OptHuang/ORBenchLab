"""Independent semantic authoring review via CLI agent sessions.

The unattended factory's semantic reviewers are real coding-agent CLI sessions
(Volc route, Claude/Codex profile), not a raw provider HTTP call: each reviewer
is an independent least-visibility ``agent_sessions.run_session`` with a fixed
model, a hard ``--max-budget-usd``, prompt/input/output digests, secret
redaction and idempotent crash-safe reuse.  The reviewer writes a strict JSON
verdict which a deterministic validator here checks; the harness never invents
the semantic judgment and never treats the mere presence of a JSON file as a
pass.  The aggregate keeps the existing ``orbenchlab.volc-authoring-review.v1``
contract the finalizer already binds, and additionally records each session's
receipt digest so promotion can re-verify them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import agent_sessions, task_authoring
from .core.errors import ORBenchError
from .volc_review import REQUIRED_REVIEW_CRITERIA


class FactoryReviewError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.volc-authoring-review.v1"
REVIEW_SCHEMA_VERSION = "orbenchlab.factory-session-review.v1"


def _value_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _model_slug(model: str) -> str:
    keep = "".join(char if char.isalnum() else "-" for char in model.lower()).strip("-")
    return (keep[:40] or "model") + "-" + hashlib.sha256(model.encode()).hexdigest()[:8]


def _review_prompt(*, model: str, output_name: str, task_id: str) -> str:
    criteria = ", ".join(sorted(REQUIRED_REVIEW_CRITERIA))
    return (
        "You are an evidence-calibrated Terminal-Bench Science task reviewer. Inspect the "
        "read-only review inputs in this directory: review-input/task/ (the candidate task "
        "tree), review-input/authoring-receipt.json (the deterministic static gate result), "
        "review-input/paper-provenance.json, and review-input/paper-derivation.json if present. "
        f"Judge the task on exactly these seven proposal criteria: {criteria}. "
        "Do not run commands, do not claim Harbor acceptance, and do not expose secrets. "
        f"Write your verdict as strict JSON to {output_name} with this shape: "
        '{"decision": "promising"|"revise"|"needs-human", "shape_complete": true, '
        '"rubric_complete": true, "criteria": [{"name": <one of the seven>, '
        '"status": "pass"|"fail"|"review", "evidence": <non-empty specific justification>} '
        "... one object per criterion, all seven present]}. Use decision 'promising' only when "
        "every criterion is a justified pass. Keep uncertain criteria as 'review' and the "
        f"decision as 'needs-human'. Task id: {task_id}. Write only {output_name}; do not modify "
        "any review-input file."
    )


def _validate_review_document(document: Any) -> dict[str, Any]:
    """Strictly validate one reviewer's JSON verdict; never trust file presence."""

    if not isinstance(document, Mapping):
        raise FactoryReviewError("reviewer verdict must be a JSON object")
    decision = document.get("decision")
    criteria = document.get("criteria")
    if (
        decision not in {"promising", "revise", "needs-human"}
        or not isinstance(document.get("shape_complete"), bool)
        or not isinstance(document.get("rubric_complete"), bool)
        or not isinstance(criteria, list)
        or len(criteria) != len(REQUIRED_REVIEW_CRITERIA)
    ):
        raise FactoryReviewError("reviewer verdict shape is invalid")
    names = set()
    for row in criteria:
        if (
            not isinstance(row, Mapping)
            or row.get("name") not in REQUIRED_REVIEW_CRITERIA
            or row.get("status") not in {"pass", "fail", "review"}
            or not isinstance(row.get("evidence"), str)
            or not row["evidence"].strip()
        ):
            raise FactoryReviewError("reviewer criterion row is invalid")
        names.add(row["name"])
    if names != REQUIRED_REVIEW_CRITERIA:
        raise FactoryReviewError("reviewer verdict is missing criteria")
    rubric_complete = all(row["status"] == "pass" for row in criteria)
    shape_complete = bool(document.get("shape_complete"))
    normalised_decision = decision
    if decision == "promising" and not rubric_complete:
        # A claimed promising verdict without a full passing rubric is
        # downgraded; the harness does not invent the judgment, it only
        # enforces internal consistency.
        normalised_decision = "needs-human"
    return {
        "decision": normalised_decision,
        "shape_complete": shape_complete and bool(document.get("shape_complete")),
        "rubric_complete": rubric_complete,
        "criteria": [
            {
                "name": row["name"],
                "status": row["status"],
                "evidence": row["evidence"].strip()[:2000],
            }
            for row in sorted(criteria, key=lambda item: item["name"])
        ],
    }


def _stage_review_inputs(
    *,
    task_dir: Path,
    static_receipt_path: Path,
    paper_provenance_path: Path,
    review_root: Path,
) -> list[Path]:
    """Build a read-only review-input tree and return the paths to protect."""

    input_root = review_root / "review-input"
    if input_root.exists():
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True)
    shutil.copytree(task_dir, input_root / "task", symlinks=False)
    shutil.copy2(static_receipt_path, input_root / "authoring-receipt.json")
    shutil.copy2(paper_provenance_path, input_root / "paper-provenance.json")
    derivation = task_dir / "data" / "paper-task-derivation.json"
    if derivation.is_file() and not derivation.is_symlink():
        shutil.copy2(derivation, input_root / "paper-derivation.json")
    for path in sorted(input_root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    input_root.chmod(0o555)
    return [input_root]


def review_task_via_sessions(
    task_dir: str | Path,
    *,
    paper_provenance_path: str | Path,
    static_receipt_path: str | Path,
    models: Sequence[str],
    provider_env: Mapping[str, str],
    out: str | Path,
    profile: str = "claude-code",
    max_budget_usd: float = 1.0,
    timeout_sec: float = 900.0,
    max_output_bytes: int = 8 * 1024 * 1024,
    round_number: int = 1,
    executable: str | Path | None = None,
) -> dict[str, Any]:
    """Run one independent CLI review session per model and aggregate them."""

    task = Path(task_dir).resolve()
    if task.is_symlink() or not task.is_dir():
        raise FactoryReviewError("task directory must be a real directory")
    selected = [str(model).strip() for model in models if str(model).strip()]
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise FactoryReviewError("authoring review requires at least two distinct reviewer models")
    static_receipt = task_authoring._load_document(Path(static_receipt_path))
    paper = task_authoring._load_document(Path(paper_provenance_path))
    task_digest = task_authoring._task_tree_digest(task)
    if static_receipt.get("task_tree_digest") != task_digest:
        raise FactoryReviewError("static receipt does not bind this task tree")
    task_id = task.name
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reviewers: list[dict[str, Any]] = []
    session_bindings: list[dict[str, Any]] = []
    for model in selected:
        slug = _model_slug(model)
        review_root = root / slug
        review_root.mkdir(parents=True, exist_ok=True)
        read_only = _stage_review_inputs(
            task_dir=task,
            static_receipt_path=Path(static_receipt_path),
            paper_provenance_path=Path(paper_provenance_path),
            review_root=review_root,
        )
        output_name = "review.json"
        output_path = review_root / output_name
        if output_path.exists() and not output_path.is_symlink():
            output_path.unlink()
        prompt = _review_prompt(model=model, output_name=output_name, task_id=task_id)
        session = agent_sessions.run_session(
            profile=profile,
            stage=f"factory-review/{slug}/round-{round_number}",
            model=model,
            prompt=prompt,
            workdir=review_root,
            out=review_root / "sessions",
            timeout_sec=timeout_sec,
            max_budget_usd=max_budget_usd,
            max_output_bytes=max_output_bytes,
            environ=provider_env,
            executable=executable,
            read_only_paths=read_only,
            allow_bash=False,
        )
        verdict: dict[str, Any]
        review_problem: str | None = None
        if session.get("status") != "completed":
            review_problem = str(session.get("failure_class") or "session_failed")
        else:
            try:
                document = json.loads(output_path.read_text(encoding="utf-8"))
                verdict = _validate_review_document(document)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                review_problem = "verdict_not_valid_json"
            except FactoryReviewError as exc:
                review_problem = f"verdict_invalid:{exc}"
        if review_problem is not None:
            verdict = {
                "decision": "needs-human",
                "shape_complete": False,
                "rubric_complete": False,
                "criteria": [],
                "problem": review_problem,
            }
        receipt_path = Path(str(session.get("receipt_path") or ""))
        reviewers.append({"model": model, "review": verdict})
        session_bindings.append(
            {
                "model": model,
                "session_id": session.get("session_id"),
                "session_receipt_digest": (
                    _file_digest(receipt_path)
                    if receipt_path.is_file() and not receipt_path.is_symlink()
                    else None
                ),
                "route_digest": session.get("identity", {}).get("route_digest"),
                "executable_digest": session.get("identity", {}).get("executable_digest"),
                "verdict_digest": _value_digest(verdict),
                "status": session.get("status"),
            }
        )
    decisions = [str(row["review"]["decision"]) for row in reviewers]
    if static_receipt.get("decision") == "blocked":
        aggregate = "blocked-static-gate"
    elif any(decision == "revise" for decision in decisions):
        aggregate = "revise"
    elif all(decision == "promising" for decision in decisions):
        aggregate = "promising-needs-harbor"
    else:
        aggregate = "needs-human"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "review_mechanism": "cli-agent-session",
        "round": int(round_number),
        "task_dir": task_id,
        "task_tree_digest": task_digest,
        "paper_digest": paper.get("source_content_digest"),
        "static_receipt_digest": static_receipt.get("receipt_digest"),
        "models": selected,
        "profile": profile,
        "max_budget_usd_per_session": max_budget_usd,
        "review_count": len(reviewers),
        "aggregate_decision": aggregate,
        "evidence_level": "E1-agent-session-review",
        "reviewers": reviewers,
        "session_bindings": session_bindings,
        "limitations": [
            "Independent CLI agent-session reviews; a proposal for the next authoring round, not TB-Science acceptance.",
            "No hidden verifier, model trajectory or Harbor runtime result was supplied to this review.",
            "Each reviewer ran as a least-visibility no-Bash session with a hard per-session budget; raw prompt/response bodies are not persisted.",
        ],
    }
    payload["review_digest"] = _value_digest(payload)
    return payload


def write_review(review: Mapping[str, Any], out: str | Path) -> Path:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "volc-authoring-review.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(review), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


__all__ = [
    "FactoryReviewError",
    "SCHEMA_VERSION",
    "review_task_via_sessions",
    "write_review",
]
