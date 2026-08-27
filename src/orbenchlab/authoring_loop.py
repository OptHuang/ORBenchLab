"""Fail-closed multi-round Volc authoring over a copied TB-Science skeleton."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import task_authoring
from .core.errors import ORBenchError
from .volc_review import (
    REQUIRED_REVIEW_CRITERIA,
    VolcConfig,
    VolcReviewError,
    _bounded_task_snapshot,
    _digest,
    _assert_public_text,
    call_reviewer,
    review_task,
    write_review,
)


class AuthoringLoopError(ORBenchError):
    """An automatic authoring round violated its bounded contract."""

    exit_code = 8


MAX_PATCH_FILES = 32
MAX_PATCH_BYTES = 256_000
BOUND_DERIVATION_PATH = "data/paper-derivation-bound.json"
_ROOT_FILES = frozenset({"task.toml", "README.md", "instruction.md", "paper-provenance.json"})
_ROOT_DIRS = frozenset({"environment", "solution", "tests", "data"})
_CREDENTIAL_NAME = re.compile(
    r"(?:^|[._-])(env|token|secret|credential|auth|private[_-]?key)(?:$|[._-])",
    re.I,
)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task_tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "content_digest": _file_digest(path),
                }
            )
    return _digest(entries)


def _safe_patch_path(value: Any) -> str:
    raw = str(value or "")
    pure = PurePosixPath(raw)
    normalized = pure.as_posix()
    if (
        not raw
        or raw != normalized
        or raw.startswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts)
        or _CREDENTIAL_NAME.search(pure.name)
    ):
        raise AuthoringLoopError(f"author patch contains unsafe path: {raw!r}")
    if normalized == BOUND_DERIVATION_PATH:
        raise AuthoringLoopError("author patch may not modify bound paper derivation evidence")
    if raw not in _ROOT_FILES and (not pure.parts or pure.parts[0] not in _ROOT_DIRS):
        raise AuthoringLoopError(f"author patch path is outside the task allowlist: {raw}")
    return normalized


def _coerce_author_response(
    value: Mapping[str, Any],
    *,
    round_number: int,
    base_digest: str,
    input_receipt_digest: str,
    previous_review_digest: str | None,
) -> tuple[Mapping[str, Any], str]:
    """Adapt the exact single-file shape emitted by some coding endpoints.

    Binding fields come from the trusted local round state, never from model
    output.  Any wider schema-less response still fails closed.
    """

    if value.get("schema_version") == "orbenchlab.author-patch.v1":
        return value, "full-envelope"
    if set(value) == {"path", "content"} and isinstance(value.get("path"), str) and isinstance(
        value.get("content"), str
    ):
        return (
            {
                "schema_version": "orbenchlab.author-patch.v1",
                "round": round_number,
                "base_task_tree_digest": base_digest,
                "input_receipt_digest": input_receipt_digest,
                "previous_review_digest": previous_review_digest,
                "files": [{"path": value["path"], "content": value["content"]}],
                "rationale": ["provider returned exact single-file shorthand"],
            },
            "single-file-shorthand",
        )
    raise AuthoringLoopError("author patch schema is unsupported")


def _normalise_patch(
    value: Mapping[str, Any],
    *,
    round_number: int,
    base_digest: str,
    input_receipt_digest: str,
    previous_review_digest: str | None,
) -> dict[str, Any]:
    if value.get("schema_version") != "orbenchlab.author-patch.v1":
        raise AuthoringLoopError("author patch schema is unsupported")
    if value.get("round") != round_number:
        raise AuthoringLoopError("author patch round does not match current round")
    if value.get("base_task_tree_digest") != base_digest:
        raise AuthoringLoopError("author patch base task-tree digest is stale")
    if value.get("input_receipt_digest") != input_receipt_digest:
        raise AuthoringLoopError("author patch input receipt digest mismatch")
    if value.get("previous_review_digest") != previous_review_digest:
        raise AuthoringLoopError("author patch previous review digest mismatch")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_PATCH_FILES:
        raise AuthoringLoopError("author patch must contain 1..32 file replacements")
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    total = 0
    for item in raw_files:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            raise AuthoringLoopError("author patch files require path and UTF-8 text content")
        path = _safe_patch_path(item.get("path"))
        if path in seen:
            raise AuthoringLoopError(f"author patch repeats path: {path}")
        seen.add(path)
        content = str(item["content"])
        if "\x00" in content:
            raise AuthoringLoopError(f"author patch contains NUL bytes: {path}")
        try:
            _assert_public_text(path, content)
        except VolcReviewError as exc:
            raise AuthoringLoopError(str(exc)) from None
        total += len(content.encode("utf-8"))
        if total > MAX_PATCH_BYTES:
            raise AuthoringLoopError("author patch exceeds 256000 UTF-8 bytes")
        files.append({"path": path, "content": content})
    rationale = value.get("rationale")
    return {
        "schema_version": "orbenchlab.author-patch.v1",
        "round": round_number,
        "base_task_tree_digest": base_digest,
        "input_receipt_digest": input_receipt_digest,
        "previous_review_digest": previous_review_digest,
        "files": files,
        "rationale": [str(item) for item in rationale] if isinstance(rationale, list) else [],
    }


def _apply_patch(task_dir: Path, patch: Mapping[str, Any]) -> str:
    before = _task_tree_digest(task_dir)
    if patch.get("base_task_tree_digest") != before:
        raise AuthoringLoopError("author patch lost its compare-and-swap task digest")
    for item in patch["files"]:
        relative = _safe_patch_path(item["path"])
        if relative == BOUND_DERIVATION_PATH:
            raise AuthoringLoopError("author patch may not modify bound paper derivation evidence")
        target = task_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise AuthoringLoopError(f"author patch target is not a regular file: {item['path']}")
        target.write_text(str(item["content"]), encoding="utf-8")
        if target.suffix == ".sh":
            target.chmod(target.stat().st_mode | 0o111)
    return _task_tree_digest(task_dir)


def _author_prompt(
    task_dir: Path,
    *,
    round_number: int,
    base_digest: str,
    input_receipt: Mapping[str, Any],
    previous_review: Mapping[str, Any] | None,
    paper_evidence: Mapping[str, Any],
) -> str:
    input_receipt_digest = str(input_receipt["receipt_digest"])
    previous_review_digest = (
        str(previous_review.get("review_digest")) if isinstance(previous_review, Mapping) else None
    )
    failures = [
        {"name": row.get("name"), "status": row.get("status"), "reason": row.get("reason")}
        for key in ("implementation_criteria", "provenance_checks")
        for row in input_receipt.get(key, [])
        if isinstance(row, Mapping) and row.get("status") in {"fail", "review"}
    ]
    payload = {
        "round": round_number,
        "base_task_tree_digest": base_digest,
        "input_receipt_digest": input_receipt_digest,
        "previous_review_digest": previous_review_digest,
        "gate_findings": failures,
        "previous_review": previous_review.get("reviewers", []) if isinstance(previous_review, Mapping) else [],
        "paper_evidence": dict(paper_evidence),
        "task_files": _bounded_task_snapshot(task_dir),
    }
    return (
        "Revise this paper-backed Terminal-Bench Science task using only supplied evidence. "
        "Return JSON only: schema_version='orbenchlab.author-patch.v1', round, "
        "base_task_tree_digest, input_receipt_digest, previous_review_digest, "
        "files=[{path,content}], rationale=[strings]. Use full replacement UTF-8 text; "
        "do not delete files, invent paper claims, weaken verifier checks, add network access, "
        "or write outside the task allowlist.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _paper_evidence(paper: Mapping[str, Any], derivation_path: Path) -> dict[str, Any]:
    if derivation_path.is_symlink() or not derivation_path.is_file():
        raise AuthoringLoopError("paper-derivation must be a regular non-symlink UTF-8 file")
    if derivation_path.stat().st_size <= 0 or derivation_path.stat().st_size > 64_000:
        raise AuthoringLoopError("paper-derivation must contain 1..64000 bytes")
    try:
        text = derivation_path.read_text(encoding="utf-8")
        _assert_public_text(derivation_path.name, text)
    except (OSError, UnicodeDecodeError, VolcReviewError) as exc:
        raise AuthoringLoopError(f"paper-derivation is not safe public UTF-8 evidence: {type(exc).__name__}") from None
    safe_paper = {
        key: paper.get(key)
        for key in (
            "title",
            "url",
            "source_content_digest",
            "license_status",
            "intake_id",
            "intake_snapshot_digest",
            "intake_item_uid",
        )
        if paper.get(key) is not None
    }
    return {
        "paper": safe_paper,
        "derivation_file": derivation_path.name,
        "derivation_digest": _file_digest(derivation_path),
        "derivation": text,
    }


def _validate_review_round(
    review: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    paper: Mapping[str, Any],
    models: Sequence[str],
    round_number: int,
) -> None:
    supplied_digest = review.get("review_digest")
    unsigned = {key: value for key, value in review.items() if key != "review_digest"}
    if (
        not isinstance(supplied_digest, str)
        or supplied_digest != _digest(unsigned)
        or review.get("schema_version") != "orbenchlab.volc-authoring-review.v1"
        or review.get("round") != round_number
        or review.get("task_tree_digest") != receipt.get("task_tree_digest")
        or review.get("static_receipt_digest") != receipt.get("receipt_digest")
        or review.get("paper_digest") != paper.get("source_content_digest")
        or review.get("review_count") != len(models)
        or review.get("models") != list(models)
    ):
        raise AuthoringLoopError("review artifact does not bind to the current round evidence")
    reviewers = review.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != len(models):
        raise AuthoringLoopError("review artifact has the wrong reviewer count")
    observed_models = [str(row.get("model")) for row in reviewers if isinstance(row, Mapping)]
    if observed_models != list(models) or len(set(observed_models)) != len(observed_models):
        raise AuthoringLoopError("reviewer models must be distinct and match the requested slots")
    if review.get("aggregate_decision") == "promising-needs-harbor":
        for reviewer in reviewers:
            result = reviewer.get("review") if isinstance(reviewer, Mapping) else None
            criteria = result.get("criteria") if isinstance(result, Mapping) else None
            if (
                not isinstance(result, Mapping)
                or result.get("decision") != "promising"
                or result.get("shape_complete") is not True
                or result.get("rubric_complete") is not True
                or not isinstance(criteria, list)
                or {row.get("name") for row in criteria if isinstance(row, Mapping)}
                != REQUIRED_REVIEW_CRITERIA
                or any(row.get("status") != "pass" for row in criteria if isinstance(row, Mapping))
            ):
                raise AuthoringLoopError("promising review lacks complete passing rubric evidence")


def _manifest(root: Path, run_path: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "run-manifest.json":
            continue
        files[path.relative_to(root).as_posix()] = _file_digest(path)
    return {
        "schema_version": "orbenchlab.authoring-loop-manifest.v1",
        "run_digest": _file_digest(run_path),
        "files": files,
    }


def iterate(
    seed_task: str | Path,
    *,
    paper_provenance: str | Path,
    paper_derivation: str | Path,
    config: VolcConfig,
    author_model: str,
    review_models: Sequence[str],
    max_rounds: int,
    out: str | Path,
    max_author_tokens: int = 2400,
    max_review_tokens: int = 2400,
) -> dict[str, Any]:
    """Run bounded author/review rounds without mutating the seed task."""

    seed = Path(seed_task)
    paper_path = Path(paper_provenance)
    derivation_path = Path(paper_derivation)
    output = Path(out)
    if seed.is_symlink() or not seed.is_dir():
        raise AuthoringLoopError("seed-task must be a real directory")
    if any(path.is_symlink() for path in seed.rglob("*")):
        raise AuthoringLoopError("seed-task may not contain symlinks")
    if not paper_path.is_file() or paper_path.is_symlink():
        raise AuthoringLoopError("paper-provenance must be a regular file")
    selected_reviewers = [str(value).strip() for value in review_models]
    if (
        not author_model.strip()
        or len(selected_reviewers) < 2
        or any(not value for value in selected_reviewers)
        or len(set(selected_reviewers)) != len(selected_reviewers)
    ):
        raise AuthoringLoopError("iterate requires one author and at least two distinct reviewer model ids")
    if max_rounds <= 0 or max_author_tokens <= 0 or max_review_tokens <= 0:
        raise AuthoringLoopError("round and token budgets must be positive")
    try:
        paper = task_authoring._load_document(paper_path)
    except task_authoring.TaskAuthoringError as exc:
        raise AuthoringLoopError(str(exc)) from None
    try:
        _assert_public_text(paper_path.name, paper_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, VolcReviewError) as exc:
        raise AuthoringLoopError(f"paper-provenance is not safe public UTF-8: {type(exc).__name__}") from None
    paper_evidence = _paper_evidence(paper, derivation_path)
    # Scan the seed before creating output artifacts and before any model call.
    try:
        _bounded_task_snapshot(seed)
    except VolcReviewError as exc:
        raise AuthoringLoopError(str(exc)) from None
    if output.exists() and any(output.iterdir()):
        raise AuthoringLoopError("authoring-loop output directory must be new or empty")
    try:
        output.resolve().relative_to(seed.resolve())
    except ValueError:
        pass
    else:
        raise AuthoringLoopError("authoring-loop output may not be inside seed-task")
    output.mkdir(parents=True, exist_ok=True)
    bound_evidence_path = output / "paper-evidence.json"
    _write_json(bound_evidence_path, paper_evidence)
    seed_digest = _task_tree_digest(seed)
    previous_receipt_path: Path | None = None
    previous_receipt: Mapping[str, Any] | None = None
    previous_review: Mapping[str, Any] | None = None
    seen_patches: set[str] = set()
    rounds: list[dict[str, Any]] = []
    final_task = seed
    status = "max-rounds"
    stop_reason = "maximum authoring rounds reached"

    for round_number in range(1, max_rounds + 1):
        round_dir = output / "rounds" / f"round-{round_number}"
        task_dir = round_dir / seed.name
        shutil.copytree(final_task, task_dir, symlinks=False)
        task_evidence = task_dir / BOUND_DERIVATION_PATH
        expected_evidence = json.dumps(
            paper_evidence, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n"
        if task_evidence.exists():
            if task_evidence.is_symlink() or task_evidence.read_text(encoding="utf-8") != expected_evidence:
                raise AuthoringLoopError("copied task contains conflicting bound paper derivation")
        else:
            task_evidence.parent.mkdir(parents=True, exist_ok=True)
            task_evidence.write_text(expected_evidence, encoding="utf-8")
        if previous_receipt is None:
            input_receipt = task_authoring.validate_task(
                task_dir,
                paper_provenance=paper_path,
                round_number=1,
            )
            task_authoring.write_receipt(input_receipt, round_dir / "preflight")
        else:
            input_receipt = previous_receipt
        base_digest = _task_tree_digest(task_dir)
        prompt = _author_prompt(
            task_dir,
            round_number=round_number,
            base_digest=base_digest,
            input_receipt=input_receipt,
            previous_review=previous_review,
            paper_evidence=paper_evidence,
        )
        try:
            api = call_reviewer(
                config,
                model=author_model,
                system=(
                    "You are a bounded TB-Science task author. Preserve scientific provenance and "
                    "strict verifier behavior. Return only the requested JSON patch."
                ),
                user=prompt,
                max_tokens=max_author_tokens,
            )
            parsed = api.pop("parsed")
            coerced, response_shape = _coerce_author_response(
                parsed,
                round_number=round_number,
                base_digest=base_digest,
                input_receipt_digest=str(input_receipt["receipt_digest"]),
                previous_review_digest=(
                    str(previous_review.get("review_digest"))
                    if isinstance(previous_review, Mapping)
                    else None
                ),
            )
            patch = _normalise_patch(
                coerced,
                round_number=round_number,
                base_digest=base_digest,
                input_receipt_digest=str(input_receipt["receipt_digest"]),
                previous_review_digest=(
                    str(previous_review.get("review_digest"))
                    if isinstance(previous_review, Mapping)
                    else None
                ),
            )
        except (VolcReviewError, AuthoringLoopError) as exc:
            status = "incomplete"
            stop_reason = f"author phase failed: {type(exc).__name__}"
            rounds.append({"round": round_number, "status": status, "phase": "author", "error_type": type(exc).__name__})
            break
        patch_digest = _digest(patch)
        if patch_digest in seen_patches:
            status = "incomplete"
            stop_reason = "repeated author patch digest"
            rounds.append({"round": round_number, "status": status, "phase": "author", "patch_digest": patch_digest})
            break
        seen_patches.add(patch_digest)
        patch_artifact = {
            **patch,
            "patch_digest": patch_digest,
            "model": author_model,
            "protocol": api.get("protocol"),
            "request_digest": api.get("request_digest"),
            "response_digest": api.get("response_digest"),
            "usage": api.get("usage", {}),
            "source_response_shape": response_shape,
        }
        _write_json(round_dir / "author-patch.json", patch_artifact)
        after_digest = _apply_patch(task_dir, patch)
        author_no_op = after_digest == base_digest
        receipt = task_authoring.validate_task(
            task_dir,
            paper_provenance=paper_path,
            round_number=round_number,
            previous_receipt=previous_receipt_path,
        )
        receipt_paths = task_authoring.write_receipt(receipt, round_dir / "authoring")
        record: dict[str, Any] = {
            "round": round_number,
            "patch_digest": patch_digest,
            "task_tree_digest": receipt["task_tree_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "static_decision": receipt["decision"],
            "author_no_op": author_no_op,
        }
        final_task = task_dir
        previous_receipt_path = receipt_paths["json"]
        previous_receipt = receipt
        if receipt["decision"] == "blocked":
            record.update({"status": "revise", "phase": "static-gate"})
            rounds.append(record)
            continue
        try:
            review = review_task(
                task_dir,
                paper_provenance=paper,
                receipt=receipt,
                config=config,
                models=selected_reviewers,
                round_number=round_number,
                max_tokens=max_review_tokens,
            )
            review_paths = write_review(review, round_dir / "review")
            previous_review = json.loads(review_paths["json"].read_text(encoding="utf-8"))
            _validate_review_round(
                previous_review,
                receipt=receipt,
                paper=paper,
                models=selected_reviewers,
                round_number=round_number,
            )
        except (
            VolcReviewError,
            AuthoringLoopError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            status = "incomplete"
            stop_reason = f"review phase failed: {type(exc).__name__}"
            record.update({"status": status, "phase": "review", "error_type": type(exc).__name__})
            rounds.append(record)
            break
        record.update(
            {
                "status": previous_review["aggregate_decision"],
                "phase": "review",
                "review_digest": previous_review["review_digest"],
                "reviewer_models": selected_reviewers,
            }
        )
        rounds.append(record)
        if previous_review["aggregate_decision"] == "promising-needs-harbor":
            status = "promising-needs-harbor"
            stop_reason = "all reviewer slots marked the statically valid task promising"
            break

    run: dict[str, Any] = {
        "schema_version": "orbenchlab.authoring-loop.v1",
        "status": status,
        "stop_reason": stop_reason,
        "seed_task": seed.name,
        "seed_task_tree_digest": seed_digest,
        "seed_unchanged": _task_tree_digest(seed) == seed_digest,
        "paper_digest": paper.get("source_content_digest"),
        "paper_derivation_digest": paper_evidence["derivation_digest"],
        "paper_evidence_artifact_digest": _file_digest(bound_evidence_path),
        "provider": config.public_dict(),
        "author_model": author_model,
        "review_models": selected_reviewers,
        "max_rounds": max_rounds,
        "rounds": rounds,
        "final_task": str(final_task),
        "final_task_tree_digest": _task_tree_digest(final_task),
        "limitations": [
            "The loop starts from an existing TB-Science skeleton; it does not generate a task from a paper alone.",
            "promising-needs-harbor is a review state, not Harbor or TB-Science acceptance.",
            "Reviewer model ids are distinct, but provider families or checkpoints may still be related.",
        ],
    }
    run["run_digest"] = _digest(run)
    run_path = output / "run.json"
    _write_json(run_path, run)
    _write_json(output / "run-manifest.json", _manifest(output, run_path))
    return run


__all__ = ["AuthoringLoopError", "BOUND_DERIVATION_PATH", "iterate"]
