"""Deterministic in-loop gates for autonomous factory stages.

A postcheck runs inside the trusted harness immediately after a stage's
required outputs validate.  It never calls a model: it recomputes local
evidence (TB-Science static authoring gate, variant-lattice conformance) and
fails the attempt when the evidence does not support the stage's claim.  A
failed postcheck keeps the stage outputs in place so the next attempt can
repair them incrementally, and its findings are written both into the
immutable evidence root and into an advisory workspace file the repair agent
can read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import task_authoring
from .core.errors import ORBenchError


class FactoryGateError(ORBenchError):
    """A postcheck could not be evaluated at all (malformed contract)."""

    exit_code = 8


POSTCHECK_SCHEMA_VERSION = "orbenchlab.factory-postcheck.v1"
POSTCHECK_NAMES = frozenset({"tb-science-static-gate", "variant-conformance"})
ADVISORY_DIR = "factory/gate"


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _directory_outputs(stage: Mapping[str, Any]) -> list[str]:
    return [
        str(output["path"])
        for output in stage.get("required_outputs", [])
        if isinstance(output, Mapping) and output.get("kind") == "directory"
    ]


def resolve_task_root(path: Path) -> Path:
    """Resolve a stage output directory to the actual strict-task root.

    Factory stages own versioned output directories (``factory/tasks/task-v2``)
    whose names cannot equal the TB-Science task slug that the static gate
    requires.  The convention is therefore: either the output directory itself
    is the task root, or it contains exactly one slug-named subdirectory with
    ``task.toml``.  Anything else is returned unchanged so the gate fails with
    an explicit missing-``task.toml`` finding.
    """

    if (path / "task.toml").is_file():
        return path
    if path.is_dir() and not path.is_symlink():
        subdirs = [
            child
            for child in sorted(path.iterdir())
            if child.is_dir() and not child.is_symlink()
        ]
        if len(subdirs) == 1 and (subdirs[0] / "task.toml").is_file():
            return subdirs[0]
    return path


def _paper_provenance_path(workspace: Path) -> Path | None:
    candidate = workspace / "factory-input" / "paper-provenance.json"
    if candidate.is_file() and not candidate.is_symlink():
        return candidate
    return None


def _failing_criteria(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ("implementation_criteria", "provenance_checks"):
        for item in receipt.get(group, []):
            if isinstance(item, Mapping) and item.get("status") == "fail":
                rows.append(
                    {
                        "name": str(item.get("name")),
                        "reason": str(item.get("reason")),
                        "evidence": [str(value) for value in item.get("evidence", [])],
                    }
                )
    return rows


def _static_gate_target(task_dir: Path, *, workspace: Path) -> dict[str, Any]:
    provenance = _paper_provenance_path(workspace)
    try:
        receipt = task_authoring.validate_task(task_dir, paper_provenance=provenance)
    except task_authoring.TaskAuthoringError as exc:
        return {
            "path": task_dir.relative_to(workspace).as_posix(),
            "decision": "blocked",
            "gate_error": str(exc),
            "failing_criteria": [],
            "passed": False,
        }
    failing = _failing_criteria(receipt)
    # The task must carry this run's exact paper provenance: downstream
    # semantic-review and finalize gates bind the paper digest from the file
    # inside the task tree, so an unchanged seed provenance would only fail
    # much later, after Harbor money was spent.
    if provenance is not None:
        task_provenance = task_dir / "paper-provenance.json"
        if (
            task_provenance.is_symlink()
            or not task_provenance.is_file()
            or task_provenance.read_bytes() != provenance.read_bytes()
        ):
            failing.append(
                {
                    "name": "workspace_paper_provenance_binding",
                    "reason": (
                        "task paper-provenance.json must be a byte-exact copy of "
                        "factory-input/paper-provenance.json"
                    ),
                    "evidence": ["paper-provenance.json"],
                }
            )
    passed = receipt.get("decision") != "blocked" and not failing
    return {
        "path": task_dir.relative_to(workspace).as_posix(),
        "decision": "blocked" if not passed else str(receipt.get("decision")),
        "task_tree_digest": receipt.get("task_tree_digest"),
        "receipt_digest": receipt.get("receipt_digest"),
        "counts": receipt.get("counts"),
        "failing_criteria": failing,
        "passed": passed,
    }


def _run_static_gate(
    stage: Mapping[str, Any], *, workspace: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = _directory_outputs(stage)
    if len(outputs) != 1:
        raise FactoryGateError(
            "tb-science-static-gate requires exactly one directory output"
        )
    task_dir = resolve_task_root(workspace / outputs[0])
    targets = [_static_gate_target(task_dir, workspace=workspace)]
    return {
        "postcheck": "tb-science-static-gate",
        "targets": targets,
        "passed": all(row["passed"] for row in targets),
    }


def _tree_rows(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rows[path.relative_to(root).as_posix()] = (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return rows


def _tree_diff(base: Mapping[str, str], other: Mapping[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(other) - set(base)),
        "removed": sorted(set(base) - set(other)),
        "changed": sorted(
            path for path in set(base) & set(other) if base[path] != other[path]
        ),
    }


def _base_task_dir(plan: Mapping[str, Any], workspace: Path) -> Path | None:
    for stage in plan.get("stages", []):
        if stage.get("id") == "task-repair-v2":
            outputs = _directory_outputs(stage)
            if len(outputs) == 1:
                candidate = resolve_task_root(workspace / outputs[0])
                if candidate.is_dir() and not candidate.is_symlink():
                    return candidate
    return None


def _run_variant_conformance(
    stage: Mapping[str, Any], *, workspace: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    from . import difficulty_matrix

    outputs = _directory_outputs(stage)
    if len(outputs) != 1:
        raise FactoryGateError("variant-conformance requires exactly one directory output")
    variants_root = workspace / outputs[0]
    manifest_path = variants_root / "variant-manifest.json"
    problems: list[str] = []
    targets: list[dict[str, Any]] = []
    try:
        manifest = difficulty_matrix.load_variant_manifest(manifest_path)
    except difficulty_matrix.DifficultyMatrixError as exc:
        return {
            "postcheck": "variant-conformance",
            "targets": [],
            "problems": [f"variant manifest is invalid: {exc}"],
            "passed": False,
        }
    base_dir = _base_task_dir(plan, workspace)
    base_rows = _tree_rows(base_dir) if base_dir is not None else None
    seen_digests: dict[str, str] = {}
    for variant in manifest["variants"]:
        variant_id = str(variant["variant_id"])
        task_dir = variants_root / variant["relative_path"]
        if not task_dir.is_dir() or task_dir.is_symlink():
            problems.append(f"variant {variant_id} task directory is missing")
            targets.append({"variant_id": variant_id, "passed": False})
            continue
        target = _static_gate_target(task_dir, workspace=workspace)
        target["variant_id"] = variant_id
        rows = _tree_rows(task_dir)
        tree_digest = _digest(sorted(rows.items()))
        target["variant_tree_digest"] = tree_digest
        if tree_digest in seen_digests:
            problems.append(
                f"variant {variant_id} is byte-identical to variant {seen_digests[tree_digest]}"
            )
            target["passed"] = False
        seen_digests.setdefault(tree_digest, variant_id)
        if base_rows is not None:
            diff = _tree_diff(base_rows, rows)
            target["diff_vs_base"] = diff
            if not any(diff.values()):
                problems.append(
                    f"variant {variant_id} is byte-identical to the base task; "
                    "a difficulty variant must change declared axis content"
                )
                target["passed"] = False
            else:
                # A declared difficulty axis must correspond to substantive
                # content, not a rename: at least one changed/added file must
                # live outside task.toml/README (the instance data, the
                # instruction, the verifier tolerance, the resource config).
                substantive = [
                    path
                    for path in (diff["added"] + diff["changed"])
                    if path.rsplit("/", 1)[-1] not in {"task.toml", "README.md"}
                ]
                target["substantive_axis_changes"] = substantive
                if not substantive:
                    problems.append(
                        f"variant {variant_id} only renames the task; its declared "
                        "difficulty axis changes no instance/instruction/verifier content"
                    )
                    target["passed"] = False
        targets.append(target)
        if not target["passed"]:
            problems.extend(
                f"variant {variant_id} static gate: {row['name']}"
                for row in target.get("failing_criteria", [])
            )
    passed = not problems and all(row.get("passed") for row in targets)
    return {
        "postcheck": "variant-conformance",
        "manifest_digest": manifest["manifest_digest"],
        "primary_axis": manifest["primary_axis"],
        "targets": targets,
        "problems": sorted(set(problems)),
        "passed": passed,
    }


_POSTCHECKS = {
    "tb-science-static-gate": _run_static_gate,
    "variant-conformance": _run_variant_conformance,
}


def run_postchecks(
    stage: Mapping[str, Any],
    *,
    workspace: Path,
    plan: Mapping[str, Any],
    attempt_number: int,
) -> dict[str, Any]:
    """Run every declared postcheck for one completed stage attempt."""

    names = [str(name) for name in stage.get("postchecks", [])]
    unknown = sorted(set(names) - POSTCHECK_NAMES)
    if unknown:
        raise FactoryGateError(f"unknown factory postcheck(s): {unknown}")
    results = [
        _POSTCHECKS[name](stage, workspace=workspace, plan=plan) for name in names
    ]
    findings: dict[str, Any] = {
        "schema_version": POSTCHECK_SCHEMA_VERSION,
        "stage_id": stage["id"],
        "attempt": int(attempt_number),
        "postchecks": results,
        "passed": all(result["passed"] for result in results),
    }
    findings["findings_digest"] = _digest(
        {key: value for key, value in findings.items() if key != "findings_digest"}
    )
    return findings


def failure_summary(findings: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for result in findings.get("postchecks", []):
        if result.get("passed"):
            continue
        name = str(result.get("postcheck"))
        problems = [str(value) for value in result.get("problems", [])]
        for target in result.get("targets", []):
            if target.get("passed"):
                continue
            criteria = ", ".join(
                str(row.get("name")) for row in target.get("failing_criteria", [])
            )
            label = target.get("variant_id") or target.get("path")
            problems.append(
                f"{label}: {criteria or target.get('gate_error') or 'gate failed'}"
            )
        parts.append(f"{name}: " + ("; ".join(problems) or "failed"))
    return " | ".join(parts)[:2000]


__all__ = [
    "ADVISORY_DIR",
    "FactoryGateError",
    "POSTCHECK_NAMES",
    "POSTCHECK_SCHEMA_VERSION",
    "failure_summary",
    "run_postchecks",
]
