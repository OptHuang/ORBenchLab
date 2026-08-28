"""Unattended outer loop for the paper-to-benchmark agent factory.

The semantic work remains inside bounded coding-agent sessions.  This module
only advances the immutable DAG, launches verifier-grounded Harbor evidence at
the two runtime barriers, installs validated read-only receipts, and resumes
the DAG.  It never treats agent prose as execution evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import (
    agent_sessions,
    agentic_factory,
    difficulty_matrix,
    factory_blueprints,
    factory_gates,
    factory_promotion,
    harbor_launcher,
    harbor_model_matrix,
    pipeline,
    session_interventions,
    task_authoring,
    volc_rollout,
)
from .core.errors import ORBenchError


class FactoryAutopilotError(ORBenchError):
    exit_code = 8


class FactoryStaticGateBlocked(FactoryAutopilotError):
    """A pre-Harbor deterministic gate blocked the run before any model spend."""

    def __init__(self, quarantine: Mapping[str, Any]):
        super().__init__(
            f"static gate blocked the autopilot: {quarantine.get('gate')}"
        )
        self.quarantine = dict(quarantine)


SCHEMA_VERSION = "orbenchlab.factory-autopilot.v1"
TRUSTED_BUNDLE_SCHEMA_VERSION = "orbenchlab.factory-trusted-bundle.v1"
REQUIRED_STAGES = frozenset(
    {
        "task-repair-v2",
        "runtime-controls",
        "pilot-frontier",
        "pilot-weak",
        "variant-author",
        "calibration",
        "final-synthesis",
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _value_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _executable_binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if resolved.is_symlink() or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FactoryAutopilotError("autopilot executable must be a real executable file")
    return {"path": str(resolved), "digest": _file_digest(resolved)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_state(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "state_digest"}
    signed = {**unsigned, "state_digest": _value_digest(unsigned)}
    _atomic_json(path, signed)
    return signed


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FactoryAutopilotError(f"trusted JSON evidence is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise FactoryAutopilotError(f"trusted JSON evidence is malformed: {path.name}") from None
    if not isinstance(value, dict):
        raise FactoryAutopilotError(f"trusted JSON evidence root is not an object: {path.name}")
    return value


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise FactoryAutopilotError(f"unsafe trusted bundle path: {value!r}")
    return path


def _tree_rows(root: Path, *, max_bytes: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    if root.is_symlink() or not root.is_dir():
        raise FactoryAutopilotError("trusted bundle source must be a real directory")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise FactoryAutopilotError(f"trusted bundle source contains a symlink: {relative}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if total > max_bytes:
            raise FactoryAutopilotError("trusted bundle exceeds its byte bound")
        rows.append(
            {
                "path": relative,
                "bytes": size,
                "content_digest": _file_digest(path),
            }
        )
    if not rows:
        raise FactoryAutopilotError("trusted bundle source is empty")
    return rows, total


def install_trusted_bundle(
    *,
    workdir: str | Path,
    relative: str,
    source: str | Path,
    source_receipts: Mapping[str, str],
    max_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    """Atomically install one validated evidence bundle under read-only inputs.

    ``source_receipts`` is deliberately supplied by the trusted caller after
    it has validated the underlying Harbor/difficulty receipts.  The manifest
    binds both those logical receipts and every copied byte.
    """

    workspace = Path(workdir).resolve()
    input_root = workspace / "factory-input"
    trusted_root = input_root / "trusted"
    if (
        input_root.is_symlink()
        or trusted_root.is_symlink()
        or not input_root.is_dir()
        or not trusted_root.is_dir()
    ):
        raise FactoryAutopilotError("factory trusted-input root is missing or unsafe")
    pure = _safe_relative(relative)
    destination = trusted_root.joinpath(*pure.parts)
    try:
        destination.resolve(strict=False).relative_to(trusted_root.resolve())
    except ValueError:
        raise FactoryAutopilotError("trusted bundle destination escaped its root") from None
    source_root = Path(source).resolve()
    rows, total = _tree_rows(source_root, max_bytes=max_bytes)
    unsigned = {
        "schema_version": TRUSTED_BUNDLE_SCHEMA_VERSION,
        "relative": pure.as_posix(),
        "source_receipts": dict(sorted(source_receipts.items())),
        "files": rows,
        "file_count": len(rows),
        "byte_count": total,
    }
    manifest = {**unsigned, "bundle_digest": _value_digest(unsigned)}
    if destination.exists():
        existing = _load_object(destination / "trusted-bundle-manifest.json")
        actual_rows, actual_total = _tree_rows(destination, max_bytes=max_bytes)
        actual_rows = [
            row for row in actual_rows if row["path"] != "trusted-bundle-manifest.json"
        ]
        if (
            existing != manifest
            or actual_rows != rows
            or actual_total - (destination / "trusted-bundle-manifest.json").stat().st_size
            != total
        ):
            raise FactoryAutopilotError("existing trusted bundle differs from validated evidence")
        return manifest
    input_root.chmod(0o755)
    trusted_root.chmod(0o755)
    temporary = trusted_root / f".{pure.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, temporary, symlinks=False)
        copied_rows, copied_total = _tree_rows(temporary, max_bytes=max_bytes)
        if copied_rows != rows or copied_total != total:
            raise FactoryAutopilotError("trusted bundle changed while being copied")
        _atomic_json(temporary / "trusted-bundle-manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        factory_blueprints._make_inputs_read_only(input_root)
    return manifest


def _validate_installed_bundle(
    *, workdir: Path, relative: str, expected_digest: str
) -> tuple[dict[str, Any], Path]:
    destination = workdir / "factory-input" / "trusted" / _safe_relative(relative)
    manifest_path = destination / "trusted-bundle-manifest.json"
    manifest = _load_object(manifest_path)
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_digest"}
    rows, total = _tree_rows(destination, max_bytes=128 * 1024 * 1024)
    rows = [row for row in rows if row["path"] != "trusted-bundle-manifest.json"]
    total -= manifest_path.stat().st_size
    if (
        manifest.get("schema_version") != TRUSTED_BUNDLE_SCHEMA_VERSION
        or manifest.get("bundle_digest") != _value_digest(unsigned)
        or manifest.get("bundle_digest") != expected_digest
        or manifest.get("files") != rows
        or manifest.get("file_count") != len(rows)
        or manifest.get("byte_count") != total
    ):
        raise FactoryAutopilotError("installed trusted bundle failed terminal validation")
    return manifest, destination


def _validate_barriers(state: Mapping[str, Any], workdir: Path) -> None:
    barriers = state.get("barriers")
    if not isinstance(barriers, Mapping):
        raise FactoryAutopilotError("autopilot barrier state is malformed")
    baseline = barriers.get("baseline")
    if "baseline" in barriers and not isinstance(baseline, Mapping):
        raise FactoryAutopilotError("baseline barrier state must be an object")
    if isinstance(baseline, Mapping):
        manifest, root = _validate_installed_bundle(
            workdir=workdir,
            relative="baseline",
            expected_digest=str(baseline.get("trusted_bundle_digest") or ""),
        )
        controls = _load_object(root / "harbor-control-screening.json")
        try:
            pipeline._validate_harbor_receipt(controls, root / "harbor-control-screening.json")
        except pipeline.PipelineError as exc:
            raise FactoryAutopilotError(f"baseline Harbor controls are invalid: {exc}") from None
        matrix = harbor_model_matrix._validated_receipt(
            _load_object(root / "harbor-model-matrix.json")
        )
        screening = _load_object(root / "screening-report.json")
        unsigned_screening = {
            key: value for key, value in screening.items() if key != "report_digest"
        }
        capability = _load_object(root / "runtime-capability.json")
        unsigned_capability = {
            key: value for key, value in capability.items() if key != "receipt_digest"
        }
        sources = manifest.get("source_receipts")
        if (
            controls.get("report_digest") != baseline.get("control_report_digest")
            or matrix.get("receipt_digest") != baseline.get("matrix_receipt_digest")
            or screening.get("report_digest") != _value_digest(unsigned_screening)
            or screening.get("report_digest") != baseline.get("screening_report_digest")
            or capability.get("receipt_digest") != _value_digest(unsigned_capability)
            or capability.get("task_tree_digest") != matrix.get("task_tree_digest")
            or not isinstance(sources, Mapping)
            or sources.get("harbor_controls") != controls.get("report_digest")
            or sources.get("harbor_model_matrix") != matrix.get("receipt_digest")
            or sources.get("screening") != screening.get("report_digest")
        ):
            raise FactoryAutopilotError("baseline trusted receipt binding failed")
        static_digest = baseline.get("static_receipt_digest")
        if static_digest is not None:
            static_doc = _load_object(root / "static-authoring-receipt.json")
            unsigned_static = {
                key: value for key, value in static_doc.items() if key != "receipt_digest"
            }
            if (
                static_doc.get("receipt_digest") != task_authoring._digest(unsigned_static)
                or static_doc.get("receipt_digest") != static_digest
                or static_doc.get("decision") == "blocked"
                or sources.get("static_authoring") != static_digest
            ):
                raise FactoryAutopilotError("baseline static-gate receipt binding failed")
    intervention = barriers.get("intervention")
    if "intervention" in barriers and not isinstance(intervention, Mapping):
        raise FactoryAutopilotError("intervention barrier state must be an object")
    if isinstance(intervention, Mapping):
        manifest, iroot = _validate_installed_bundle(
            workdir=workdir,
            relative="intervention",
            expected_digest=str(intervention.get("trusted_bundle_digest") or ""),
        )
        capability = _load_object(iroot / "runtime-capability.json")
        unsigned_capability = {
            key: value for key, value in capability.items() if key != "receipt_digest"
        }
        sources = manifest.get("source_receipts")
        if (
            capability.get("schema_version") != "orbenchlab.runtime-capability.v1"
            or capability.get("receipt_digest") != _value_digest(unsigned_capability)
            or capability.get("receipt_digest")
            != intervention.get("capability_receipt_digest")
            or capability.get("checkpoint_capability") is not False
            or capability.get("harbor_native") is not False
            or not isinstance(sources, Mapping)
            or sources.get("runtime_capability") != capability.get("receipt_digest")
        ):
            raise FactoryAutopilotError("intervention capability receipt binding failed")
        study_digest = intervention.get("study_receipt_digest")
        if study_digest is not None:
            study = _load_object(iroot / "intervention-study.json")
            unsigned_study = {
                key: value for key, value in study.items() if key != "receipt_digest"
            }
            if (
                study.get("schema_version")
                != session_interventions.STUDY_SCHEMA_VERSION
                or study.get("receipt_digest")
                != session_interventions._digest(unsigned_study)
                or study.get("receipt_digest") != study_digest
                or sources.get("intervention_study") != study_digest
                or capability.get("study_evidence_level") != study.get("evidence_level")
                or (
                    capability.get("causal_intervention_claim_available")
                    and study.get("evidence_level")
                    != "E4-controlled-same-session-intervention"
                )
            ):
                raise FactoryAutopilotError("intervention study receipt binding failed")
    difficulty = barriers.get("difficulty")
    if "difficulty" in barriers and not isinstance(difficulty, Mapping):
        raise FactoryAutopilotError("difficulty barrier state must be an object")
    if isinstance(difficulty, Mapping):
        manifest, root = _validate_installed_bundle(
            workdir=workdir,
            relative="difficulty",
            expected_digest=str(difficulty.get("trusted_bundle_digest") or ""),
        )
        receipt = _load_object(root / "difficulty-matrix.json")
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        sources = manifest.get("source_receipts")
        preregistration_digest = receipt.get("preregistration_digest")
        preregistration_valid = preregistration_digest is None
        if isinstance(preregistration_digest, str):
            preregistration = _load_object(root / "difficulty-preregistration.json")
            unsigned_preregistration = {
                key: value
                for key, value in preregistration.items()
                if key != "preregistration_digest"
            }
            preregistration_valid = (
                preregistration.get("schema_version")
                == difficulty_matrix.PREREGISTRATION_SCHEMA_VERSION
                and preregistration.get("preregistration_digest")
                == difficulty_matrix._digest(unsigned_preregistration)
                == preregistration_digest
                and isinstance(sources, Mapping)
                and sources.get("difficulty_preregistration")
                == preregistration_digest
            )
        if (
            receipt.get("schema_version") != difficulty_matrix.SCHEMA_VERSION
            or receipt.get("receipt_digest") != difficulty_matrix._digest(unsigned)
            or receipt.get("receipt_digest") != difficulty.get("difficulty_receipt_digest")
            or receipt.get("evidence_level") != "E3"
            or receipt.get("checkpoint_capability") is not False
            or not isinstance(sources, Mapping)
            or sources.get("difficulty_matrix") != receipt.get("receipt_digest")
            or not preregistration_valid
        ):
            raise FactoryAutopilotError("difficulty trusted receipt binding failed")


def _stage_output_path(
    plan: Mapping[str, Any], workdir: Path, stage_id: str, *, kind: str
) -> Path:
    stage = next((row for row in plan["stages"] if row["id"] == stage_id), None)
    outputs = stage.get("required_outputs") if isinstance(stage, Mapping) else None
    matches = [row for row in outputs or [] if row.get("kind") == kind]
    if len(matches) != 1:
        raise FactoryAutopilotError(f"stage {stage_id} must own exactly one {kind} output")
    return agentic_factory._artifact_path(workdir, matches[0]["path"])


def _runtime_capability(task_digest: str) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "orbenchlab.runtime-capability.v1",
        "task_tree_digest": task_digest,
        "trajectory_evidence_level": "E3",
        "checkpoint_capability": False,
        "same_checkpoint_hint_injection": False,
        "causal_intervention_claim_available": False,
        "limitations": [
            "Harbor trials are independent full restarts.",
            "Restart-with-hint cannot establish an E4 same-checkpoint causal effect.",
            (
                "A separate same-session injection channel exists via "
                "'orbench intervention-study' (claude-code stream-json stdin); it was "
                "not used for these Harbor trials and confers no E4 status here."
            ),
        ],
    }
    receipt["receipt_digest"] = _value_digest(receipt)
    return receipt


def _static_gate(
    task: Path, *, workdir: Path, out: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the deterministic TB-Science gate before any Harbor spend.

    Returns ``(receipt, quarantine)``; ``quarantine`` is a machine-readable
    reason object when the task is blocked, so an unattended run stops with an
    explainable state instead of launching paid trials against a broken task.
    """

    provenance = workdir / "factory-input" / "paper-provenance.json"
    receipt = task_authoring.validate_task(
        task,
        paper_provenance=provenance if provenance.is_file() else None,
    )
    task_authoring.write_receipt(receipt, out)
    task_provenance = task / "paper-provenance.json"
    if provenance.is_file() and (
        task_provenance.is_symlink()
        or not task_provenance.is_file()
        or task_provenance.read_bytes() != provenance.read_bytes()
    ):
        return receipt, {
            "reason": "static-gate-blocked",
            "gate": label,
            "task": task.name,
            "receipt_digest": receipt["receipt_digest"],
            "receipt_path": str(out / "authoring-receipt.json"),
            "failing_criteria": ["workspace_paper_provenance_binding"],
        }
    if receipt["decision"] == "blocked":
        failing = sorted(
            str(row.get("name"))
            for group in ("implementation_criteria", "provenance_checks")
            for row in receipt.get(group, [])
            if isinstance(row, Mapping) and row.get("status") == "fail"
        )
        return receipt, {
            "reason": "static-gate-blocked",
            "gate": label,
            "task": task.name,
            "receipt_digest": receipt["receipt_digest"],
            "receipt_path": str(out / "authoring-receipt.json"),
            "failing_criteria": failing,
        }
    return receipt, None


def _ensure_baseline(
    *,
    plan: Mapping[str, Any],
    workdir: Path,
    evidence_root: Path,
    harbor_executable: str | Path,
    claude_executable: str | Path,
    models: Sequence[str],
    repetitions: int,
    provider_env: Mapping[str, str],
    max_budget_usd: float,
    max_turns: int,
    harbor_timeout_sec: float,
    max_job_attempts: int,
) -> dict[str, Any]:
    task = factory_gates.resolve_task_root(
        _stage_output_path(plan, workdir, "task-repair-v2", kind="directory")
    )
    task_digest = volc_rollout._task_tree_digest(task)
    root = evidence_root / "baseline"
    static_receipt, quarantine = _static_gate(
        task, workdir=workdir, out=root / "static-gate", label="baseline"
    )
    if quarantine is not None:
        raise FactoryStaticGateBlocked(quarantine)
    controls = harbor_launcher.launch_controls(
        task,
        harbor_executable=harbor_executable,
        out=root / "controls",
        timeout_sec=harbor_timeout_sec,
    )
    matrix = harbor_model_matrix.launch_matrix(
        task,
        harbor_executable=harbor_executable,
        claude_executable=claude_executable,
        out=root / "matrix",
        models=models,
        repetitions=repetitions,
        provider_env=provider_env,
        max_budget_usd=max_budget_usd,
        max_turns=max_turns,
        timeout_sec=harbor_timeout_sec,
        max_job_attempts=max_job_attempts,
    )
    trace = harbor_model_matrix.write_trace_bundle(
        matrix,
        matrix_root=root / "matrix",
        out=root / "trusted-source" / "trace-bundle",
        secret_values=[
            value
            for name, value in provider_env.items()
            if "KEY" in name.upper() or "TOKEN" in name.upper()
        ],
    )
    screening = harbor_model_matrix.build_screening_report(
        matrix, harbor_controls=controls, out=root / "matrix"
    )
    trusted_source = root / "trusted-source"
    trusted_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "controls" / "harbor-control-screening.json", trusted_source)
    shutil.copy2(root / "matrix" / "harbor-model-matrix.json", trusted_source)
    shutil.copy2(root / "matrix" / "screening-report.json", trusted_source)
    shutil.copy2(
        root / "static-gate" / "authoring-receipt.json",
        trusted_source / "static-authoring-receipt.json",
    )
    _atomic_json(trusted_source / "runtime-capability.json", _runtime_capability(task_digest))
    bundle = install_trusted_bundle(
        workdir=workdir,
        relative="baseline",
        source=trusted_source,
        source_receipts={
            "harbor_controls": controls["report_digest"],
            "harbor_model_matrix": matrix["receipt_digest"],
            "screening": screening["report_digest"],
            "static_authoring": static_receipt["receipt_digest"],
            "trace_manifest": trace["manifest_digest"],
        },
    )
    return {
        "task_tree_digest": task_digest,
        "control_report_digest": controls["report_digest"],
        "matrix_receipt_digest": matrix["receipt_digest"],
        "screening_report_digest": screening["report_digest"],
        "static_receipt_digest": static_receipt["receipt_digest"],
        "trace_manifest_digest": trace["manifest_digest"],
        "trusted_bundle_digest": bundle["bundle_digest"],
        "observed_usage": _usage_summary(matrix),
    }


def _usage_summary(matrix: Mapping[str, Any]) -> dict[str, Any]:
    totals: dict[str, float | int] = {
        "n_input_tokens": 0,
        "n_cache_tokens": 0,
        "n_output_tokens": 0,
        "cost_usd": 0.0,
        "trials_with_reported_cost": 0,
    }
    for trial in matrix.get("trials", []):
        usage = trial.get("usage") if isinstance(trial, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
        cost = usage.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            totals["cost_usd"] += float(cost)
            totals["trials_with_reported_cost"] += 1
    totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
    return totals


def _ensure_difficulty(
    *,
    plan: Mapping[str, Any],
    workdir: Path,
    evidence_root: Path,
    harbor_executable: str | Path,
    claude_executable: str | Path,
    frontier_model: str,
    weak_model: str,
    repetitions: int,
    provider_env: Mapping[str, str],
    max_budget_usd: float,
    max_turns: int,
    harbor_timeout_sec: float,
    max_variants: int,
    held_out: bool,
    max_job_attempts: int,
) -> dict[str, Any]:
    variants_root = _stage_output_path(plan, workdir, "variant-author", kind="directory")
    manifest_path = variants_root / "variant-manifest.json"
    manifest = difficulty_matrix.load_variant_manifest(manifest_path)
    if len(manifest["variants"]) > max_variants:
        raise FactoryAutopilotError("agent-authored variant count exceeds the autopilot bound")
    preregistration_path: Path | None = None
    preregistration_digest: str | None = None
    if held_out:
        preregistration_path = evidence_root / "difficulty" / "difficulty-preregistration.json"
        variant_evidence_root = evidence_root / "difficulty" / "variants"
        if (
            not preregistration_path.exists()
            and variant_evidence_root.exists()
            and any(variant_evidence_root.rglob("*"))
        ):
            raise FactoryAutopilotError(
                "held-out preregistration must be frozen before any variant Harbor evidence"
            )
        _, route_digest = agent_sessions._session_env("claude-code", provider_env)
        preregistration = difficulty_matrix.build_preregistration(
            manifest_path=manifest_path,
            variants_root=variants_root,
            frontier_model=frontier_model,
            weak_model=weak_model,
            repetitions=repetitions,
            max_budget_usd=max_budget_usd,
            max_turns=max_turns,
            max_job_attempts=max_job_attempts,
            provider_route_digest=route_digest,
            claude_executable_digest=_file_digest(Path(claude_executable)),
        )
        preregistration_path = difficulty_matrix.write_preregistration(
            preregistration, evidence_root / "difficulty"
        )
        preregistration_digest = preregistration["preregistration_digest"]
    evidence_map: dict[str, dict[str, str]] = {}
    observed = []
    variant_static_receipts: dict[str, str] = {}
    for variant in manifest["variants"]:
        variant_id = variant["variant_id"]
        task = variants_root / variant["relative_path"]
        root = evidence_root / "difficulty" / "variants" / variant_id
        static_receipt, quarantine = _static_gate(
            task,
            workdir=workdir,
            out=root / "static-gate",
            label=f"variant:{variant_id}",
        )
        if quarantine is not None:
            raise FactoryStaticGateBlocked(quarantine)
        variant_static_receipts[variant_id] = static_receipt["receipt_digest"]
        controls = harbor_launcher.launch_controls(
            task,
            harbor_executable=harbor_executable,
            out=root / "controls",
            timeout_sec=harbor_timeout_sec,
        )
        matrix = harbor_model_matrix.launch_matrix(
            task,
            harbor_executable=harbor_executable,
            claude_executable=claude_executable,
            out=root / "matrix",
            models=[frontier_model, weak_model],
            repetitions=repetitions,
            provider_env=provider_env,
            max_budget_usd=max_budget_usd,
            max_turns=max_turns,
            timeout_sec=harbor_timeout_sec,
            max_job_attempts=max_job_attempts,
            preregistration_digest=preregistration_digest,
        )
        harbor_model_matrix.write_trace_bundle(
            matrix,
            matrix_root=root / "matrix",
            out=root / "trace-bundle",
            secret_values=[
                value
                for name, value in provider_env.items()
                if "KEY" in name.upper() or "TOKEN" in name.upper()
            ],
        )
        evidence_map[variant_id] = {
            "controls": str(root / "controls" / "harbor-control-screening.json"),
            "model_matrix": str(root / "matrix" / "harbor-model-matrix.json"),
        }
        observed.append({"variant_id": variant_id, **_usage_summary(matrix)})
    base_task = factory_gates.resolve_task_root(
        _stage_output_path(plan, workdir, "task-repair-v2", kind="directory")
    )
    receipt = difficulty_matrix.build_receipt(
        manifest_path=manifest_path,
        variants_root=variants_root,
        evidence=evidence_map,
        frontier_model=frontier_model,
        weak_model=weak_model,
        held_out=held_out,
        preregistration_path=preregistration_path,
        base_task_tree_digest=volc_rollout._task_tree_digest(base_task),
    )
    output = difficulty_matrix.write_receipt(receipt, evidence_root / "difficulty")
    trusted_source = evidence_root / "difficulty" / "trusted-source"
    trusted_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, trusted_source / "difficulty-matrix.json")
    if preregistration_path is not None:
        shutil.copy2(preregistration_path, trusted_source / preregistration_path.name)
    bundle = install_trusted_bundle(
        workdir=workdir,
        relative="difficulty",
        source=trusted_source,
        source_receipts={
            "difficulty_matrix": receipt["receipt_digest"],
            **(
                {"difficulty_preregistration": preregistration_digest}
                if preregistration_digest
                else {}
            ),
        },
    )
    return {
        "variant_manifest_digest": manifest["manifest_digest"],
        "variant_count": len(manifest["variants"]),
        "variant_static_receipts": variant_static_receipts,
        "difficulty_receipt_digest": receipt["receipt_digest"],
        "decision": receipt["decision"],
        "evidence_level": receipt["evidence_level"],
        "preregistration_digest": preregistration_digest,
        "trusted_bundle_digest": bundle["bundle_digest"],
        "observed_usage": observed,
    }


def _load_policy(path: Path) -> dict[str, Any]:
    """Normalize the agent-authored policy into a strict machine policy.

    The harness validates the schema; it never invents trigger/hint content.
    hint_level may arrive as a JSON number, which we coerce to int without
    changing its value.
    """

    document = _load_object(path)
    policy = {
        "trigger": document.get("trigger"),
        "hint_level": document.get("hint_level"),
        "hint_text": document.get("hint_text"),
    }
    level = policy["hint_level"]
    if isinstance(level, float) and level.is_integer():
        policy["hint_level"] = int(level)
    trigger = policy["trigger"]
    if isinstance(trigger, Mapping):
        value = trigger.get("value")
        if isinstance(value, float) and value.is_integer() and trigger.get("kind") in {
            "assistant-event-index"
        }:
            policy = {**policy, "trigger": {**trigger, "value": int(value)}}
    try:
        return session_interventions._validate_policy(policy)
    except session_interventions.SessionInterventionError as exc:
        raise FactoryAutopilotError(
            f"agent-authored intervention policy is not a valid machine policy: {exc}"
        ) from None


def _ensure_intervention(
    *,
    plan: Mapping[str, Any],
    workdir: Path,
    evidence_root: Path,
    claude_executable: str | Path,
    model: str,
    provider_env: Mapping[str, str],
    max_budget_usd: float,
    enabled: bool,
    verifier_argv: Sequence[str],
    n_control: int,
    n_treatment: int,
    timeout_sec: float,
    max_output_bytes: int,
) -> dict[str, Any]:
    """Probe the same-session injection channel and, when supported and enabled,
    run a crash-safe controlled study on the frozen task, feeding trusted input."""

    task = factory_gates.resolve_task_root(
        _stage_output_path(plan, workdir, "task-repair-v2", kind="directory")
    )
    task_digest = volc_rollout._task_tree_digest(task)
    policy_path = agentic_factory._artifact_path(
        workdir, "factory/analysis/intervention-policy.json"
    )
    policy = _load_policy(policy_path)
    capability = session_interventions.probe_capability(
        profile="claude-code", runtime="agent-session"
    )
    root = evidence_root / "intervention"
    trusted_source = root / "trusted-source"
    trusted_source.mkdir(parents=True, exist_ok=True)
    study_receipt: dict[str, Any] | None = None
    study_reason = None
    if not capability["same_session_hint_injection"]:
        study_reason = "runtime-unsupported"
    elif not enabled:
        study_reason = "disabled-by-configuration"
    elif not verifier_argv:
        study_reason = "no-verifier-command-configured"
    else:
        study_receipt = session_interventions.run_intervention_study(
            profile="claude-code",
            model=model,
            prompt=_intervention_prompt(task),
            template_workdir=task,
            out=root / "study",
            environ=provider_env,
            verifier_argv=list(verifier_argv),
            policy=policy,
            n_control=n_control,
            n_treatment=n_treatment,
            timeout_sec=timeout_sec,
            max_budget_usd=max_budget_usd,
            executable=claude_executable,
            max_output_bytes=max_output_bytes,
        )
    capability_receipt = {
        "schema_version": "orbenchlab.runtime-capability.v1",
        "task_tree_digest": task_digest,
        "trajectory_evidence_level": "E3",
        "channel": "claude-code-stream-json-same-session",
        "harbor_native": False,
        "checkpoint_capability": False,
        "same_checkpoint_hint_injection": False,
        "same_session_hint_injection": bool(
            capability["same_session_hint_injection"]
        ),
        "policy_digest": session_interventions._digest(policy),
        "study_status": (
            study_receipt["status"] if study_receipt is not None else "not-run"
        ),
        "study_evidence_level": (
            study_receipt["evidence_level"] if study_receipt is not None else None
        ),
        "study_reason": study_reason,
        "causal_intervention_claim_available": bool(
            study_receipt is not None
            and study_receipt.get("evidence_level")
            == "E4-controlled-same-session-intervention"
        ),
        "limitations": [
            "Same-session stdin injection is a turn-boundary continuation, not a "
            "mid-token checkpoint restore; it is not Harbor-native.",
            "Harbor task trials remain independent restarts (restart-with-hint is E3).",
        ],
    }
    capability_receipt["receipt_digest"] = _value_digest(
        {k: v for k, v in capability_receipt.items() if k != "receipt_digest"}
    )
    _atomic_json(trusted_source / "runtime-capability.json", capability_receipt)
    source_receipts = {"runtime_capability": capability_receipt["receipt_digest"]}
    if study_receipt is not None:
        shutil.copy2(
            root / "study" / "intervention-study.json",
            trusted_source / "intervention-study.json",
        )
        source_receipts["intervention_study"] = study_receipt["receipt_digest"]
    bundle = install_trusted_bundle(
        workdir=workdir,
        relative="intervention",
        source=trusted_source,
        source_receipts=source_receipts,
    )
    return {
        "task_tree_digest": task_digest,
        "capability_receipt_digest": capability_receipt["receipt_digest"],
        "same_session_hint_injection": capability_receipt["same_session_hint_injection"],
        "study_status": capability_receipt["study_status"],
        "study_evidence_level": capability_receipt["study_evidence_level"],
        "study_reason": study_reason,
        "study_receipt_digest": (
            study_receipt["receipt_digest"] if study_receipt is not None else None
        ),
        "trusted_bundle_digest": bundle["bundle_digest"],
    }


def _intervention_prompt(task: Path) -> str:
    task_id = volc_rollout._task_id(task)
    return (
        "You are solving a Terminal-Bench Science operations-research task in the "
        "current directory. Read instruction.md and the data files, then create the "
        "required solution artifacts exactly as the instruction specifies. Task id: "
        f"{task_id}. If a follow-up user message arrives, incorporate it exactly. "
        "Do not access the network."
    )


def _selected_task(workdir: Path) -> str | None:
    path = workdir / "factory" / "final" / "task-review-summary.json"
    if not path.is_file() or path.is_symlink():
        return None
    value = _load_object(path).get("selected_task")
    if not isinstance(value, str):
        return None
    pure = _safe_relative(value)
    selected = workdir.joinpath(*pure.parts).resolve()
    if not selected.is_relative_to(workdir) or selected.is_symlink() or not selected.is_dir():
        raise FactoryAutopilotError("final summary selected an unsafe or missing task")
    return selected.relative_to(workdir).as_posix()


def run(
    plan: Mapping[str, Any],
    *,
    workdir: str | Path,
    factory_out: str | Path,
    out: str | Path,
    harbor_executable: str | Path,
    claude_executable: str | Path,
    frontier_model: str,
    weak_model: str,
    provider_env: Mapping[str, str],
    repetitions: int = 5,
    max_budget_usd: float = 0.5,
    max_turns: int = 40,
    harbor_timeout_sec: float = 10_800,
    max_variants: int = 6,
    max_harbor_liability_usd: float = 100.0,
    max_job_attempts: int = 2,
    held_out: bool = False,
    promote: bool = True,
    promotion_review_timeout_sec: float = 600.0,
    promotion_max_review_tokens: int = 2400,
    intervention_study: bool = False,
    intervention_verifier_argv: Sequence[str] = (),
    intervention_control: int = 3,
    intervention_treatment: int = 3,
    intervention_timeout_sec: float = 900.0,
    max_intervention_liability_usd: float = 20.0,
) -> dict[str, Any]:
    """Run or resume the complete semantic-and-runtime factory state machine."""

    checked = agentic_factory.validate_plan(plan)
    agentic_factory._require_hard_budget_profiles(checked)
    stage_ids = {stage["id"] for stage in checked["stages"]}
    if not REQUIRED_STAGES <= stage_ids:
        raise FactoryAutopilotError("factory plan lacks required autopilot barrier stages")
    models = [str(frontier_model).strip(), str(weak_model).strip()]
    if (
        not all(models)
        or models[0] == models[1]
        or repetitions < 5
        or max_variants < 3
        or max_turns < 1
        or not 0 < max_budget_usd <= 100
        or harbor_timeout_sec <= 0
        or max_harbor_liability_usd <= 0
        or not 1 <= max_job_attempts <= 5
    ):
        raise FactoryAutopilotError("autopilot model, repetition or budget bounds are invalid")
    maximum_harbor_liability = (
        (1 + max_variants)
        * len(models)
        * repetitions
        * max_budget_usd
        * max_job_attempts
    )
    if maximum_harbor_liability > max_harbor_liability_usd:
        raise FactoryAutopilotError("maximum Harbor model liability exceeds the configured cap")
    workspace = Path(workdir).resolve()
    factory_root = Path(factory_out).resolve()
    root = Path(out).resolve()
    boundaries = {
        "workspace": workspace,
        "factory_out": factory_root,
        "autopilot_out": root,
    }
    boundary_rows = list(boundaries.items())
    for index, (left_name, left) in enumerate(boundary_rows):
        for right_name, right in boundary_rows[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise FactoryAutopilotError(
                    f"{left_name} and {right_name} must be non-overlapping boundaries"
                )
    root.mkdir(parents=True, exist_ok=True)
    harbor_binding = _executable_binding(harbor_executable)
    claude_binding = _executable_binding(claude_executable)
    _, route_digest = agent_sessions._session_env("claude-code", provider_env)
    identity = {
        "plan_digest": checked["plan_digest"],
        "workspace_binding": _value_digest(str(workspace)),
        "factory_out_binding": _value_digest(str(factory_root)),
        "harbor_executable": harbor_binding,
        "claude_executable": claude_binding,
        "frontier_model": models[0],
        "weak_model": models[1],
        "repetitions": repetitions,
        "max_budget_usd_per_trial": max_budget_usd,
        "max_turns_per_trial": max_turns,
        "harbor_timeout_sec": harbor_timeout_sec,
        "max_variants": max_variants,
        "maximum_harbor_liability_usd": maximum_harbor_liability,
        "max_job_attempts_per_model": max_job_attempts,
        "provider_route_digest": route_digest,
        "held_out_confirmation": bool(held_out),
        "intervention": {
            "enabled": bool(intervention_study),
            "verifier_argv": [str(item) for item in intervention_verifier_argv],
            "n_control": intervention_control,
            "n_treatment": intervention_treatment,
            "timeout_sec": intervention_timeout_sec,
            "max_liability_usd": max_intervention_liability_usd,
        },
    }
    has_intervention_stage = any(
        stage["id"] == "intervention-policy" for stage in checked["stages"]
    )
    if intervention_study:
        if not has_intervention_stage:
            raise FactoryAutopilotError(
                "intervention study requires an intervention-policy stage in the plan"
            )
        if intervention_control < 3 or intervention_treatment < 3:
            raise FactoryAutopilotError(
                "an intervention study needs at least three control and treatment trials"
            )
        intervention_liability = (
            (intervention_control + intervention_treatment) * max_budget_usd
        )
        if intervention_liability > max_intervention_liability_usd:
            raise FactoryAutopilotError(
                "worst-case intervention liability exceeds its configured cap"
            )
    identity_digest = _value_digest(identity)
    state_path = root / "autopilot-state.json"
    lock_path = root / ".autopilot.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if state_path.exists():
            state = _load_object(state_path)
            unsigned_state = {
                key: value for key, value in state.items() if key != "state_digest"
            }
            if (
                state.get("schema_version") != SCHEMA_VERSION
                or state.get("identity_digest") != identity_digest
                or state.get("state_digest") != _value_digest(unsigned_state)
            ):
                raise FactoryAutopilotError("autopilot output binds another immutable run")
        else:
            state = {
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "identity_digest": identity_digest,
                "status": "active",
                "barriers": {},
            }
            state = _write_state(state_path, state)
        environments = {"claude-code": dict(provider_env)}
        executables = {"claude-code": str(claude_binding["path"])}
        if (
            state.get("status") == "quarantined"
            and isinstance(state.get("quarantine"), Mapping)
            and state["quarantine"].get("reason") == "static-gate-blocked"
        ):
            # The blocking task tree belongs to an immutable completed stage;
            # rerunning cannot change the deterministic gate result.
            return dict(state)
        if state.get("status") == "promoted":
            return dict(state)
        for _ in range(128):
            _validate_barriers(state, workspace)
            factory_run, _ = agentic_factory.initialise(
                checked, factory_root, workspace=workspace
            )
            if factory_run["status"] in {"semantic-complete-e1", "quarantined"}:
                required_barriers = {"baseline", "difficulty"}
                if has_intervention_stage:
                    required_barriers.add("intervention")
                if factory_run["status"] == "semantic-complete-e1" and set(
                    state["barriers"]
                ) != required_barriers:
                    raise FactoryAutopilotError(
                        "semantic-complete factory lacks both trusted runtime barriers"
                    )
                state["status"] = factory_run["status"]
                state["factory_run_digest"] = factory_run["run_digest"]
                state["selected_task"] = _selected_task(workspace)
                state = _write_state(state_path, state)
                if factory_run["status"] == "semantic-complete-e1" and promote:
                    promotion = factory_promotion.run_promotion(
                        plan=checked,
                        workdir=workspace,
                        factory_out=factory_root,
                        evidence_root=root,
                        out=root / "promotion",
                        provider_env=provider_env,
                        state=state,
                        review_timeout_sec=promotion_review_timeout_sec,
                        max_review_tokens=promotion_max_review_tokens,
                    )
                    state["promotion"] = {
                        key: promotion.get(key)
                        for key in (
                            "selected_task",
                            "task_id",
                            "promoted",
                            "decision",
                            "gates",
                            "final_report",
                            "final_report_digest",
                            "promotion_digest",
                        )
                    }
                    state["status"] = (
                        "promoted" if promotion["promoted"] else "promotion-blocked"
                    )
                    state = _write_state(state_path, state)
                return dict(state)
            ready = agentic_factory.ready_stages(checked, factory_run)
            if not ready:
                raise FactoryAutopilotError("active factory has no ready stage")
            next_stage = ready[0]
            try:
                if next_stage == "runtime-controls" and "baseline" not in state["barriers"]:
                    state["barriers"]["baseline"] = _ensure_baseline(
                        plan=checked,
                        workdir=workspace,
                        evidence_root=root,
                        harbor_executable=harbor_binding["path"],
                        claude_executable=claude_binding["path"],
                        models=models,
                        repetitions=repetitions,
                        provider_env=provider_env,
                        max_budget_usd=max_budget_usd,
                        max_turns=max_turns,
                        harbor_timeout_sec=harbor_timeout_sec,
                        max_job_attempts=max_job_attempts,
                    )
                    state = _write_state(state_path, state)
                if (
                    next_stage == "intervention-study"
                    and has_intervention_stage
                    and "intervention" not in state["barriers"]
                ):
                    state["barriers"]["intervention"] = _ensure_intervention(
                        plan=checked,
                        workdir=workspace,
                        evidence_root=root,
                        claude_executable=claude_binding["path"],
                        model=models[0],
                        provider_env=provider_env,
                        max_budget_usd=max_budget_usd,
                        enabled=bool(intervention_study),
                        verifier_argv=intervention_verifier_argv,
                        n_control=intervention_control,
                        n_treatment=intervention_treatment,
                        timeout_sec=intervention_timeout_sec,
                        max_output_bytes=32 * 1024 * 1024,
                    )
                    state = _write_state(state_path, state)
                if next_stage == "calibration" and "difficulty" not in state["barriers"]:
                    state["barriers"]["difficulty"] = _ensure_difficulty(
                        plan=checked,
                        workdir=workspace,
                        evidence_root=root,
                        harbor_executable=harbor_binding["path"],
                        claude_executable=claude_binding["path"],
                        frontier_model=models[0],
                        weak_model=models[1],
                        repetitions=repetitions,
                        provider_env=provider_env,
                        max_budget_usd=max_budget_usd,
                        max_turns=max_turns,
                        harbor_timeout_sec=harbor_timeout_sec,
                        max_variants=max_variants,
                        held_out=held_out,
                        max_job_attempts=max_job_attempts,
                    )
                    state = _write_state(state_path, state)
            except FactoryStaticGateBlocked as blocked:
                state["status"] = "quarantined"
                state["quarantine"] = blocked.quarantine
                state = _write_state(state_path, state)
                return dict(state)
            result = agentic_factory.run_factory(
                checked,
                workdir=workspace,
                out=factory_root,
                environments=environments,
                executables=executables,
                max_new_stages=1,
            )
            state["factory_status"] = result["status"]
            state["factory_run_digest"] = result["run_digest"]
            state = _write_state(state_path, state)
        raise FactoryAutopilotError("autopilot exceeded its bounded transition count")


__all__ = [
    "FactoryAutopilotError",
    "SCHEMA_VERSION",
    "install_trusted_bundle",
    "run",
]
