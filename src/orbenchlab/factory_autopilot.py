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


class FactoryRuntimeQuarantine(FactoryAutopilotError):
    """A runtime control failure could not be repaired within its round cap."""

    def __init__(self, quarantine: Mapping[str, Any]):
        super().__init__(
            f"runtime control quarantine: {quarantine.get('failure_class')}"
        )
        self.quarantine = dict(quarantine)


class FactoryInfraRetry(FactoryAutopilotError):
    """A transient infrastructure failure; the barrier should resume, not repair."""

    def __init__(self, bundle: Mapping[str, Any]):
        super().__init__("Harbor infrastructure failure; resume to retry")
        self.bundle = dict(bundle)


def _controls_with_repair(
    *,
    task: Path,
    workdir: Path,
    root: Path,
    harbor_executable: str | Path,
    claude_executable: str | Path,
    model: str,
    provider_env: Mapping[str, str],
    harbor_timeout_sec: float,
    max_repair_rounds: int,
    repair_max_budget_usd: float,
    scope: str,
    credential_relay: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Launch Oracle/NOP controls, repairing a task defect within a round cap.

    Returns ``(task, controls_receipt, repair_state)``.  Infrastructure
    failures raise :class:`FactoryInfraRetry` (resume, never mutate the task);
    an unrepairable or repair-exhausted task defect raises
    :class:`FactoryRuntimeQuarantine`.
    """

    from . import factory_runtime_repair

    paper_ancestors = [
        workdir / "factory-input" / "paper-provenance.json",
        workdir / "factory-input" / "paper.txt",
    ]
    current = task
    repair_receipts: list[dict[str, Any]] = []
    for attempt in range(max_repair_rounds + 1):
        controls_out = root / "controls" if attempt == 0 else root / "repair" / f"round-{attempt}" / "controls"
        try:
            controls = harbor_launcher.launch_controls(
                current,
                harbor_executable=harbor_executable,
                out=controls_out,
                timeout_sec=harbor_timeout_sec,
            )
            return current, controls, {
                "repaired": attempt > 0,
                "rounds": attempt,
                "repair_receipts": repair_receipts,
                "final_task_tree_digest": volc_rollout._task_tree_digest(current),
            }
        except harbor_launcher.HarborLauncherError as exc:
            stderr_tail = getattr(exc, "stderr_tail", "") or ""
            failure_class = factory_runtime_repair.classify_failure(
                str(exc), stderr=stderr_tail
            )
            bundle_out = (
                root / "failure" if attempt == 0 else root / "repair" / f"round-{attempt}" / "failure"
            )
            bundle = factory_runtime_repair.save_failure_bundle(
                out=bundle_out,
                scope=scope,
                task=current,
                controls_root=controls_out,
                failure_class=failure_class,
                # Record the real stderr so the failure bundle and any repair
                # session diagnose the actual defect, not the coarse class.
                message=(stderr_tail.strip() or str(exc)),
                attempt=attempt,
                reserved_liability_usd=0.0,
            )
            if failure_class == "infra":
                # Transient: never mutate the task; the barrier resumes and the
                # launcher's own reuse/top-up preserves any confirmed jobs.
                raise FactoryInfraRetry(bundle) from None
            if attempt >= max_repair_rounds:
                raise FactoryRuntimeQuarantine(
                    {
                        "reason": "runtime-control-unrepaired",
                        "scope": scope,
                        "failure_class": failure_class,
                        "failure_bundle_digest": bundle["bundle_digest"],
                        "failure_bundle_path": str(bundle_out / "failure-bundle.json"),
                        "rounds": attempt,
                        "repair_receipts": [r["receipt_digest"] for r in repair_receipts],
                    }
                ) from None
            repair = factory_runtime_repair.repair_task_once(
                task=current,
                failure_bundle_path=bundle_out / "failure-bundle.json",
                paper_ancestors=[p for p in paper_ancestors if p.is_file()],
                out=root / "repair" / f"round-{attempt + 1}",
                claude_executable=claude_executable,
                model=model,
                provider_env=provider_env,
                max_budget_usd=repair_max_budget_usd,
                timeout_sec=harbor_timeout_sec,
                round_number=attempt + 1,
                parent_task_digest=volc_rollout._task_tree_digest(current),
                failure_bundle_digest=bundle["bundle_digest"],
                credential_relay=credential_relay,
            )
            repair_receipts.append(repair)
            if repair["status"] != "produced" or repair["static_decision"] == "blocked":
                raise FactoryRuntimeQuarantine(
                    {
                        "reason": "runtime-repair-failed",
                        "scope": scope,
                        "failure_class": failure_class,
                        "repair_status": repair["status"],
                        "static_decision": repair["static_decision"],
                        "failure_bundle_digest": bundle["bundle_digest"],
                        "rounds": attempt + 1,
                    }
                ) from None
            current = Path(repair["repaired_task_path"])
    raise FactoryRuntimeQuarantine(
        {"reason": "runtime-repair-exhausted", "scope": scope, "rounds": max_repair_rounds}
    )


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
        harbor_native = bool(capability.get("harbor_native"))
        if (
            capability.get("schema_version") != "orbenchlab.runtime-capability.v1"
            or capability.get("receipt_digest") != _value_digest(unsigned_capability)
            or capability.get("receipt_digest")
            != intervention.get("capability_receipt_digest")
            or capability.get("checkpoint_capability") is not False
            or not isinstance(sources, Mapping)
            or sources.get("runtime_capability") != capability.get("receipt_digest")
        ):
            raise FactoryAutopilotError("intervention capability receipt binding failed")
        study_digest = intervention.get("study_receipt_digest")
        if study_digest is not None and harbor_native:
            # Harbor-native live study: bind the live-intervention-study receipt.
            from . import harbor_intervention_study as _his

            study = _load_object(iroot / "live-intervention-study.json")
            unsigned_study = {k: v for k, v in study.items() if k != "receipt_digest"}
            unsigned_study["arms"] = [
                {k: v for k, v in arm.items() if k != "reused"} for arm in study.get("arms", [])
            ]
            if (
                study.get("schema_version") != _his.STUDY_SCHEMA_VERSION
                or study.get("receipt_digest") != _his._digest(unsigned_study)
                or study.get("receipt_digest") != study_digest
                or sources.get("live_intervention_study") != study_digest
                or (
                    capability.get("causal_intervention_claim_available")
                    and study.get("evidence_level")
                    != "E4-controlled-same-session-intervention"
                )
            ):
                raise FactoryAutopilotError("live intervention study receipt binding failed")
        elif study_digest is not None:
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


RUNTIME_REPAIR_ADOPTION_SCHEMA = "orbenchlab.runtime-repair-adoption.v1"


def _adoption_records(workdir: Path) -> list[dict[str, Any]]:
    """Return validated runtime-repair adoption receipts, ascending by version.

    A recorded adoption whose canonical tree is missing or has drifted from its
    bound digest is a hard error, never a silent fall-back to the stale task.
    """

    root = workdir / "factory" / "runtime-repair"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for receipt_path in sorted(root.glob("adoption-v*.json")):
        if receipt_path.is_symlink():
            raise FactoryAutopilotError("runtime-repair adoption receipt is a symlink")
        record = _load_object(receipt_path)
        if record.get("schema_version") != RUNTIME_REPAIR_ADOPTION_SCHEMA:
            raise FactoryAutopilotError("runtime-repair adoption receipt schema mismatch")
        unsigned = {k: v for k, v in record.items() if k != "adoption_digest"}
        if record.get("adoption_digest") != _value_digest(unsigned):
            raise FactoryAutopilotError("runtime-repair adoption receipt digest mismatch")
        adopted = workdir.joinpath(*_safe_relative(str(record["adopted_task_relpath"])).parts)
        if adopted.is_symlink() or not adopted.is_dir():
            raise FactoryAutopilotError("adopted runtime-repair task tree is missing")
        if volc_rollout._task_tree_digest(adopted) != record["adopted_task_tree_digest"]:
            raise FactoryAutopilotError("adopted runtime-repair task tree has drifted")
        records.append(record)
    records.sort(key=lambda row: int(row["version"]))
    return records


def _current_task_root(plan: Mapping[str, Any], workdir: Path) -> Path:
    """Resolve the task the pipeline currently operates on.

    This is the latest canonically adopted runtime-repair version when one
    exists, so every barrier after a baseline repair (difficulty, intervention)
    and the promoted provenance reference the repaired tree, not the original
    control-failing ``task-repair-v2`` output.
    """

    records = _adoption_records(workdir)
    if records:
        adopted = workdir.joinpath(
            *_safe_relative(str(records[-1]["adopted_task_relpath"])).parts
        )
        return factory_gates.resolve_task_root(adopted)
    return factory_gates.resolve_task_root(
        _stage_output_path(plan, workdir, "task-repair-v2", kind="directory")
    )


def _adopt_repaired_task(
    *,
    workdir: Path,
    scope: str,
    original_task: Path,
    repaired_task: Path,
    repair_state: Mapping[str, Any],
    failure_bundle_digest: str | None,
    static_receipt_digest: str | None,
) -> dict[str, Any]:
    """Canonically adopt a repaired task as a new runtime-repair DAG version.

    Copies the repaired tree to ``factory/runtime-repair/v{N}/<slug>`` (harness
    owned, read-only) and writes a lineage receipt binding parent and adopted
    digests plus the repair receipt chain. Idempotent: an existing adoption for
    the same (scope, parent, adopted) triple is reused so resume never forks a
    duplicate version.
    """

    slug = repaired_task.name
    adopted_digest = volc_rollout._task_tree_digest(repaired_task)
    parent_digest = volc_rollout._task_tree_digest(original_task)
    existing = _adoption_records(workdir)
    for record in existing:
        if (
            record.get("scope") == scope
            and record.get("parent_task_tree_digest") == parent_digest
            and record.get("adopted_task_tree_digest") == adopted_digest
        ):
            return record
    version = (max((int(r["version"]) for r in existing), default=0)) + 1
    repair_root = workdir / "factory" / "runtime-repair"
    version_dir = repair_root / f"v{version}"
    canonical = version_dir / slug
    version_dir.mkdir(parents=True, exist_ok=True)
    if canonical.exists():
        _make_writable_tree(canonical)
        shutil.rmtree(canonical)
    temporary = version_dir / f".{slug}.{uuid.uuid4().hex}.tmp"
    shutil.copytree(repaired_task, temporary, symlinks=False)
    if volc_rollout._task_tree_digest(temporary) != adopted_digest:
        _make_writable_tree(temporary)
        shutil.rmtree(temporary)
        raise FactoryAutopilotError("adopted runtime-repair task copy changed content")
    os.replace(temporary, canonical)
    record = {
        "schema_version": RUNTIME_REPAIR_ADOPTION_SCHEMA,
        "version": version,
        "scope": scope,
        "parent_task_tree_digest": parent_digest,
        "adopted_task_tree_digest": adopted_digest,
        "adopted_task_relpath": canonical.relative_to(workdir).as_posix(),
        "repair_rounds": int(repair_state.get("rounds", 0)),
        "repair_receipt_digests": [
            r["receipt_digest"] for r in repair_state.get("repair_receipts", [])
        ],
        "failure_bundle_digest": failure_bundle_digest,
        "repaired_static_receipt_digest": static_receipt_digest,
    }
    record["adoption_digest"] = _value_digest(record)
    _atomic_json(repair_root / f"adoption-v{version}.json", record)
    _read_only_tree(canonical)
    return record


def _make_writable_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o755)
        except OSError:
            pass
    try:
        root.chmod(0o755)
    except OSError:
        pass


def _read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        try:
            path.chmod(0o555 if path.is_dir() else 0o444)
        except OSError:
            pass
    try:
        root.chmod(0o555)
    except OSError:
        pass


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

    # A task that still resolves to its version directory (rather than a unique
    # slug child) is a structural defect; name it explicitly instead of failing
    # obscurely on task_name / no_extraneous_files.
    if (task / "task.toml").is_file() and task.parent.name.startswith("task-v"):
        classified = factory_gates.classify_task_root(task.parent)
        if classified["kind"] != "single-child":
            return {}, {
                "reason": "static-gate-blocked",
                "gate": label,
                "task": task.name,
                "failing_criteria": ["task_root_layout"],
                "task_root_kind": classified["kind"],
                "detail": classified["detail"],
            }
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
    harbor_relay_host: str = "127.0.0.1",
    harbor_relay_bind_host: str = "127.0.0.1",
    max_repair_rounds: int = 0,
    repair_max_budget_usd: float = 4.0,
    credential_relay: bool = False,
) -> dict[str, Any]:
    original_task = factory_gates.resolve_task_root(
        _stage_output_path(plan, workdir, "task-repair-v2", kind="directory")
    )
    root = evidence_root / "baseline"
    static_receipt, quarantine = _static_gate(
        original_task, workdir=workdir, out=root / "static-gate", label="baseline"
    )
    if quarantine is not None:
        raise FactoryStaticGateBlocked(quarantine)
    task, controls, repair_state = _controls_with_repair(
        task=original_task,
        workdir=workdir,
        root=root,
        harbor_executable=harbor_executable,
        claude_executable=claude_executable,
        model=models[0],
        provider_env=provider_env,
        harbor_timeout_sec=harbor_timeout_sec,
        max_repair_rounds=max_repair_rounds,
        repair_max_budget_usd=repair_max_budget_usd,
        scope="baseline",
        credential_relay=credential_relay,
    )
    task_digest = volc_rollout._task_tree_digest(task)
    adoption: dict[str, Any] | None = None
    if task is not original_task:
        # A repaired task passed controls: re-run the static gate over the
        # repaired tree so the trusted bundle binds a fully gated task.
        repaired_static, repaired_quarantine = _static_gate(
            task, workdir=workdir, out=root / "static-gate-repaired", label="baseline-repaired"
        )
        if repaired_quarantine is not None:
            raise FactoryStaticGateBlocked(repaired_quarantine)
        static_receipt = repaired_static
        # Canonically adopt the repaired tree as a new runtime-repair DAG
        # version with a lineage receipt, so the model matrix and every barrier
        # after this one operate on the repaired task, not the original.
        repair_receipts = repair_state.get("repair_receipts", [])
        adoption = _adopt_repaired_task(
            workdir=workdir,
            scope="baseline",
            original_task=original_task,
            repaired_task=task,
            repair_state=repair_state,
            failure_bundle_digest=(
                repair_receipts[0].get("failure_bundle_digest") if repair_receipts else None
            ),
            static_receipt_digest=repaired_static["receipt_digest"],
        )
        task = _current_task_root(plan, workdir)
        task_digest = volc_rollout._task_tree_digest(task)
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
        relay_host=harbor_relay_host,
        relay_bind_host=harbor_relay_bind_host,
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
    # Auto-feedback: classify the discrimination pattern deterministically. A
    # 'controls pass but no model solves' result is fed back for contract review
    # rather than silently promoted or blamed on model quality.
    discrimination_feedback = harbor_model_matrix.classify_discrimination(screening)
    _atomic_json(root / "matrix" / "discrimination-feedback.json", discrimination_feedback)
    trusted_source = root / "trusted-source"
    trusted_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "matrix" / "discrimination-feedback.json", trusted_source)
    shutil.copy2(root / "controls" / "harbor-control-screening.json", trusted_source)
    shutil.copy2(root / "matrix" / "harbor-model-matrix.json", trusted_source)
    shutil.copy2(root / "matrix" / "screening-report.json", trusted_source)
    static_gate_dir = "static-gate-repaired" if repair_state["repaired"] else "static-gate"
    shutil.copy2(
        root / static_gate_dir / "authoring-receipt.json",
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
            "discrimination_feedback": discrimination_feedback["feedback_digest"],
            "static_authoring": static_receipt["receipt_digest"],
            "trace_manifest": trace["manifest_digest"],
            **(
                {"runtime_repair_adoption": adoption["adoption_digest"]}
                if adoption is not None
                else {}
            ),
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
        "discrimination": {
            "kind": discrimination_feedback["kind"],
            "contract_review_required": discrimination_feedback["contract_review_required"],
            "feedback_digest": discrimination_feedback["feedback_digest"],
            "reason": discrimination_feedback["reason"],
        },
        "runtime_repair": {
            "repaired": repair_state["repaired"],
            "rounds": repair_state["rounds"],
            "final_task_tree_digest": repair_state["final_task_tree_digest"],
            "repaired_task_path": (
                repair_state["repair_receipts"][-1]["repaired_task_path"]
                if repair_state["repaired"]
                else None
            ),
            "repair_receipt_digests": [
                r["receipt_digest"] for r in repair_state["repair_receipts"]
            ],
            "adoption_digest": adoption["adoption_digest"] if adoption else None,
            "adopted_task_relpath": adoption["adopted_task_relpath"] if adoption else None,
            "adopted_version": adoption["version"] if adoption else None,
        },
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
    harbor_relay_host: str = "127.0.0.1",
    harbor_relay_bind_host: str = "127.0.0.1",
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
        from . import factory_runtime_repair

        try:
            controls = harbor_launcher.launch_controls(
                task,
                harbor_executable=harbor_executable,
                out=root / "controls",
                timeout_sec=harbor_timeout_sec,
            )
        except harbor_launcher.HarborLauncherError as exc:
            # A variant whose controls fail is isolated with a machine bundle
            # instead of crashing the run; infrastructure failures resume.
            failure_class = factory_runtime_repair.classify_failure(str(exc))
            bundle = factory_runtime_repair.save_failure_bundle(
                out=root / "failure",
                scope=f"variant:{variant_id}",
                task=task,
                controls_root=root / "controls",
                failure_class=failure_class,
                message=str(exc),
                attempt=0,
                reserved_liability_usd=0.0,
            )
            if failure_class == "infra":
                raise FactoryInfraRetry(bundle) from None
            raise FactoryRuntimeQuarantine(
                {
                    "reason": "variant-control-failed",
                    "scope": f"variant:{variant_id}",
                    "failure_class": failure_class,
                    "failure_bundle_digest": bundle["bundle_digest"],
                    "failure_bundle_path": str(root / "failure" / "failure-bundle.json"),
                }
            ) from None
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
            relay_host=harbor_relay_host,
            relay_bind_host=harbor_relay_bind_host,
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
    base_task = _current_task_root(plan, workdir)
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


# Task-tree entries that must never be visible to a solver agent: the
# reference solution, the hidden verifier, and oracle-derived data files.
_INTERVENTION_HIDDEN_DIRS = frozenset({"solution", "tests"})
_INTERVENTION_HIDDEN_DATA = (
    "reference-bounds.json",
    "difficulty-matrix.json",
    "paper-task-derivation.json",
)


def _build_agent_visible_template(task: Path, out: Path) -> dict[str, Any]:
    """Copy the task into an agent-visible template stripped of oracle material.

    The returned template contains only what a Harbor agent environment would
    expose (instruction, environment, public input data, task.toml) plus an
    empty ``submission/`` directory.  ``solution/``, ``tests/``, the paper
    provenance/derivation and oracle data files are removed, and both the
    visible and removed (hidden) tree digests are recorded so the receipt can
    prove the boundary.
    """

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(task, out, symlinks=False)
    hidden_rows: list[dict[str, str]] = []

    def record_hidden(path: Path) -> None:
        for candidate in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if candidate.is_file() and not candidate.is_symlink():
                hidden_rows.append(
                    {
                        "path": candidate.relative_to(out).as_posix(),
                        "content_digest": _file_digest(candidate),
                    }
                )

    for name in _INTERVENTION_HIDDEN_DIRS:
        target = out / name
        if target.exists():
            record_hidden(target)
            shutil.rmtree(target)
    for name in ("paper-provenance.json",):
        target = out / name
        if target.is_file():
            record_hidden(target)
            target.unlink()
    data_dir = out / "data"
    if data_dir.is_dir():
        for name in _INTERVENTION_HIDDEN_DATA:
            target = data_dir / name
            if target.is_file():
                record_hidden(target)
                target.unlink()
    submission = out / "submission"
    submission.mkdir(exist_ok=True)
    (submission / ".keep").write_text("", encoding="utf-8")
    visible_rows = [
        {
            "path": path.relative_to(out).as_posix(),
            "content_digest": _file_digest(path),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    visible_names = sorted(
        {row["path"].split("/", 1)[0] for row in visible_rows}
        - {"submission"}
    )
    return {
        "template": out,
        "visible_tree_digest": _value_digest(visible_rows),
        "hidden_tree_digest": _value_digest(sorted(
            hidden_rows, key=lambda row: row["path"]
        )),
        "hidden_file_count": len(hidden_rows),
        "agent_read_only_names": visible_names,
    }


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
    verifier_adapter: "session_interventions.GradeCallable | None" = None,
    live_arm_executor=None,
    live_levels: Sequence[str] = ("L1", "L2", "L3"),
    live_repeats: int = 5,
) -> dict[str, Any]:
    """Probe the same-session injection channel and honestly record its evidence.

    A verifier-grounded controlled study runs only when a *trusted* Harbor-
    equivalent, secret-safe verifier adapter is supplied (``verifier_adapter``).
    The autopilot ships no such adapter — an untrusted solver agent that needs
    both a shell and the provider credential cannot be made secret-safe or
    Harbor-truthful outside a container — so by default it records a
    machine-readable E0/E1 capability receipt with a precise reason and never
    fabricates an E4 causal claim from a host-shell verifier.
    """

    task = _current_task_root(plan, workdir)
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
    live_study_receipt: dict[str, Any] | None = None
    study_reason = None
    template_info: dict[str, Any] | None = None
    if not capability["same_session_hint_injection"]:
        study_reason = "runtime-unsupported"
    elif not enabled:
        study_reason = "disabled-by-configuration"
    elif live_arm_executor is not None:
        # Harbor-native live intervention: each arm performs the real
        # interrupt/hint handshake in a no-network container and is graded by
        # the SEPARATE frozen verifier, so the study reaches E4 honestly.
        from . import harbor_intervention_study as _his

        template_info = _build_agent_visible_template(task, root / "agent-template")
        live_study_receipt = _his.run_live_intervention_study(
            task_id=volc_rollout._task_id(task),
            model=model,
            levels=list(live_levels),
            repeats=live_repeats,
            out=root / "live-study",
            arm_executor=live_arm_executor,
            max_budget_usd_per_arm=max_budget_usd,
        )
        study_reason = "harbor-native-live-intervention"
    elif verifier_adapter is None:
        # No Harbor-equivalent, secret-safe, isolated verifier adapter is
        # available; a host-shell verifier would score an empty submission as
        # pass and fake E4, so the study is honestly not run.
        study_reason = "no-harbor-grounded-verifier-adapter"
        template_info = _build_agent_visible_template(task, root / "agent-template")
    else:
        template_info = _build_agent_visible_template(task, root / "agent-template")
        study_receipt = session_interventions.run_intervention_study(
            profile="claude-code",
            model=model,
            prompt=_intervention_prompt(task),
            template_workdir=template_info["template"],
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
            agent_read_only_names=template_info["agent_read_only_names"],
            agent_allow_bash=True,
            grade=verifier_adapter,
        )
    # Unify evidence fields across the honest E0/E1 path, the caller-adapter
    # path, and the Harbor-native live path.
    if live_study_receipt is not None:
        _study_valid = bool(live_study_receipt.get("study_valid"))
        u_harbor_native = True
        u_study_status = "completed" if _study_valid else "invalid"
        u_study_level = (
            live_study_receipt["evidence_level"]
            if _study_valid
            else "E1-live-study-no-valid-arms"
        )
        u_causal = bool(
            _study_valid
            and live_study_receipt.get("evidence_level")
            == "E4-controlled-same-session-intervention"
        )
        u_verifier_mech = "harbor-native-container-verifier"
    elif study_receipt is not None:
        u_harbor_native = False
        u_study_status = study_receipt["status"]
        u_study_level = study_receipt["evidence_level"]
        u_causal = bool(
            study_receipt.get("evidence_level") == "E4-controlled-same-session-intervention"
        )
        u_verifier_mech = "caller-supplied-trusted-adapter"
    else:
        u_harbor_native = False
        u_study_status = "not-run"
        u_study_level = (
            "E0-unsupported" if study_reason == "runtime-unsupported" else "E1-capability-only-no-study"
        )
        u_causal = False
        u_verifier_mech = None
    capability_receipt = {
        "schema_version": "orbenchlab.runtime-capability.v1",
        "task_tree_digest": task_digest,
        "trajectory_evidence_level": "E3",
        "channel": "claude-code-stream-json-same-session",
        "harbor_native": u_harbor_native,
        "checkpoint_capability": False,
        "same_checkpoint_hint_injection": False,
        "same_session_hint_injection": bool(
            capability["same_session_hint_injection"]
        ),
        "agent_visible_tree_digest": (
            template_info["visible_tree_digest"] if template_info else None
        ),
        "hidden_from_agent_tree_digest": (
            template_info["hidden_tree_digest"] if template_info else None
        ),
        "hidden_from_agent_file_count": (
            template_info["hidden_file_count"] if template_info else None
        ),
        "verifier_mechanism": u_verifier_mech,
        "policy_digest": session_interventions._digest(policy),
        "study_status": u_study_status,
        "study_evidence_level": u_study_level,
        "study_reason": study_reason,
        "causal_intervention_claim_available": u_causal,
        "limitations": (
            [
                "Live intervention interrupts the model at a tool/assistant "
                "checkpoint on the same session and grades with the separate "
                "no-network container verifier.",
                "Harbor task matrix trials remain independent restarts (restart-with-hint is E3).",
            ]
            if u_harbor_native
            else [
                "Same-session stdin injection is a turn-boundary continuation, not a "
                "mid-token checkpoint restore; it is not Harbor-native.",
                "Harbor task trials remain independent restarts (restart-with-hint is E3).",
            ]
        ),
    }
    capability_receipt["receipt_digest"] = _value_digest(
        {k: v for k, v in capability_receipt.items() if k != "receipt_digest"}
    )
    _atomic_json(trusted_source / "runtime-capability.json", capability_receipt)
    source_receipts = {"runtime_capability": capability_receipt["receipt_digest"]}
    if live_study_receipt is not None:
        shutil.copy2(
            root / "live-study" / "live-intervention-study.json",
            trusted_source / "live-intervention-study.json",
        )
        source_receipts["live_intervention_study"] = live_study_receipt["receipt_digest"]
    elif study_receipt is not None:
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
        "harbor_native": capability_receipt["harbor_native"],
        "causal_intervention_claim_available": capability_receipt[
            "causal_intervention_claim_available"
        ],
        "study_receipt_digest": (
            live_study_receipt["receipt_digest"]
            if live_study_receipt is not None
            else (study_receipt["receipt_digest"] if study_receipt is not None else None)
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
    intervention_study: bool = True,
    intervention_verifier_argv: Sequence[str] = (),
    intervention_control: int = 3,
    intervention_treatment: int = 3,
    intervention_timeout_sec: float = 900.0,
    max_intervention_liability_usd: float = 20.0,
    max_runtime_repair_rounds: int = 1,
    repair_max_budget_usd: float = 4.0,
    harbor_relay_host: str = "127.0.0.1",
    harbor_relay_bind_host: str = "127.0.0.1",
    credential_relay: bool = True,
    live_arm_executor_factory=None,
    live_intervention_levels: Sequence[str] = ("L1", "L2", "L3"),
    live_intervention_repeats: int = 5,
) -> dict[str, Any]:
    """Run or resume the complete semantic-and-runtime factory state machine.

    ``credential_relay`` defaults to on: every claude-code agent session in the
    unattended chain (paper stages, runtime repair, promotion review) receives a
    revocable loopback-scoped token from a host-side relay instead of the real
    provider credential, and each session fails closed if the real token is found
    in any artifact or process argv.
    """

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
    if (
        not isinstance(max_runtime_repair_rounds, int)
        or isinstance(max_runtime_repair_rounds, bool)
        or not 0 <= max_runtime_repair_rounds <= 5
        or not 0 < repair_max_budget_usd <= 100
    ):
        raise FactoryAutopilotError("runtime-repair round or budget bounds are invalid")
    maximum_harbor_liability = (
        (1 + max_variants)
        * len(models)
        * repetitions
        * max_budget_usd
        * max_job_attempts
    )
    if maximum_harbor_liability > max_harbor_liability_usd:
        raise FactoryAutopilotError("maximum Harbor model liability exceeds the configured cap")
    # Worst-case spend is not the model matrix alone: every control scope
    # (baseline plus each variant) may drive up to max_runtime_repair_rounds
    # repair sessions, the intervention study runs control+treatment sessions,
    # and promotion runs one review session per reviewer model. Bind the whole
    # ledger into identity and cap it so a resume cannot silently widen it.
    maximum_repair_liability = (
        (1 + max_variants) * max_runtime_repair_rounds * repair_max_budget_usd
    )
    maximum_intervention_liability = (
        (intervention_control + intervention_treatment) * max_budget_usd
        if intervention_study
        else 0.0
    )
    maximum_promotion_liability = (
        len(factory_promotion._reviewer_models(checked)) * max_budget_usd if promote else 0.0
    )
    maximum_total_liability = round(
        maximum_harbor_liability
        + maximum_repair_liability
        + maximum_intervention_liability
        + maximum_promotion_liability,
        6,
    )
    configured_total_cap = round(
        max_harbor_liability_usd
        + maximum_repair_liability
        + (max_intervention_liability_usd if intervention_study else 0.0)
        + maximum_promotion_liability,
        6,
    )
    if maximum_total_liability > configured_total_cap:
        raise FactoryAutopilotError("maximum total factory liability exceeds the configured caps")
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
        "credential_transport": (
            "host-side-relay-per-session-scoped-token" if credential_relay else "direct-provider-env"
        ),
        "runtime_repair": {
            "max_runtime_repair_rounds": max_runtime_repair_rounds,
            "repair_max_budget_usd": repair_max_budget_usd,
        },
        "liability_ledger": {
            "harbor_matrix_usd": round(maximum_harbor_liability, 6),
            "runtime_repair_usd": round(maximum_repair_liability, 6),
            "intervention_usd": round(maximum_intervention_liability, 6),
            "promotion_review_usd": round(maximum_promotion_liability, 6),
            "maximum_total_usd": maximum_total_liability,
        },
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
                        review_executable=claude_binding["path"],
                        review_timeout_sec=promotion_review_timeout_sec,
                        review_max_budget_usd=max_budget_usd,
                        max_review_tokens=promotion_max_review_tokens,
                        credential_relay=credential_relay,
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
                        harbor_relay_host=harbor_relay_host,
                        harbor_relay_bind_host=harbor_relay_bind_host,
                        max_repair_rounds=max_runtime_repair_rounds,
                        repair_max_budget_usd=repair_max_budget_usd,
                        credential_relay=credential_relay,
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
                        live_arm_executor=(
                            live_arm_executor_factory(_current_task_root(checked, workspace))
                            if live_arm_executor_factory is not None
                            else None
                        ),
                        live_levels=live_intervention_levels,
                        live_repeats=live_intervention_repeats,
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
                        harbor_relay_host=harbor_relay_host,
                        harbor_relay_bind_host=harbor_relay_bind_host,
                    )
                    state = _write_state(state_path, state)
            except (FactoryStaticGateBlocked, FactoryRuntimeQuarantine) as blocked:
                state["status"] = "quarantined"
                state["quarantine"] = blocked.quarantine
                state = _write_state(state_path, state)
                return dict(state)
            except FactoryInfraRetry as infra:
                # Transient infrastructure failure: record the machine bundle,
                # keep the run active, and return so a resume retries without
                # mutating the task or discarding confirmed jobs.
                state["status"] = "active"
                state["last_infra_retry"] = infra.bundle
                state = _write_state(state_path, state)
                return dict(state)
            result = agentic_factory.run_factory(
                checked,
                workdir=workspace,
                out=factory_root,
                environments=environments,
                executables=executables,
                max_new_stages=1,
                credential_relay=credential_relay,
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
