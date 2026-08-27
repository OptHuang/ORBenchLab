"""Resumable control plane for autonomous paper-to-benchmark agent sessions.

Agents own semantic work: reading papers, proposing tasks, authoring files,
reviewing, repairing, analysing trajectories and proposing difficulty variants.
This module owns only the trustworthy outer loop: a hash-bound DAG, bounded
session invocations, hard budgets, required-output contracts and atomic state.

Benchmark execution and grading remain delegated to Harbor and the task's
verifier.  A completed semantic stage is therefore not, by itself, benchmark
acceptance; plans must include the existing deterministic/Harbor receipts
before their final promotion stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import fcntl
import copy
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .agent_sessions import DEFAULT_MAX_OUTPUT_BYTES, AgentSessionError, run_session
from .core.errors import ORBenchError


class AgenticFactoryError(ORBenchError):
    """A factory plan, session or output violated its evidence contract."""

    exit_code = 8


PLAN_SCHEMA_VERSION = "orbenchlab.agentic-factory-plan.v1"
RUN_SCHEMA_VERSION = "orbenchlab.agentic-factory-run.v1"
ATTEMPT_SCHEMA_VERSION = "orbenchlab.agentic-factory-attempt.v1"
_STAGE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PROFILES = frozenset({"codex", "claude-code"})
_OUTPUT_KINDS = frozenset({"file", "json", "directory"})
DEFAULT_MAX_WORKSPACE_BYTES = 1024 * 1024 * 1024
MAX_WORKSPACE_FILES = 20_000
MAX_WORKSPACE_FILE_BYTES = 256 * 1024 * 1024


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _value_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_run(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "run_digest"}
    signed = {**unsigned, "run_digest": _value_digest(unsigned)}
    _atomic_json(path, signed)
    return signed


def _safe_relative_path(value: Any) -> str:
    raw = str(value or "")
    pure = PurePosixPath(raw)
    normalized = pure.as_posix()
    if (
        not raw
        or raw != normalized
        or raw.startswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.parts[0].startswith(".")
    ):
        raise AgenticFactoryError(f"unsafe required-output path: {raw!r}")
    return normalized


def _normalise_output(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AgenticFactoryError("required_outputs entries must be objects")
    unknown = sorted(set(value) - {"path", "kind"})
    if unknown:
        raise AgenticFactoryError(f"required output has unsupported key(s): {unknown}")
    kind = str(value.get("kind", ""))
    if kind not in _OUTPUT_KINDS:
        raise AgenticFactoryError(f"required output kind must be one of {sorted(_OUTPUT_KINDS)}")
    return {"path": _safe_relative_path(value.get("path")), "kind": kind}


def _normalise_stage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticFactoryError("factory stages must be objects")
    allowed = {
        "id",
        "role",
        "profile",
        "model",
        "prompt",
        "depends_on",
        "timeout_sec",
        "max_attempts",
        "max_budget_usd",
        "max_output_bytes",
        "required_outputs",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AgenticFactoryError(f"factory stage has unsupported key(s): {unknown}")
    stage_id = str(value.get("id", ""))
    if not _STAGE_ID.fullmatch(stage_id):
        raise AgenticFactoryError(f"invalid stage id: {stage_id!r}")
    role = str(value.get("role", "")).strip()
    model = str(value.get("model", "")).strip()
    prompt = str(value.get("prompt", "")).strip()
    profile = str(value.get("profile", ""))
    if not role or len(role) > 160:
        raise AgenticFactoryError(f"stage {stage_id} requires a bounded non-empty role")
    if profile not in _PROFILES:
        raise AgenticFactoryError(f"stage {stage_id} has unsupported profile {profile!r}")
    if not model or len(model) > 256:
        raise AgenticFactoryError(f"stage {stage_id} requires a bounded model id")
    if not prompt or len(prompt.encode("utf-8")) > 256_000:
        raise AgenticFactoryError(f"stage {stage_id} requires a prompt of at most 256000 bytes")
    dependencies = value.get("depends_on", [])
    if (
        not isinstance(dependencies, list)
        or any(not isinstance(item, str) or not _STAGE_ID.fullmatch(item) for item in dependencies)
        or len(set(dependencies)) != len(dependencies)
        or stage_id in dependencies
    ):
        raise AgenticFactoryError(f"stage {stage_id} has invalid dependencies")
    timeout_sec = value.get("timeout_sec")
    max_attempts = value.get("max_attempts")
    max_budget_usd = value.get("max_budget_usd")
    max_output_bytes = value.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)
    if not isinstance(timeout_sec, int) or isinstance(timeout_sec, bool) or not 1 <= timeout_sec <= 86_400:
        raise AgenticFactoryError(f"stage {stage_id} timeout_sec must be in 1..86400")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 5:
        raise AgenticFactoryError(f"stage {stage_id} max_attempts must be in 1..5")
    if (
        not isinstance(max_budget_usd, (int, float))
        or isinstance(max_budget_usd, bool)
        or not 0 < float(max_budget_usd) <= 100
    ):
        raise AgenticFactoryError(f"stage {stage_id} max_budget_usd must be in (0,100]")
    if (
        not isinstance(max_output_bytes, int)
        or isinstance(max_output_bytes, bool)
        or not 1 <= max_output_bytes <= 256 * 1024 * 1024
    ):
        raise AgenticFactoryError(f"stage {stage_id} max_output_bytes must be in 1..268435456")
    raw_outputs = value.get("required_outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise AgenticFactoryError(f"stage {stage_id} requires at least one output contract")
    outputs = [_normalise_output(item) for item in raw_outputs]
    paths = [item["path"] for item in outputs]
    if len(set(paths)) != len(paths):
        raise AgenticFactoryError(f"stage {stage_id} repeats a required-output path")
    return {
        "id": stage_id,
        "role": role,
        "profile": profile,
        "model": model,
        "prompt": prompt,
        "depends_on": list(dependencies),
        "timeout_sec": timeout_sec,
        "max_attempts": max_attempts,
        "max_budget_usd": float(max_budget_usd),
        "max_output_bytes": max_output_bytes,
        "required_outputs": outputs,
    }


def _assert_acyclic(stages: Sequence[Mapping[str, Any]]) -> None:
    dependencies = {str(stage["id"]): set(stage["depends_on"]) for stage in stages}
    unknown = sorted({dep for deps in dependencies.values() for dep in deps} - set(dependencies))
    if unknown:
        raise AgenticFactoryError(f"factory plan references unknown dependencies: {unknown}")
    remaining = {key: set(value) for key, value in dependencies.items()}
    resolved: set[str] = set()
    while remaining:
        ready = sorted(key for key, value in remaining.items() if value <= resolved)
        if not ready:
            raise AgenticFactoryError("factory stage graph contains a dependency cycle")
        for key in ready:
            resolved.add(key)
            remaining.pop(key)


def compile_plan(
    *,
    name: str,
    source_binding_digest: str,
    stages: Sequence[Mapping[str, Any]],
    workspace_manifest: str | None = None,
    max_workspace_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES,
) -> dict[str, Any]:
    """Validate and hash an immutable autonomous-session DAG."""

    clean_name = str(name).strip()
    if not clean_name or len(clean_name) > 160:
        raise AgenticFactoryError("factory plan name must contain 1..160 characters")
    if not _DIGEST.fullmatch(str(source_binding_digest)):
        raise AgenticFactoryError("source_binding_digest must be sha256:<64 lowercase hex>")
    if (
        not isinstance(max_workspace_bytes, int)
        or isinstance(max_workspace_bytes, bool)
        or not 1 <= max_workspace_bytes <= 10 * 1024 * 1024 * 1024
    ):
        raise AgenticFactoryError("max_workspace_bytes must be in 1..10737418240")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
        raise AgenticFactoryError("factory plan requires at least one stage")
    normalised = [_normalise_stage(stage) for stage in stages]
    ids = [stage["id"] for stage in normalised]
    if len(set(ids)) != len(ids):
        raise AgenticFactoryError("factory stage ids must be unique")
    output_owners: dict[str, str] = {}
    for stage in normalised:
        for output in stage["required_outputs"]:
            path = output["path"]
            if path in output_owners:
                raise AgenticFactoryError(
                    f"required output {path!r} is owned by both {output_owners[path]} and {stage['id']}"
                )
            output_owners[path] = stage["id"]
    owned_paths = sorted(output_owners)
    for index, left in enumerate(owned_paths):
        for right in owned_paths[index + 1 :]:
            if right.startswith(left.rstrip("/") + "/"):
                raise AgenticFactoryError(
                    f"required outputs may not overlap: {left!r} contains {right!r}"
                )
    _assert_acyclic(normalised)
    maximum_liability = round(
        sum(stage["max_budget_usd"] * stage["max_attempts"] for stage in normalised),
        6,
    )
    if maximum_liability > 100:
        raise AgenticFactoryError("factory maximum model liability exceeds 100 USD")
    unsigned = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "name": clean_name,
        "source_binding_digest": str(source_binding_digest),
        "workspace_manifest": (
            _safe_relative_path(workspace_manifest) if workspace_manifest else None
        ),
        "maximum_model_liability_usd": maximum_liability,
        "max_workspace_bytes": max_workspace_bytes,
        "stages": normalised,
    }
    plan_digest = _value_digest(unsigned)
    return {
        **unsigned,
        "factory_id": f"factory-{plan_digest.removeprefix('sha256:')[:20]}",
        "plan_digest": plan_digest,
    }


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompile a persisted plan and reject any stale or extra fields."""

    allowed = {
        "schema_version",
        "name",
        "source_binding_digest",
        "workspace_manifest",
        "maximum_model_liability_usd",
        "max_workspace_bytes",
        "stages",
        "factory_id",
        "plan_digest",
    }
    unknown = sorted(set(value) - allowed)
    if unknown or value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise AgenticFactoryError("factory plan has unsupported schema or fields")
    compiled = compile_plan(
        name=str(value.get("name", "")),
        source_binding_digest=str(value.get("source_binding_digest", "")),
        stages=value.get("stages", []),
        workspace_manifest=(
            str(value["workspace_manifest"]) if value.get("workspace_manifest") else None
        ),
        max_workspace_bytes=int(value.get("max_workspace_bytes", 0)),
    )
    if value.get("factory_id") != compiled["factory_id"] or value.get("plan_digest") != compiled["plan_digest"]:
        raise AgenticFactoryError("factory plan identity/digest does not match its contents")
    if value.get("maximum_model_liability_usd") != compiled["maximum_model_liability_usd"]:
        raise AgenticFactoryError("factory plan maximum model liability is stale")
    return compiled


def load_plan(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AgenticFactoryError("factory plan is not valid UTF-8 JSON") from None
    if not isinstance(value, Mapping):
        raise AgenticFactoryError("factory plan root must be an object")
    return validate_plan(value)


def write_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    checked = validate_plan(plan)
    destination = Path(path)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise AgenticFactoryError("refusing to overwrite a malformed factory plan") from None
        if existing != checked:
            raise AgenticFactoryError("refusing to overwrite a different factory plan")
        return destination
    _atomic_json(destination, checked)
    return destination


def _new_run(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "factory_id": plan["factory_id"],
        "plan_digest": plan["plan_digest"],
        "status": "active",
        "evidence_level": "E1-agent-session-process",
        "limitations": [
            "Semantic factory completion is not static-gate, verifier, Harbor, E4, or promotion evidence."
        ],
        "stages": {
            stage["id"]: {"status": "pending", "attempts": [], "output_artifacts": []}
            for stage in plan["stages"]
        },
    }


def _validate_session_receipt(
    root: Path,
    plan: Mapping[str, Any],
    stage: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    workspace: Path | None,
) -> None:
    session_id = attempt.get("session_id")
    expected_digest = attempt.get("session_receipt_digest")
    if session_id is None and expected_digest is None:
        if attempt.get("status") == "completed":
            raise AgenticFactoryError("completed factory attempt has no agent-session receipt")
        return
    if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise AgenticFactoryError("factory attempt has an invalid session id")
    path = root / "sessions" / session_id / "receipt.json"
    if not path.is_file() or path.is_symlink() or _file_digest(path) != expected_digest:
        raise AgenticFactoryError("factory attempt session receipt digest mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AgenticFactoryError("factory attempt session receipt is malformed") from None
    if not isinstance(receipt, Mapping):
        raise AgenticFactoryError("factory attempt session receipt root must be an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    identity = receipt.get("identity")
    attempt_number = int(attempt["attempt"])
    if (
        receipt.get("receipt_digest") != _value_digest(unsigned)
        or receipt.get("session_id") != session_id
        or (attempt.get("status") == "completed" and receipt.get("status") != "completed")
        or not isinstance(identity, Mapping)
        or identity.get("stage")
        != f"{plan['factory_id']}/{stage['id']}/attempt-{attempt_number}"
        or identity.get("profile") != stage["profile"]
        or identity.get("model") != stage["model"]
        or identity.get("prompt_digest")
        != "sha256:" + hashlib.sha256(_stage_prompt(plan, stage).encode("utf-8")).hexdigest()
        or identity.get("timeout_sec") != stage["timeout_sec"]
        or identity.get("max_output_bytes") != stage["max_output_bytes"]
        or identity.get("max_budget_usd") != stage["max_budget_usd"]
    ):
        raise AgenticFactoryError("factory attempt session receipt failed its signed binding")
    session_root = path.parent
    stdout_path = session_root / "stdout.bin"
    stderr_path = session_root / "stderr.bin"
    if (
        not stdout_path.is_file()
        or stdout_path.is_symlink()
        or not stderr_path.is_file()
        or stderr_path.is_symlink()
        or receipt.get("stdout_digest") != _file_digest(stdout_path)
        or receipt.get("stderr_digest") != _file_digest(stderr_path)
    ):
        raise AgenticFactoryError("factory attempt session output digest mismatch")
    if workspace is not None and identity.get("workdir_binding") != _value_digest(
        str(workspace.resolve())
    ):
        raise AgenticFactoryError("factory attempt session is bound to another workdir")


def _validate_run_chain(
    root: Path,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    workspace: Path | None = None,
) -> None:
    stage_specs = {stage["id"]: stage for stage in plan["stages"]}
    for stage_id, state in run["stages"].items():
        if not isinstance(state.get("attempts"), list) or not isinstance(
            state.get("output_artifacts"), list
        ):
            raise AgenticFactoryError(f"factory run has malformed attempt state for {stage_id}")
        attempts = state["attempts"]
        for index, summary in enumerate(attempts, start=1):
            if not isinstance(summary, Mapping) or summary.get("attempt") != index:
                raise AgenticFactoryError(f"factory run has a broken attempt chain for {stage_id}")
            expected_relative = f"stages/{stage_id}/attempt-{index:03d}.json"
            if summary.get("receipt") != expected_relative:
                raise AgenticFactoryError(f"factory run has an unsafe attempt path for {stage_id}")
            path = root / expected_relative
            if not path.is_file() or path.is_symlink():
                raise AgenticFactoryError(f"factory run is missing an attempt receipt for {stage_id}")
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise AgenticFactoryError(f"factory attempt receipt is malformed for {stage_id}") from None
            if not isinstance(receipt, Mapping):
                raise AgenticFactoryError(f"factory attempt receipt root is invalid for {stage_id}")
            unsigned = {key: value for key, value in receipt.items() if key != "attempt_digest"}
            if (
                receipt.get("attempt_digest") != _value_digest(unsigned)
                or summary.get("attempt_digest") != receipt.get("attempt_digest")
                or summary.get("status") != receipt.get("status")
                or receipt.get("factory_id") != plan["factory_id"]
                or receipt.get("plan_digest") != plan["plan_digest"]
                or receipt.get("stage_id") != stage_id
                or receipt.get("attempt") != index
            ):
                raise AgenticFactoryError(f"factory attempt receipt binding failed for {stage_id}")
            _validate_session_receipt(
                root,
                plan,
                stage_specs[stage_id],
                receipt,
                workspace=workspace,
            )
        if state["status"] == "completed":
            if not attempts or attempts[-1].get("status") != "completed":
                raise AgenticFactoryError(f"completed stage has no completed receipt: {stage_id}")
            last_path = root / str(attempts[-1]["receipt"])
            last = json.loads(last_path.read_text(encoding="utf-8"))
            if state["output_artifacts"] != last.get("output_artifacts"):
                raise AgenticFactoryError(f"completed stage output chain mismatch: {stage_id}")
            if workspace is not None:
                actual = _snapshot_outputs(workspace, stage_specs[stage_id])
                expected = {row["path"]: {"content_digest": row["content_digest"], "file_count": row["file_count"]} for row in state["output_artifacts"]}
                if actual != expected:
                    raise AgenticFactoryError(f"completed stage output changed after receipt: {stage_id}")
        if state["status"] == "failed" and (not attempts or attempts[-1].get("status") != "failed"):
            raise AgenticFactoryError(f"failed stage has no failed receipt: {stage_id}")


def _load_run(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AgenticFactoryError("factory run state is not valid UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise AgenticFactoryError("factory run state root must be an object")
    if (
        value.get("schema_version") != RUN_SCHEMA_VERSION
        or value.get("factory_id") != plan["factory_id"]
        or value.get("plan_digest") != plan["plan_digest"]
        or set(value.get("stages", {})) != {stage["id"] for stage in plan["stages"]}
    ):
        raise AgenticFactoryError("factory run state is stale or bound to another plan")
    unsigned = {key: item for key, item in value.items() if key != "run_digest"}
    if value.get("run_digest") != _value_digest(unsigned):
        raise AgenticFactoryError("factory run state digest does not match its contents")
    status = value.get("status")
    if status not in {"active", "semantic-complete-e1", "quarantined"}:
        raise AgenticFactoryError("factory run has an unsupported top-level status")
    for stage_id, state in value["stages"].items():
        if not isinstance(state, Mapping) or state.get("status") not in {
            "pending",
            "running",
            "completed",
            "failed",
        }:
            raise AgenticFactoryError(f"factory run has invalid state for {stage_id}")
    stage_statuses = [state["status"] for state in value["stages"].values()]
    completion_payload = {
        "factory_id": plan["factory_id"],
        "plan_digest": plan["plan_digest"],
        "stages": value["stages"],
        "evidence_level": "E1-agent-session-process",
    }
    if status == "semantic-complete-e1":
        if (
            any(item != "completed" for item in stage_statuses)
            or value.get("completion_digest") != _value_digest(completion_payload)
            or "quarantine" in value
        ):
            raise AgenticFactoryError("semantic-complete factory run violates terminal invariants")
    elif status == "quarantined":
        quarantine = value.get("quarantine")
        failed = [
            stage_id
            for stage_id, state in value["stages"].items()
            if state["status"] == "failed"
        ]
        if (
            not isinstance(quarantine, Mapping)
            or len(failed) != 1
            or quarantine.get("stage_id") != failed[0]
            or "completion_digest" in value
        ):
            raise AgenticFactoryError("quarantined factory run violates terminal invariants")
    elif (
        any(item == "failed" for item in stage_statuses)
        or "completion_digest" in value
        or "quarantine" in value
    ):
        raise AgenticFactoryError("active factory run violates state invariants")
    _validate_run_chain(path.parent, plan, value)
    return value


def _reconcile_orphan_attempts(
    root: Path,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach an attempt receipt written just before a process interruption."""

    recovered = copy.deepcopy(dict(run))
    specs = {stage["id"]: stage for stage in plan["stages"]}
    for stage_id, state in recovered["stages"].items():
        if state["status"] not in {"pending", "running"}:
            continue
        attempt_number = len(state["attempts"]) + 1
        relative = f"stages/{stage_id}/attempt-{attempt_number:03d}.json"
        path = root / relative
        if not path.exists():
            if state["status"] == "running":
                state["status"] = "pending"
            continue
        if path.is_symlink() or not path.is_file():
            raise AgenticFactoryError(f"orphan attempt receipt is unsafe for {stage_id}")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise AgenticFactoryError(f"orphan attempt receipt is malformed for {stage_id}") from None
        if not isinstance(receipt, Mapping):
            raise AgenticFactoryError(f"orphan attempt receipt root is invalid for {stage_id}")
        summary = {
            "attempt": attempt_number,
            "status": receipt.get("status"),
            "receipt": relative,
            "attempt_digest": receipt.get("attempt_digest"),
        }
        state["attempts"].append(summary)
        if receipt.get("status") == "completed":
            state["status"] = "completed"
            state["output_artifacts"] = receipt.get("output_artifacts", [])
        elif receipt.get("status") == "failed":
            if attempt_number >= specs[stage_id]["max_attempts"]:
                state["status"] = "failed"
                recovered["status"] = "quarantined"
                recovered["quarantine"] = {
                    "stage_id": stage_id,
                    "failure_class": receipt.get("failure_class"),
                    "attempt_receipt": relative,
                }
            else:
                state["status"] = "pending"
        else:
            raise AgenticFactoryError(f"orphan attempt has invalid status for {stage_id}")
    _validate_run_chain(root, plan, recovered)
    return recovered


def initialise(plan: Mapping[str, Any], out: str | Path) -> tuple[dict[str, Any], bool]:
    checked = validate_plan(plan)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "factory-plan.json"
    run_path = root / "factory-run.json"
    write_plan(checked, plan_path)
    if run_path.exists():
        run = _load_run(run_path, checked)
        run = _reconcile_orphan_attempts(root, checked, run)
        run = _write_run(run_path, run)
        return run, True
    run = _new_run(checked)
    run = _write_run(run_path, run)
    return run, False


def ready_stages(plan: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    checked = validate_plan(plan)
    states = run.get("stages")
    if not isinstance(states, Mapping):
        raise AgenticFactoryError("factory run has no stage state mapping")
    ready = []
    for stage in checked["stages"]:
        state = states.get(stage["id"])
        if not isinstance(state, Mapping) or state.get("status") != "pending":
            continue
        if all(states.get(dep, {}).get("status") == "completed" for dep in stage["depends_on"]):
            ready.append(stage["id"])
    return ready


def _directory_digest(path: Path) -> tuple[str, int]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        if candidate.is_symlink():
            raise AgenticFactoryError(f"required output directory contains a symlink: {relative}")
        if candidate.is_file():
            rows.append({"path": relative, "content_digest": _file_digest(candidate)})
    if not rows:
        raise AgenticFactoryError(f"required output directory is empty: {path.name}")
    return _value_digest(rows), len(rows)


def _artifact_path(workdir: Path, relative: str) -> Path:
    """Resolve an artifact without allowing a parent symlink to escape cwd."""

    root = workdir.resolve()
    pure = PurePosixPath(_safe_relative_path(relative))
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise AgenticFactoryError(f"required output traverses a symlink: {relative}")
    try:
        current.resolve(strict=False).relative_to(root)
    except ValueError:
        raise AgenticFactoryError(f"required output escapes factory workdir: {relative}") from None
    return current


def _validate_workspace_binding(workdir: Path, plan: Mapping[str, Any]) -> None:
    relative = plan.get("workspace_manifest")
    if relative is None:
        return
    path = _artifact_path(workdir, str(relative))
    if path.is_symlink() or not path.is_file():
        raise AgenticFactoryError("factory workspace manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AgenticFactoryError("factory workspace manifest is malformed") from None
    if not isinstance(manifest, Mapping):
        raise AgenticFactoryError("factory workspace manifest root must be an object")
    unsigned = {key: value for key, value in manifest.items() if key != "workspace_binding_digest"}
    if (
        manifest.get("workspace_binding_digest") != _value_digest(unsigned)
        or manifest.get("workspace_binding_digest") != plan["source_binding_digest"]
    ):
        raise AgenticFactoryError("factory workspace binding does not match the plan")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise AgenticFactoryError("factory workspace manifest has no input map")
    required = {"paper", "paper_text", "paper_provenance", "seed_task"}
    if set(inputs) != required:
        raise AgenticFactoryError("factory workspace manifest input map is unsupported")
    paper = _artifact_path(workdir, str(inputs["paper"]))
    paper_text = _artifact_path(workdir, str(inputs["paper_text"]))
    provenance = _artifact_path(workdir, str(inputs["paper_provenance"]))
    seed = _artifact_path(workdir, str(inputs["seed_task"]))
    if (
        not paper.is_file()
        or paper.is_symlink()
        or not paper_text.is_file()
        or paper_text.is_symlink()
        or not provenance.is_file()
        or provenance.is_symlink()
        or not seed.is_dir()
        or seed.is_symlink()
    ):
        raise AgenticFactoryError("factory workspace bound inputs are missing")
    seed_digest, _ = _directory_digest(seed)
    if (
        manifest.get("paper_content_digest") != _file_digest(paper)
        or manifest.get("paper_text_digest") != _file_digest(paper_text)
        or manifest.get("paper_provenance_digest") != _file_digest(provenance)
        or manifest.get("seed_task_tree_digest") != seed_digest
    ):
        raise AgenticFactoryError("factory workspace bound input digest mismatch")


def _validate_workspace_limits(
    workdir: Path,
    *,
    max_bytes: int,
    secret_values: Sequence[str] = (),
) -> dict[str, int]:
    total = 0
    count = 0
    secrets = [value.encode("utf-8") for value in secret_values if len(value) >= 8]
    for path in sorted(workdir.rglob("*")):
        relative = path.relative_to(workdir).as_posix()
        if ".git" in path.relative_to(workdir).parts:
            continue
        if path.is_symlink():
            raise AgenticFactoryError(f"factory workspace contains a symlink: {relative}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        count += 1
        total += size
        if size > MAX_WORKSPACE_FILE_BYTES:
            raise AgenticFactoryError(f"factory workspace file exceeds 256 MiB: {relative}")
        if count > MAX_WORKSPACE_FILES or total > max_bytes:
            raise AgenticFactoryError("factory workspace exceeded its file-count or byte budget")
        if secrets:
            data = path.read_bytes()
            if any(secret in data for secret in secrets):
                raise AgenticFactoryError(f"factory workspace output contains a provider credential: {relative}")
    return {"file_count": count, "byte_count": total}


@contextmanager
def _factory_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".factory.lock"
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _artifact(path: Path, kind: str) -> dict[str, Any] | None:
    if path.is_symlink():
        raise AgenticFactoryError(f"required output is a symlink: {path.name}")
    if kind in {"file", "json"}:
        if not path.is_file():
            return None
        if kind == "json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise AgenticFactoryError(f"required JSON output is malformed: {path.name}") from None
        return {"content_digest": _file_digest(path), "file_count": 1}
    if not path.is_dir():
        return None
    digest, count = _directory_digest(path)
    return {"content_digest": digest, "file_count": count}


def _snapshot_outputs(workdir: Path, stage: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    return {
        output["path"]: _artifact(_artifact_path(workdir, output["path"]), output["kind"])
        for output in stage["required_outputs"]
    }


def _validate_outputs(
    workdir: Path,
    stage: Mapping[str, Any],
    before: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for output in stage["required_outputs"]:
        relative = output["path"]
        after = _artifact(_artifact_path(workdir, relative), output["kind"])
        if after is None:
            raise AgenticFactoryError(f"stage {stage['id']} did not create required output {relative}")
        if before.get(relative) == after:
            raise AgenticFactoryError(f"stage {stage['id']} left required output unchanged: {relative}")
        artifacts.append({"path": relative, "kind": output["kind"], **after})
    return artifacts


def _stage_prompt(plan: Mapping[str, Any], stage: Mapping[str, Any]) -> str:
    contract = {
        "factory_id": plan["factory_id"],
        "plan_digest": plan["plan_digest"],
        "stage_id": stage["id"],
        "role": stage["role"],
        "dependencies": stage["depends_on"],
        "required_outputs": stage["required_outputs"],
        "rule": (
            "Work autonomously inside the supplied workspace. Create every required output. "
            "Do not claim that semantic output is Harbor/verifier acceptance."
        ),
    }
    return stage["prompt"] + "\n\nORBENCH_FACTORY_CONTRACT\n" + json.dumps(
        contract,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def _write_attempt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AgenticFactoryError(f"refusing to overwrite attempt receipt: {path.name}")
    signed = dict(payload)
    signed["attempt_digest"] = _value_digest(payload)
    _atomic_json(path, signed)


def _run_factory_locked(
    plan: Mapping[str, Any],
    *,
    workdir: str | Path,
    out: str | Path,
    environments: Mapping[str, Mapping[str, str]] | None = None,
    executables: Mapping[str, str | Path] | None = None,
    max_new_stages: int | None = None,
) -> dict[str, Any]:
    """Run or resume ready agent stages until completion or quarantine.

    ``max_new_stages`` is an operational checkpoint, not a semantic success
    condition.  It is useful for cron/worker leases; a later invocation resumes
    from the atomically persisted run state.
    """

    checked = validate_plan(plan)
    requested_workspace = Path(workdir)
    if requested_workspace.is_symlink() or not requested_workspace.is_dir():
        raise AgenticFactoryError("factory workdir must be a regular directory")
    workspace = requested_workspace.resolve()
    root = Path(out).resolve()
    try:
        root.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise AgenticFactoryError("factory evidence output must be outside the agent workdir")
    try:
        workspace.relative_to(root)
    except ValueError:
        pass
    else:
        raise AgenticFactoryError("agent workdir must be outside the factory evidence output")
    _validate_workspace_binding(workspace, checked)
    _validate_workspace_limits(workspace, max_bytes=checked["max_workspace_bytes"])
    run, resumed = initialise(checked, root)
    _validate_run_chain(root, checked, run, workspace=workspace)
    if run["status"] in {"semantic-complete-e1", "quarantined"}:
        return {**run, "resumed": True, "run_path": str(root / "factory-run.json")}
    stage_by_id = {stage["id"]: stage for stage in checked["stages"]}
    executed = 0
    while True:
        ready = ready_stages(checked, run)
        if not ready or (max_new_stages is not None and executed >= max_new_stages):
            break
        stage_id = ready[0]
        stage = stage_by_id[stage_id]
        _validate_workspace_binding(workspace, checked)
        _validate_run_chain(root, checked, run, workspace=workspace)
        _validate_workspace_limits(workspace, max_bytes=checked["max_workspace_bytes"])
        state = run["stages"][stage_id]
        attempt_number = len(state["attempts"]) + 1
        state["status"] = "running"
        run = _write_run(root / "factory-run.json", run)
        before = _snapshot_outputs(workspace, stage)
        attempt_path = root / "stages" / stage_id / f"attempt-{attempt_number:03d}.json"
        session: Mapping[str, Any] | None = None
        failure_class: str | None = None
        failure_detail: str | None = None
        artifacts: list[dict[str, Any]] = []
        workspace_usage: dict[str, int] | None = None
        stage_environment = (environments or {}).get(stage["profile"], {})
        try:
            session = run_session(
                profile=stage["profile"],
                stage=f"{checked['factory_id']}/{stage_id}/attempt-{attempt_number}",
                model=stage["model"],
                prompt=_stage_prompt(checked, stage),
                workdir=workspace,
                out=root / "sessions",
                timeout_sec=stage["timeout_sec"],
                max_budget_usd=stage["max_budget_usd"],
                max_output_bytes=stage["max_output_bytes"],
                environ=stage_environment,
                executable=(executables or {}).get(stage["profile"]),
            )
            workspace_usage = _validate_workspace_limits(
                workspace,
                max_bytes=checked["max_workspace_bytes"],
                secret_values=[
                    value
                    for name, value in stage_environment.items()
                    if "KEY" in name.upper() or "TOKEN" in name.upper()
                ],
            )
            if session.get("status") != "completed":
                failure_class = str(session.get("failure_class") or "agent_process_failure")
            else:
                artifacts = _validate_outputs(workspace, stage, before)
        except (AgentSessionError, AgenticFactoryError) as exc:
            failure_class = "session_contract_failure"
            failure_detail = str(exc)
        receipt_path = Path(str(session.get("receipt_path"))) if session and session.get("receipt_path") else None
        attempt = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "factory_id": checked["factory_id"],
            "plan_digest": checked["plan_digest"],
            "stage_id": stage_id,
            "attempt": attempt_number,
            "status": "completed" if failure_class is None else "failed",
            "evidence_level": "E1-agent-session-process",
            "max_budget_usd": stage["max_budget_usd"],
            "failure_class": failure_class,
            "failure_detail": failure_detail,
            "session_id": session.get("session_id") if session else None,
            "session_receipt_digest": (
                _file_digest(receipt_path) if receipt_path and receipt_path.is_file() else None
            ),
            "output_artifacts": artifacts,
            "workspace_usage": workspace_usage,
        }
        # Trusted inputs and already-completed outputs are rechecked before the
        # attempt receipt is made immutable. If this check fails, there is no
        # orphan receipt to collide with the same attempt number on recovery.
        _validate_workspace_binding(workspace, checked)
        _validate_run_chain(root, checked, run, workspace=workspace)
        _write_attempt(attempt_path, attempt)
        state["attempts"].append(
            {
                "attempt": attempt_number,
                "status": attempt["status"],
                "receipt": attempt_path.relative_to(root).as_posix(),
                "attempt_digest": _value_digest(attempt),
            }
        )
        if failure_class is None:
            state["status"] = "completed"
            state["output_artifacts"] = artifacts
        elif attempt_number >= stage["max_attempts"]:
            state["status"] = "failed"
            run["status"] = "quarantined"
            run["quarantine"] = {
                "stage_id": stage_id,
                "failure_class": failure_class,
                "attempt_receipt": attempt_path.relative_to(root).as_posix(),
            }
        else:
            state["status"] = "pending"
        run = _write_run(root / "factory-run.json", run)
        executed += 1
        if run["status"] == "quarantined":
            break
    if run["status"] == "active" and all(
        state["status"] == "completed" for state in run["stages"].values()
    ):
        _validate_run_chain(root, checked, run, workspace=workspace)
        run["status"] = "semantic-complete-e1"
        run["completion_digest"] = _value_digest(
            {
                "factory_id": checked["factory_id"],
                "plan_digest": checked["plan_digest"],
                "stages": run["stages"],
                "evidence_level": "E1-agent-session-process",
            }
        )
        run = _write_run(root / "factory-run.json", run)
    return {
        **run,
        "resumed": resumed,
        "new_stage_attempts": executed,
        "run_path": str(root / "factory-run.json"),
    }


def run_factory(
    plan: Mapping[str, Any],
    *,
    workdir: str | Path,
    out: str | Path,
    environments: Mapping[str, Mapping[str, str]] | None = None,
    executables: Mapping[str, str | Path] | None = None,
    max_new_stages: int | None = None,
) -> dict[str, Any]:
    """Serialize one factory state chain and run or resume its ready stages."""

    root = Path(out).resolve()
    with _factory_lock(root):
        return _run_factory_locked(
            plan,
            workdir=workdir,
            out=root,
            environments=environments,
            executables=executables,
            max_new_stages=max_new_stages,
        )


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "AgenticFactoryError",
    "PLAN_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "compile_plan",
    "initialise",
    "load_plan",
    "ready_stages",
    "run_factory",
    "validate_plan",
    "write_plan",
]
