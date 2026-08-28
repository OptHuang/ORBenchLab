"""Bounded, receipt-bound runtime repair for Harbor control failures.

When an Oracle/NOP control or the model matrix fails at a runtime barrier, the
autopilot must not crash or silently quarantine.  This module classifies the
failure (infrastructure vs. task defect), writes a sanitized machine-readable
failure bundle, and — for a repairable task defect — drives a bounded repair
agent session that produces a new task version, re-runs the deterministic
static gate and re-runs the controls.  Every artifact is digest-bound into a
resumable receipt chain; an infrastructure failure never mutates the task.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    agent_sessions,
    factory_gates,
    harbor_launcher,
    task_authoring,
    volc_rollout,
)
from .core.errors import ORBenchError


class RuntimeRepairError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.runtime-repair.v1"
FAILURE_BUNDLE_SCHEMA = "orbenchlab.runtime-failure-bundle.v1"

# Launcher-level failure classes that are transient by construction: the
# command was killed by a resource bound or never launched. These are infra
# regardless of stderr (the autopilot resumes, never mutating the task).
_INFRA_LAUNCHER_MARKERS = ("timeout", "output_limit_exceeded", "launch")
# Descriptive control-validation messages the launcher raises for a task defect.
_TASK_MARKERS = (
    "did not produce a valid control job",
    "reward and CTRF",
    "did not pass",
    "control job task identity",
    "artifact manifest",
)
# Genuine infrastructure signatures in a command's real stderr: a nonzero exit
# carrying one of these is a transient host problem, not a task defect.
_INFRA_STDERR_MARKERS = (
    "cannot connect to the docker daemon",
    "docker: error during connect",
    "is the docker daemon running",
    "connection refused",
    "network is unreachable",
    "temporary failure in name resolution",
    "no space left on device",
    "i/o timeout",
    "context deadline exceeded",
    "permission denied while trying to connect",
)
# Task-build / verifier / contract signatures in a command's real stderr: a
# nonzero exit carrying one of these is a repairable task defect.
_TASK_STDERR_MARKERS = (
    "task_dockerfile_build_failed",
    "dockerfile",
    "failed to solve",
    "build failed",
    "verifier",
    "ctrf",
    "reward",
    "test.sh",
    "solution",
    "traceback (most recent call last)",
)


def _value_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _has_completed_session(sessions_root: Path) -> bool:
    """True if a prior agent session under ``sessions_root`` completed."""

    if not sessions_root.is_dir():
        return False
    for receipt in sessions_root.glob("*/receipt.json"):
        try:
            doc = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if doc.get("status") == "completed":
            return True
    return False


def classify_failure(message: str, stderr: str = "") -> str:
    """Classify a Harbor failure as 'infra', 'task', or 'unknown'.

    Classification is grounded in the command's real stderr, not only the
    launcher's coarse failure class: a ``nonzero_exit`` carrying a Docker-build,
    verifier or contract signature is a repairable task defect, while one
    carrying a Docker-daemon/network signature is a transient the autopilot
    resumes without mutating the task. Only genuine launcher transients
    (timeout, output-limit, launch failure) are infra irrespective of stderr.
    """

    lowered_msg = message.lower()
    blob = (message + "\n" + stderr).lower()
    # Descriptive control-validation messages are always task defects.
    if any(marker in message for marker in _TASK_MARKERS):
        return "task"
    # Genuine infrastructure signatures in the real stderr win first.
    if any(marker in blob for marker in _INFRA_STDERR_MARKERS):
        return "infra"
    # Launcher-level transients (killed by a bound or never launched).
    if any(marker in lowered_msg for marker in _INFRA_LAUNCHER_MARKERS):
        return "infra"
    # Task-build / verifier / contract signatures in the real stderr.
    if any(marker in blob for marker in _TASK_STDERR_MARKERS):
        return "task"
    # A control command that ran and exited nonzero with no infra signature is a
    # task/verifier defect, not a transient: repair rather than resume forever.
    if "nonzero_exit" in lowered_msg:
        return "task"
    return "unknown"


def _control_job_evidence(jobs_dir: Path, *, task_id: str) -> list[dict[str, Any]]:
    """Collect sanitized digests of any control jobs that were produced."""

    rows: list[dict[str, Any]] = []
    if not jobs_dir.is_dir():
        return rows
    for control in ("oracle", "nop"):
        for job in sorted(jobs_dir.glob(f"{task_id}-{control}-attempt-*")):
            result = job / "result.json"
            if not result.is_file():
                continue
            row: dict[str, Any] = {
                "control": control,
                "job_name": job.name,
                "job_result_digest": _file_digest(result),
            }
            trials = sorted(job.glob("*/verifier/reward.txt"))
            if trials:
                reward_path = trials[0]
                try:
                    row["reward"] = float(reward_path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    row["reward"] = None
                ctrf = reward_path.parent / "ctrf.json"
                if ctrf.is_file():
                    row["ctrf_digest"] = _file_digest(ctrf)
                trial_result = reward_path.parent.parent / "result.json"
                if trial_result.is_file():
                    try:
                        doc = json.loads(trial_result.read_text(encoding="utf-8"))
                        exc = doc.get("exception_info")
                        if isinstance(exc, Mapping):
                            row["exception_type"] = str(exc.get("exception_type"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        pass
            rows.append(row)
    return rows


def save_failure_bundle(
    *,
    out: Path,
    scope: str,
    task: Path,
    controls_root: Path,
    failure_class: str,
    message: str,
    attempt: int,
    reserved_liability_usd: float,
) -> dict[str, Any]:
    """Atomically persist a sanitized machine-readable failure bundle."""

    task_id = volc_rollout._task_id(task)
    bundle = {
        "schema_version": FAILURE_BUNDLE_SCHEMA,
        "scope": scope,
        "task_id": task_id,
        "task_tree_digest": volc_rollout._task_tree_digest(task),
        "authoring_task_tree_digest": task_authoring._task_tree_digest(task),
        "failure_class": failure_class,
        "failure_message": message[:2000],
        "attempt": int(attempt),
        "reserved_liability_usd": round(float(reserved_liability_usd), 6),
        "control_jobs": _control_job_evidence(controls_root / "jobs", task_id=task_id),
    }
    bundle["bundle_digest"] = _value_digest(bundle)
    _atomic_json(out / "failure-bundle.json", bundle)
    return bundle


def _repair_prompt(scope: str, failure_bundle: Mapping[str, Any]) -> str:
    return (
        f"A Harbor {scope} control run failed on this task. The trusted harness has "
        "placed the machine-readable failure evidence at repair-input/failure-bundle.json "
        "and the current task tree at repair-input/task/<slug>/. Diagnose the failure and "
        "repair the task by copying repair-input/task/<slug> to ./task-vnext/<slug> (keeping "
        "the same slug directory name) and editing it there. Typical causes: the reference "
        "solution does not pass the verifier "
        "(Oracle reward 0), the NOP/empty submission unexpectedly passes (weak verifier), or "
        "a nondeterministic or resource-bound verifier. Keep the task paper-faithful and "
        "strict: preserve the slug-named layout, the separate no-network verifier, CTRF "
        "reporting and resource bounds. Do not weaken the verifier just to make Oracle pass. "
        "Read the paper evidence in repair-input/ if present. Write only under ./task-vnext/."
    )


def _stage_repair_inputs(
    *,
    task: Path,
    failure_bundle_path: Path,
    paper_ancestors: Sequence[Path],
    repair_root: Path,
) -> list[Path]:
    input_root = repair_root / "repair-input"
    if input_root.exists():
        for path in sorted(input_root.rglob("*"), reverse=True):
            try:
                path.chmod(0o755)
            except OSError:
                pass
        shutil.rmtree(input_root, ignore_errors=True)
    input_root.mkdir(parents=True)
    # Preserve the slug-named task directory so the repaired copy keeps the
    # TB-Science layout the static gate requires.
    shutil.copytree(task, input_root / "task" / task.name, symlinks=False)
    shutil.copy2(failure_bundle_path, input_root / "failure-bundle.json")
    for ancestor in paper_ancestors:
        if ancestor.is_file() and not ancestor.is_symlink():
            shutil.copy2(ancestor, input_root / ancestor.name)
    for path in sorted(input_root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    input_root.chmod(0o555)
    return [input_root]


def repair_task_once(
    *,
    task: Path,
    failure_bundle_path: Path,
    paper_ancestors: Sequence[Path],
    out: Path,
    claude_executable: str | Path,
    model: str,
    provider_env: Mapping[str, str],
    max_budget_usd: float,
    timeout_sec: float,
    round_number: int,
    parent_task_digest: str,
    failure_bundle_digest: str,
    credential_relay: bool = False,
) -> dict[str, Any]:
    """Run one bounded repair agent session and static-gate its output."""

    out.mkdir(parents=True, exist_ok=True)
    # A repair round adopts only output this session actually writes. Any
    # pre-existing task-vnext with no completed session behind it is stale
    # (a crashed prior attempt or a planted tree) and must be cleared before
    # the session runs; a genuine resume keeps its completed session's output.
    vnext = out / "task-vnext"
    if vnext.exists() and not _has_completed_session(out / "sessions"):
        if vnext.is_dir() and not vnext.is_symlink():
            shutil.rmtree(vnext)
        else:
            vnext.unlink()
    read_only = _stage_repair_inputs(
        task=task,
        failure_bundle_path=failure_bundle_path,
        paper_ancestors=paper_ancestors,
        repair_root=out,
    )
    slug = task.name
    session = agent_sessions.run_session(
        profile="claude-code",
        stage=f"runtime-repair/round-{round_number}",
        model=model,
        prompt=_repair_prompt("baseline", {}),
        workdir=out,
        out=out / "sessions",
        timeout_sec=timeout_sec,
        max_budget_usd=max_budget_usd,
        environ=provider_env,
        executable=claude_executable,
        read_only_paths=read_only,
        allow_bash=False,
        credential_relay=credential_relay,
    )
    session_ok = session.get("status") == "completed"
    produced = out / "task-vnext" / slug
    if produced.is_dir() and not produced.is_symlink():
        repaired = produced
    else:
        # Accept either task-vnext/<slug> or task-vnext being the task root.
        alt = out / "task-vnext"
        repaired = alt if (alt / "task.toml").is_file() else produced
    if not session_ok:
        # A failed or crashed repair session produces no adoptable task,
        # regardless of any files left in the workspace.
        status = "session-failed"
    elif repaired.is_dir() and (repaired / "task.toml").is_file():
        status = "produced"
    else:
        status = "no-task"
    static: dict[str, Any] | None = None
    static_decision = None
    if status == "produced":
        provenance = None
        for ancestor in paper_ancestors:
            if ancestor.name == "paper-provenance.json":
                provenance = repaired / "paper-provenance.json"
        static = task_authoring.validate_task(
            repaired,
            paper_provenance=provenance if provenance and provenance.is_file() else None,
        )
        task_authoring.write_receipt(static, out / "static")
        static_decision = static["decision"]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "round": round_number,
        "session_status": session.get("status"),
        "session_id": session.get("session_id"),
        "session_receipt_digest": (
            _file_digest(Path(str(session["receipt_path"])))
            if session.get("receipt_path")
            and Path(str(session["receipt_path"])).is_file()
            else None
        ),
        "parent_task_tree_digest": parent_task_digest,
        "failure_bundle_digest": failure_bundle_digest,
        "status": status,
        "repaired_task_path": str(repaired) if status == "produced" else None,
        "repaired_task_tree_digest": (
            volc_rollout._task_tree_digest(repaired) if status == "produced" else None
        ),
        "static_decision": static_decision,
        "static_receipt_digest": static.get("receipt_digest") if static else None,
    }
    receipt["receipt_digest"] = _value_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _atomic_json(out / "repair-receipt.json", receipt)
    return receipt


__all__ = [
    "RuntimeRepairError",
    "SCHEMA_VERSION",
    "classify_failure",
    "repair_task_once",
    "save_failure_bundle",
]
