"""Deterministic unattended control-plane for a completed agent factory.

Only the Harbor and calibration adapters execute external commands.  Static
authoring, task cards, and finalization call trusted package code directly.
Every stage is locked, content-bound and resumable; agent prose is never a
substitute for an expected receipt.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import agent_sessions, factory_finalize, pipeline, task_authoring, volc_rollout
from .core.errors import ORBenchError


class FactorySupervisorError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.factory-supervisor.v1"


def _digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{value}"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    os.replace(temporary, path)


def _command(
    executable: str | None,
    arguments: Sequence[str],
    *,
    timeout_sec: float,
    cwd: Path,
    child_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not executable:
        return {"status": "blocked", "failure_class": "external_dependency_missing", "argv": None}
    argv = [str(executable), *arguments]
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, text=False,
        env=dict(child_env) if child_env is not None else {"PATH": os.environ.get("PATH", "")},
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
        failure = None if process.returncode == 0 else "nonzero_exit"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        failure = "timeout"
    return {
        "status": "passed" if failure is None else "blocked",
        "failure_class": failure,
        "argv": argv,
        "exit_code": process.returncode,
        "elapsed_sec": round(time.monotonic() - started, 6),
        "stdout_digest": "sha256:" + hashlib.sha256(stdout).hexdigest(),
        "stderr_digest": "sha256:" + hashlib.sha256(stderr).hexdigest(),
    }


def _executable_binding(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).resolve()
    return {"path": str(path), "digest": _digest(path) if path.is_file() else None}


def run(
    *,
    plan_path: str | Path,
    factory_run_path: str | Path,
    workdir: str | Path,
    task_dir: str | Path,
    task_genome: str | Path,
    paper_provenance: str | Path,
    out: str | Path,
    harbor_executable: str | None,
    harbor_inputs: Mapping[str, str],
    semantic_review_executable: str | None,
    semantic_review_models: Sequence[str],
    calibration_executable: str | None,
    calibration_models: Sequence[str],
    test_image: str,
    provider_env: Mapping[str, str] | None = None,
    repetitions: int = 5,
    timeout_sec: float = 600,
) -> dict[str, Any]:
    """Execute or resume the fixed five-stage control-plane."""

    if repetitions < 5 or len(set(calibration_models)) < 2 or len(set(semantic_review_models)) < 2 or timeout_sec <= 0:
        raise FactorySupervisorError("calibration requires two distinct models, >=5 repetitions and positive timeout")
    root, task, workspace, genome_path = Path(out), Path(task_dir), Path(workdir), Path(task_genome)
    provider_child, route_digest = agent_sessions._session_env("claude-code", provider_env or {})
    genome = task_authoring._load_document(genome_path)
    if str(genome.get("family") or "") != volc_rollout._task_id(task) or not genome.get("difficulty_axes"):
        raise FactorySupervisorError("task genome must bind task identity and declare difficulty axes")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".supervisor.lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        identity = {
            "plan_digest": _digest(Path(plan_path)), "factory_run_digest": _digest(Path(factory_run_path)),
            "task_tree_digest": task_authoring._task_tree_digest(task),
            "paper_provenance_digest": _digest(Path(paper_provenance)),
            "task_genome_digest": _digest(genome_path),
            "models": list(calibration_models), "repetitions": repetitions,
            "test_image": test_image,
            "harbor_adapter": _executable_binding(harbor_executable),
            "harbor_inputs": dict(sorted(harbor_inputs.items())),
            "calibration_adapter": _executable_binding(calibration_executable),
            "semantic_review_adapter": _executable_binding(semantic_review_executable),
            "semantic_review_models": list(semantic_review_models),
            "timeout_sec": timeout_sec,
            "provider_route_digest": route_digest,
        }
        identity_digest = "sha256:" + hashlib.sha256(_canonical(identity)).hexdigest()
        state_path = root / "supervisor-state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("identity_digest") != identity_digest:
                raise FactorySupervisorError("supervisor output belongs to another immutable input set")
        else:
            state = {"schema_version": SCHEMA_VERSION, "identity": identity, "identity_digest": identity_digest, "stages": {}}

        def record(name: str, result: Mapping[str, Any]) -> None:
            state["stages"][name] = dict(result)
            state["status"] = "active"
            _atomic(state_path, state)

        def reuse(name: str, path: Path) -> None:
            actual = _digest(path)
            previous = state["stages"].get(name)
            if previous and previous.get("output_digest") not in (None, actual):
                raise FactorySupervisorError(f"completed {name} output changed after receipt")
            record(name, {"status": "reused", "output": str(path), "output_digest": actual})

        static_json = root / "static" / "authoring-receipt.json"
        if not static_json.exists():
            receipt = task_authoring.validate_task(task, paper_provenance=paper_provenance)
            task_authoring.write_receipt(receipt, root / "static")
            record("static", {"status": "passed" if receipt["decision"] != "blocked" else "blocked", "output": str(static_json), "output_digest": _digest(static_json)})
        else:
            reuse("static", static_json)

        semantic_json = root / "semantic" / "volc-authoring-review.json"
        if semantic_json.exists():
            reuse("semantic_review", semantic_json)
        else:
            args = ["--task-dir", str(task), "--paper-provenance", str(paper_provenance), "--receipt", str(static_json), "--round", "1", "--models", ",".join(semantic_review_models), "--out", str(root / "semantic")]
            semantic = _command(semantic_review_executable, args, timeout_sec=timeout_sec, cwd=workspace, child_env=provider_child)
            if semantic.get("status") == "passed" and not semantic_json.is_file():
                semantic = {**semantic, "status": "blocked", "failure_class": "expected_receipt_missing"}
            if semantic_json.is_file(): semantic = {**semantic, "output": str(semantic_json), "output_digest": _digest(semantic_json)}
            record("semantic_review", semantic)

        harbor_json = root / "harbor" / "harbor-control-screening.json"
        if harbor_json.exists():
            reuse("harbor", harbor_json)
        else:
            required = ("executed_task_dir", "oracle_job", "nop_job")
            if any(not harbor_inputs.get(key) for key in required):
                harbor = {"status": "blocked", "failure_class": "harbor_inputs_missing"}
            else:
                args = ["--task-dir", str(task), "--executed-task-dir", harbor_inputs["executed_task_dir"], "--oracle-job", harbor_inputs["oracle_job"], "--nop-job", harbor_inputs["nop_job"], "--out", str(root / "harbor")]
                harbor = _command(harbor_executable, args, timeout_sec=timeout_sec, cwd=workspace)
            if harbor.get("status") == "passed" and not harbor_json.is_file():
                harbor = {**harbor, "status": "blocked", "failure_class": "expected_receipt_missing"}
            if harbor_json.is_file(): harbor = {**harbor, "output": str(harbor_json), "output_digest": _digest(harbor_json)}
            record("harbor", harbor)

        calibration_json = root / "calibration" / "screening-report.json"
        if calibration_json.exists():
            reuse("calibration", calibration_json)
        else:
            args = ["--task-dir", str(task), "--test-image", test_image, "--out", str(root / "calibration"), "--models", ",".join(calibration_models), "--repetitions", str(repetitions), "--hint-level", "0", "--controls", "oracle,nop"]
            calibration = _command(calibration_executable, args, timeout_sec=timeout_sec, cwd=workspace, child_env=provider_child)
            if calibration.get("status") == "passed" and not calibration_json.is_file():
                calibration = {**calibration, "status": "blocked", "failure_class": "expected_receipt_missing"}
            if calibration_json.is_file(): calibration = {**calibration, "output": str(calibration_json), "output_digest": _digest(calibration_json)}
            record("calibration", calibration)

        cards_json = root / "cards" / "task-cards.json"
        evidence = [path for path in (harbor_json, calibration_json) if path.is_file()]
        if not cards_json.exists():
            pipeline.run(out=root / "cards", task_inputs=[genome_path], screening_inputs=evidence)
            record("cards", {"status": "passed", "output": str(cards_json), "output_digest": _digest(cards_json)})
        else:
            reuse("cards", cards_json)

        prerequisites = all(path.is_file() for path in (static_json, semantic_json, harbor_json, calibration_json, cards_json))
        final_path = root / "final" / "factory-finalization.json"
        if prerequisites:
            final = factory_finalize.build_receipt(plan_path=plan_path, factory_run_path=factory_run_path, workdir=workspace, task_dir=task, static_receipt_path=static_json, semantic_review_path=semantic_json, harbor_receipt_path=harbor_json, calibration_receipt_path=calibration_json, final_summary_path=cards_json)
            factory_finalize.write_receipt(final, root / "final")
        else:
            final = {"decision": "not-promoted", "promoted": False, "evidence_level": "E1", "failure_class": "prerequisite_blocked"}
            _atomic(final_path, final)
        record("finalize", {"status": "passed" if final["promoted"] else "blocked", "output": str(final_path), "output_digest": _digest(final_path)})
        state["status"] = "eligible-for-human-release-review" if final["promoted"] else "quarantined"
        state["promoted"] = bool(final["promoted"])
        _atomic(state_path, state)
        return dict(state)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


__all__ = ["FactorySupervisorError", "SCHEMA_VERSION", "run"]
