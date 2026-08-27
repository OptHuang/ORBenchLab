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
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    agent_sessions,
    agentic_factory,
    factory_finalize,
    harbor_launcher,
    pipeline,
    task_authoring,
    volc_rollout,
    volc_review,
)
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
    max_output_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    if not executable:
        return {"status": "blocked", "failure_class": "external_dependency_missing", "argv": None}
    argv = [str(executable), *arguments]
    started = time.monotonic()
    stdout, stderr, return_code, failure = agent_sessions._bounded_process(
        argv,
        cwd=cwd,
        env=(
            dict(child_env)
            if child_env is not None
            else {"PATH": os.environ.get("PATH", "")}
        ),
        stdin=b"",
        timeout_sec=timeout_sec,
        max_output_bytes=max_output_bytes,
    )
    if failure is None and return_code != 0:
        failure = "nonzero_exit"
    return {
        "status": "passed" if failure is None else "blocked",
        "failure_class": failure,
        "argv": argv,
        "exit_code": return_code,
        "elapsed_sec": round(time.monotonic() - started, 6),
        "max_output_bytes": max_output_bytes,
        "stdout_digest": "sha256:" + hashlib.sha256(stdout).hexdigest(),
        "stderr_digest": "sha256:" + hashlib.sha256(stderr).hexdigest(),
    }


def _executable_binding(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).resolve()
    return {"path": str(path), "digest": _digest(path) if path.is_file() else None}


def _bind_factory_outputs(
    *,
    plan_path: Path,
    factory_run_path: Path,
    workspace: Path,
    task: Path,
    genome_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Bind supervisor inputs to immutable completed agent-stage artifacts."""

    plan = agentic_factory.load_plan(plan_path)
    run = agentic_factory._load_run(factory_run_path, plan)
    if run.get("status") != "semantic-complete-e1":
        raise FactorySupervisorError("factory run is not semantic-complete-e1")
    agentic_factory._validate_run_chain(
        factory_run_path.parent,
        plan,
        run,
        workspace=workspace,
    )
    stage_specs = {stage["id"]: stage for stage in plan["stages"]}
    task_roots: list[Path] = []
    for stage_id in ("task-repair-v2", "variant-author"):
        state = run["stages"].get(stage_id)
        spec = stage_specs.get(stage_id)
        if not isinstance(state, Mapping) or state.get("status") != "completed" or spec is None:
            continue
        for output in spec["required_outputs"]:
            if output["kind"] == "directory":
                task_roots.append(agentic_factory._artifact_path(workspace, output["path"]))
    if not task_roots:
        raise FactorySupervisorError("factory run has no completed task artifact")
    requested_task = task.resolve()
    if task.is_symlink() or not task.is_dir():
        raise FactorySupervisorError("selected task must be a real factory artifact directory")
    if not any(
        requested_task == root.resolve() or requested_task.is_relative_to(root.resolve())
        for root in task_roots
    ):
        raise FactorySupervisorError("selected task is not owned by a completed factory stage")

    final_spec = stage_specs.get("final-synthesis")
    final_state = run["stages"].get("final-synthesis")
    if (
        final_spec is None
        or not isinstance(final_state, Mapping)
        or final_state.get("status") != "completed"
    ):
        raise FactorySupervisorError("factory run has no completed final-synthesis stage")
    outputs = {output["path"]: output for output in final_spec["required_outputs"]}
    genome_relative = "factory/final/task-genome.json"
    summary_relative = "factory/final/task-review-summary.json"
    if genome_relative not in outputs or summary_relative not in outputs:
        raise FactorySupervisorError("final-synthesis contract lacks genome or review summary")
    expected_genome = agentic_factory._artifact_path(workspace, genome_relative)
    summary_path = agentic_factory._artifact_path(workspace, summary_relative)
    if genome_path.resolve() != expected_genome.resolve():
        raise FactorySupervisorError("task genome is not the completed final-synthesis artifact")
    if not expected_genome.is_file() or not summary_path.is_file():
        raise FactorySupervisorError("completed final-synthesis artifacts are missing")
    return run, summary_path


def _validate_selected_genome(
    genome: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    workspace: Path,
    task: Path,
    paper_provenance: Path,
) -> None:
    selected = task.relative_to(workspace).as_posix()
    if genome.get("selected_task") != selected or summary.get("selected_task") != selected:
        raise FactorySupervisorError("genome and summary do not bind the selected task path")
    if str(genome.get("family") or "") != volc_rollout._task_id(task):
        raise FactorySupervisorError("task genome family does not bind selected task identity")
    if any(not isinstance(genome.get(key), str) or not str(genome[key]).strip() for key in ("title", "design_goal")):
        raise FactorySupervisorError("task genome requires non-empty title and design_goal")
    source = genome.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("paper_provenance_digest") != _digest(paper_provenance)
    ):
        raise FactorySupervisorError("task genome source does not bind paper provenance")
    axes = genome.get("difficulty_axes")
    if not isinstance(axes, Mapping) or not axes:
        raise FactorySupervisorError("task genome must declare difficulty axes")
    for name, axis in axes.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(axis, Mapping)
            or not isinstance(axis.get("levels"), list)
            or len(axis["levels"]) < 2
            or not isinstance(axis.get("meaning"), str)
            or not axis["meaning"].strip()
            or not isinstance(axis.get("expected_direction"), str)
            or not axis["expected_direction"].strip()
        ):
            raise FactorySupervisorError(f"task genome difficulty axis is incomplete: {name!r}")


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
    harbor_cli_executable: str | None = None,
    builtin_volc: bool = False,
) -> dict[str, Any]:
    """Execute or resume the fixed five-stage control-plane."""

    if repetitions < 5 or len(set(calibration_models)) < 2 or len(set(semantic_review_models)) < 2 or timeout_sec <= 0:
        raise FactorySupervisorError("calibration requires two distinct models, >=5 repetitions and positive timeout")
    root = Path(out)
    workspace = Path(workdir).resolve()
    task = Path(task_dir).resolve()
    genome_path = Path(task_genome).resolve()
    paper_path = Path(paper_provenance).resolve()
    _, summary_path = _bind_factory_outputs(
        plan_path=Path(plan_path),
        factory_run_path=Path(factory_run_path),
        workspace=workspace,
        task=task,
        genome_path=genome_path,
    )
    genome = task_authoring._load_document(genome_path)
    summary = task_authoring._load_document(summary_path)
    _validate_selected_genome(
        genome,
        summary=summary,
        workspace=workspace,
        task=task,
        paper_provenance=paper_path,
    )
    provider_child, route_digest = agent_sessions._session_env("claude-code", provider_env or {})
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".supervisor.lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        identity = {
            "plan_digest": _digest(Path(plan_path)), "factory_run_digest": _digest(Path(factory_run_path)),
            "task_tree_digest": task_authoring._task_tree_digest(task),
            "paper_provenance_digest": _digest(paper_path),
            "factory_summary_digest": _digest(summary_path),
            "task_genome_digest": _digest(genome_path),
            "models": list(calibration_models), "repetitions": repetitions,
            "test_image": test_image,
            "harbor_adapter": _executable_binding(harbor_executable),
            "harbor_cli": _executable_binding(harbor_cli_executable),
            "harbor_inputs": dict(sorted(harbor_inputs.items())),
            "calibration_adapter": _executable_binding(calibration_executable),
            "semantic_review_adapter": _executable_binding(semantic_review_executable),
            "semantic_review_models": list(semantic_review_models),
            "builtin_volc": bool(builtin_volc),
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
            receipt = task_authoring.validate_task(task, paper_provenance=paper_path)
            task_authoring.write_receipt(receipt, root / "static")
            record("static", {"status": "passed" if receipt["decision"] != "blocked" else "blocked", "output": str(static_json), "output_digest": _digest(static_json)})
        else:
            reuse("static", static_json)
        static_receipt = json.loads(static_json.read_text(encoding="utf-8"))
        static_passed = static_receipt.get("decision") != "blocked"

        semantic_json = root / "semantic" / "volc-authoring-review.json"
        if semantic_json.exists():
            reuse("semantic_review", semantic_json)
        elif not static_passed:
            semantic = {"status": "blocked", "failure_class": "upstream_static_blocked"}
            record("semantic_review", semantic)
        elif builtin_volc:
            try:
                config = volc_review.VolcConfig(
                    base_url=str((provider_env or {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/"),
                    token=str((provider_env or {}).get("ANTHROPIC_AUTH_TOKEN", "")),
                    default_model="ark-code-latest",
                    timeout_sec=max(1, int(timeout_sec)),
                )
                review = volc_review.review_task(
                    task,
                    paper_provenance=task_authoring._load_document(paper_path),
                    receipt=static_receipt,
                    config=config,
                    models=semantic_review_models,
                    round_number=1,
                )
                volc_review.write_review(review, root / "semantic")
                semantic = {
                    "status": "passed",
                    "mode": "builtin-volc",
                    "output": str(semantic_json),
                    "output_digest": _digest(semantic_json),
                }
            except volc_review.VolcReviewError as exc:
                semantic = {
                    "status": "blocked",
                    "failure_class": "volc_review_failed",
                    "failure_detail": str(exc),
                }
            record("semantic_review", semantic)
        else:
            args = ["--task-dir", str(task), "--paper-provenance", str(paper_path), "--receipt", str(static_json), "--round", "1", "--models", ",".join(semantic_review_models), "--out", str(root / "semantic")]
            semantic = _command(semantic_review_executable, args, timeout_sec=timeout_sec, cwd=workspace, child_env=provider_child)
            if semantic.get("status") == "passed" and not semantic_json.is_file():
                semantic = {**semantic, "status": "blocked", "failure_class": "expected_receipt_missing"}
            if semantic_json.is_file(): semantic = {**semantic, "output": str(semantic_json), "output_digest": _digest(semantic_json)}
            record("semantic_review", semantic)
        semantic_passed = False
        if semantic_json.is_file():
            semantic_receipt = json.loads(semantic_json.read_text(encoding="utf-8"))
            semantic_passed = semantic_receipt.get("aggregate_decision") == "promising-needs-harbor"

        harbor_json = root / "harbor" / "harbor-control-screening.json"
        if harbor_json.exists():
            reuse("harbor", harbor_json)
        elif not semantic_passed:
            harbor = {"status": "blocked", "failure_class": "upstream_semantic_blocked"}
            record("harbor", harbor)
        else:
            required = ("executed_task_dir", "oracle_job", "nop_job")
            if harbor_cli_executable:
                try:
                    harbor_launcher.launch_controls(
                        task,
                        harbor_executable=harbor_cli_executable,
                        out=root / "harbor",
                        timeout_sec=timeout_sec,
                    )
                    harbor = {
                        "status": "passed",
                        "mode": "launched",
                        "output": str(harbor_json),
                        "output_digest": _digest(harbor_json),
                    }
                except harbor_launcher.HarborLauncherError as exc:
                    harbor = {
                        "status": "blocked",
                        "failure_class": "harbor_launch_failed",
                        "failure_detail": str(exc),
                    }
            elif any(not harbor_inputs.get(key) for key in required):
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
        elif not semantic_passed:
            calibration = {"status": "blocked", "failure_class": "upstream_semantic_blocked"}
            record("calibration", calibration)
        elif builtin_volc:
            try:
                config = volc_review.VolcConfig(
                    base_url=str((provider_env or {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/"),
                    token=str((provider_env or {}).get("ANTHROPIC_AUTH_TOKEN", "")),
                    default_model="ark-code-latest",
                    timeout_sec=max(1, int(timeout_sec)),
                )
                volc_rollout.run_rollout(
                    task,
                    config=config,
                    models=calibration_models,
                    test_image=test_image,
                    out=root / "calibration",
                    repetitions=repetitions,
                    hint_levels=[0],
                    controls=["oracle", "nop"],
                    timeout_sec=max(1, int(timeout_sec)),
                )
                calibration = {
                    "status": "passed",
                    "mode": "builtin-volc",
                    "output": str(calibration_json),
                    "output_digest": _digest(calibration_json),
                }
            except (volc_review.VolcReviewError, volc_rollout.VolcRolloutError) as exc:
                calibration = {
                    "status": "blocked",
                    "failure_class": "volc_calibration_failed",
                    "failure_detail": str(exc),
                }
            record("calibration", calibration)
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
