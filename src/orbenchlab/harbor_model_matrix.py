"""Launch and validate repeated Harbor coding-agent trials with ATIF traces.

Unlike :mod:`volc_rollout`, this path runs the real Harbor agent lifecycle and
requires a verifier-grounded ATIF trajectory for every completed trial.  It is
descriptive E3 evidence; it does not claim same-checkpoint causal intervention.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from . import agent_sessions, harbor_launcher
from .core.errors import ORBenchError
from . import volc_rollout
from .volc_rollout import _digest, _task_id, _task_tree_digest


class HarborModelMatrixError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.harbor-model-matrix.v1"
_MODEL_SLUG = re.compile(r"[^a-z0-9]+")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise HarborModelMatrixError(f"invalid Harbor JSON evidence: {path}") from None
    if not isinstance(value, Mapping):
        raise HarborModelMatrixError(f"Harbor JSON evidence must be an object: {path}")
    return value


def _model_key(model: str) -> str:
    slug = _MODEL_SLUG.sub("-", model.lower()).strip("-")[:36] or "model"
    suffix = hashlib.sha256(model.encode()).hexdigest()[:10]
    return f"{slug}-{suffix}"


def _trajectory_paths(trial: Path, result: Mapping[str, Any]) -> list[Path]:
    step_results = result.get("step_results")
    if isinstance(step_results, list) and step_results:
        paths = []
        for index, step in enumerate(step_results, start=1):
            name = (
                str(step.get("step_name") or f"step{index}")
                if isinstance(step, Mapping)
                else f"step{index}"
            )
            paths.append(trial / "steps" / name / "agent" / "trajectory.json")
        return paths
    return [trial / "agent" / "trajectory.json"]


def _ctrf_summary(path: Path) -> dict[str, int]:
    document = _load_json(path)
    results = document.get("results")
    summary = results.get("summary") if isinstance(results, Mapping) else None
    keys = ("tests", "passed", "failed", "skipped", "pending", "other")
    if not isinstance(summary, Mapping) or any(
        not isinstance(summary.get(key), int) or isinstance(summary.get(key), bool)
        for key in keys
    ):
        raise HarborModelMatrixError("Harbor model CTRF summary is malformed")
    counts = {key: int(summary[key]) for key in keys}
    if counts["tests"] <= 0 or any(value < 0 for value in counts.values()) or sum(
        counts[key] for key in keys[1:]
    ) != counts["tests"]:
        raise HarborModelMatrixError("Harbor model CTRF counts are inconsistent")
    return counts


def _validate_model_job(
    job_dir: Path,
    *,
    task_id: str,
    model: str,
    repetitions: int,
) -> list[dict[str, Any]]:
    job_result_path = job_dir / "result.json"
    job_result = _load_json(job_result_path)
    stats = job_result.get("stats")
    if (
        job_result.get("n_total_trials") != repetitions
        or not isinstance(stats, Mapping)
        or stats.get("n_completed_trials") != repetitions
        or not isinstance(stats.get("n_errored_trials"), int)
        or not 0 <= int(stats["n_errored_trials"]) <= repetitions
    ):
        raise HarborModelMatrixError("Harbor model job is not a complete repeated rectangle")
    trials = sorted(
        path.parent.parent
        for path in job_dir.glob("*/verifier/ctrf.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(trials) != repetitions or len(set(trials)) != repetitions:
        raise HarborModelMatrixError("Harbor model job has missing or duplicate completed trials")
    rows: list[dict[str, Any]] = []
    for attempt, trial in enumerate(trials, start=1):
        result_path = trial / "result.json"
        reward_path = trial / "verifier" / "reward.txt"
        ctrf_path = trial / "verifier" / "ctrf.json"
        result = _load_json(result_path)
        exception = result.get("exception_info")
        if exception is not None and not isinstance(exception, Mapping):
            raise HarborModelMatrixError("Harbor model trial exception evidence is malformed")
        agent_result = result.get("agent_result")
        usage = {}
        if isinstance(agent_result, Mapping):
            for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
                value = agent_result.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[key] = value
            cost = agent_result.get("cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                usage["cost_usd"] = float(cost)
        observed_task = str(result.get("task_name") or "").rsplit("/", 1)[-1].replace("-", "_")
        if observed_task != task_id or not reward_path.is_file():
            raise HarborModelMatrixError("Harbor model trial task identity or reward is missing")
        try:
            reward = float(reward_path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeDecodeError, ValueError):
            raise HarborModelMatrixError("Harbor model reward is malformed") from None
        if reward not in {0.0, 1.0}:
            raise HarborModelMatrixError("Harbor model reward must be binary")
        rewards = result.get("verifier_result")
        reward_map = rewards.get("rewards") if isinstance(rewards, Mapping) else None
        counts = _ctrf_summary(ctrf_path)
        if (
            not isinstance(reward_map, Mapping)
            or reward_map.get("reward") != reward
            or (reward == 1.0 and counts["passed"] != counts["tests"])
            or (reward == 0.0 and counts["failed"] <= 0)
        ):
            raise HarborModelMatrixError("Harbor model reward and CTRF disagree")
        trace_rows = []
        for trace_path in _trajectory_paths(trial, result):
            trace = _load_json(trace_path)
            steps = trace.get("steps")
            if not str(trace.get("schema_version") or "").startswith("ATIF-") or not isinstance(
                steps, list
            ) or not steps:
                raise HarborModelMatrixError("Harbor model trajectory is not complete ATIF")
            trace_rows.append(
                {
                    "path": trace_path.relative_to(job_dir).as_posix(),
                    "digest": _file_digest(trace_path),
                    "steps": len(steps),
                }
            )
        rows.append(
            {
                "trial_id": _digest(
                    {
                        "job_result_digest": _file_digest(job_result_path),
                        "trial_result_digest": _file_digest(result_path),
                        "model": model,
                        "attempt": attempt,
                    }
                ),
                "model_id": model,
                "attempt": attempt,
                "status": "pass" if reward == 1.0 else "fail",
                "agent_failure_class": (
                    str(exception.get("exception_type") or "AgentError")
                    if isinstance(exception, Mapping)
                    else None
                ),
                "usage": usage,
                "reward": reward,
                "trial_name": trial.name,
                "trial_result_digest": _file_digest(result_path),
                "reward_digest": _file_digest(reward_path),
                "ctrf_digest": _file_digest(ctrf_path),
                "ctrf_summary": counts,
                "trajectories": trace_rows,
            }
        )
    return rows


def _next_job(jobs: Path, *, task_id: str, model: str) -> tuple[str, int]:
    prefix = f"{task_id}-{_model_key(model)}-attempt-"
    attempts = []
    for path in jobs.glob(prefix + "*"):
        try:
            attempts.append(int(path.name.removeprefix(prefix)))
        except ValueError:
            continue
    number = max(attempts, default=0) + 1
    return f"{prefix}{number:03d}", number


def _reserve_job(
    root: Path,
    *,
    task_id: str,
    task_tree_digest: str,
    model: str,
    repetitions: int,
    max_budget_usd: float,
    max_turns: int,
    provider_route_digest: str,
    max_job_attempts: int,
    preregistration_digest: str | None,
) -> tuple[str, int, dict[str, Any]]:
    """Crash-safely charge one whole Harbor job before its subprocess starts."""

    prefix = f"{task_id}-{_model_key(model)}-attempt-"
    reservations = root / "reservations"
    reservations.mkdir(parents=True, exist_ok=True)
    jobs = root / "jobs"
    while True:
        numbers = []
        for path in [*reservations.glob(prefix + "*.json"), *jobs.glob(prefix + "*")]:
            raw = path.name.removesuffix(".json").removeprefix(prefix)
            try:
                numbers.append(int(raw))
            except ValueError:
                continue
        attempt = max(numbers, default=0) + 1
        if attempt > max_job_attempts:
            raise HarborModelMatrixError(
                f"crash-safe Harbor job attempt cap reached for model {model}"
            )
        job_name = f"{prefix}{attempt:03d}"
        unsigned = {
            "schema_version": "orbenchlab.harbor-job-reservation.v1",
            "task": task_id,
            "task_tree_digest": task_tree_digest,
            "model_id": model,
            "job_name": job_name,
            "attempt": attempt,
            "repetitions": repetitions,
            "max_budget_usd_per_trial": max_budget_usd,
            "max_turns_per_trial": max_turns,
            "provider_route_digest": provider_route_digest,
            "preregistration_digest": preregistration_digest,
            "reserved_liability_usd": round(repetitions * max_budget_usd, 6),
            "accounting": "full job liability charged before Harbor subprocess launch",
        }
        reservation = {**unsigned, "reservation_digest": _digest(unsigned)}
        path = reservations / f"{job_name}.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(reservation, stream, indent=2, sort_keys=True, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            continue
        return job_name, attempt, reservation


def _valid_job(
    jobs: Path,
    *,
    task_id: str,
    model: str,
    repetitions: int,
) -> tuple[Path, list[dict[str, Any]]] | None:
    prefix = f"{task_id}-{_model_key(model)}-attempt-"
    for job in sorted(jobs.glob(prefix + "*"), reverse=True):
        try:
            return job, _validate_model_job(
                job, task_id=task_id, model=model, repetitions=repetitions
            )
        except HarborModelMatrixError:
            continue
    return None


def launch_matrix(
    task_dir: str | Path,
    *,
    harbor_executable: str | Path,
    claude_executable: str | Path,
    out: str | Path,
    models: Sequence[str],
    repetitions: int,
    provider_env: Mapping[str, str],
    max_budget_usd: float = 1.0,
    max_turns: int = 40,
    timeout_sec: float = 10_800,
    max_output_bytes: int = 32 * 1024 * 1024,
    max_job_attempts: int = 2,
    preregistration_digest: str | None = None,
) -> dict[str, Any]:
    """Run or resume a rectangular Harbor model matrix for one frozen task."""

    selected = [str(model).strip() for model in models if str(model).strip()]
    if (
        not selected
        or len(selected) != len(set(selected))
        or repetitions < 1
        or max_turns < 1
        or not 0 < max_budget_usd <= 100
        or timeout_sec <= 0
        or max_output_bytes <= 0
        or not 1 <= max_job_attempts <= 5
        or (
            preregistration_digest is not None
            and not re.fullmatch(r"sha256:[0-9a-f]{64}", preregistration_digest)
        )
    ):
        raise HarborModelMatrixError("Harbor model matrix bounds or models are invalid")
    task = Path(task_dir).resolve()
    harbor = Path(harbor_executable).resolve()
    claude = Path(claude_executable).resolve()
    if task.is_symlink() or not task.is_dir():
        raise HarborModelMatrixError("task directory must be real")
    for executable in (harbor, claude):
        if executable.is_symlink() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise HarborModelMatrixError("Harbor and Claude executables must be real executable files")
    child_env, route_digest = agent_sessions._session_env("claude-code", provider_env)
    route = str(provider_env.get("ANTHROPIC_BASE_URL", ""))
    route_host = urlsplit(route).hostname or ""
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_id = _task_id(task)
    snapshot = harbor_launcher._snapshot_task(task, root / "executed-task" / task.name)
    task_digest = _task_tree_digest(snapshot)
    jobs = root / "jobs"
    all_trials: list[dict[str, Any]] = []
    job_rows = []
    secrets = tuple(
        str(value)
        for name, value in provider_env.items()
        if ("KEY" in name.upper() or "TOKEN" in name.upper()) and str(value)
    )
    for model in selected:
        reusable = _valid_job(
            jobs, task_id=task_id, model=model, repetitions=repetitions
        )
        if reusable is None:
            job_name, attempt, reservation = _reserve_job(
                root,
                task_id=task_id,
                task_tree_digest=task_digest,
                model=model,
                repetitions=repetitions,
                max_budget_usd=max_budget_usd,
                max_turns=max_turns,
                provider_route_digest=route_digest,
                max_job_attempts=max_job_attempts,
                preregistration_digest=preregistration_digest,
            )
            harbor_launcher._bounded_command(
                [
                    str(harbor),
                    "run",
                    "--path",
                    str(snapshot),
                    "--agent",
                    "claude-code",
                    "--model",
                    model,
                    "--agent-kwarg",
                    f"max_budget_usd={max_budget_usd:.12g}",
                    "--agent-kwarg",
                    f"max_turns={max_turns}",
                    "--mounts",
                    json.dumps(
                        [
                            {
                                "type": "bind",
                                "source": str(claude),
                                "target": "/usr/local/bin/claude",
                                "read_only": True,
                            }
                        ]
                    ),
                    "--allow-agent-host",
                    route_host,
                    "--jobs-dir",
                    str(jobs),
                    "--job-name",
                    job_name,
                    "--n-attempts",
                    str(repetitions),
                    "--n-concurrent",
                    "1",
                    "--max-retries",
                    "0",
                    "--yes",
                    "--quiet",
                ],
                cwd=root,
                log_root=root / "commands" / f"{_model_key(model)}-attempt-{attempt:03d}",
                timeout_sec=timeout_sec,
                max_output_bytes=max_output_bytes,
                extra_env=child_env,
                secret_values=secrets,
            )
            job = jobs / job_name
            trials = _validate_model_job(
                job, task_id=task_id, model=model, repetitions=repetitions
            )
        else:
            job, trials = reusable
            reservation_path = root / "reservations" / f"{job.name}.json"
            reservation = _load_json(reservation_path)
            unsigned_reservation = {
                key: value for key, value in reservation.items() if key != "reservation_digest"
            }
            if (
                reservation.get("reservation_digest") != _digest(unsigned_reservation)
                or reservation.get("job_name") != job.name
                or reservation.get("task_tree_digest") != task_digest
                or reservation.get("model_id") != model
                or reservation.get("repetitions") != repetitions
                or reservation.get("max_budget_usd_per_trial") != max_budget_usd
                or reservation.get("max_turns_per_trial") != max_turns
                or reservation.get("provider_route_digest") != route_digest
                or reservation.get("preregistration_digest") != preregistration_digest
            ):
                raise HarborModelMatrixError("reused Harbor job reservation is malformed")
        all_trials.extend(trials)
        job_rows.append(
            {
                "model_id": model,
                "job_name": job.name,
                "job_result_digest": _file_digest(job / "result.json"),
                "reservation_digest": reservation["reservation_digest"],
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id,
        "task_tree_digest": task_digest,
        "models": selected,
        "repetitions": repetitions,
        "rectangular": len(all_trials) == len(selected) * repetitions,
        "trials": all_trials,
        "jobs": job_rows,
        "provider_route_digest": route_digest,
        "preregistration_digest": preregistration_digest,
        "agent": {
            "name": "claude-code",
            "executable_digest": _file_digest(claude),
            "max_budget_usd_per_trial": max_budget_usd,
            "max_turns_per_trial": max_turns,
            "budget_enforcement": "claude-cli-max-budget-usd",
            "max_job_attempts_per_model": max_job_attempts,
            "maximum_model_liability_usd": round(
                len(selected) * repetitions * max_budget_usd * max_job_attempts,
                6,
            ),
            "liability_accounting": "crash-safe whole-job reservation before subprocess launch",
        },
        "evidence_level": "E3",
        "checkpoint_capability": False,
        "limitations": [
            "Verifier-grounded Harbor trajectories; not TB-Science acceptance.",
            "Independent full restarts; no same-checkpoint E4 causal intervention claim.",
        ],
    }
    receipt["receipt_digest"] = _digest(receipt)
    harbor_launcher._atomic_json(root / "harbor-model-matrix.json", receipt)
    return receipt


def _validated_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    supplied = receipt.pop("receipt_digest", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or supplied != _digest(receipt)
        or value.get("rectangular") is not True
        or not isinstance(value.get("trials"), list)
        or not isinstance(value.get("jobs"), list)
    ):
        raise HarborModelMatrixError("Harbor model matrix receipt is malformed or unbound")
    return dict(value)


def write_trace_bundle(
    matrix: Mapping[str, Any],
    *,
    matrix_root: str | Path,
    out: str | Path,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Copy only digest-verified ATIF files into a bounded analysis bundle."""

    checked = _validated_receipt(matrix)
    source_root = Path(matrix_root).resolve()
    destination = Path(out).resolve()
    if destination.exists() and any(destination.iterdir()):
        manifest_path = destination / "trace-manifest.json"
        if manifest_path.is_file():
            manifest = _load_json(manifest_path)
            if manifest.get("matrix_receipt_digest") == checked["receipt_digest"]:
                return dict(manifest)
        raise HarborModelMatrixError("refusing to overwrite another trace bundle")
    destination.mkdir(parents=True, exist_ok=True)
    jobs = {
        str(row.get("model_id")): str(row.get("job_name"))
        for row in checked["jobs"]
        if isinstance(row, Mapping)
    }
    secrets = [str(value).encode() for value in secret_values if len(str(value)) >= 8]
    rows = []
    try:
        for trial in checked["trials"]:
            if not isinstance(trial, Mapping):
                raise HarborModelMatrixError("matrix trial row is malformed")
            model = str(trial.get("model_id") or "")
            job_name = jobs.get(model)
            trajectories = trial.get("trajectories")
            if not job_name or not isinstance(trajectories, list) or not trajectories:
                raise HarborModelMatrixError("matrix trial lacks a bound trajectory")
            job = (source_root / "jobs" / job_name).resolve()
            for index, trajectory in enumerate(trajectories, start=1):
                if not isinstance(trajectory, Mapping):
                    raise HarborModelMatrixError("matrix trajectory row is malformed")
                relative = Path(str(trajectory.get("path") or ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise HarborModelMatrixError("matrix trajectory path is unsafe")
                source = (job / relative).resolve()
                if (
                    not source.is_relative_to(job)
                    or source.is_symlink()
                    or not source.is_file()
                    or source.stat().st_size > 16 * 1024 * 1024
                    or _file_digest(source) != trajectory.get("digest")
                ):
                    raise HarborModelMatrixError("matrix trajectory source failed digest validation")
                data = source.read_bytes()
                if any(secret in data for secret in secrets):
                    raise HarborModelMatrixError("provider credential appeared in ATIF trajectory")
                target = destination / _model_key(model) / str(trial["trial_id"]).removeprefix(
                    "sha256:"
                )[:16] / f"trajectory-{index:02d}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                rows.append(
                    {
                        "model_id": model,
                        "trial_id": trial["trial_id"],
                        "path": target.relative_to(destination).as_posix(),
                        "digest": _file_digest(target),
                        "steps": trajectory.get("steps"),
                    }
                )
        manifest: dict[str, Any] = {
            "schema_version": "orbenchlab.harbor-trace-bundle.v1",
            "matrix_receipt_digest": checked["receipt_digest"],
            "task_tree_digest": checked["task_tree_digest"],
            "trajectories": rows,
        }
        manifest["manifest_digest"] = _digest(manifest)
        harbor_launcher._atomic_json(destination / "trace-manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def build_screening_report(
    matrix: Mapping[str, Any],
    *,
    harbor_controls: Mapping[str, Any],
    out: str | Path,
) -> dict[str, Any]:
    """Convert a validated Harbor matrix into the common screening envelope."""

    checked = _validated_receipt(matrix)
    controls_unsigned = {
        key: value for key, value in harbor_controls.items() if key != "report_digest"
    }
    if (
        harbor_controls.get("harbor_receipt_schema_version")
        != "orbenchlab.harbor-controls.v1"
        or harbor_controls.get("report_digest") != _digest(controls_unsigned)
        or harbor_controls.get("task_tree_digest") != checked["task_tree_digest"]
        or not isinstance(harbor_controls.get("tasks"), list)
        or len(harbor_controls["tasks"]) != 1
    ):
        raise HarborModelMatrixError("Harbor controls do not bind the model matrix task")
    control_gates = harbor_controls["tasks"][0].get("control_gates")
    if not isinstance(control_gates, Mapping) or any(
        not isinstance(control_gates.get(name), Mapping)
        or control_gates[name].get("gate") != "pass"
        for name in ("oracle", "nop")
    ):
        raise HarborModelMatrixError("Harbor controls did not pass")
    raw_trials = []
    for trial in checked["trials"]:
        raw_trials.append(
            {
                "model": trial["model_id"],
                "trial": trial["attempt"],
                "hint_level": 0,
                "status": trial["status"],
                "phase": "harbor-verifier",
                "agent_failure_class": trial.get("agent_failure_class"),
                "usage": trial.get("usage", {}),
                "harbor_trial_id": trial["trial_id"],
                "trial_result_digest": trial["trial_result_digest"],
                "reward_digest": trial["reward_digest"],
                "ctrf_digest": trial["ctrf_digest"],
                "trajectories": trial["trajectories"],
                "verifier": {
                    "receipt_valid": True,
                    "status": trial["status"],
                    "reward": trial["reward"],
                    "ctrf": {
                        "summary": trial["ctrf_summary"],
                        "digest": trial["ctrf_digest"],
                    },
                },
            }
        )
    arms = volc_rollout._summarize_trials(raw_trials)
    discrimination = volc_rollout._discrimination_summary(
        arms,
        checked["models"],
        repetitions=int(checked["repetitions"]),
    )
    decision = "review-promising" if discrimination["promising"] else "collect-more-evidence"
    task_row = {
        "task": checked["task"],
        "family": checked["task"],
        "arms": arms,
        "control_gates": dict(control_gates),
        "discrimination_index_observed_gap": discrimination["observed_gap"],
        "discrimination": discrimination,
        "decision": decision,
        "evidence_level": "E3",
        "limitations": list(checked["limitations"]),
    }
    report: dict[str, Any] = {
        "schema_version": "orbenchlab.screening-report.v1",
        "harbor_model_matrix_schema_version": SCHEMA_VERSION,
        "task": checked["task"],
        "tasks": [task_row],
        "trials": raw_trials,
        "models": checked["models"],
        "hint_levels": [0],
        "run_contract": {
            "models": checked["models"],
            "repetitions": checked["repetitions"],
            "hint_levels": [0],
            "max_budget_usd_per_trial": checked["agent"]["max_budget_usd_per_trial"],
            "max_turns_per_trial": checked["agent"]["max_turns_per_trial"],
        },
        "task_tree_digest": checked["task_tree_digest"],
        "harbor_model_matrix_digest": checked["receipt_digest"],
        "harbor_control_report_digest": harbor_controls["report_digest"],
    }
    report["report_digest"] = _digest(report)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    harbor_launcher._atomic_json(output / "screening-report.json", report)
    return report


__all__ = [
    "HarborModelMatrixError",
    "SCHEMA_VERSION",
    "build_screening_report",
    "launch_matrix",
    "write_trace_bundle",
]
