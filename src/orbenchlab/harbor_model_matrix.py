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
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from . import agent_sessions, harbor_launcher
from .core.errors import ORBenchError
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
            job_name, attempt = _next_job(jobs, task_id=task_id, model=model)
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
        all_trials.extend(trials)
        job_rows.append(
            {
                "model_id": model,
                "job_name": job.name,
                "job_result_digest": _file_digest(job / "result.json"),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id,
        "task_tree_digest": _task_tree_digest(snapshot),
        "models": selected,
        "repetitions": repetitions,
        "rectangular": len(all_trials) == len(selected) * repetitions,
        "trials": all_trials,
        "jobs": job_rows,
        "provider_route_digest": route_digest,
        "agent": {
            "name": "claude-code",
            "executable_digest": _file_digest(claude),
            "max_budget_usd_per_trial": max_budget_usd,
            "max_turns_per_trial": max_turns,
            "budget_enforcement": "claude-cli-max-budget-usd",
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


__all__ = ["HarborModelMatrixError", "SCHEMA_VERSION", "launch_matrix"]
