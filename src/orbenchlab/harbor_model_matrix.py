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
import threading
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


def _job_trials(
    job_dir: Path,
    *,
    task_id: str,
    model: str,
    requested: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one Harbor job trial-by-trial.

    Returns ``(valid_rows, excluded_rows)``.  A trial whose verifier evidence
    is *absent* (no CTRF or no reward — a setup crash or verifier timeout left
    no conclusive grade) is excluded as inconclusive and disclosed, so one
    such trial no longer discards its four completed siblings.  Evidence that
    is *present but inconsistent* stays a hard error: that is corruption, not
    an infrastructure gap.
    """

    job_result_path = job_dir / "result.json"
    job_result = _load_json(job_result_path)
    stats = job_result.get("stats")
    if (
        job_result.get("n_total_trials") != requested
        or not isinstance(stats, Mapping)
        or not isinstance(stats.get("n_completed_trials"), int)
    ):
        raise HarborModelMatrixError("Harbor model job result does not match its request")
    trial_dirs = sorted(
        path.parent
        for path in job_dir.glob("*/result.json")
        if path.is_file() and not path.is_symlink() and path.parent.parent == job_dir
    )
    if len(trial_dirs) > requested:
        raise HarborModelMatrixError("Harbor model job contains surplus trials")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for trial in trial_dirs:
        result_path = trial / "result.json"
        reward_path = trial / "verifier" / "reward.txt"
        ctrf_path = trial / "verifier" / "ctrf.json"
        if (
            not reward_path.is_file()
            or reward_path.is_symlink()
            or not ctrf_path.is_file()
            or ctrf_path.is_symlink()
        ):
            excluded.append(
                {
                    "job_name": job_dir.name,
                    "trial_name": trial.name,
                    "model_id": model,
                    "reason": "inconclusive-no-verifier-evidence",
                    "trial_result_digest": _file_digest(result_path),
                }
            )
            continue
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
                        "trial_name": trial.name,
                    }
                ),
                "model_id": model,
                "status": "pass" if reward == 1.0 else "fail",
                "agent_failure_class": (
                    str(exception.get("exception_type") or "AgentError")
                    if isinstance(exception, Mapping)
                    else None
                ),
                "usage": usage,
                "reward": reward,
                "job_name": job_dir.name,
                "trial_name": trial.name,
                "trial_result_digest": _file_digest(result_path),
                "reward_digest": _file_digest(reward_path),
                "ctrf_digest": _file_digest(ctrf_path),
                "ctrf_summary": counts,
                "trajectories": trace_rows,
            }
        )
    return rows, excluded


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


def _existing_jobs(jobs: Path, *, task_id: str, model: str) -> list[Path]:
    prefix = f"{task_id}-{_model_key(model)}-attempt-"
    return sorted(
        path
        for path in jobs.glob(prefix + "*")
        if path.is_dir()
        and not path.is_symlink()
        and (path / "result.json").is_file()
    )


def _validated_reservation(
    root: Path,
    job: Path,
    *,
    task_digest: str,
    model: str,
    max_budget_usd: float,
    max_turns: int,
    route_digest: str,
    preregistration_digest: str | None,
) -> dict[str, Any]:
    reservation_path = root / "reservations" / f"{job.name}.json"
    reservation = _load_json(reservation_path)
    unsigned = {
        key: value for key, value in reservation.items() if key != "reservation_digest"
    }
    requested = reservation.get("repetitions")
    if (
        reservation.get("reservation_digest") != _digest(unsigned)
        or reservation.get("job_name") != job.name
        or reservation.get("task_tree_digest") != task_digest
        or reservation.get("model_id") != model
        or not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 1
        or reservation.get("max_budget_usd_per_trial") != max_budget_usd
        or reservation.get("max_turns_per_trial") != max_turns
        or reservation.get("provider_route_digest") != route_digest
        or reservation.get("preregistration_digest") != preregistration_digest
    ):
        raise HarborModelMatrixError("Harbor job reservation is missing or malformed")
    return dict(reservation)


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
    credential_transport: str = "relay",
    relay_host: str = "127.0.0.1",
    relay_bind_host: str = "127.0.0.1",
    relay_prober=None,
) -> dict[str, Any]:
    """Run or resume a rectangular Harbor model matrix for one frozen task.

    ``credential_transport`` controls how the provider credential reaches the
    Harbor agent.  ``relay`` (default) starts a host-side credential relay so
    the real key never enters Harbor's environment/argv/container; ``direct``
    is refused because Harbor forwards agent env via ``docker compose exec -e
    KEY=value`` onto the host argv.  ``relay_bind_host`` is the interface the
    relay listens on (``0.0.0.0`` on a real host so a container can reach it),
    and ``relay_host`` is the address the container connects to (the Docker
    bridge gateway or ``host.docker.internal``).  ``relay_prober`` — required
    for a real paid run — is a callable ``(url, token) -> (ok, detail)`` that
    proves the relay is reachable from inside the actual Harbor container
    before any trial is bought.
    """

    from . import harbor_credentials

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
    # Validate and bind the real Volc route for the receipt, but never hand the
    # real token to Harbor: it would land on the host argv.
    _, route_digest = agent_sessions._session_env("claude-code", provider_env)
    route = str(provider_env.get("ANTHROPIC_BASE_URL", ""))
    if credential_transport != "relay":
        raise harbor_credentials.CredentialRelayError(
            "Harbor forwards agent env onto the host argv; only credential_transport='relay' is allowed"
        )
    secrets = tuple(
        str(value)
        for name, value in provider_env.items()
        if ("KEY" in name.upper() or "TOKEN" in name.upper()) and str(value)
    )
    real_token = str(
        provider_env.get("ANTHROPIC_AUTH_TOKEN")
        or provider_env.get("ANTHROPIC_API_KEY")
        or ""
    )
    # Caps derive from the worst-case work, not a fixed constant: one message
    # request per turn per trial per job attempt, with generous headroom.
    per_job_request_cap = max(8, (max_turns + 4))
    total_request_cap = per_job_request_cap * len(selected) * repetitions * max_job_attempts
    relay = harbor_credentials.CredentialRelay(
        real_base_url=route,
        real_token=real_token,
        bind_host=relay_bind_host,
        advertise_host=relay_host,
        max_requests=total_request_cap,
    )
    relay.start()
    try:
        return _launch_matrix_relayed(
            relay=relay,
            harbor=harbor,
            claude=claude,
            task=task,
            selected=selected,
            out=out,
            repetitions=repetitions,
            max_budget_usd=max_budget_usd,
            max_turns=max_turns,
            timeout_sec=timeout_sec,
            max_output_bytes=max_output_bytes,
            max_job_attempts=max_job_attempts,
            preregistration_digest=preregistration_digest,
            route_digest=route_digest,
            secrets=secrets,
            relay_prober=relay_prober,
            per_job_request_cap=per_job_request_cap,
        )
    finally:
        relay.stop()


def _launch_matrix_relayed(
    *,
    relay: Any,
    harbor: Path,
    claude: Path,
    task: Path,
    selected: Sequence[str],
    out: str | Path,
    repetitions: int,
    max_budget_usd: float,
    max_turns: int,
    timeout_sec: float,
    max_output_bytes: int,
    max_job_attempts: int,
    preregistration_digest: str | None,
    route_digest: str,
    secrets: tuple[str, ...],
    relay_prober=None,
    per_job_request_cap: int = 64,
) -> dict[str, Any]:
    from . import harbor_credentials

    route_host = relay.advertise_host
    reachability: dict[str, Any] | None = None
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_id = _task_id(task)
    snapshot = harbor_launcher._snapshot_task(task, root / "executed-task" / task.name)
    task_digest = _task_tree_digest(snapshot)
    jobs = root / "jobs"
    all_trials: list[dict[str, Any]] = []
    job_rows = []
    excluded_trials: list[dict[str, Any]] = []
    surplus_trials = 0
    for model in selected:
        valid: list[dict[str, Any]] = []

        def consume_job(job: Path, requested: int, reservation: Mapping[str, Any]) -> None:
            rows, skipped = _job_trials(
                job, task_id=task_id, model=model, requested=requested
            )
            valid.extend(rows)
            excluded_trials.extend(skipped)
            job_rows.append(
                {
                    "model_id": model,
                    "job_name": job.name,
                    "job_result_digest": _file_digest(job / "result.json"),
                    "reservation_digest": reservation["reservation_digest"],
                    "requested_trials": requested,
                    "valid_trials": len(rows),
                    "excluded_trials": len(skipped),
                }
            )

        # Confirmed trials from earlier (possibly partial) jobs are never
        # re-bought: every existing job is consumed against its reservation
        # before any new liability is reserved.
        for job in _existing_jobs(jobs, task_id=task_id, model=model):
            reservation = _validated_reservation(
                root,
                job,
                task_digest=task_digest,
                model=model,
                max_budget_usd=max_budget_usd,
                max_turns=max_turns,
                route_digest=route_digest,
                preregistration_digest=preregistration_digest,
            )
            consume_job(job, int(reservation["repetitions"]), reservation)
        while len(valid) < repetitions:
            missing = repetitions - len(valid)
            job_name, attempt, reservation = _reserve_job(
                root,
                task_id=task_id,
                task_tree_digest=task_digest,
                model=model,
                repetitions=missing,
                max_budget_usd=max_budget_usd,
                max_turns=max_turns,
                provider_route_digest=route_digest,
                max_job_attempts=max_job_attempts,
                preregistration_digest=preregistration_digest,
            )
            command_log = root / "commands" / f"{_model_key(model)}-attempt-{attempt:03d}"
            # A per-job, short-lived, revocable token bound to this exact
            # model/task/job/budget scope — a leaked token cannot buy the whole
            # matrix, only this job's capped requests.
            job_token = relay.issue_token(
                {
                    "model": model,
                    "task_tree_digest": task_digest,
                    "job_name": job_name,
                    "max_budget_usd_per_trial": max_budget_usd,
                    "request_cap": per_job_request_cap,
                }
            )
            child_env = relay.relay_env(job_token)
            if relay_prober is not None and reachability is None:
                reachability = harbor_credentials.probe_reachability(
                    prober=relay_prober,
                    advertise_host=relay.advertise_host,
                    port=relay.port,
                    route_path=urlsplit(child_env["ANTHROPIC_BASE_URL"]).path,
                    token=job_token,
                )
            argv = [
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
                str(missing),
                "--n-concurrent",
                "1",
                "--max-retries",
                "0",
                "--yes",
                "--quiet",
            ]
            # A background scanner watches host process argv for the real
            # credential while Harbor runs; with the relay the real key is never
            # in Harbor's env, so this stays empty and proves the transport.
            proc_leak: list[int] = []
            stop_scanner = threading.Event()

            def _scan() -> None:
                while not stop_scanner.wait(0.5):
                    proc_leak.extend(harbor_credentials.scan_proc_for_secret(secrets))

            scanner = threading.Thread(target=_scan, daemon=True)
            scanner.start()
            try:
                harbor_launcher._bounded_command(
                    argv,
                    cwd=root,
                    log_root=command_log,
                    timeout_sec=timeout_sec,
                    max_output_bytes=max_output_bytes,
                    extra_env=child_env,
                    secret_values=secrets,
                )
            finally:
                stop_scanner.set()
                scanner.join(timeout=5)
                # The per-job token is revoked the moment the job ends, so it
                # cannot be replayed even if it leaked into the container.
                relay.revoke_token(job_token)
            # Fail closed on any real-credential leak in argv or artifacts.
            harbor_credentials.assert_no_credential_leak(
                secret_values=secrets,
                artifact_roots=[command_log, jobs / job_name],
                scan_proc=False,
            )
            if proc_leak:
                raise harbor_credentials.CredentialSecurityBarrier(
                    "provider credential leak detected in host process argv"
                )
            consume_job(jobs / job_name, missing, reservation)
        # Deterministic order and stable renumbering; a resumed run that finds
        # more than ``repetitions`` valid trials keeps the earliest ones and
        # discloses the surplus rather than silently changing the rectangle.
        valid.sort(key=lambda row: (row["job_name"], row["trial_name"]))
        surplus_trials += max(0, len(valid) - repetitions)
        for attempt_number, row in enumerate(valid[:repetitions], start=1):
            all_trials.append({**row, "attempt": attempt_number})
    # Final full-tree scan across the whole matrix output, plus the relay's
    # non-secret security receipt (policy digest, endpoint allowlist, token
    # scope digests, request/byte counts, scan results).
    artifact_scan = harbor_credentials.assert_no_credential_leak(
        secret_values=secrets, artifact_roots=[root], scan_proc=False
    )
    security_receipt = relay.security_receipt(
        scan_results={"artifact_scan": artifact_scan, "reachability": reachability}
    )
    # The full security receipt carries per-run token digests and counts, which
    # are not part of the deterministic matrix identity; write it beside the
    # matrix receipt and bind only its stable policy digest.
    harbor_launcher._atomic_json(root / "credential-security.json", security_receipt)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id,
        "task_tree_digest": task_digest,
        "models": selected,
        "repetitions": repetitions,
        "rectangular": len(all_trials) == len(selected) * repetitions,
        "trials": all_trials,
        "excluded_trials": excluded_trials,
        "surplus_valid_trials": surplus_trials,
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
            "credential_transport": "host-side-relay-per-job-scoped-token",
            "credential_security_policy_digest": security_receipt["policy_digest"],
        },
        "evidence_level": "E3",
        "checkpoint_capability": False,
        "limitations": [
            "Verifier-grounded Harbor trajectories; not TB-Science acceptance.",
            "Independent full restarts; no same-checkpoint E4 causal intervention claim.",
            *(
                [
                    f"{len(excluded_trials)} trial(s) had no conclusive verifier evidence "
                    "and were excluded from the rectangle as inconclusive; exclusions are "
                    "itemised in excluded_trials and may bias rates if failures correlate "
                    "with infrastructure loss."
                ]
                if excluded_trials
                else []
            ),
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
    model_jobs = {
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
            job_name = str(trial.get("job_name") or "") or model_jobs.get(model)
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
    excluded = checked.get("excluded_trials") or []
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
        "excluded_trial_count": len(excluded),
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


DISCRIMINATION_FEEDBACK_SCHEMA = "orbenchlab.discrimination-feedback.v1"


def classify_discrimination(screening: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a screening report's model-discrimination pattern.

    This is a deterministic pattern classifier, not a semantic verdict: it
    reports *what the numbers show* and whether that warrants a contract review,
    without itself deciding the instruction/verifier contract is broken. When
    the controls pass yet no model solves the task (``degenerate-all-fail``),
    that is the strongest machine-detectable signal of an instruction/verifier
    contract mismatch or an over-hard task, and it must be fed back for review
    rather than silently promoted or blamed on model quality.
    """

    tasks = screening.get("tasks") if isinstance(screening, Mapping) else None
    task_row = tasks[0] if isinstance(tasks, list) and tasks else {}
    discrimination = task_row.get("discrimination") if isinstance(task_row, Mapping) else None
    cells = (
        list(discrimination.get("models") or [])
        if isinstance(discrimination, Mapping)
        else []
    )
    controls_passed = False
    control_gates = task_row.get("control_gates") if isinstance(task_row, Mapping) else None
    if isinstance(control_gates, Mapping):
        controls_passed = all(
            isinstance(control_gates.get(name), Mapping)
            and control_gates[name].get("gate") == "pass"
            for name in ("oracle", "nop")
        )
    rates = [
        float(cell["solve_rate"])
        for cell in cells
        if isinstance(cell, Mapping) and isinstance(cell.get("solve_rate"), (int, float))
    ]
    rectangular = bool(discrimination.get("rectangular")) if isinstance(discrimination, Mapping) else False
    promising = bool(discrimination.get("promising")) if isinstance(discrimination, Mapping) else False

    if not rectangular or len(rates) < 2:
        kind = "insufficient-evidence"
    elif all(rate <= 0.0 for rate in rates):
        kind = "degenerate-all-fail"
    elif all(rate >= 1.0 for rate in rates):
        kind = "degenerate-all-pass"
    elif promising:
        kind = "discriminating"
    else:
        kind = "weak-signal"

    contract_review_required = kind == "degenerate-all-fail" and controls_passed
    reasons = {
        "insufficient-evidence": "model arms are not a complete equal-budget rectangle",
        "degenerate-all-fail": (
            "controls pass but no model solved any trial: suspect an instruction/"
            "verifier contract mismatch or an over-hard task; do not attribute to "
            "model quality without a contract review"
            if controls_passed
            else "no model solved any trial"
        ),
        "degenerate-all-pass": "every model solved every trial: the task is too easy to discriminate",
        "discriminating": "positive conservative model separation on the unassisted cell",
        "weak-signal": "model arms are complete but separation is not significant",
    }
    verdict = {
        "schema_version": DISCRIMINATION_FEEDBACK_SCHEMA,
        "kind": kind,
        "controls_passed": controls_passed,
        "contract_review_required": contract_review_required,
        "model_solve_rates": [
            {"model": cell.get("model"), "solve_rate": cell.get("solve_rate")}
            for cell in cells
            if isinstance(cell, Mapping)
        ],
        "observed_gap": discrimination.get("observed_gap") if isinstance(discrimination, Mapping) else None,
        "reason": reasons[kind],
        "screening_report_digest": screening.get("report_digest"),
    }
    verdict["feedback_digest"] = _digest(
        {key: value for key, value in verdict.items() if key != "feedback_digest"}
    )
    return verdict


__all__ = [
    "DISCRIMINATION_FEEDBACK_SCHEMA",
    "HarborModelMatrixError",
    "SCHEMA_VERSION",
    "build_screening_report",
    "classify_discrimination",
    "launch_matrix",
    "write_trace_bundle",
]
