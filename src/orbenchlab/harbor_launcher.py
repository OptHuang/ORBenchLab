"""Bounded, idempotent Harbor Oracle/NOP launcher for factory promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from . import harbor_controls
from .core.errors import ORBenchError
from .volc_rollout import _task_id, _task_tree_digest


class HarborLauncherError(ORBenchError):
    exit_code = 8


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _bounded_command(
    argv: list[str],
    *,
    cwd: Path,
    log_root: Path,
    timeout_sec: float,
    max_output_bytes: int,
    extra_env: Mapping[str, str] | None = None,
    secret_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "stdout.bin"
    stderr_path = log_root / "stderr.bin"
    started = time.monotonic()
    child_env = {
        name: os.environ[name]
        for name in ("PATH", "DOCKER_HOST", "XDG_RUNTIME_DIR")
        if name in os.environ
    }
    child_env.update({str(key): str(value) for key, value in (extra_env or {}).items()})
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=child_env,
        )
        failure: str | None = None
        while process.poll() is None:
            elapsed = time.monotonic() - started
            size = stdout_path.stat().st_size + stderr_path.stat().st_size
            if size > max_output_bytes:
                failure = "output_limit_exceeded"
            elif elapsed > timeout_sec:
                failure = "timeout"
            if failure:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(0.05)
        return_code = process.wait()
    redacted = False
    for path in (stdout_path, stderr_path):
        data = path.read_bytes()
        for secret in secret_values:
            if secret and secret.encode() in data:
                data = data.replace(secret.encode(), b"[REDACTED_PROVIDER_CREDENTIAL]")
                redacted = True
        path.write_bytes(data)
    if failure is None and return_code != 0:
        failure = "nonzero_exit"
    receipt = {
        "schema_version": "orbenchlab.harbor-command.v1",
        "argv": argv,
        "executable_digest": _file_digest(Path(argv[0]).resolve()) if Path(argv[0]).is_file() else None,
        "status": "passed" if failure is None else "blocked",
        "failure_class": failure,
        "exit_code": return_code,
        "elapsed_sec": round(time.monotonic() - started, 6),
        "max_output_bytes": max_output_bytes,
        "stdout_digest": _file_digest(stdout_path),
        "stderr_digest": _file_digest(stderr_path),
        "provider_credential_redacted": redacted,
    }
    _atomic_json(log_root / "command-receipt.json", receipt)
    if failure:
        raise HarborLauncherError(f"Harbor command failed: {failure}")
    return receipt


def _snapshot_task(task: Path, destination: Path) -> Path:
    source_digest = _task_tree_digest(task)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise HarborLauncherError("executed Harbor task snapshot path is unsafe")
        if _task_tree_digest(destination) != source_digest:
            raise HarborLauncherError("existing executed Harbor task snapshot has drifted")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copytree(task, temporary, symlinks=False)
    if _task_tree_digest(temporary) != source_digest:
        raise HarborLauncherError("executed Harbor task snapshot copy changed content")
    os.replace(temporary, destination)
    return destination


def _valid_completed_job(jobs: Path, *, task_id: str, control: str) -> Path | None:
    candidates = sorted(jobs.glob(f"{task_id}-{control}-attempt-*"), reverse=True)
    for job in candidates:
        if not (job / "result.json").is_file():
            continue
        try:
            harbor_controls._validate_job(job, control=control)
        except harbor_controls.HarborControlError:
            continue
        return job
    return None


def _next_job(jobs: Path, *, task_id: str, control: str) -> tuple[str, int]:
    prefix = f"{task_id}-{control}-attempt-"
    numbers = []
    for path in jobs.glob(prefix + "*"):
        try:
            numbers.append(int(path.name.removeprefix(prefix)))
        except ValueError:
            continue
    attempt = max(numbers, default=0) + 1
    return f"{prefix}{attempt:03d}", attempt


def launch_controls(
    task_dir: str | Path,
    *,
    harbor_executable: str | Path,
    out: str | Path,
    timeout_sec: float = 7200,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Run or resume one exact Oracle and NOP Harbor control trial."""

    if timeout_sec <= 0 or max_output_bytes <= 0:
        raise HarborLauncherError("Harbor timeout and output bounds must be positive")
    task = Path(task_dir).resolve()
    executable = Path(harbor_executable).resolve()
    if task.is_symlink() or not task.is_dir():
        raise HarborLauncherError("task directory must be a real directory")
    if executable.is_symlink() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise HarborLauncherError("Harbor executable must be a real executable file")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_id = _task_id(task)
    snapshot = _snapshot_task(task, root / "executed-task" / task.name)
    jobs = root / "jobs"
    selected_jobs: dict[str, Path] = {}
    for control in ("oracle", "nop"):
        completed = _valid_completed_job(jobs, task_id=task_id, control=control)
        if completed is not None:
            selected_jobs[control] = completed
            continue
        job_name, attempt = _next_job(jobs, task_id=task_id, control=control)
        _bounded_command(
            [
                str(executable),
                "run",
                "--path",
                str(snapshot),
                "--agent",
                control,
                "--jobs-dir",
                str(jobs),
                "--job-name",
                job_name,
                "--n-concurrent",
                "1",
                "--max-retries",
                "0",
                "--yes",
            ],
            cwd=root,
            log_root=root / "commands" / f"{control}-attempt-{attempt:03d}",
            timeout_sec=timeout_sec,
            max_output_bytes=max_output_bytes,
        )
        job = jobs / job_name
        try:
            harbor_controls._validate_job(job, control=control)
        except harbor_controls.HarborControlError as exc:
            raise HarborLauncherError(
                f"Harbor {control} attempt did not produce a valid control job: {exc}"
            ) from None
        selected_jobs[control] = job
    receipt = harbor_controls.build_receipt(
        task,
        executed_task_dir=snapshot,
        oracle_job=selected_jobs["oracle"],
        nop_job=selected_jobs["nop"],
    )
    harbor_controls.write_receipt(receipt, root)
    return receipt


__all__ = ["HarborLauncherError", "launch_controls"]
