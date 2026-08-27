"""Bounded, provider-pinned sessions for autonomous coding-agent CLI stages.

The agent owns semantic work.  This module owns only the process boundary and
its evidence: deterministic identity, a narrow environment, a hard deadline,
and a fail-closed receipt.  A completed receipt is not verifier evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .core.errors import ORBenchError


class AgentSessionError(ORBenchError):
    exit_code = 9


_PROFILES = {
    "codex": {
        "route": "OPENAI_BASE_URL",
        "token": "OPENAI_API_KEY",
        "allowed": frozenset({"OPENAI_BASE_URL", "OPENAI_API_KEY"}),
    },
    "claude-code": {
        "route": "ANTHROPIC_BASE_URL",
        "token": "ANTHROPIC_AUTH_TOKEN",
        "allowed": frozenset(
            {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"}
        ),
    },
}
_SAFE_HOST_ENV = ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def _argv(profile: str, executable: str, model: str) -> list[str]:
    if profile == "codex":
        return [executable, "exec", "--json", "--model", model, "-"]
    return [
        executable,
        "--print",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--model",
        model,
        "--tools",
        "",
    ]


def _stdin(profile: str, prompt: str) -> bytes:
    if profile == "codex":
        return prompt.encode()
    return (
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": prompt}},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def _session_env(profile: str, supplied: Mapping[str, str]) -> tuple[dict[str, str], str]:
    spec = _PROFILES[profile]
    unknown = set(supplied) - spec["allowed"]
    if unknown:
        raise AgentSessionError("agent environment contains non-allowlisted names")
    route = str(supplied.get(spec["route"], "")).strip().rstrip("/")
    parsed = urlsplit(route)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not (host == "volces.com" or host.endswith(".volces.com"))
        or not parsed.path.rstrip("/").endswith("/api/coding")
    ):
        raise AgentSessionError("agent session requires the Volc HTTPS coding route")
    if not str(supplied.get(spec["token"], "")).strip():
        raise AgentSessionError("agent session requires its profile credential")
    child = {name: os.environ[name] for name in _SAFE_HOST_ENV if name in os.environ}
    child.update({key: str(value) for key, value in supplied.items()})
    return child, _digest({"host": host, "path": parsed.path.rstrip("/")})


def _valid_reuse(path: Path, session_id: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        root = path.parent
        workdir = Path(str(receipt.get("workdir_runtime_path", "")))
        output_root = Path(str(receipt.get("output_runtime_path", "")))
        if (
            receipt.get("receipt_digest") != _digest(unsigned)
            or receipt.get("session_id") != session_id
            or receipt.get("status") != "completed"
            or receipt.get("stdout_digest") != _digest_bytes((root / "stdout.bin").read_bytes())
            or receipt.get("stderr_digest") != _digest_bytes((root / "stderr.bin").read_bytes())
            or not workdir.is_dir()
            or receipt.get("result_tree_digest") != _tree_digest(workdir, exclude=output_root)
        ):
            return None
        return receipt
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def _tree_digest(root: Path, *, exclude: Path | None = None) -> str:
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if exclude is not None and (path == exclude or exclude in path.parents):
            continue
        if ".git" in relative.parts or not path.is_file() or path.is_symlink():
            continue
        rows.append((relative.as_posix(), _digest_bytes(path.read_bytes())))
    return _digest(rows)


def run_session(
    *,
    profile: str,
    stage: str,
    model: str,
    prompt: str,
    workdir: str | Path,
    out: str | Path,
    timeout_sec: float,
    environ: Mapping[str, str],
    executable: str | Path | None = None,
) -> dict[str, Any]:
    """Run or safely reuse one deterministic autonomous-agent session."""

    if profile not in _PROFILES:
        raise AgentSessionError("profile must be codex or claude-code")
    if not stage.strip() or not model.strip() or not prompt or timeout_sec <= 0:
        raise AgentSessionError("stage, model, prompt and timeout must be positive")
    cwd = Path(workdir).resolve()
    if not cwd.is_dir() or cwd.is_symlink():
        raise AgentSessionError("workdir must be a real directory")
    child_env, route_digest = _session_env(profile, environ)
    requested = str(executable or ("codex" if profile == "codex" else "claude"))
    resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
    if not resolved or not Path(resolved).is_file():
        raise AgentSessionError("agent CLI executable is unavailable")
    command = _argv(profile, str(Path(resolved).resolve()), model.strip())
    output_root = Path(out).resolve()
    input_tree_digest = _tree_digest(cwd, exclude=output_root)
    identity = {
        "schema_version": "orbenchlab.agent-session.identity.v1",
        "profile": profile,
        "stage": stage.strip(),
        "model": model.strip(),
        "prompt_digest": _digest_bytes(prompt.encode()),
        "route_digest": route_digest,
        "argv_template": command[1:],
        "workdir_binding": _digest(str(cwd)),
        "executable_digest": _digest_bytes(Path(resolved).read_bytes()),
    }
    session_id = _digest(identity).removeprefix("sha256:")[:32]
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = session_dir / "receipt.json"
    reused = _valid_reuse(receipt_path, session_id)
    if reused is not None:
        return {**reused, "receipt_path": str(receipt_path), "reused": True}

    started = time.monotonic()
    stdout = b""
    stderr = b""
    exit_code: int | None = None
    failure_class: str | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(_stdin(profile, prompt), timeout=timeout_sec)
            exit_code = process.returncode
            if exit_code != 0:
                failure_class = "agent_exit_nonzero"
        except subprocess.TimeoutExpired:
            failure_class = "wall_clock_timeout"
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            exit_code = process.returncode
    except OSError:
        failure_class = "launch_error"

    _atomic_bytes(session_dir / "stdout.bin", stdout)
    _atomic_bytes(session_dir / "stderr.bin", stderr)
    receipt: dict[str, Any] = {
        "schema_version": "orbenchlab.agent-session.receipt.v1",
        "session_id": session_id,
        "identity": identity,
        # Runtime-only local path supports safe reuse checks. Downstream public
        # exporters must continue to omit host paths, as they do for run roots.
        "workdir_runtime_path": str(cwd),
        "output_runtime_path": str(output_root),
        "input_tree_digest": input_tree_digest,
        "result_tree_digest": _tree_digest(cwd, exclude=output_root),
        "status": "completed" if failure_class is None else "failed",
        "failure_class": failure_class,
        "exit_code": exit_code,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout_digest": _digest_bytes(stdout),
        "stderr_digest": _digest_bytes(stderr),
        "trace_digest": _digest(
            {
                "session_id": session_id,
                "stdout": _digest_bytes(stdout),
                "stderr": _digest_bytes(stderr),
                "exit_code": exit_code,
                "failure_class": failure_class,
            }
        ),
        "usage": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        "evidence_level": "E1-agent-session-process",
        "limitations": ["Agent completion is not static-gate, verifier, or Harbor evidence."],
    }
    receipt["receipt_digest"] = _digest(receipt)
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path), "reused": False}


__all__ = ["AgentSessionError", "run_session"]
