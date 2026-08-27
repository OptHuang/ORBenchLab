"""Bounded, provider-pinned sessions for autonomous coding-agent CLI stages.

The agent owns semantic work.  This module owns only the process boundary and
its evidence: deterministic identity, a narrow environment, a hard deadline,
and a fail-closed receipt.  A completed receipt is not verifier evidence.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping
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
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


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


def _argv(
    profile: str,
    executable: str,
    model: str,
    *,
    max_budget_usd: float,
) -> list[str]:
    if profile == "codex":
        return [executable, "exec", "--json", "--model", model, "-"]
    coding_tools = "Read,Glob,Grep,Edit,Write,Bash"
    return [
        executable,
        "--print",
        "--verbose",
        "--safe-mode",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--max-budget-usd",
        format(max_budget_usd, ".12g"),
        "--tools",
        coding_tools,
        "--allowedTools",
        coding_tools,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--model",
        model,
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


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _bounded_process(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdin: bytes,
    timeout_sec: float,
    max_output_bytes: int,
    on_chunk: Callable[[str, bytes], None] | None = None,
) -> tuple[bytes, bytes, int | None, str | None]:
    """Drain both pipes incrementally and kill the group at either hard bound."""

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        return b"", b"", None, "launch_error"
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    streams = selectors.DefaultSelector()
    for stream in (process.stdin, process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
    streams.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    input_offset = 0
    captured = 0
    failure: str | None = None
    deadline = time.monotonic() + timeout_sec
    try:
        while streams.get_map():
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                failure = "wall_clock_timeout"
                break
            for key, _ in streams.select(min(remaining_time, 0.1)):
                stream = key.fileobj
                if key.data == "stdin":
                    try:
                        written = os.write(stream.fileno(), stdin[input_offset : input_offset + 65_536])
                    except BrokenPipeError:
                        written = 0
                        input_offset = len(stdin)
                    input_offset += written
                    if input_offset >= len(stdin):
                        streams.unregister(stream)
                        stream.close()
                    continue
                try:
                    chunk = os.read(stream.fileno(), min(65_536, max_output_bytes - captured + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    streams.unregister(stream)
                    stream.close()
                    continue
                room = max_output_bytes - captured
                if len(chunk) > room:
                    if room:
                        bounded = chunk[:room]
                        chunks[key.data].append(bounded)
                        if on_chunk is not None:
                            on_chunk(str(key.data), bounded)
                        captured += room
                    failure = "output_limit_exceeded"
                    break
                chunks[key.data].append(chunk)
                if on_chunk is not None:
                    on_chunk(str(key.data), chunk)
                captured += len(chunk)
            if failure is not None:
                break
        if failure is not None:
            _kill_process_group(process)
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()) if failure is None else 5)
        except subprocess.TimeoutExpired:
            failure = failure or "wall_clock_timeout"
            _kill_process_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    finally:
        streams.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()
    exit_code = process.returncode
    if failure is None and exit_code != 0:
        failure = "agent_exit_nonzero"
    return b"".join(chunks["stdout"]), b"".join(chunks["stderr"]), exit_code, failure


def _null_usage() -> dict[str, int | float | None]:
    return {
        "input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }


def _parse_usage(
    profile: str, stdout: bytes
) -> tuple[dict[str, int | float | None], dict[str, Any]]:
    """Extract only accounting metadata from a complete Claude final event."""

    usage = _null_usage()
    parser: dict[str, Any] = {
        "protocol": (
            "claude-stream-json-final-result-v1" if profile == "claude-code" else None
        ),
        "status": "unsupported" if profile != "claude-code" else "incomplete",
    }
    if profile != "claude-code":
        return usage, parser
    try:
        lines = [line for line in stdout.decode("utf-8").splitlines() if line.strip()]
        events = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError):
        parser["status"] = "invalid"
        return usage, parser
    if not events or any(not isinstance(event, Mapping) for event in events):
        parser["status"] = "invalid" if events else "incomplete"
        return usage, parser
    result_indices = [index for index, event in enumerate(events) if event.get("type") == "result"]
    if result_indices != [len(events) - 1]:
        return usage, parser
    result = events[-1]
    raw_usage = result.get("usage")
    cost = result.get("total_cost_usd")
    fields = {
        "input_tokens": raw_usage.get("input_tokens") if isinstance(raw_usage, Mapping) else None,
        "cache_creation_input_tokens": (
            raw_usage.get("cache_creation_input_tokens", 0)
            if isinstance(raw_usage, Mapping)
            else None
        ),
        "cache_read_input_tokens": (
            raw_usage.get("cache_read_input_tokens", 0)
            if isinstance(raw_usage, Mapping)
            else None
        ),
        "output_tokens": raw_usage.get("output_tokens") if isinstance(raw_usage, Mapping) else None,
    }
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in fields.values()
        )
        or not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        parser["status"] = "invalid"
        return usage, parser
    usage.update(fields)
    usage["cost_usd"] = float(cost)
    parser["status"] = "parsed"
    subtype = result.get("subtype")
    if isinstance(subtype, str):
        parser["result_subtype"] = subtype
    return usage, parser


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
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_budget_usd: float | None = None,
) -> dict[str, Any]:
    """Run or safely reuse one deterministic autonomous-agent session."""

    if profile not in _PROFILES:
        raise AgentSessionError("profile must be codex or claude-code")
    if (
        not stage.strip()
        or not model.strip()
        or not prompt
        or timeout_sec <= 0
        or not isinstance(max_output_bytes, int)
        or isinstance(max_output_bytes, bool)
        or not 1 <= max_output_bytes <= 256 * 1024 * 1024
    ):
        raise AgentSessionError("stage, model, prompt and timeout must be positive")
    if (
        max_budget_usd is None
        or isinstance(max_budget_usd, bool)
        or not isinstance(max_budget_usd, (int, float))
        or not math.isfinite(float(max_budget_usd))
        or not 0 < float(max_budget_usd) <= 100.0
    ):
        raise AgentSessionError("max_budget_usd must be a finite value in (0, 100]")
    budget = float(max_budget_usd)
    cwd = Path(workdir).resolve()
    if not cwd.is_dir() or cwd.is_symlink():
        raise AgentSessionError("workdir must be a real directory")
    child_env, route_digest = _session_env(profile, environ)
    requested = str(executable or ("codex" if profile == "codex" else "claude"))
    resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
    if not resolved or not Path(resolved).is_file():
        raise AgentSessionError("agent CLI executable is unavailable")
    command = _argv(
        profile,
        str(Path(resolved).resolve()),
        model.strip(),
        max_budget_usd=budget,
    )
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
        "timeout_sec": timeout_sec,
        "max_output_bytes": max_output_bytes,
        "max_budget_usd": budget,
        "budget_enforcement": (
            "claude-cli-max-budget-usd"
            if profile == "claude-code"
            else "unsupported-codex-cli"
        ),
    }
    session_id = _digest(identity).removeprefix("sha256:")[:32]
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = session_dir / "receipt.json"
    lock_path = session_dir / "session.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reused = _valid_reuse(receipt_path, session_id)
        if reused is not None:
            return {**reused, "receipt_path": str(receipt_path), "reused": True}

        started = time.monotonic()
        live_paths = {
            "stdout": session_dir / "stdout.live",
            "stderr": session_dir / "stderr.live",
        }
        with live_paths["stdout"].open("wb") as live_stdout, live_paths["stderr"].open(
            "wb"
        ) as live_stderr:
            live_streams = {"stdout": live_stdout, "stderr": live_stderr}

            def write_live(name: str, chunk: bytes) -> None:
                stream = live_streams[name]
                stream.write(chunk)
                stream.flush()

            stdout, stderr, exit_code, failure_class = _bounded_process(
                command,
                cwd=cwd,
                env=child_env,
                stdin=_stdin(profile, prompt),
                timeout_sec=timeout_sec,
                max_output_bytes=max_output_bytes,
                on_chunk=write_live,
            )
        usage, usage_parser = _parse_usage(profile, stdout)

        _atomic_bytes(session_dir / "stdout.bin", stdout)
        _atomic_bytes(session_dir / "stderr.bin", stderr)
        for path in live_paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        receipt: dict[str, Any] = {
            "schema_version": "orbenchlab.agent-session.receipt.v1",
            "session_id": session_id,
            "identity": identity,
            # Runtime-only paths support safe reuse checks. Public exporters
            # must continue to omit them, as they do for benchmark run roots.
            "workdir_runtime_path": str(cwd),
            "output_runtime_path": str(output_root),
            "input_tree_digest": input_tree_digest,
            "result_tree_digest": _tree_digest(cwd, exclude=output_root),
            "status": "completed" if failure_class is None else "failed",
            "failure_class": failure_class,
            "exit_code": exit_code,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "max_output_bytes": max_output_bytes,
            "captured_output_bytes": len(stdout) + len(stderr),
            "budget": {
                "max_budget_usd": budget,
                "enforcement": identity["budget_enforcement"],
                "hard_enforced_by_cli": profile == "claude-code",
            },
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
            "live_monitoring": {
                "during_execution": ["stdout.live", "stderr.live"],
                "completed": ["stdout.bin", "stderr.bin"],
                "hint_injection_supported": False,
            },
            "usage": usage,
            "usage_parser": usage_parser,
            "evidence_level": "E1-agent-session-process",
            "limitations": [
                "Agent completion is not static-gate, verifier, or Harbor evidence.",
                (
                    "The workdir is the process cwd and evidence boundary, not an OS filesystem "
                    "sandbox; enabled coding tools, especially Bash, retain the host account's "
                    "filesystem permissions. Run untrusted stages in an external container/worktree sandbox."
                ),
                "This process harness does not provide network isolation.",
                "Live trace files support monitoring, but this runner does not inject hints into an active checkpoint.",
            ],
        }
        receipt["receipt_digest"] = _digest(receipt)
        _atomic_json(receipt_path, receipt)
        return {**receipt, "receipt_path": str(receipt_path), "reused": False}


__all__ = ["AgentSessionError", "DEFAULT_MAX_OUTPUT_BYTES", "run_session"]
