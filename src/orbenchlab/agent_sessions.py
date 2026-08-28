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
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
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
_REDACTION = b"[REDACTED_PROVIDER_CREDENTIAL]"


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
    allow_bash: bool,
) -> list[str]:
    if profile == "codex":
        return [executable, "exec", "--json", "--model", model, "-"]
    coding_tools = "Read,Glob,Grep,Edit,Write" + (",Bash" if allow_bash else "")
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


def _binding_spec(value: Any) -> tuple[Path, tuple[str, ...]]:
    """Normalize a read-only path spec to (path, digest-excluded child names).

    A directory may declare harness-owned append-only children (for example
    ``factory-input/trusted``) whose contents are digest-bound separately; the
    parent's digest then stays stable while bundles are installed.
    """

    if isinstance(value, Mapping):
        excluded = tuple(sorted(str(name) for name in value.get("digest_exclude", ())))
        return Path(str(value["path"])), excluded
    return Path(value), ()


def _read_only_command(
    command: Sequence[str],
    *,
    cwd: Path,
    paths: Sequence[Any],
    hidden_paths: Sequence[str | Path] = (),
) -> tuple[list[str], dict[str, Any] | None]:
    """Wrap a command with immutable visible inputs and unreadable hidden paths."""

    if not paths and not hidden_paths:
        return list(command), None
    cwd = cwd.resolve()
    def bindings_for(values: Sequence[Any], *, label: str) -> list[dict[str, Any]]:
        rows = []
        for value in values:
            requested, digest_exclude = _binding_spec(value)
            if requested.is_symlink() or not requested.exists():
                raise AgentSessionError(f"{label} session path must exist and not be a symlink")
            resolved = requested.resolve()
            try:
                relative = resolved.relative_to(cwd).as_posix()
            except ValueError:
                raise AgentSessionError(f"{label} session path must be inside the workdir") from None
            row = {
                "path": relative,
                "kind": "directory" if resolved.is_dir() else "file",
                "content_digest": (
                    _tree_digest(resolved, exclude_children=digest_exclude)
                    if resolved.is_dir()
                    else _digest_bytes(resolved.read_bytes())
                ),
            }
            if digest_exclude:
                row["digest_exclude"] = list(digest_exclude)
            rows.append(row)
        if len({row["path"] for row in rows}) != len(rows):
            raise AgentSessionError(f"{label} session paths must be unique")
        return rows

    bindings = bindings_for(paths, label="read-only")
    hidden_bindings = bindings_for(hidden_paths, label="hidden")
    visible_names = {row["path"] for row in bindings}
    hidden_names = {row["path"] for row in hidden_bindings}
    if visible_names & hidden_names:
        raise AgentSessionError("session paths cannot be both visible and hidden")
    for left in visible_names:
        left_path = PurePosixPath(left)
        for right in hidden_names:
            right_path = PurePosixPath(right)
            if left_path in right_path.parents or right_path in left_path.parents:
                raise AgentSessionError("visible and hidden session paths may not overlap")
    resolved_paths = [cwd / row["path"] for row in bindings]
    resolved_hidden = [(cwd / row["path"], row["kind"]) for row in hidden_bindings]
    if sys.platform.startswith("linux"):
        sandbox = shutil.which("bwrap")
        if not sandbox:
            raise AgentSessionError("Bubblewrap is required for read-only factory inputs")
        sandbox_path = Path(sandbox).resolve()
        wrapped = [
            str(sandbox_path),
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(cwd),
            str(cwd),
            "--proc",
            "/proc",
            "--dev-bind",
            "/dev",
            "/dev",
        ]
        for path in resolved_paths:
            wrapped.extend(["--ro-bind", str(path), str(path)])
        for path, path_kind in resolved_hidden:
            if path_kind == "directory":
                wrapped.extend(["--tmpfs", str(path), "--remount-ro", str(path)])
            else:
                wrapped.extend(["--ro-bind", "/dev/null", str(path)])
        # The private /tmp tmpfs would otherwise shadow an executable that lives
        # under /tmp (e.g. a pytest tmp_path fixture). Re-materialize the exact
        # executable path read-only so callers never need to stage binaries
        # outside /tmp to be visible inside the sandbox.
        executable_path = Path(command[0])
        if executable_path.is_file() and not executable_path.is_symlink():
            executable_resolved = str(executable_path.resolve())
            wrapped.extend(["--ro-bind", executable_resolved, executable_resolved])
        wrapped.extend(["--chdir", str(cwd), "--", *command])
        kind = "bubblewrap-read-only-bindings-v1"
        policy = "root-ro-workdir-rw-protected-ro-private-tmp-v1"
    elif sys.platform == "darwin":
        sandbox = shutil.which("sandbox-exec")
        if not sandbox:
            raise AgentSessionError("sandbox-exec is required for read-only factory inputs")
        sandbox_path = Path(sandbox).resolve()
        quoted = [str(path).replace('"', '\\"') for path in resolved_paths]
        hidden_quoted = [str(path).replace('"', '\\"') for path, _ in resolved_hidden]
        writable = str(cwd).replace('"', '\\"')
        profile = (
            "(version 1)\n(allow default)\n(deny file-write*)\n"
            f'(allow file-write* (subpath "{writable}"))\n'
            + "\n".join(
                f'(deny file-write* (subpath "{path}"))' for path in quoted
            )
            + "\n"
            + "\n".join(
                f'(deny file-read* file-write* (subpath "{path}"))'
                for path in hidden_quoted
            )
        )
        wrapped = [str(sandbox_path), "-p", profile, *command]
        kind = "sandbox-exec-read-only-subpaths-v1"
        policy = "default-deny-write-workdir-rw-protected-ro-v1"
    else:
        raise AgentSessionError("this platform has no supported read-only factory sandbox")
    contract = {
        "kind": kind,
        "policy": policy,
        "executable_digest": _digest_bytes(sandbox_path.read_bytes()),
        "read_only_bindings": bindings,
        "hidden_bindings": hidden_bindings,
        "hard_enforced": True,
    }
    return wrapped, contract


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


def _tree_digest(
    root: Path,
    *,
    exclude: Path | None = None,
    exclude_children: Sequence[str] = (),
) -> str:
    rows: list[tuple[str, str]] = []
    excluded = {str(name) for name in exclude_children}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if exclude is not None and (path == exclude or exclude in path.parents):
            continue
        if relative.parts and relative.parts[0] in excluded:
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


class _StreamingSecretRedactor:
    """Redact exact credentials without leaking chunk-boundary prefixes."""

    def __init__(self, secrets: Sequence[bytes]):
        self.secrets = tuple(
            sorted({value for value in secrets if len(value) >= 8}, key=len, reverse=True)
        )
        self.pending = b""
        self.redacted = False
        self.maximum = max((len(value) for value in self.secrets), default=1)

    def feed(self, chunk: bytes, *, final: bool = False) -> bytes:
        data = self.pending + chunk
        held = 0
        if not final and self.secrets:
            for length in range(1, min(self.maximum - 1, len(data)) + 1):
                suffix = data[-length:]
                if any(secret.startswith(suffix) for secret in self.secrets):
                    held = length
        limit = len(data) - held
        output = bytearray()
        index = 0
        while index < limit:
            match = next(
                (secret for secret in self.secrets if data.startswith(secret, index)),
                None,
            )
            if match is not None:
                output.extend(_REDACTION)
                self.redacted = True
                index += len(match)
            else:
                output.append(data[index])
                index += 1
        self.pending = data[index:]
        if final:
            while self.pending:
                match = next(
                    (
                        secret
                        for secret in self.secrets
                        if self.pending.startswith(secret)
                    ),
                    None,
                )
                if match is not None:
                    output.extend(_REDACTION)
                    self.redacted = True
                    self.pending = self.pending[len(match) :]
                else:
                    output.append(self.pending[0])
                    self.pending = self.pending[1:]
        return bytes(output)


def _redact_bytes(value: bytes, secrets: Sequence[bytes]) -> tuple[bytes, bool]:
    redactor = _StreamingSecretRedactor(secrets)
    return redactor.feed(value, final=True), redactor.redacted


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
    read_only_paths: Sequence[str | Path] = (),
    hidden_paths: Sequence[str | Path] = (),
    allow_bash: bool = True,
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
    secret_values = tuple(
        str(value).encode()
        for name, value in environ.items()
        if ("KEY" in name.upper() or "TOKEN" in name.upper()) and str(value)
    )
    requested = str(executable or ("codex" if profile == "codex" else "claude"))
    resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
    if not resolved or not Path(resolved).is_file():
        raise AgentSessionError("agent CLI executable is unavailable")
    agent_command = _argv(
        profile,
        str(Path(resolved).resolve()),
        model.strip(),
        max_budget_usd=budget,
        allow_bash=allow_bash,
    )
    command, filesystem_sandbox = _read_only_command(
        agent_command,
        cwd=cwd,
        paths=read_only_paths,
        hidden_paths=hidden_paths,
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
        "argv_template": agent_command[1:],
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
        "filesystem_sandbox": filesystem_sandbox,
        "bash_tool_enabled": bool(allow_bash),
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
            live_redactors = {
                name: _StreamingSecretRedactor(secret_values)
                for name in live_streams
            }

            def write_live(name: str, chunk: bytes) -> None:
                stream = live_streams[name]
                stream.write(live_redactors[name].feed(chunk))
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
            for name, stream in live_streams.items():
                stream.write(live_redactors[name].feed(b"", final=True))
                stream.flush()
        usage, usage_parser = _parse_usage(profile, stdout)
        stdout, stdout_redacted = _redact_bytes(stdout, secret_values)
        stderr, stderr_redacted = _redact_bytes(stderr, secret_values)

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
            "provider_credential_redacted": stdout_redacted or stderr_redacted,
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
                    "Declared read-only inputs are protected by an inherited OS sandbox, but other "
                    "host-account paths and the writable workdir are not isolated."
                    if filesystem_sandbox is not None
                    else "The workdir is the process cwd and evidence boundary, not an OS filesystem "
                    "sandbox; enabled coding tools, especially Bash, retain the host account's "
                    "filesystem permissions. Run untrusted stages in an external container/worktree sandbox."
                ),
                (
                    "Bash is disabled; semantic factory agents cannot read the provider credential "
                    "from a child shell or initiate arbitrary shell-network egress."
                    if not allow_bash
                    else "Bash inherits the agent process environment; use only on an externally isolated worker."
                ),
                "This process harness does not provide network isolation.",
                "Live trace files support monitoring, but this runner does not inject hints into an active checkpoint.",
            ],
        }
        receipt["receipt_digest"] = _digest(receipt)
        _atomic_json(receipt_path, receipt)
        return {**receipt, "receipt_path": str(receipt_path), "reused": False}


__all__ = ["AgentSessionError", "DEFAULT_MAX_OUTPUT_BYTES", "run_session"]
