"""Monitored agent sessions with same-session hint injection.

The Claude Code CLI accepts a stream of user messages on stdin when run with
``--input-format stream-json``.  That gives the trusted harness a real
mid-session intervention channel: it can watch the event stream, and at a
fixed, predeclared policy point append one hint message to the *same running
session* instead of restarting the agent.  This module owns that channel and
its evidence:

- a machine-readable capability receipt (fail-closed: Codex's ``exec`` mode
  and Harbor container trials have no such channel and are recorded as
  unsupported rather than silently downgraded);
- a bounded monitored session whose receipt records every event's arrival
  time, the exact injection instant, pre/post-checkpoint event counts and
  whether the runtime *confirmed* the injected turn;
- a controlled study runner that pairs injection sessions with no-injection
  controls under one frozen contract and grades the result conservatively:
  the E4 label is only emitted when the capability is supported, every
  treatment injection was confirmed by the runtime, and both arms have
  enough verifier-grounded trials.

A restart-with-hint arm is *not* this; it stays E3 elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import agent_sessions
from .agent_sessions import (
    _argv,
    _kill_process_group,
    _session_env,
    _StreamingSecretRedactor,
    _tree_digest,
)
from .core.errors import ORBenchError


class SessionInterventionError(ORBenchError):
    exit_code = 9


CAPABILITY_SCHEMA_VERSION = "orbenchlab.session-intervention-capability.v1"
SESSION_SCHEMA_VERSION = "orbenchlab.intervention-session.v1"
STUDY_SCHEMA_VERSION = "orbenchlab.intervention-study.v1"
INJECTION_CONTRACT = "claude-stream-json-multi-turn-stdin-v1"
_TRIGGER_KINDS = frozenset({"assistant-event-index", "elapsed-sec", "stdout-pattern"})
_MAX_EVENTS = 20_000


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def probe_capability(*, profile: str, runtime: str = "agent-session") -> dict[str, Any]:
    """Return the fail-closed injection capability for one execution path."""

    if runtime not in {"agent-session", "harbor-trial"}:
        raise SessionInterventionError("unknown intervention runtime")
    receipt: dict[str, Any] = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "profile": profile,
        "runtime": runtime,
        "same_session_hint_injection": False,
        "injection_contract": None,
        "per_session_confirmation_required": True,
        "reason": None,
    }
    if runtime == "harbor-trial":
        receipt["reason"] = (
            "Harbor trials run the agent inside a task container as independent "
            "full restarts; the harness has no stdin channel into the running "
            "agent, so same-checkpoint injection is unsupported (restart-with-hint "
            "remains E3)."
        )
    elif profile == "codex":
        receipt["reason"] = (
            "codex exec reads one complete prompt and exposes no mid-session "
            "user-message channel."
        )
    elif profile == "claude-code":
        receipt["same_session_hint_injection"] = True
        receipt["injection_contract"] = INJECTION_CONTRACT
        receipt["reason"] = (
            "claude --print --input-format stream-json accepts additional user "
            "messages on the open stdin of the same session; every injection "
            "must still be confirmed by a runtime event after the hint."
        )
    else:
        receipt["reason"] = f"unknown agent profile {profile!r}"
    receipt["receipt_digest"] = _digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    return receipt


def _validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    trigger = policy.get("trigger")
    hint_text = policy.get("hint_text")
    hint_level = policy.get("hint_level")
    if (
        not isinstance(trigger, Mapping)
        or trigger.get("kind") not in _TRIGGER_KINDS
        or not isinstance(hint_text, str)
        or not hint_text.strip()
        or len(hint_text.encode()) > 64_000
        or not isinstance(hint_level, int)
        or isinstance(hint_level, bool)
        or not 1 <= hint_level <= 5
    ):
        raise SessionInterventionError(
            "intervention policy requires a known trigger, hint_level 1..5 and bounded hint_text"
        )
    kind = str(trigger["kind"])
    value = trigger.get("value")
    if kind == "assistant-event-index":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SessionInterventionError("assistant-event-index trigger needs value >= 1")
    elif kind == "elapsed-sec":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise SessionInterventionError("elapsed-sec trigger needs a positive value")
    else:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise SessionInterventionError("stdout-pattern trigger needs a bounded string")
    return {
        "trigger": {"kind": kind, "value": value},
        "hint_level": hint_level,
        "hint_text": hint_text,
    }


class _EventMonitor:
    """Incrementally parse stream-json stdout lines into a timed event log."""

    def __init__(self) -> None:
        self.buffer = b""
        self.events: list[dict[str, Any]] = []
        self.assistant_events = 0
        self.result_seen = False
        self.truncated = False

    def feed(self, chunk: bytes, at_sec: float) -> list[dict[str, Any]]:
        self.buffer += chunk
        fresh: list[dict[str, Any]] = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                value = json.loads(text)
                event_type = (
                    str(value.get("type")) if isinstance(value, Mapping) else "non-object"
                )
            except json.JSONDecodeError:
                value, event_type = None, "non-json"
            if event_type == "assistant":
                self.assistant_events += 1
            if event_type == "result":
                self.result_seen = True
            event = {
                "index": len(self.events),
                "at_sec": round(at_sec, 4),
                "type": event_type,
                "raw_text": text if value is None else None,
            }
            if isinstance(value, Mapping) and isinstance(value.get("subtype"), str):
                event["subtype"] = value["subtype"]
            if len(self.events) < _MAX_EVENTS:
                self.events.append(event)
            else:
                self.truncated = True
            fresh.append({**event, "line_text": text})
        return fresh


def _trigger_fired(
    policy: Mapping[str, Any],
    monitor: _EventMonitor,
    fresh: Sequence[Mapping[str, Any]],
    elapsed: float,
) -> bool:
    trigger = policy["trigger"]
    kind = trigger["kind"]
    if kind == "assistant-event-index":
        return monitor.assistant_events >= int(trigger["value"])
    if kind == "elapsed-sec":
        return elapsed >= float(trigger["value"])
    pattern = str(trigger["value"])
    return any(pattern in str(event.get("line_text", "")) for event in fresh)


def run_intervention_session(
    *,
    profile: str,
    stage: str,
    model: str,
    prompt: str,
    workdir: str | Path,
    out: str | Path,
    timeout_sec: float,
    environ: Mapping[str, str],
    max_budget_usd: float,
    policy: Mapping[str, Any] | None = None,
    executable: str | Path | None = None,
    max_output_bytes: int = agent_sessions.DEFAULT_MAX_OUTPUT_BYTES,
    trigger_timeout_sec: float | None = None,
    allow_bash: bool = True,
) -> dict[str, Any]:
    """Run one monitored session, optionally injecting a hint mid-session.

    ``policy=None`` runs the identically instrumented control arm: same
    runner, same monitoring, stdin closed after the initial message.
    """

    capability = probe_capability(profile=profile, runtime="agent-session")
    if policy is not None and not capability["same_session_hint_injection"]:
        raise SessionInterventionError(
            f"profile {profile!r} does not support same-session injection: "
            f"{capability['reason']}"
        )
    checked_policy = _validate_policy(policy) if policy is not None else None
    if (
        not stage.strip()
        or not model.strip()
        or not prompt
        or timeout_sec <= 0
        or not 0 < float(max_budget_usd) <= 100
        or not 1 <= max_output_bytes <= 256 * 1024 * 1024
    ):
        raise SessionInterventionError("stage, model, prompt and bounds must be valid")
    cwd = Path(workdir).resolve()
    if not cwd.is_dir() or cwd.is_symlink():
        raise SessionInterventionError("workdir must be a real directory")
    child_env, route_digest = _session_env(profile, environ)
    secrets = tuple(
        str(value).encode()
        for name, value in environ.items()
        if ("KEY" in name.upper() or "TOKEN" in name.upper()) and str(value)
    )
    requested = str(executable or "claude")
    resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
    if not resolved or not Path(resolved).is_file():
        raise SessionInterventionError("agent CLI executable is unavailable")
    command = _argv(
        profile,
        str(Path(resolved).resolve()),
        model.strip(),
        max_budget_usd=float(max_budget_usd),
        allow_bash=allow_bash,
    )
    initial = agent_sessions._stdin(profile, prompt)
    hint_bytes = b""
    if checked_policy is not None:
        hint_bytes = agent_sessions._stdin(profile, checked_policy["hint_text"])
    trigger_deadline = (
        float(trigger_timeout_sec)
        if trigger_timeout_sec is not None
        else max(1.0, float(timeout_sec) * 0.8)
    )
    identity = {
        "schema_version": "orbenchlab.intervention-session.identity.v1",
        "profile": profile,
        "stage": stage.strip(),
        "model": model.strip(),
        "prompt_digest": _digest_bytes(prompt.encode()),
        "route_digest": route_digest,
        "workdir_binding": _digest(str(cwd)),
        "executable_digest": _digest_bytes(Path(resolved).read_bytes()),
        "timeout_sec": float(timeout_sec),
        "trigger_timeout_sec": trigger_deadline,
        "max_output_bytes": max_output_bytes,
        "max_budget_usd": float(max_budget_usd),
        "injection_contract": (
            INJECTION_CONTRACT if checked_policy is not None else None
        ),
        "policy": checked_policy,
        "policy_digest": _digest(checked_policy) if checked_policy else None,
        "capability_digest": capability["receipt_digest"],
    }
    session_id = _digest(identity).removeprefix("sha256:")[:32]
    session_dir = Path(out).resolve() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    monitor = _EventMonitor()
    injection: dict[str, Any] = {
        "requested": checked_policy is not None,
        "fired": False,
        "fired_at_sec": None,
        "pre_injection_events": None,
        "pre_injection_assistant_events": None,
        "post_injection_events": 0,
        "injection_confirmed": False,
        "stdin_closed_reason": None,
        "hint_digest": _digest_bytes(hint_bytes) if hint_bytes else None,
    }
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(child_env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        raise SessionInterventionError("agent CLI launch failed") from None
    assert process.stdin and process.stdout and process.stderr
    for stream in (process.stdin, process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdin_open = True
    pending = initial
    pending_offset = 0
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    captured = 0
    failure: str | None = None
    deadline = started + float(timeout_sec)

    def close_stdin(reason: str) -> None:
        nonlocal stdin_open
        if stdin_open:
            stdin_open = False
            injection["stdin_closed_reason"] = injection["stdin_closed_reason"] or reason
            try:
                process.stdin.close()
            except OSError:
                pass

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                failure = "wall_clock_timeout"
                break
            if process.poll() is not None and not streams.get_map():
                break
            # Drive stdin outside the selector: write whatever is pending.
            if stdin_open and pending_offset < len(pending):
                try:
                    written = os.write(
                        process.stdin.fileno(),
                        pending[pending_offset : pending_offset + 65_536],
                    )
                    pending_offset += written
                except BlockingIOError:
                    pass
                except (BrokenPipeError, OSError):
                    close_stdin("agent-closed-stdin")
            if stdin_open and pending_offset >= len(pending):
                if checked_policy is None:
                    close_stdin("no-injection-control")
                elif injection["fired"]:
                    close_stdin("hint-delivered")
            ready = streams.select(0.05)
            fresh_events: list[Mapping[str, Any]] = []
            for key, _ in ready:
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 65_536)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    streams.unregister(stream)
                    stream.close()
                    continue
                room = max_output_bytes - captured
                if len(chunk) > room:
                    chunks[key.data].append(chunk[:room])
                    captured += room
                    failure = "output_limit_exceeded"
                    break
                chunks[key.data].append(chunk)
                captured += len(chunk)
                if key.data == "stdout":
                    fresh_events = monitor.feed(chunk, time.monotonic() - started)
                    if injection["fired"]:
                        injection["post_injection_events"] += len(fresh_events)
                        if any(
                            event.get("type") in {"assistant", "result", "user"}
                            for event in fresh_events
                        ):
                            injection["injection_confirmed"] = True
            if failure is not None:
                break
            elapsed = time.monotonic() - started
            if checked_policy is not None and stdin_open and not injection["fired"]:
                if monitor.result_seen:
                    close_stdin("completed-before-trigger")
                elif elapsed >= trigger_deadline:
                    close_stdin("trigger-timeout")
                elif pending_offset >= len(pending) and _trigger_fired(
                    checked_policy, monitor, fresh_events, elapsed
                ):
                    injection["fired"] = True
                    injection["fired_at_sec"] = round(elapsed, 4)
                    injection["pre_injection_events"] = len(monitor.events)
                    injection["pre_injection_assistant_events"] = monitor.assistant_events
                    pending = hint_bytes
                    pending_offset = 0
            if process.poll() is not None and not streams.get_map():
                break
        if failure is not None:
            _kill_process_group(process)
        close_stdin("session-end")
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
        for stream in (process.stdout, process.stderr):
            if not stream.closed:
                stream.close()
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
    exit_code = process.returncode
    if failure is None and exit_code != 0:
        failure = "agent_exit_nonzero"
    stdout_raw = b"".join(chunks["stdout"])
    stderr_raw = b"".join(chunks["stderr"])
    stdout, redacted_out = agent_sessions._redact_bytes(stdout_raw, secrets)
    stderr, redacted_err = agent_sessions._redact_bytes(stderr_raw, secrets)
    agent_sessions._atomic_bytes(session_dir / "stdout.bin", stdout)
    agent_sessions._atomic_bytes(session_dir / "stderr.bin", stderr)
    redactor = _StreamingSecretRedactor(secrets)
    events_path = session_dir / "events.jsonl"
    with events_path.open("wb") as stream:
        for event in monitor.events:
            line = json.dumps(event, sort_keys=True, ensure_ascii=False).encode() + b"\n"
            stream.write(redactor.feed(line))
        stream.write(redactor.feed(b"", final=True))
    intervention_class = (
        "same-session-continuation"
        if injection["fired"] and injection["injection_confirmed"]
        else "none"
    )
    receipt: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "identity": identity,
        "status": "completed" if failure is None else "failed",
        "failure_class": failure,
        "exit_code": exit_code,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "event_count": len(monitor.events),
        "assistant_event_count": monitor.assistant_events,
        "events_truncated": monitor.truncated,
        "events_digest": _digest_bytes(events_path.read_bytes()),
        "stdout_digest": _digest_bytes(stdout),
        "stderr_digest": _digest_bytes(stderr),
        "provider_credential_redacted": redacted_out or redacted_err,
        "injection": injection,
        "intervention_class": intervention_class,
        "capability": capability,
        "evidence_level": "E1-session-process",
        "limitations": [
            "One monitored session is process evidence, not a causal claim.",
            "E4 requires a controlled study with confirmed injections and no-injection controls.",
            "The verifier, not this receipt, grades task outcomes.",
        ],
    }
    receipt["receipt_digest"] = _digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _atomic_json(session_dir / "receipt.json", receipt)
    return {**receipt, "receipt_path": str(session_dir / "receipt.json")}


def _run_verifier(
    argv: Sequence[str], *, cwd: Path, timeout_sec: float, max_output_bytes: int
) -> dict[str, Any]:
    stdout, stderr, exit_code, failure = agent_sessions._bounded_process(
        [str(item) for item in argv],
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", "")},
        stdin=b"",
        timeout_sec=timeout_sec,
        max_output_bytes=max_output_bytes,
    )
    status = "pass" if failure is None and exit_code == 0 else "fail"
    if failure not in (None, "agent_exit_nonzero"):
        status = "infra_error"
    return {
        "status": status,
        "exit_code": exit_code,
        "failure_class": failure,
        "stdout_digest": _digest_bytes(stdout),
        "stderr_digest": _digest_bytes(stderr),
    }


def run_intervention_study(
    *,
    profile: str,
    model: str,
    prompt: str,
    template_workdir: str | Path,
    out: str | Path,
    environ: Mapping[str, str],
    verifier_argv: Sequence[str],
    policy: Mapping[str, Any],
    n_control: int = 3,
    n_treatment: int = 3,
    timeout_sec: float = 600.0,
    max_budget_usd: float = 1.0,
    executable: str | Path | None = None,
    verifier_timeout_sec: float = 300.0,
    max_output_bytes: int = agent_sessions.DEFAULT_MAX_OUTPUT_BYTES,
) -> dict[str, Any]:
    """Run a controlled same-session intervention experiment with verifier outcomes."""

    if n_control < 1 or n_treatment < 1 or not verifier_argv:
        raise SessionInterventionError("study needs both arms and a verifier command")
    template = Path(template_workdir).resolve()
    if template.is_symlink() or not template.is_dir():
        raise SessionInterventionError("template workdir must be a real directory")
    checked_policy = _validate_policy(policy)
    capability = probe_capability(profile=profile, runtime="agent-session")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not capability["same_session_hint_injection"]:
        receipt = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "status": "unsupported-capability",
            "capability": capability,
            "policy": checked_policy,
            "evidence_level": "E0-unsupported",
            "trials": [],
            "limitations": [
                "The selected profile/runtime has no same-session injection channel; "
                "no sessions were run and no E-level above E0 may be claimed.",
            ],
        }
        receipt["receipt_digest"] = _digest(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        _atomic_json(root / "intervention-study.json", receipt)
        return receipt
    trials: list[dict[str, Any]] = []
    arms = [("control", index, None) for index in range(1, n_control + 1)] + [
        ("treatment", index, checked_policy) for index in range(1, n_treatment + 1)
    ]
    for arm, index, arm_policy in arms:
        trial_dir = root / "trials" / f"{arm}-{index:02d}"
        workdir = trial_dir / "workdir"
        if not workdir.exists():
            trial_dir.mkdir(parents=True, exist_ok=True)
            temporary = trial_dir / f".workdir.{uuid.uuid4().hex}.tmp"
            shutil.copytree(template, temporary, symlinks=False)
            os.replace(temporary, workdir)
        session = run_intervention_session(
            profile=profile,
            stage=f"intervention-study/{arm}/{index:02d}",
            model=model,
            prompt=prompt,
            workdir=workdir,
            out=trial_dir / "sessions",
            timeout_sec=timeout_sec,
            environ=environ,
            max_budget_usd=max_budget_usd,
            policy=arm_policy,
            executable=executable,
            max_output_bytes=max_output_bytes,
        )
        verifier = _run_verifier(
            verifier_argv,
            cwd=workdir,
            timeout_sec=verifier_timeout_sec,
            max_output_bytes=max_output_bytes,
        )
        trials.append(
            {
                "arm": arm,
                "index": index,
                "session_id": session["session_id"],
                "session_receipt_digest": session["receipt_digest"],
                "session_status": session["status"],
                "intervention_class": session["intervention_class"],
                "injection": session["injection"],
                "verifier": verifier,
                "workdir_tree_digest": _tree_digest(workdir),
            }
        )

    def arm_summary(name: str) -> dict[str, Any]:
        rows = [row for row in trials if row["arm"] == name]
        graded = [row for row in rows if row["verifier"]["status"] in {"pass", "fail"}]
        passed = sum(row["verifier"]["status"] == "pass" for row in graded)
        return {
            "n": len(rows),
            "graded": len(graded),
            "passed": passed,
            "pass_rate": round(passed / len(graded), 6) if graded else None,
            "infra_errors": sum(
                row["verifier"]["status"] == "infra_error" for row in rows
            ),
        }

    treatment_rows = [row for row in trials if row["arm"] == "treatment"]
    all_confirmed = bool(treatment_rows) and all(
        row["intervention_class"] == "same-session-continuation"
        for row in treatment_rows
    )
    graded_complete = all(
        row["verifier"]["status"] in {"pass", "fail"} for row in trials
    )
    if all_confirmed and graded_complete and n_control >= 3 and n_treatment >= 3:
        evidence_level = "E4-controlled-same-session-intervention"
    elif all_confirmed and graded_complete:
        evidence_level = "E3-underpowered-same-session-intervention"
    else:
        evidence_level = "E1-incomplete-intervention-evidence"
    receipt = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "status": "completed",
        "capability": capability,
        "policy": checked_policy,
        "policy_digest": _digest(checked_policy),
        "model": model,
        "prompt_digest": _digest_bytes(prompt.encode()),
        "template_tree_digest": _tree_digest(template),
        "verifier_argv": [str(item) for item in verifier_argv],
        "arms": {"control": arm_summary("control"), "treatment": arm_summary("treatment")},
        "trials": trials,
        "all_treatment_injections_confirmed": all_confirmed,
        "evidence_level": evidence_level,
        "limitations": [
            "Injection lands at a turn boundary of the same session; it is a "
            "same-session continuation, not a mid-token checkpoint restore.",
            "Effect claims require the paired no-injection control arm shown here.",
        ],
    }
    receipt["receipt_digest"] = _digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _atomic_json(root / "intervention-study.json", receipt)
    return receipt


__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "INJECTION_CONTRACT",
    "SESSION_SCHEMA_VERSION",
    "STUDY_SCHEMA_VERSION",
    "SessionInterventionError",
    "probe_capability",
    "run_intervention_session",
    "run_intervention_study",
]
