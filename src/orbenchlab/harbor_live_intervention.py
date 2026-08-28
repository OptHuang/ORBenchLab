"""Harbor-native, same-session *live* intervention over the Claude stream-json
control protocol.

The stock one-shot runner (and the earlier ``session_interventions`` multi-turn
runner) cannot establish an E4 same-checkpoint causal claim: naively queueing a
user message while a tool is running does not interrupt the model at a
checkpoint.  This module drives a single persistent Claude ``stream-json``
session and performs the real control handshake:

    observe an assistant tool_use / target event
      -> send a ``control_request`` ``interrupt``
      -> require a ``control_response`` success ack
      -> require the interrupted ``result`` boundary
      -> verify no tool is still in flight (quiescence)
      -> freeze a snapshot (workspace + raw-stream prefix digests)
      -> send the hint as a NEW user turn on the SAME session
      -> require the hint to be replayed, then continue to the final result

Every step is journalled with an ``intervention_id`` so a crash/retry reuses a
completed journal instead of injecting twice.  The transport is injectable:
tests drive a faithful fake emitter subprocess speaking the same protocol, and
the identical state machine runs the real Claude CLI on the execution host.

This module owns only the protocol, the journal and the ATIF projection.  The
credential-safe transport (host-side relay) and the container tool proxy are
supplied by the caller; the secret must never reach this process's child argv,
env, or any journalled byte, which the caller-provided ``secret_values`` scrub
enforces here as defence in depth.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .core.errors import ORBenchError


class LiveInterventionError(ORBenchError):
    exit_code = 8


LIVE_SCHEMA_VERSION = "orbenchlab.live-intervention.v1"
ATIF_SCHEMA_VERSION = "orbenchlab.live-intervention.atif.v1"
CONTROL_CONTRACT = "claude-stream-json-interrupt-ack-result-hint-v1"

# A single stream-json session must never exceed these safety bounds.
_MAX_EVENTS = 20000
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def _tree_digest(paths: Sequence[Path]) -> str:
    """Order-independent digest of the concatenated file contents under paths."""

    hasher = hashlib.sha256()
    files: list[Path] = []
    for root in paths:
        root = Path(root)
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    for file in sorted(files, key=lambda p: str(p)):
        try:
            payload = file.read_bytes()
        except OSError:
            payload = b"<unreadable>"
        hasher.update(str(file).encode())
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(payload).digest())
    return "sha256:" + hasher.hexdigest()


def _scrub(text: str, secret_values: Sequence[str]) -> str:
    for secret in secret_values:
        if secret:
            text = text.replace(secret, "[REDACTED_SECRET]")
    return text


# ---------------------------------------------------------------------------
# Policy


@dataclass(frozen=True)
class Trigger:
    """When to interrupt the model. ``kind`` is one of:

    - ``tool-use``: the first assistant ``tool_use`` block (optionally matching
      ``value`` as the tool name; empty value matches any tool).
    - ``assistant-index``: the Nth assistant event (``value`` is the count).
    - ``stdout-pattern``: the first event whose raw line contains ``value``.
    """

    kind: str
    value: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"tool-use", "assistant-index", "stdout-pattern"}:
            raise LiveInterventionError(f"unknown trigger kind {self.kind!r}")
        if self.kind == "assistant-index" and int(self.value or 0) < 1:
            raise LiveInterventionError("assistant-index trigger needs a positive value")


@dataclass(frozen=True)
class InterventionPolicy:
    """A baseline or L1/L2/L3 intervention policy.

    ``level`` ``baseline`` runs the identical instrumented session with no
    interrupt or hint.  ``L1``/``L2``/``L3`` differ only in the ``hint_text``
    (increasing specificity); each must carry a unique ``hint_marker`` echoed
    by the model so replay/effect can be verified.
    """

    level: str
    hint_text: str = ""
    hint_marker: str = ""
    trigger: Trigger = field(default_factory=lambda: Trigger("tool-use"))

    def __post_init__(self) -> None:
        if self.level not in {"baseline", "L1", "L2", "L3"}:
            raise LiveInterventionError(f"unknown intervention level {self.level!r}")
        if self.level != "baseline":
            if not self.hint_text.strip() or not self.hint_marker.strip():
                raise LiveInterventionError("an intervention level needs hint_text and hint_marker")
            if self.hint_marker not in self.hint_text:
                raise LiveInterventionError("hint_text must contain the hint_marker")


# ---------------------------------------------------------------------------
# Transport abstraction


class ClaudeProc:
    """Minimal async process handle the state machine drives.

    The default implementation wraps ``asyncio.create_subprocess_exec``; tests
    inject a fake emitter subprocess with the same interface.
    """

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def send(self, payload: Mapping[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await self._proc.stdin.drain()

    async def readline(self, timeout: float) -> bytes:
        assert self._proc.stdout is not None
        return await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)

    def close_stdin(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            self._proc.stdin.close()

    async def wait(self, timeout: float) -> int | None:
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._proc.kill()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        return self._proc.returncode

    async def read_stderr(self, limit: int = 8192) -> bytes:
        if self._proc.stderr is None:
            return b""
        try:
            return await asyncio.wait_for(self._proc.stderr.read(limit), timeout=2)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return b""


SpawnFn = Callable[[], Awaitable[ClaudeProc]]


def _default_spawn(argv: Sequence[str], env: Mapping[str, str], cwd: str | Path) -> SpawnFn:
    async def spawn() -> ClaudeProc:
        proc = await asyncio.create_subprocess_exec(
            *[str(part) for part in argv],
            cwd=str(cwd),
            env={str(k): str(v) for k, v in env.items()},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return ClaudeProc(proc)

    return spawn


# ---------------------------------------------------------------------------
# Protocol helpers


def _user_event(text: str) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
        "session_id": "default",
    }


def _interrupt_request(request_id: str) -> dict[str, Any]:
    return {
        "type": "control_request",
        "request_id": request_id,
        "request": {"subtype": "interrupt"},
    }


def _blocks(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, list):
            return [block for block in content if isinstance(block, Mapping)]
    return []


def _trigger_matches(trigger: Trigger, event: Mapping[str, Any], raw: str, assistant_count: int) -> bool:
    kind = event.get("type")
    if trigger.kind == "assistant-index":
        return kind == "assistant" and assistant_count >= int(trigger.value)
    if trigger.kind == "stdout-pattern":
        return trigger.value in raw
    # tool-use
    if kind != "assistant":
        return False
    for block in _blocks(event):
        if block.get("type") == "tool_use":
            if not trigger.value or block.get("name") == trigger.value:
                return True
    return False


# ---------------------------------------------------------------------------
# Core state machine


async def _drive_session(
    *,
    spawn: SpawnFn,
    policy: InterventionPolicy,
    initial_prompt: str,
    intervention_id: str,
    timeout_sec: float,
    read_timeout_sec: float,
    quiescence_grace_sec: float,
    snapshot_paths: Sequence[Path],
    secret_values: Sequence[str],
    raw_journal_path: Path,
) -> dict[str, Any]:
    proc = await spawn()
    start_wall = time.time()
    started = time.monotonic()
    deadline = started + float(timeout_sec)

    assistant_count = 0
    result_count = 0
    events: list[dict[str, Any]] = []
    truncated = False
    cli_session_ids: list[str] = []
    pending_tool_use: set[str] = set()
    interrupt_request_id = f"orbench-intv-{intervention_id}"
    interrupt_sent = False
    interrupt_acked = False
    ack_event_index: int | None = None
    boundary_index: int | None = None
    hint_sent = False
    hint_replayed = False
    snapshot: dict[str, Any] | None = None
    quiescence_ok: bool | None = None
    error: str | None = None
    raw_bytes = 0
    is_intervention = policy.level != "baseline"

    raw_journal_path.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = raw_journal_path.open("wb")
    try:
        await proc.send(_user_event(initial_prompt))
        while True:
            if time.monotonic() > deadline:
                error = error or "wall_clock_timeout"
                break
            try:
                line = await proc.readline(timeout=min(read_timeout_sec, max(0.1, deadline - time.monotonic())))
            except asyncio.TimeoutError:
                error = error or "stream_read_timeout"
                break
            if not line:
                break
            if raw_bytes < _MAX_JOURNAL_BYTES:
                raw_handle.write(line)
                raw_handle.flush()
                raw_bytes += len(line)
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            kind = event.get("type")
            sid = event.get("session_id")
            if kind in {"system", "assistant", "result"} and isinstance(sid, str) and sid and sid != "default":
                if sid not in cli_session_ids:
                    cli_session_ids.append(sid)
            if kind == "assistant":
                assistant_count += 1
                for block in _blocks(event):
                    if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                        pending_tool_use.add(block["id"])
            if kind in {"user", "tool_result"}:
                for block in _blocks(event):
                    if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                        pending_tool_use.discard(block["tool_use_id"])
            # Record a compact event row.
            if len(events) < _MAX_EVENTS:
                row: dict[str, Any] = {"index": len(events), "type": kind}
                if isinstance(event.get("subtype"), str):
                    row["subtype"] = event["subtype"]
                if isinstance(sid, str) and sid:
                    row["session_id"] = sid
                events.append(row)
            else:
                truncated = True
            event_index = len(events) - 1

            # Hint replay detection (independent user step echoed by --replay-user-messages).
            if is_intervention and kind == "user" and policy.hint_marker and policy.hint_marker in raw:
                hint_replayed = True

            # Trigger -> interrupt.
            if is_intervention and not interrupt_sent and _trigger_matches(policy.trigger, event, raw, assistant_count):
                await proc.send(_interrupt_request(interrupt_request_id))
                interrupt_sent = True

            # Interrupt ack.
            if kind == "control_response":
                response = event.get("response") or {}
                if response.get("request_id") == interrupt_request_id and response.get("subtype") == "success":
                    interrupt_acked = True
                    ack_event_index = event_index

            # Result boundary.
            if kind == "result":
                result_count += 1
                if is_intervention and interrupt_acked and not hint_sent:
                    # Interrupted-result boundary: require quiescence before the hint.
                    grace_deadline = time.monotonic() + quiescence_grace_sec
                    while pending_tool_use and time.monotonic() < grace_deadline:
                        try:
                            extra = await proc.readline(timeout=0.2)
                        except asyncio.TimeoutError:
                            break
                        if not extra:
                            break
                        if raw_bytes < _MAX_JOURNAL_BYTES:
                            raw_handle.write(extra)
                            raw_handle.flush()
                            raw_bytes += len(extra)
                        try:
                            ev2 = json.loads(extra.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                        if isinstance(ev2, Mapping):
                            for block in _blocks(ev2):
                                if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                                    pending_tool_use.discard(block["tool_use_id"])
                    quiescence_ok = not pending_tool_use
                    boundary_index = event_index
                    raw_handle.flush()
                    snapshot = {
                        "at_event_index": boundary_index,
                        "workspace_digest": _tree_digest(snapshot_paths),
                        "raw_stream_prefix_digest": _digest_bytes(raw_journal_path.read_bytes()),
                        "no_in_flight_tool": bool(quiescence_ok),
                    }
                    if quiescence_ok:
                        await proc.send(_user_event(policy.hint_text))
                        hint_sent = True
                    else:
                        error = error or "quiescence_violation_tool_in_flight"
                        break
                    continue
                # Baseline completes at the first result; an intervention completes
                # at the post-hint result.
                if not is_intervention:
                    break
                if hint_sent and result_count >= 2:
                    break

        proc.close_stdin()
        returncode = await proc.wait(timeout=10)
        stderr_tail = _scrub((await proc.read_stderr()).decode("utf-8", errors="replace"), secret_values)
    finally:
        raw_handle.close()

    # Scrub the raw journal on disk (defence in depth; the secret must never be here).
    raw_text = raw_journal_path.read_text(encoding="utf-8", errors="replace")
    scrubbed = _scrub(raw_text, secret_values)
    if scrubbed != raw_text:
        _atomic_write(raw_journal_path, scrubbed.encode())

    if is_intervention:
        protocol_satisfied = bool(
            interrupt_sent
            and interrupt_acked
            and boundary_index is not None
            and quiescence_ok
            and hint_sent
            and hint_replayed
            and result_count >= 2
        )
    else:
        protocol_satisfied = result_count >= 1 and error is None

    return {
        "pid": proc.pid,
        "start_wall_epoch": round(start_wall, 3),
        "returncode": returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "assistant_events": assistant_count,
        "result_count": result_count,
        "events": events,
        "events_truncated": truncated,
        "cli_session_ids": cli_session_ids,
        "single_session": len(cli_session_ids) <= 1,
        "interrupt": {
            "sent": interrupt_sent,
            "request_id": interrupt_request_id if interrupt_sent else None,
            "acked": interrupt_acked,
            "ack_event_index": ack_event_index,
        },
        "quiescent_snapshot": snapshot,
        "hint": {
            "sent": hint_sent,
            "replayed": hint_replayed,
            "marker": policy.hint_marker or None,
            "text_digest": _digest_bytes(policy.hint_text.encode()) if policy.hint_text else None,
        },
        "protocol_satisfied": protocol_satisfied,
        "error": error,
        "stderr_tail": stderr_tail[-2000:],
        "raw_stream_digest": _digest_bytes(raw_journal_path.read_bytes()),
    }


def _build_atif(*, drive: Mapping[str, Any], policy: InterventionPolicy, intervention_id: str) -> dict[str, Any]:
    """Project the driven session into an ATIF-style trajectory.

    The interrupted tool/result boundary and the independent user hint step are
    preserved as first-class steps carrying the intervention_id, so a downstream
    consumer can tell an interrupt-then-hint from a naive queued message.
    """

    steps: list[dict[str, Any]] = []
    for row in drive["events"]:
        step = {"index": row["index"], "kind": row["type"]}
        if row.get("subtype"):
            step["subtype"] = row["subtype"]
        interrupt = drive["interrupt"]
        if interrupt["acked"] and row["index"] == interrupt["ack_event_index"]:
            step["interrupt_ack"] = True
            step["intervention_id"] = intervention_id
        snapshot = drive.get("quiescent_snapshot")
        if snapshot and row["index"] == snapshot["at_event_index"]:
            step["interrupted_result_boundary"] = True
        steps.append(step)
    if drive["hint"]["sent"]:
        steps.append(
            {
                "index": len(steps),
                "kind": "user-hint",
                "independent_intervention": True,
                "intervention_id": intervention_id,
                "hint_marker": policy.hint_marker,
                "hint_text_digest": drive["hint"]["text_digest"],
                "replayed_by_model": drive["hint"]["replayed"],
            }
        )
    atif = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "intervention_id": intervention_id,
        "level": policy.level,
        "control_contract": CONTROL_CONTRACT,
        "steps": steps,
        "preserves_interrupt_and_independent_hint": bool(
            drive["interrupt"]["acked"] and (drive["hint"]["sent"] or policy.level == "baseline")
        ),
    }
    atif["atif_digest"] = _digest({k: v for k, v in atif.items() if k != "atif_digest"})
    return atif


# ---------------------------------------------------------------------------
# Public entry point


def run_live_intervention(
    *,
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | Path,
    policy: InterventionPolicy,
    initial_prompt: str,
    intervention_id: str,
    journal_dir: str | Path,
    timeout_sec: float = 900.0,
    read_timeout_sec: float = 60.0,
    quiescence_grace_sec: float = 15.0,
    snapshot_paths: Sequence[str | Path] = (),
    secret_values: Sequence[str] = (),
    harbor_identity: Mapping[str, Any] | None = None,
    budget_max_usd: float | None = None,
    spawn: SpawnFn | None = None,
) -> dict[str, Any]:
    """Run (or reuse) one live-intervention session and journal it.

    Idempotent: a completed journal for ``intervention_id`` is reused verbatim,
    so a crash/retry never injects the hint twice.  ``argv``/``env`` build the
    default subprocess transport; tests pass ``spawn`` directly.  The returned
    receipt carries the interrupt/ack/boundary/quiescence/hint evidence, the
    single-session guarantee, the Harbor trial/task/container identity, budget
    counters and the ATIF projection.
    """

    if not intervention_id.strip():
        raise LiveInterventionError("intervention_id is required")
    journal_root = Path(journal_dir)
    journal_path = journal_root / "live-intervention.json"
    raw_journal_path = journal_root / "raw-stream.jsonl"
    if journal_path.is_file() and not journal_path.is_symlink():
        existing = json.loads(journal_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == LIVE_SCHEMA_VERSION
            and existing.get("intervention_id") == intervention_id
            and existing.get("status") in {"completed", "failed"}
        ):
            unsigned = {k: v for k, v in existing.items() if k != "journal_digest"}
            if existing.get("journal_digest") == _digest(unsigned):
                return {**existing, "reused": True}
            raise LiveInterventionError("existing intervention journal digest mismatch")

    if spawn is None:
        if not argv:
            raise LiveInterventionError("argv is required when no spawn transport is given")
        spawn = _default_spawn(argv, env or {}, cwd)

    drive = asyncio.run(
        _drive_session(
            spawn=spawn,
            policy=policy,
            initial_prompt=initial_prompt,
            intervention_id=intervention_id,
            timeout_sec=timeout_sec,
            read_timeout_sec=read_timeout_sec,
            quiescence_grace_sec=quiescence_grace_sec,
            snapshot_paths=[Path(p) for p in snapshot_paths],
            secret_values=list(secret_values),
            raw_journal_path=raw_journal_path,
        )
    )
    atif = _build_atif(drive=drive, policy=policy, intervention_id=intervention_id)
    _atomic_json(journal_root / "atif.json", atif)

    status = "completed" if drive["protocol_satisfied"] else "failed"
    journal: dict[str, Any] = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "control_contract": CONTROL_CONTRACT,
        "intervention_id": intervention_id,
        "level": policy.level,
        "status": status,
        "protocol_satisfied": drive["protocol_satisfied"],
        "error": drive["error"],
        "trigger": {"kind": policy.trigger.kind, "value": policy.trigger.value},
        "interrupt": drive["interrupt"],
        "quiescent_snapshot": drive["quiescent_snapshot"],
        "hint": drive["hint"],
        "continuation": {
            "final_result_seen": drive["result_count"] >= (2 if policy.level != "baseline" else 1),
            "result_count": drive["result_count"],
        },
        "process": {"pid": drive["pid"], "start_wall_epoch": drive["start_wall_epoch"], "returncode": drive["returncode"]},
        "claude_session_id": drive["cli_session_ids"][0] if drive["cli_session_ids"] else None,
        "single_session": drive["single_session"],
        "harbor_identity": dict(harbor_identity) if harbor_identity else None,
        "budget": {
            "max_usd": budget_max_usd,
        },
        "raw_stream_journal": raw_journal_path.name,
        "raw_stream_digest": drive["raw_stream_digest"],
        "atif_digest": atif["atif_digest"],
        "events_truncated": drive["events_truncated"],
        "elapsed_sec": drive["elapsed_sec"],
        "reused": False,
    }
    journal["journal_digest"] = _digest({k: v for k, v in journal.items() if k != "journal_digest"})
    _atomic_json(journal_path, journal)
    return journal


__all__ = [
    "ATIF_SCHEMA_VERSION",
    "CONTROL_CONTRACT",
    "ClaudeProc",
    "InterventionPolicy",
    "LIVE_SCHEMA_VERSION",
    "LiveInterventionError",
    "Trigger",
    "run_live_intervention",
]
