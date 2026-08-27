#!/usr/bin/env python3
"""Observe a Harbor job without changing its execution.

The script is intentionally an observe-only companion to Harbor.  It polls a
job directory, records lifecycle/trajectory facts as JSONL, and prints a small
status line suitable for a long-running screen session.  It never sends input
to an agent, edits a trial, or kills a process.  A future intervention driver
must use an explicit checkpoint/resume contract; treating a log tail as a
prompt-injection channel would invalidate the evidence.

Harbor writes result.json atomically only at trial completion, while the agent
stream log is available during execution.  Consequently the live status is
E1/E2 observation evidence; completed verifier rewards are copied as E3 facts
but are not interpreted as model comparisons here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stream_summary(path: Path) -> dict[str, Any]:
    """Summarize a Claude stream log without retaining prompt/tool contents."""

    counts: Counter[str] = Counter()
    last_timestamp: str | None = None
    last_kind: str | None = None
    assistant_turns = 0
    tool_uses = 0
    error_events = 0
    try:
        # Reading the complete file is acceptable for the modest monitor log;
        # malformed/truncated final lines are ignored.  Keep only counters.
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if isinstance(kind, str):
                    counts[kind] += 1
                    last_kind = kind
                ts = event.get("timestamp")
                if isinstance(ts, str):
                    last_timestamp = ts
                if kind == "assistant":
                    assistant_turns += 1
                    message = event.get("message")
                    if isinstance(message, dict):
                        blocks = message.get("content")
                        if isinstance(blocks, list):
                            tool_uses += sum(
                                isinstance(block, dict)
                                and block.get("type") == "tool_use"
                                for block in blocks
                            )
                if kind == "error" or event.get("is_error") is True:
                    error_events += 1
    except OSError:
        pass
    return {
        "bytes": path.stat().st_size if path.exists() else 0,
        "event_counts": dict(counts),
        "assistant_turns": assistant_turns,
        "tool_uses": tool_uses,
        "error_events": error_events,
        "last_event_type": last_kind,
        "last_event_timestamp": last_timestamp,
    }


def _trial_record(trial_dir: Path) -> dict[str, Any]:
    result = _read_json(trial_dir / "result.json")
    stream = _stream_summary(trial_dir / "agent" / "claude-code.txt")
    if result is not None:
        status = "completed" if result.get("finished_at") else "result-written"
        exc = result.get("exception_info")
        rewards = (result.get("verifier_result") or {}).get("rewards")
        agent = result.get("config", {}).get("agent", {})
        return {
            "trial_name": trial_dir.name,
            "status": status,
            "task_name": result.get("task_name"),
            "model": agent.get("model_name") or (result.get("agent_info") or {}).get("model_info", {}).get("name"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "exception": (
                (exc or {}).get("exception_type")
                or (exc or {}).get("type")
                or (exc or {}).get("name")
            )
            if isinstance(exc, dict)
            else None,
            "rewards": rewards if isinstance(rewards, dict) else None,
            "trace_present": (trial_dir / "agent" / "trajectory.json").exists(),
            "stream": stream,
        }

    # A directory can be created before Harbor starts the trial.  Distinguish
    # that from a live agent by the presence of its stream log.
    status = "running" if stream["bytes"] else "pending"
    return {
        "trial_name": trial_dir.name,
        "status": status,
        "task_name": None,
        "model": None,
        "started_at": None,
        "finished_at": None,
        "exception": None,
        "rewards": None,
        "trace_present": (trial_dir / "agent" / "trajectory.json").exists(),
        "stream": stream,
    }


def snapshot(job_dir: Path) -> dict[str, Any]:
    aggregate = _read_json(job_dir / "result.json") or {}
    trials = [
        _trial_record(p)
        for p in sorted(job_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    stats = aggregate.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    # Harbor's aggregate is the source of truth for pending/running counts;
    # local trial inspection is retained as a cross-check, not substituted.
    return {
        "observed_at": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "evidence_level": "E1" if aggregate.get("finished_at") is None else "E3",
        "job_dir": str(job_dir),
        "job_result_sha256": _sha256(job_dir / "result.json"),
        "job_id": aggregate.get("id"),
        "job_started_at": aggregate.get("started_at"),
        "job_finished_at": aggregate.get("finished_at"),
        "stats": stats,
        "trials": trials,
    }


def _print_snapshot(value: dict[str, Any]) -> None:
    stats = value.get("stats") or {}
    trials = value.get("trials") or []
    counts = Counter(t.get("status") for t in trials)
    print(
        f"[{value['observed_at']}] "
        f"completed={stats.get('n_completed_trials', counts.get('completed', 0))} "
        f"running={stats.get('n_running_trials', counts.get('running', 0))} "
        f"pending={stats.get('n_pending_trials', counts.get('pending', 0))} "
        f"errored={stats.get('n_errored_trials', 0)} "
        f"local={dict(counts)}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSONL observation file")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    job_dir = args.job_dir.expanduser().resolve()
    if not job_dir.is_dir():
        parser.error(f"job directory does not exist: {job_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    while True:
        value = snapshot(job_dir)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _print_snapshot(value)
        if args.once or value.get("job_finished_at") is not None:
            return 0
        time.sleep(max(0.25, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
