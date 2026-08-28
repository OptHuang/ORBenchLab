from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orbenchlab import harbor_live_intervention as hli

FAKE = Path(__file__).resolve().parent / "fake_claude_stream.py"
SID = "sid-live-1"
MARKER = "LIVE_HINT_MARKER_20260828"


def _spawn(mode: str, *, sid: str = SID, marker: str = MARKER, canary: str = ""):
    argv = [sys.executable, str(FAKE), mode, sid, marker, canary]

    async def spawn():
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return hli.ClaudeProc(proc)

    return spawn


def _policy(level: str = "L1") -> hli.InterventionPolicy:
    if level == "baseline":
        return hli.InterventionPolicy(level="baseline")
    return hli.InterventionPolicy(
        level=level,
        hint_text=f"{MARKER}: write HINTED instead and confirm the marker.",
        hint_marker=MARKER,
        trigger=hli.Trigger("tool-use", "Bash"),
    )


def test_same_session_interrupt_hint_continuation(tmp_path: Path):
    # Acceptance 1: one session through interrupt -> ack -> boundary -> hint;
    # hint appears in the raw stream and ATIF; single session id preserved.
    rec = hli.run_live_intervention(
        cwd=tmp_path,
        policy=_policy("L1"),
        initial_prompt="do the task",
        intervention_id="iv-1",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
        harbor_identity={"trial_id": "t1", "task_id": "task", "container_id": "c1"},
        spawn=_spawn("honor"),
    )
    assert rec["status"] == "completed"
    assert rec["protocol_satisfied"] is True
    assert rec["interrupt"]["sent"] and rec["interrupt"]["acked"]
    assert rec["quiescent_snapshot"]["no_in_flight_tool"] is True
    assert rec["hint"]["sent"] and rec["hint"]["replayed"]
    assert rec["single_session"] is True
    assert rec["claude_session_id"] == SID
    assert rec["harbor_identity"]["container_id"] == "c1"
    # Raw stream on disk contains the replayed marker.
    raw = (tmp_path / "j" / "raw-stream.jsonl").read_text()
    assert MARKER in raw
    # ATIF preserves the interrupt boundary and an independent hint step.
    atif = json.loads((tmp_path / "j" / "atif.json").read_text())
    assert atif["preserves_interrupt_and_independent_hint"] is True
    steps = atif["steps"]
    assert any(s.get("interrupt_ack") for s in steps)
    assert any(s.get("interrupted_result_boundary") for s in steps)
    hint_steps = [s for s in steps if s.get("kind") == "user-hint"]
    assert len(hint_steps) == 1
    assert hint_steps[0]["intervention_id"] == "iv-1"
    assert hint_steps[0]["replayed_by_model"] is True


def test_naive_queue_without_interrupt_does_not_count(tmp_path: Path):
    # Acceptance 2a: if the model never acks the interrupt, the runner must not
    # send the hint and must not claim a satisfied protocol.
    rec = hli.run_live_intervention(
        cwd=tmp_path,
        policy=_policy("L1"),
        initial_prompt="do the task",
        intervention_id="iv-2",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
        spawn=_spawn("ignore-interrupt"),
    )
    assert rec["interrupt"]["sent"] is True
    assert rec["interrupt"]["acked"] is False
    assert rec["hint"]["sent"] is False
    assert rec["protocol_satisfied"] is False
    assert rec["status"] == "failed"


def test_quiescence_gate_blocks_hint_when_tool_in_flight(tmp_path: Path):
    # Acceptance 2b: an interrupted boundary with a tool still in flight must
    # block the hint (no injection into a non-quiescent checkpoint).
    rec = hli.run_live_intervention(
        cwd=tmp_path,
        policy=_policy("L2"),
        initial_prompt="do the task",
        intervention_id="iv-3",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
        quiescence_grace_sec=1.0,
        spawn=_spawn("tool-in-flight"),
    )
    assert rec["interrupt"]["acked"] is True
    assert rec["quiescent_snapshot"]["no_in_flight_tool"] is False
    assert rec["hint"]["sent"] is False
    assert rec["error"] == "quiescence_violation_tool_in_flight"
    assert rec["protocol_satisfied"] is False


def test_crash_retry_reuses_journal_without_reinjecting(tmp_path: Path):
    # Acceptance 2c: a completed journal is reused; the second call must not
    # spawn a session or inject again.
    kwargs = dict(
        cwd=tmp_path,
        policy=_policy("L1"),
        initial_prompt="do the task",
        intervention_id="iv-4",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
    )
    first = hli.run_live_intervention(spawn=_spawn("honor"), **kwargs)
    assert first["reused"] is False and first["protocol_satisfied"]

    # A spawn that would raise if invoked proves reuse does not re-run.
    async def exploding_spawn():
        raise AssertionError("spawn must not be called on reuse")

    second = hli.run_live_intervention(spawn=exploding_spawn, **kwargs)
    assert second["reused"] is True
    assert second["journal_digest"] == first["journal_digest"]
    assert second["hint"]["sent"] is True


def test_baseline_arm_runs_without_interrupt(tmp_path: Path):
    rec = hli.run_live_intervention(
        cwd=tmp_path,
        policy=_policy("baseline"),
        initial_prompt="do the task",
        intervention_id="iv-base",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
        spawn=_spawn("baseline"),
    )
    assert rec["status"] == "completed"
    assert rec["interrupt"]["sent"] is False
    assert rec["hint"]["sent"] is False
    assert rec["continuation"]["result_count"] == 1


def test_secret_canary_is_scrubbed_from_journal(tmp_path: Path):
    # Acceptance 3 (journal half): even if the model stream echoes the secret,
    # the runner scrubs it from the raw journal and never records it.
    canary = "REAL-live-canary-secret-0123456789abcdef"
    rec = hli.run_live_intervention(
        cwd=tmp_path,
        policy=_policy("L1"),
        initial_prompt="do the task",
        intervention_id="iv-5",
        journal_dir=tmp_path / "j",
        timeout_sec=30,
        secret_values=[canary],
        spawn=_spawn("leak-secret", canary=canary),
    )
    assert rec["protocol_satisfied"] is True
    raw = (tmp_path / "j" / "raw-stream.jsonl").read_text()
    assert canary not in raw
    assert "[REDACTED_SECRET]" in raw
    assert canary not in json.dumps(rec)
    assert canary not in (tmp_path / "j" / "atif.json").read_text()
