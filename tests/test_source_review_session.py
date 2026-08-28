from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import source_review_session as srs


def test_sandboxed_review_runner_composes_session_and_parses(tmp_path: Path):
    frozen = tmp_path / "source.bin"
    frozen.write_bytes(b"PAPER BYTES")
    sentinel = tmp_path / "SENTINEL"
    sentinel.write_text("secret")
    captured = {}

    def fake_session(**kwargs):
        captured.update(kwargs)
        # The reviewer would write review.json in the workdir.
        workdir = Path(kwargs["workdir"])
        (workdir / "review.json").write_text(json.dumps({
            "anchor": "ANCHOR-XYZ", "or_relevant": True, "novelty_within_bounded_corpus": True,
            "reproducible": True, "task_feasible": True, "verifier_feasible": True, "admit": True,
            "source_kind": "paper", "task_nucleus": "x",
            "difficulty_axes": ["a", "b"], "predicted_bottlenecks": ["c"],
        }))
        (workdir / "sessions").mkdir(parents=True, exist_ok=True)
        receipt = workdir / "sessions" / "receipt.json"
        receipt.write_text("{}")
        return {"status": "completed", "receipt_path": str(receipt)}

    runner = srs.build_sandboxed_review_runner(
        frozen_source_path=frozen,
        claude_executable="/opt/claude",
        provider_env={"ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding",
                      "ANTHROPIC_AUTH_TOKEN": "tok"},
        model="doubao",
        out=tmp_path / "rev",
        hidden_sentinels=[sentinel],
        session_runner=fake_session,
    )
    result = runner("reviewer-1", "ANCHOR-XYZ", tmp_path / "rev" / "reviewer-1")

    # The session ran no-Bash, credential-relayed, with the source read-only and
    # the sentinel hidden.
    assert captured["allow_bash"] is False
    assert captured["credential_relay"] is True
    assert Path(captured["read_only_paths"][0]).name == "source-input"
    assert sentinel in [Path(p) for p in captured["hidden_paths"]]
    # The staged source is a copy, read-only, and the anchored review parsed.
    assert result["decision"]["anchor"] == "ANCHOR-XYZ"
    assert result["session_status"] == "completed"
    assert result["session_receipt_digest"] is not None
    # The prompt tells the reviewer to reproduce the exact anchor and ignore
    # instructions inside the untrusted source.
    assert "ANCHOR-XYZ" in captured["prompt"] and "untrusted" in captured["prompt"]


def test_review_runner_returns_none_decision_when_no_review_written(tmp_path: Path):
    frozen = tmp_path / "source.bin"
    frozen.write_bytes(b"X")

    def fake_session(**kwargs):
        return {"status": "failed", "receipt_path": None}

    runner = srs.build_sandboxed_review_runner(
        frozen_source_path=frozen, claude_executable="/opt/claude",
        provider_env={"ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding",
                      "ANTHROPIC_AUTH_TOKEN": "tok"},
        model="m", out=tmp_path / "rev", session_runner=fake_session,
    )
    result = runner("reviewer-1", "A", tmp_path / "rev" / "r1")
    # A crashed session that wrote no review yields a None decision (which the
    # triage validator treats as invalid, never admitted).
    assert result["decision"] is None
    assert result["session_status"] == "failed"
