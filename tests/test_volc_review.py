from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab.volc_review import (
    VolcConfig,
    VolcReviewError,
    _digest,
    _normalize_review,
    _task_tree_digest,
    call_reviewer,
    review_task,
    write_review,
)


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_volc_config_requires_volc_host(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/api")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    monkeypatch.setenv("ANTHROPIC_MODEL", "ark-code-latest")
    with pytest.raises(VolcReviewError, match="non-Volcengine"):
        VolcConfig.from_env()


def test_call_reviewer_records_digests_without_token(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(
            {
                "type": "message",
                "content": [{"type": "text", "text": '{"decision":"promising"}'}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            }
        ),
    )
    config = VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret-token", "ark-code-latest")
    result = call_reviewer(config, model="ark-code-latest", system="system", user="user")
    assert result["review"]["decision"] if "review" in result else result["response_digest"]
    assert result["usage"]["input_tokens"] == 12
    assert "secret-token" not in json.dumps(result)


def test_call_reviewer_routes_explicit_model_through_anthropic_coding_gateway(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["api_key"] = request.headers.get("X-api-key")
        return _Response(
            {
                "type": "message",
                "content": [{"type": "text", "text": '{"decision":"needs-human"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 2},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    config = VolcConfig(
        "https://ark.cn-beijing.volces.com/api/coding",
        "secret-token",
        "ark-code-latest",
    )

    result = call_reviewer(
        config,
        model="deepseek-v3-1-250821",
        system="system",
        user="user",
    )

    assert observed["url"] == "https://ark.cn-beijing.volces.com/api/coding/v1/messages"
    assert observed["api_key"] == "secret-token"
    assert result["protocol"] == "anthropic"
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 2}
    assert "secret-token" not in json.dumps(result)


def test_promising_review_without_complete_rubric_is_downgraded():
    review = _normalize_review(
        {
            "decision": "promising",
            "task_summary": "summary",
            "blocking_findings": [],
            "difficulty_axes": [],
            "criteria": [],
            "suggested_edits": [],
        }
    )
    assert review["decision"] == "needs-human"
    assert review["rubric_complete"] is False


def test_review_task_is_blocked_by_static_receipt_and_writes_summary(tmp_path: Path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("public instruction", encoding="utf-8")
    config = VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret-token", "ark-code-latest")

    def fake_call(*args, **kwargs):
        return {
            "model": kwargs["model"],
            "elapsed_sec": 0.01,
            "request_digest": "sha256:" + "a" * 64,
            "response_digest": "sha256:" + "b" * 64,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "parsed": {"decision": "promising", "blocking_findings": [], "suggested_edits": []},
        }

    monkeypatch.setattr("orbenchlab.volc_review.call_reviewer", fake_call)
    paper_digest = "sha256:" + "c" * 64
    receipt = {
        "authoring_schema_version": "orbenchlab.tbscience-authoring.v1",
        "task_dir": task.name,
        "decision": "blocked",
        "round": 1,
        "task_tree_digest": _task_tree_digest(task),
        "paper": {"source_content_digest": paper_digest},
        "counts": {"fail": 1, "pass": 0, "review": 0},
        "implementation_criteria": [],
        "provenance_checks": [],
    }
    receipt["receipt_digest"] = _digest(receipt)
    review = review_task(
        task,
        paper_provenance={"source_content_digest": paper_digest},
        receipt=receipt,
        config=config,
        models=["ark-code-latest", "deepseek-v4-flash"],
        round_number=1,
    )
    assert review["aggregate_decision"] == "blocked-static-gate"
    assert review["review_count"] == 2
    paths = write_review(review, tmp_path / "out")
    written = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert written["review_digest"].startswith("sha256:")
    assert "secret-token" not in paths["json"].read_text(encoding="utf-8")


def test_review_rejects_stale_receipt_before_provider_call(tmp_path: Path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("public")
    paper_digest = "sha256:" + "c" * 64
    receipt = {
        "authoring_schema_version": "orbenchlab.tbscience-authoring.v1",
        "task_dir": task.name,
        "decision": "ready-for-human-review",
        "round": 1,
        "task_tree_digest": _task_tree_digest(task),
        "paper": {"source_content_digest": paper_digest},
    }
    receipt["receipt_digest"] = _digest(receipt)
    (task / "instruction.md").write_text("changed after receipt")
    called = []
    monkeypatch.setattr("orbenchlab.volc_review.call_reviewer", lambda *a, **k: called.append(True))
    with pytest.raises(VolcReviewError, match="stale"):
        review_task(
            task,
            paper_provenance={"source_content_digest": paper_digest},
            receipt=receipt,
            config=VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"),
            models=["reviewer-a", "reviewer-b"],
            round_number=1,
        )
    assert called == []


def test_review_snapshot_rejects_api_key_file_before_provider(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("public")
    (task / "api_key.json").write_text('{"api_key":"LIVE-SECRET-123"}')
    paper_digest = "sha256:" + "c" * 64
    receipt = {
        "authoring_schema_version": "orbenchlab.tbscience-authoring.v1",
        "task_dir": task.name,
        "decision": "ready-for-human-review",
        "round": 1,
        "task_tree_digest": _task_tree_digest(task),
        "paper": {"source_content_digest": paper_digest},
    }
    receipt["receipt_digest"] = _digest(receipt)
    with pytest.raises(VolcReviewError, match="credential-like"):
        review_task(
            task,
            paper_provenance={"source_content_digest": paper_digest},
            receipt=receipt,
            config=VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"),
            models=["reviewer-a", "reviewer-b"],
            round_number=1,
        )


def test_review_requires_two_distinct_models_before_provider(tmp_path: Path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("public")
    paper_digest = "sha256:" + "c" * 64
    receipt = {
        "authoring_schema_version": "orbenchlab.tbscience-authoring.v1",
        "task_dir": task.name,
        "decision": "ready-for-human-review",
        "round": 1,
        "task_tree_digest": _task_tree_digest(task),
        "paper": {"source_content_digest": paper_digest},
    }
    receipt["receipt_digest"] = _digest(receipt)
    called = []
    monkeypatch.setattr("orbenchlab.volc_review.call_reviewer", lambda *a, **k: called.append(True))
    with pytest.raises(VolcReviewError, match="two distinct"):
        review_task(
            task,
            paper_provenance={"source_content_digest": paper_digest},
            receipt=receipt,
            config=VolcConfig(
                "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
            ),
            models=["ark-code-latest", "ark-code-latest"],
            round_number=1,
        )
    assert called == []
