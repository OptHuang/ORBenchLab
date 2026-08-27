from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab.volc_review import VolcConfig, VolcReviewError, call_reviewer, review_task, write_review


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


def test_call_reviewer_routes_explicit_ark_model_id_through_openai_compat(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.headers.get("Authorization")
        return _Response(
            {
                "choices": [{"message": {"content": '{"decision":"needs-human"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
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

    assert observed["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert observed["authorization"] == "Bearer secret-token"
    assert result["protocol"] == "openai"
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 2}
    assert "secret-token" not in json.dumps(result)


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
    review = review_task(
        task,
        paper_provenance={"source_content_digest": "sha256:" + "c" * 64},
        receipt={
            "decision": "blocked",
            "round": 1,
            "receipt_digest": "sha256:" + "d" * 64,
            "task_tree_digest": "sha256:" + "e" * 64,
            "counts": {"fail": 1, "pass": 0, "review": 0},
            "implementation_criteria": [],
            "provenance_checks": [],
        },
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
