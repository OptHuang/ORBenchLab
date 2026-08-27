from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import authoring_loop
from orbenchlab.volc_review import REQUIRED_REVIEW_CRITERIA, VolcConfig


ROOT = Path(__file__).parents[1]


def test_author_patch_rejects_traversal_and_stale_digest():
    common = {
        "schema_version": "orbenchlab.author-patch.v1",
        "round": 1,
        "base_task_tree_digest": "sha256:" + "a" * 64,
        "input_receipt_digest": "sha256:" + "b" * 64,
        "previous_review_digest": None,
        "files": [{"path": "../README.md", "content": "bad"}],
    }
    with pytest.raises(authoring_loop.AuthoringLoopError, match="unsafe path"):
        authoring_loop._normalise_patch(
            common,
            round_number=1,
            base_digest="sha256:" + "a" * 64,
            input_receipt_digest="sha256:" + "b" * 64,
            previous_review_digest=None,
        )
    for path, message in (
        ("data/paper-derivation-bound.json", "bound paper derivation"),
        ("data//paper-derivation-bound.json", "unsafe path"),
        ("data/./paper-derivation-bound.json", "unsafe path"),
    ):
        with pytest.raises(authoring_loop.AuthoringLoopError, match=message):
            authoring_loop._normalise_patch(
                {**common, "files": [{"path": path, "content": "forged evidence"}]},
                round_number=1,
                base_digest="sha256:" + "a" * 64,
                input_receipt_digest="sha256:" + "b" * 64,
                previous_review_digest=None,
            )
    with pytest.raises(authoring_loop.AuthoringLoopError, match="credential-like"):
        authoring_loop._normalise_patch(
            {
                **common,
                "files": [{"path": "data/config.json", "content": 'api_key="LIVE-SECRET-123"'}],
            },
            round_number=1,
            base_digest="sha256:" + "a" * 64,
            input_receipt_digest="sha256:" + "b" * 64,
            previous_review_digest=None,
        )
    with pytest.raises(authoring_loop.AuthoringLoopError, match="stale"):
        authoring_loop._normalise_patch(
            {**common, "files": [{"path": "README.md", "content": "ok"}]},
            round_number=1,
            base_digest="sha256:" + "c" * 64,
            input_receipt_digest="sha256:" + "b" * 64,
            previous_review_digest=None,
        )


def test_exact_single_file_author_response_gets_trusted_round_binding():
    base = "sha256:" + "a" * 64
    receipt = "sha256:" + "b" * 64
    coerced, shape = authoring_loop._coerce_author_response(
        {"path": "README.md", "content": "bounded edit"},
        round_number=2,
        base_digest=base,
        input_receipt_digest=receipt,
        previous_review_digest="sha256:" + "c" * 64,
    )
    patch = authoring_loop._normalise_patch(
        coerced,
        round_number=2,
        base_digest=base,
        input_receipt_digest=receipt,
        previous_review_digest="sha256:" + "c" * 64,
    )
    assert shape == "single-file-shorthand"
    assert patch["files"] == [{"path": "README.md", "content": "bounded edit"}]

    with pytest.raises(authoring_loop.AuthoringLoopError, match="schema is unsupported"):
        authoring_loop._coerce_author_response(
            {"path": "README.md", "content": "edit", "untrusted_round": 9},
            round_number=2,
            base_digest=base,
            input_receipt_digest=receipt,
            previous_review_digest=None,
        )
    coerced_reserved, _ = authoring_loop._coerce_author_response(
        {"path": authoring_loop.BOUND_DERIVATION_PATH, "content": "forged"},
        round_number=2,
        base_digest=base,
        input_receipt_digest=receipt,
        previous_review_digest=None,
    )
    with pytest.raises(authoring_loop.AuthoringLoopError, match="bound paper derivation"):
        authoring_loop._normalise_patch(
            coerced_reserved,
            round_number=2,
            base_digest=base,
            input_receipt_digest=receipt,
            previous_review_digest=None,
        )


def test_iterate_runs_two_rounds_without_mutating_seed(monkeypatch, tmp_path):
    seed = ROOT / "examples/tasks/alphaevolve-scheduling"
    paper = seed / "paper-provenance.json"
    derivation = seed / "data/paper-task-derivation.json"
    seed_readme = (seed / "README.md").read_text(encoding="utf-8")
    author_calls = []

    def fake_author(config, *, model, system, user, max_tokens):
        payload = json.loads(user.split("\n\n", 1)[1])
        assert payload["paper_evidence"]["derivation"]
        assert payload["paper_evidence"]["paper"]["source_content_digest"].startswith("sha256:")
        readme = next(row["preview"] for row in payload["task_files"] if row["path"] == "README.md")
        author_calls.append(payload["round"])
        return {
            "model": model,
            "protocol": "anthropic",
            "request_digest": "sha256:" + str(payload["round"]) * 64,
            "response_digest": "sha256:" + str(payload["round"] + 2) * 64,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "parsed": {
                "schema_version": "orbenchlab.author-patch.v1",
                "round": payload["round"],
                "base_task_tree_digest": payload["base_task_tree_digest"],
                "input_receipt_digest": payload["input_receipt_digest"],
                "previous_review_digest": payload["previous_review_digest"],
                "files": [
                    {
                        "path": "README.md",
                        "content": readme + f"\n\nAuthoring round {payload['round']}.\n",
                    }
                ],
                "rationale": ["bounded fixture edit"],
            },
        }

    review_calls = []

    def fake_review(task_dir, *, paper_provenance, receipt, config, models, round_number, max_tokens):
        review_calls.append(round_number)
        decision = "revise" if round_number == 1 else "promising-needs-harbor"
        reviewer_decision = "revise" if round_number == 1 else "promising"
        return {
            "schema_version": "orbenchlab.volc-authoring-review.v1",
            "round": round_number,
            "models": list(models),
            "review_count": len(models),
            "task_tree_digest": receipt["task_tree_digest"],
            "static_receipt_digest": receipt["receipt_digest"],
            "paper_digest": paper_provenance["source_content_digest"],
            "aggregate_decision": decision,
            "evidence_level": "E1-model-review",
            "reviewers": [
                {
                    "model": model,
                    "review": {
                        "decision": reviewer_decision,
                        "shape_complete": True,
                        "rubric_complete": True,
                        "criteria": [
                            {"name": name, "status": "pass", "evidence": "fixture evidence"}
                            for name in sorted(REQUIRED_REVIEW_CRITERIA)
                        ],
                        "blocking_findings": [],
                        "suggested_edits": [],
                    },
                }
                for model in models
            ],
            "limitations": [],
        }

    monkeypatch.setattr(authoring_loop, "call_reviewer", fake_author)
    monkeypatch.setattr(authoring_loop, "review_task", fake_review)
    run = authoring_loop.iterate(
        seed,
        paper_provenance=paper,
        paper_derivation=derivation,
        config=VolcConfig(
            "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
        ),
        author_model="ark-code-latest",
        review_models=["reviewer-a", "reviewer-b"],
        max_rounds=3,
        out=tmp_path / "run",
    )

    assert run["status"] == "promising-needs-harbor"
    assert run["seed_unchanged"] is True
    assert author_calls == [1, 2]
    assert review_calls == [1, 2]
    assert (seed / "README.md").read_text(encoding="utf-8") == seed_readme
    final_readme = (Path(run["final_task"]) / "README.md").read_text(encoding="utf-8")
    assert "Authoring round 1" in final_readme
    assert "Authoring round 2" in final_readme
    assert (Path(run["final_task"]) / authoring_loop.BOUND_DERIVATION_PATH).is_file()
    assert run["paper_derivation_digest"].startswith("sha256:")
    manifest = json.loads((tmp_path / "run/run-manifest.json").read_text())
    assert manifest["run_digest"].startswith("sha256:")


def test_iterate_requires_two_reviewer_slots(tmp_path):
    seed = ROOT / "examples/tasks/alphaevolve-scheduling"
    with pytest.raises(authoring_loop.AuthoringLoopError, match="two distinct reviewer"):
        authoring_loop.iterate(
            seed,
            paper_provenance=seed / "paper-provenance.json",
            paper_derivation=seed / "data/paper-task-derivation.json",
            config=VolcConfig(
                "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
            ),
            author_model="ark-code-latest",
            review_models=["ark-code-latest"],
            max_rounds=1,
            out=tmp_path / "run",
        )
    with pytest.raises(authoring_loop.AuthoringLoopError, match="distinct reviewer"):
        authoring_loop.iterate(
            seed,
            paper_provenance=seed / "paper-provenance.json",
            paper_derivation=seed / "data/paper-task-derivation.json",
            config=VolcConfig(
                "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
            ),
            author_model="ark-code-latest",
            review_models=["ark-code-latest", "ark-code-latest"],
            max_rounds=1,
            out=tmp_path / "run-duplicate",
        )


def test_iterate_provider_failure_writes_incomplete_run_without_touching_seed(monkeypatch, tmp_path):
    seed = ROOT / "examples/tasks/alphaevolve-scheduling"
    before = authoring_loop._task_tree_digest(seed)

    def fail(*args, **kwargs):
        from orbenchlab.volc_review import VolcReviewError

        raise VolcReviewError("provider unavailable")

    monkeypatch.setattr(authoring_loop, "call_reviewer", fail)
    run = authoring_loop.iterate(
        seed,
        paper_provenance=seed / "paper-provenance.json",
        paper_derivation=seed / "data/paper-task-derivation.json",
        config=VolcConfig(
            "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
        ),
        author_model="ark-code-latest",
        review_models=["reviewer-a", "reviewer-b"],
        max_rounds=2,
        out=tmp_path / "run",
    )

    assert run["status"] == "incomplete"
    assert run["rounds"][0]["phase"] == "author"
    assert authoring_loop._task_tree_digest(seed) == before
    assert (tmp_path / "run/run.json").is_file()


def test_iterate_rejects_secret_seed_and_missing_derivation_before_model(monkeypatch, tmp_path):
    original = ROOT / "examples/tasks/alphaevolve-scheduling"
    seed = tmp_path / "alphaevolve-scheduling"
    shutil.copytree(original, seed)
    (seed / "api_key.json").write_text('{"api_key":"LIVE-SECRET-123"}')
    called = []
    monkeypatch.setattr(authoring_loop, "call_reviewer", lambda *a, **k: called.append(True))

    with pytest.raises(authoring_loop.AuthoringLoopError, match="paper-derivation"):
        authoring_loop.iterate(
            seed,
            paper_provenance=seed / "paper-provenance.json",
            paper_derivation=tmp_path / "missing.md",
            config=VolcConfig(
                "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
            ),
            author_model="ark-code-latest",
            review_models=["reviewer-a", "reviewer-b"],
            max_rounds=1,
            out=tmp_path / "missing-run",
        )
    with pytest.raises(authoring_loop.AuthoringLoopError, match="credential-like"):
        authoring_loop.iterate(
            seed,
            paper_provenance=seed / "paper-provenance.json",
            paper_derivation=seed / "data/paper-task-derivation.json",
            config=VolcConfig(
                "https://ark.cn-beijing.volces.com/api/coding", "secret", "ark-code-latest"
            ),
            author_model="ark-code-latest",
            review_models=["reviewer-a", "reviewer-b"],
            max_rounds=1,
            out=tmp_path / "secret-run",
        )
    assert called == []
    assert not (tmp_path / "missing-run").exists()
    assert not (tmp_path / "secret-run").exists()
