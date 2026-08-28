from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orbenchlab import factory_batch, factory_blueprints

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _provenance_for(paper: Path) -> dict:
    unsigned = {
        "paper_provenance_schema_version": "orbenchlab.paper-provenance.v1",
        "title": "Fixture paper",
        "url": "https://example.org/paper",
        "source_content_digest": "sha256:" + hashlib.sha256(paper.read_bytes()).hexdigest(),
        "license_status": "pending-human",
    }
    return {
        **unsigned,
        "binding_digest": "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _candidate(tmp_path: Path, name: str, *, corrupt: bool = False) -> dict:
    home = tmp_path / name
    home.mkdir(parents=True, exist_ok=True)
    paper = home / "paper.pdf"
    paper.write_bytes(f"%PDF-1.4 fixture {name}\n".encode())
    provenance = _provenance_for(paper)
    if corrupt:
        paper.write_bytes(b"%PDF-1.4 tampered\n")
    provenance_path = home / "paper-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "id": name,
        "paper_file": str(paper),
        "paper_provenance": str(provenance_path),
        "seed_task": str(GOOD_TASK),
    }


def _spec(candidates: list[dict]) -> dict:
    return {
        "schema_version": factory_batch.SCHEMA_VERSION,
        "provider": {"kind": "explicit-list", "candidates": candidates},
        "models": {
            "author_model": "author",
            "reviewer_models": ["rev-a", "rev-b"],
            "frontier_model": "frontier",
            "weak_model": "weak",
        },
    }


def test_spec_and_provider_validation(tmp_path: Path):
    with pytest.raises(factory_batch.FactoryBatchError):
        factory_batch.load_batch_spec(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: nope\n", encoding="utf-8")
    with pytest.raises(factory_batch.FactoryBatchError):
        factory_batch.load_batch_spec(bad)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(_spec([_candidate(tmp_path, "alpha")])), encoding="utf-8"
    )
    spec = factory_batch.load_batch_spec(good)
    rows = factory_batch.discover_candidates(spec)
    assert [row["id"] for row in rows] == ["alpha"]


def test_paper_binding_dir_provider_discovers_bound_candidates(tmp_path: Path):
    root = tmp_path / "bindings"
    for name in ("alpha", "beta"):
        _candidate(root, name)
    spec = {
        "kind": "paper-binding-dir",
        "root": str(root),
        "seed_task": str(GOOD_TASK),
    }
    rows = factory_batch._provider_paper_binding_dir(spec)
    assert [row["id"] for row in rows] == ["alpha", "beta"]
    assert all(row["paper_file"].endswith("paper.pdf") for row in rows)


def test_screening_rejects_broken_bindings_without_model_calls(tmp_path: Path):
    ok = factory_batch.screen_candidate(_candidate(tmp_path, "good-candidate"))
    assert ok["promising"] is True
    assert ok["license_status"] == "pending-human"
    tampered = factory_batch.screen_candidate(
        _candidate(tmp_path, "tampered-candidate", corrupt=True)
    )
    assert tampered["promising"] is False
    assert any("do not match" in reason for reason in tampered["reasons"])
    missing = factory_batch.screen_candidate(
        {
            "id": "missing",
            "paper_file": str(tmp_path / "nope.pdf"),
            "paper_provenance": str(tmp_path / "nope.json"),
            "seed_task": str(tmp_path / "noseed"),
        }
    )
    assert missing["promising"] is False
    assert len(missing["reasons"]) == 3


def test_batch_isolates_candidates_and_survives_failures(tmp_path: Path, monkeypatch):
    candidates = [
        _candidate(tmp_path, "alpha"),
        _candidate(tmp_path, "beta"),
        _candidate(tmp_path, "tampered", corrupt=True),
    ]
    spec = _spec(candidates)

    def fake_prepare(**kwargs):
        workdir = Path(kwargs["workdir"])
        workdir.mkdir(parents=True, exist_ok=True)
        return {"workspace_binding_digest": "sha256:" + "1" * 64}

    def fake_plan(**kwargs):
        from orbenchlab import agentic_factory

        return agentic_factory.compile_plan(
            name="batch fixture plan",
            source_binding_digest=kwargs["source_binding_digest"],
            stages=[
                {
                    "id": "only-stage",
                    "role": "agent",
                    "profile": "claude-code",
                    "model": kwargs["author_model"],
                    "prompt": "p",
                    "depends_on": [],
                    "timeout_sec": 5,
                    "max_attempts": 1,
                    "max_budget_usd": 0.1,
                    "required_outputs": [{"path": "factory/out.json", "kind": "json"}],
                }
            ],
        )

    def fake_autopilot(plan, **kwargs):
        candidate_root = Path(kwargs["out"]).parent.name
        if candidate_root == "beta":
            raise factory_batch.factory_autopilot.FactoryAutopilotError("beta broke")
        return {
            "status": "promoted",
            "promotion": {
                "decision": "eligible-for-human-release-review",
                "final_report": {"markdown": str(Path(kwargs["out"]) / "promotion/final-report.md")},
            },
        }

    monkeypatch.setattr(factory_batch.factory_blueprints, "prepare_workspace", fake_prepare)
    monkeypatch.setattr(
        factory_batch.factory_blueprints, "paper_to_benchmark_plan", fake_plan
    )
    monkeypatch.setattr(factory_batch.agentic_factory, "write_plan", lambda plan, path: path)
    monkeypatch.setattr(factory_batch.factory_autopilot, "run", fake_autopilot)
    state = factory_batch.run_batch(
        spec=spec,
        out=tmp_path / "batch",
        provider_env={},
        harbor_executable="/bin/true",
        claude_executable="/bin/true",
        max_total_liability_usd=1000.0,
    )
    assert state["admitted"] == ["alpha", "beta"]
    assert state["skipped"] == ["tampered"]
    by_id = {row["id"]: row for row in state["candidates"]}
    assert by_id["alpha"]["status"] == "promoted"
    assert by_id["alpha"]["promotion_decision"] == "eligible-for-human-release-review"
    assert by_id["beta"]["status"] == "error"
    assert by_id["beta"]["error_class"] == "FactoryAutopilotError"
    persisted = json.loads((tmp_path / "batch" / "batch-state.json").read_text())
    assert persisted["batch_digest"] == state["batch_digest"]
    assert persisted["status_counts"] == {"error": 1, "promoted": 1}


def test_batch_refuses_to_start_over_the_total_liability_cap(tmp_path: Path):
    spec = _spec([_candidate(tmp_path, "alpha"), _candidate(tmp_path, "beta")])
    with pytest.raises(factory_batch.FactoryBatchError, match="liability"):
        factory_batch.run_batch(
            spec=spec,
            out=tmp_path / "batch",
            provider_env={},
            harbor_executable="/bin/true",
            claude_executable="/bin/true",
            max_total_liability_usd=50.0,
        )
