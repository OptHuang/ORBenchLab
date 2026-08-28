from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import agentic_factory, factory_autopilot


PROVIDER = {
    "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding",
    "ANTHROPIC_AUTH_TOKEN": "fixture-secret",
}


def _executable(path: Path, name: str) -> Path:
    value = path / name
    value.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    value.chmod(0o755)
    return value


def _workspace(path: Path) -> Path:
    root = path / "work"
    trusted = root / "factory-input" / "trusted"
    trusted.mkdir(parents=True)
    trusted.chmod(0o555)
    trusted.parent.chmod(0o555)
    return root


def _plan() -> dict:
    previous: list[str] = []
    stages = []
    kinds = {
        "task-repair-v2": "directory",
        "variant-author": "directory",
    }
    for name in (
        "task-repair-v2",
        "runtime-controls",
        "pilot-frontier",
        "pilot-weak",
        "variant-author",
        "calibration",
        "final-synthesis",
    ):
        stages.append(
            {
                "id": name,
                "role": name,
                "profile": "claude-code",
                "model": "fixture-model",
                "prompt": f"write {name}",
                "depends_on": list(previous),
                "timeout_sec": 5,
                "max_attempts": 1,
                "max_budget_usd": 0.1,
                "required_outputs": [
                    {
                        "path": f"factory/{name}",
                        "kind": kinds.get(name, "json"),
                    }
                ],
            }
        )
        previous = [name]
    return agentic_factory.compile_plan(
        name="autopilot fixture",
        source_binding_digest="sha256:" + "a" * 64,
        stages=stages,
    )


def test_trusted_bundle_is_atomic_read_only_and_idempotent(tmp_path: Path):
    work = _workspace(tmp_path)
    source = tmp_path / "source"
    (source / "trace").mkdir(parents=True)
    (source / "receipt.json").write_text('{"ok":true}\n', encoding="utf-8")
    (source / "trace" / "trajectory.json").write_text('{"steps":[1]}\n', encoding="utf-8")
    first = factory_autopilot.install_trusted_bundle(
        workdir=work,
        relative="baseline",
        source=source,
        source_receipts={"matrix": "sha256:" + "b" * 64},
    )
    second = factory_autopilot.install_trusted_bundle(
        workdir=work,
        relative="baseline",
        source=source,
        source_receipts={"matrix": "sha256:" + "b" * 64},
    )
    destination = work / "factory-input/trusted/baseline"
    assert first == second
    assert first["file_count"] == 2
    assert destination.joinpath("trusted-bundle-manifest.json").is_file()
    assert destination.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in destination.rglob("*") if path.is_file())

    (source / "receipt.json").write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="differs"):
        factory_autopilot.install_trusted_bundle(
            workdir=work,
            relative="baseline",
            source=source,
            source_receipts={"matrix": "sha256:" + "b" * 64},
        )


def test_trusted_bundle_rejects_source_symlink(tmp_path: Path):
    work = _workspace(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    target.write_text("not trusted", encoding="utf-8")
    source.joinpath("escape").symlink_to(target)
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="symlink"):
        factory_autopilot.install_trusted_bundle(
            workdir=work,
            relative="baseline",
            source=source,
            source_receipts={},
        )


def test_autopilot_injects_each_runtime_barrier_once_and_resumes(
    tmp_path: Path, monkeypatch
):
    plan = _plan()
    work = _workspace(tmp_path)
    harbor = _executable(tmp_path, "harbor")
    claude = _executable(tmp_path, "claude")
    order = [
        "runtime-controls",
        "pilot-frontier",
        "pilot-weak",
        "variant-author",
        "calibration",
        "final-synthesis",
    ]
    cursor = {"value": 0}
    calls = {"baseline": 0, "difficulty": 0, "factory": 0}

    def fake_initialise(plan_value, out, **kwargs):
        status = "semantic-complete-e1" if cursor["value"] == len(order) else "active"
        return {
            "status": status,
            "run_digest": "sha256:" + "c" * 64,
            "stages": {},
        }, cursor["value"] > 0

    def fake_ready(plan_value, run_value):
        return [] if run_value["status"] != "active" else [order[cursor["value"]]]

    def fake_run(*args, **kwargs):
        calls["factory"] += 1
        cursor["value"] += 1
        return {
            "status": "active",
            "run_digest": "sha256:" + f"{cursor['value']:064x}",
        }

    def fake_baseline(**kwargs):
        calls["baseline"] += 1
        return {"matrix_receipt_digest": "sha256:" + "d" * 64}

    def fake_difficulty(**kwargs):
        calls["difficulty"] += 1
        return {"difficulty_receipt_digest": "sha256:" + "e" * 64}

    monkeypatch.setattr(factory_autopilot.agentic_factory, "initialise", fake_initialise)
    monkeypatch.setattr(factory_autopilot.agentic_factory, "ready_stages", fake_ready)
    monkeypatch.setattr(factory_autopilot.agentic_factory, "run_factory", fake_run)
    monkeypatch.setattr(factory_autopilot, "_ensure_baseline", fake_baseline)
    monkeypatch.setattr(factory_autopilot, "_ensure_difficulty", fake_difficulty)
    monkeypatch.setattr(factory_autopilot, "_validate_barriers", lambda *args: None)
    promotions = []

    def fake_promotion(**kwargs):
        promotions.append(kwargs)
        return {
            "selected_task": "factory/tasks/task-v2/demo",
            "task_id": "demo",
            "promoted": True,
            "decision": "eligible-for-human-release-review",
            "gates": {},
            "final_report": {"markdown": str(tmp_path / "autopilot/promotion/final-report.md")},
            "final_report_digest": "sha256:" + "9" * 64,
            "promotion_digest": "sha256:" + "8" * 64,
        }

    monkeypatch.setattr(
        factory_autopilot.factory_promotion, "run_promotion", fake_promotion
    )
    monkeypatch.setattr(factory_autopilot, "_selected_task", lambda workspace: None)
    kwargs = dict(
        workdir=work,
        factory_out=tmp_path / "factory-run",
        out=tmp_path / "autopilot",
        harbor_executable=harbor,
        claude_executable=claude,
        frontier_model="frontier",
        weak_model="weak",
        provider_env=PROVIDER,
        repetitions=5,
        max_budget_usd=0.5,
        max_variants=3,
        intervention_study=False,
    )
    result = factory_autopilot.run(plan, **kwargs)
    assert result["status"] == "promoted"
    assert result["promotion"]["decision"] == "eligible-for-human-release-review"
    assert calls == {"baseline": 1, "difficulty": 1, "factory": 6}
    assert len(promotions) == 1
    assert promotions[0]["evidence_root"] == (tmp_path / "autopilot").resolve()
    # A resumed promoted run returns immediately without re-running promotion.
    resumed = factory_autopilot.run(plan, **kwargs)
    assert resumed["status"] == "promoted"
    assert len(promotions) == 1
    persisted = json.loads((tmp_path / "autopilot/autopilot-state.json").read_text())
    assert set(persisted["barriers"]) == {"baseline", "difficulty"}
    persisted["barriers"]["baseline"]["matrix_receipt_digest"] = "sha256:" + "f" * 64
    (tmp_path / "autopilot/autopilot-state.json").write_text(json.dumps(persisted))
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="immutable run"):
        factory_autopilot.run(plan, **kwargs)


def test_autopilot_identity_binds_repair_bounds_and_liability(tmp_path: Path, monkeypatch):
    # The repair round/budget bounds and the full liability ledger are part of
    # the autopilot identity, so a resume that widens them is rejected as drift
    # rather than silently accepted.
    plan = _plan()
    work = _workspace(tmp_path)
    harbor = _executable(tmp_path, "harbor")
    claude = _executable(tmp_path, "claude")

    order = [
        "runtime-controls",
        "pilot-frontier",
        "pilot-weak",
        "variant-author",
        "calibration",
        "final-synthesis",
    ]
    cursor = {"value": 0}

    def fake_initialise(plan_value, out, **kwargs):
        status = "semantic-complete-e1" if cursor["value"] == len(order) else "active"
        return {"status": status, "run_digest": "sha256:" + "c" * 64, "stages": {}}, cursor["value"] > 0

    def fake_ready(plan_value, run_value):
        return [] if run_value["status"] != "active" else [order[cursor["value"]]]

    def fake_run(*args, **kwargs):
        cursor["value"] += 1
        return {"status": "active", "run_digest": "sha256:" + f"{cursor['value']:064x}"}

    monkeypatch.setattr(factory_autopilot.agentic_factory, "initialise", fake_initialise)
    monkeypatch.setattr(factory_autopilot.agentic_factory, "ready_stages", fake_ready)
    monkeypatch.setattr(factory_autopilot.agentic_factory, "run_factory", fake_run)
    monkeypatch.setattr(
        factory_autopilot, "_ensure_baseline", lambda **k: {"matrix_receipt_digest": "sha256:" + "d" * 64}
    )
    monkeypatch.setattr(
        factory_autopilot, "_ensure_difficulty", lambda **k: {"difficulty_receipt_digest": "sha256:" + "e" * 64}
    )
    monkeypatch.setattr(factory_autopilot, "_validate_barriers", lambda *a: None)
    monkeypatch.setattr(
        factory_autopilot.factory_promotion,
        "run_promotion",
        lambda **k: {
            "selected_task": "factory/tasks/task-v2/demo",
            "task_id": "demo",
            "promoted": True,
            "decision": "eligible-for-human-release-review",
            "gates": {},
            "final_report": {"markdown": str(tmp_path / "autopilot/promotion/final-report.md")},
            "final_report_digest": "sha256:" + "9" * 64,
            "promotion_digest": "sha256:" + "8" * 64,
        },
    )
    monkeypatch.setattr(factory_autopilot, "_selected_task", lambda workspace: None)
    kwargs = dict(
        workdir=work,
        factory_out=tmp_path / "factory-run",
        out=tmp_path / "autopilot",
        harbor_executable=harbor,
        claude_executable=claude,
        frontier_model="frontier",
        weak_model="weak",
        provider_env=PROVIDER,
        repetitions=5,
        max_budget_usd=0.5,
        max_variants=3,
        intervention_study=False,
    )
    factory_autopilot.run(plan, **kwargs, max_runtime_repair_rounds=1, repair_max_budget_usd=1.0)
    state = json.loads((tmp_path / "autopilot/autopilot-state.json").read_text())
    identity = state["identity"]
    assert identity["runtime_repair"] == {
        "max_runtime_repair_rounds": 1,
        "repair_max_budget_usd": 1.0,
    }
    ledger = identity["liability_ledger"]
    assert ledger["runtime_repair_usd"] == round((1 + 3) * 1 * 1.0, 6)
    assert ledger["maximum_total_usd"] >= ledger["harbor_matrix_usd"]
    assert identity["credential_transport"] == "host-side-relay-per-session-scoped-token"
    # A resume that widens the repair bound is a different immutable run.
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="immutable run"):
        factory_autopilot.run(
            plan, **kwargs, max_runtime_repair_rounds=2, repair_max_budget_usd=1.0
        )
    # An absurd repair bound is rejected outright.
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="runtime-repair"):
        factory_autopilot.run(
            plan, **kwargs, max_runtime_repair_rounds=99, repair_max_budget_usd=1.0
        )


def test_autopilot_rejects_worst_case_harbor_liability_before_launch(tmp_path: Path):
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="liability"):
        factory_autopilot.run(
            _plan(),
            workdir=tmp_path / "missing-work",
            factory_out=tmp_path / "factory-run",
            out=tmp_path / "autopilot",
            harbor_executable=tmp_path / "missing-harbor",
            claude_executable=tmp_path / "missing-claude",
            frontier_model="frontier",
            weak_model="weak",
            provider_env=PROVIDER,
            repetitions=5,
            max_budget_usd=1.0,
            max_variants=3,
            max_harbor_liability_usd=1.0,
        )


def test_autopilot_rejects_overlapping_evidence_boundaries(tmp_path: Path):
    work = tmp_path / "work"
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="non-overlapping"):
        factory_autopilot.run(
            _plan(),
            workdir=work,
            factory_out=tmp_path / "factory-run",
            out=work / "autopilot",
            harbor_executable=tmp_path / "missing-harbor",
            claude_executable=tmp_path / "missing-claude",
            frontier_model="frontier",
            weak_model="weak",
            provider_env=PROVIDER,
            repetitions=5,
            max_budget_usd=0.5,
            max_variants=3,
        )


def test_barrier_shape_fails_closed_before_filesystem_lookup(tmp_path: Path):
    with pytest.raises(factory_autopilot.FactoryAutopilotError, match="must be an object"):
        factory_autopilot._validate_barriers(
            {"barriers": {"baseline": "forged", "difficulty": "forged"}},
            tmp_path / "missing",
        )
