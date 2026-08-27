from __future__ import annotations

import json
from pathlib import Path

from orbenchlab import agentic_factory, factory_finalize, task_authoring, volc_rollout
from orbenchlab.cli import main


VOLC = "https://ark.cn-beijing.volces.com/api/coding"
DIGEST = "sha256:" + "a" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _signed(value: dict, field: str) -> dict:
    value[field] = factory_finalize._value_digest(value)
    return value


def _factory(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    work = tmp_path / "work"
    work.mkdir()
    executable = tmp_path / "fixture-agent"
    executable.write_text(
        "#!/bin/sh\ncat >/dev/null\nmkdir -p factory/tasks/task-v2\n"
        "printf '[task]\\nname = \"fixture_task\"\\n' > factory/tasks/task-v2/task.toml\n"
        "printf 'fixture task\\n' > factory/tasks/task-v2/README.md\nprintf done\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    plan = agentic_factory.compile_plan(
        name="finalize fixture",
        source_binding_digest=DIGEST,
        stages=[
            {
                "id": "author",
                "role": "author",
                "profile": "codex",
                "model": "fixture",
                "prompt": "author task",
                "depends_on": [],
                "timeout_sec": 5,
                "max_attempts": 1,
                "max_budget_usd": 0.25,
                "required_outputs": [{"path": "factory/tasks/task-v2", "kind": "directory"}],
            }
        ],
    )
    root = tmp_path / "run"
    plan_path = agentic_factory.write_plan(plan, root / "input-plan.json")
    result = agentic_factory.run_factory(
        plan,
        workdir=work,
        out=root,
        environments={"codex": {"OPENAI_BASE_URL": VOLC, "OPENAI_API_KEY": "fixture"}},
        executables={"codex": executable},
    )
    assert result["status"] == "semantic-complete-e1"
    return plan_path, root / "factory-run.json", work, work / "factory/tasks/task-v2"


def _evidence(tmp_path: Path, task: Path) -> tuple[Path, Path, Path, Path]:
    authoring_digest = task_authoring._task_tree_digest(task)
    runtime_digest = volc_rollout._task_tree_digest(task)
    task_id = volc_rollout._task_id(task)
    criteria = [
        {"name": name, "status": "pass", "reason": "fixture"}
        for name in task_authoring.IMPLEMENTATION_CRITERIA
    ]
    static = _signed(
        {
            "authoring_schema_version": task_authoring.AUTHORING_SCHEMA_VERSION,
            "task_tree_digest": authoring_digest,
            "decision": "ready-for-harbor-validation",
            "implementation_criteria": criteria,
            "provenance_checks": [{"name": "paper", "status": "pass", "reason": "fixture"}],
        },
        "receipt_digest",
    )
    digest_fields = {
        name: "sha256:" + character * 64
        for name, character in zip(
            ("job_result_digest", "trial_result_digest", "ctrf_digest", "reward_digest", "artifact_manifest_digest"),
            "bcdef",
        )
    }
    controls = {
        "oracle": {"gate": "pass", "control": "oracle", "task_name": task_id, "reward": 1.0, "ctrf_summary": {"tests": 2, "passed": 2, "failed": 0, "skipped": 0, "pending": 0, "other": 0}, **digest_fields},
        "nop": {"gate": "pass", "control": "nop", "task_name": task_id, "reward": 0.0, "ctrf_summary": {"tests": 2, "passed": 0, "failed": 2, "skipped": 0, "pending": 0, "other": 0}, **digest_fields},
    }
    harbor = _signed(
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
            "task_tree_digest": runtime_digest,
            "authoring_task_tree_digest": runtime_digest,
            "executed_task_tree_digest": runtime_digest,
            "tasks": [{"task": task_id, "family": task_id, "arms": {}, "control_gates": controls, "decision": "collect-more-evidence", "evidence_level": "E3"}],
        },
        "report_digest",
    )
    arms = {
        "frontier@hint-0": {"model_id": "frontier", "hint_level": 0, "solve_n": 5, "infra_exceptions": [], "solve_rate": 0.8},
        "weak@hint-0": {"model_id": "weak", "hint_level": 0, "solve_n": 5, "infra_exceptions": [], "solve_rate": 0.0},
    }
    calibration = _signed(
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "task_tree_digest": runtime_digest,
            "tasks": [{"task": task_id, "family": task_id, "arms": arms, "discrimination": {"rectangular": True, "promising": True}, "decision": "review-promising", "evidence_level": "E3"}],
        },
        "report_digest",
    )
    return (
        _write(tmp_path / "static.json", static),
        _write(tmp_path / "harbor.json", harbor),
        _write(tmp_path / "calibration.json", calibration),
        _write(tmp_path / "summary.json", {"summary": "human review packet"}),
    )


def test_finalize_promotes_only_complete_independent_evidence(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, harbor, calibration, summary = _evidence(tmp_path, task)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is True
    assert receipt["decision"] == "eligible-for-human-release-review"
    assert receipt["evidence_level"] == "E3"
    assert all(gate["status"] == "pass" for gate in receipt["gates"])
    assert receipt["receipt_digest"].startswith("sha256:")


def test_finalize_missing_or_forged_gate_is_not_promoted(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, harbor, calibration, summary = _evidence(tmp_path, task)
    harbor_doc = json.loads(harbor.read_text())
    harbor_doc["tasks"][0]["control_gates"]["nop"]["reward"] = 1.0
    _write(harbor, harbor_doc)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        harbor_receipt_path=harbor,
        calibration_receipt_path=tmp_path / "missing-calibration.json",
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert receipt["decision"] == "not-promoted"
    failed = {gate["name"] for gate in receipt["gates"] if gate["status"] == "fail"}
    assert {"harbor_oracle_nop", "model_calibration"} <= failed


def test_agent_summary_cannot_override_a_blocked_static_gate(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, harbor, calibration, summary = _evidence(tmp_path, task)
    static_doc = json.loads(static.read_text())
    static_doc["decision"] = "blocked"
    static_doc["receipt_digest"] = factory_finalize._value_digest(
        {key: value for key, value in static_doc.items() if key != "receipt_digest"}
    )
    _write(static, static_doc)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert next(g for g in receipt["gates"] if g["name"] == "final_summary")["status"] == "pass"


def test_static_gate_requires_real_passing_provenance_rows(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, harbor, calibration, summary = _evidence(tmp_path, task)
    static_doc = json.loads(static.read_text())
    static_doc["provenance_checks"] = ["not-a-check"]
    static_doc["receipt_digest"] = factory_finalize._value_digest(
        {key: value for key, value in static_doc.items() if key != "receipt_digest"}
    )
    _write(static, static_doc)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert next(g for g in receipt["gates"] if g["name"] == "static_authoring")["status"] == "fail"


def test_finalize_cli_writes_not_promoted_receipt_when_gate_is_missing(
    tmp_path: Path, capsys
):
    plan, run, work, task = _factory(tmp_path)
    static, harbor, _, summary = _evidence(tmp_path, task)
    out = tmp_path / "final"
    code = main(
        [
            "agent-factory", "finalize", "--plan", str(plan), "--factory-run", str(run),
            "--workdir", str(work), "--task-dir", str(task), "--static-receipt", str(static),
            "--harbor-receipt", str(harbor), "--calibration-receipt", str(tmp_path / "missing.json"),
            "--final-summary", str(summary), "--out", str(out),
        ]
    )
    assert code == 8
    assert json.loads(capsys.readouterr().out)["promoted"] is False
    written = json.loads((out / "factory-finalization.json").read_text(encoding="utf-8"))
    assert written["decision"] == "not-promoted"
