from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orbenchlab import agentic_factory, factory_finalize, pipeline, task_authoring, volc_rollout
from orbenchlab.cli import main
from orbenchlab.volc_review import REQUIRED_REVIEW_CRITERIA


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
        "printf 'fixture task\\n' > factory/tasks/task-v2/README.md\n"
        "printf '{\"source_content_digest\":\"sha256:8888888888888888888888888888888888888888888888888888888888888888\"}' > factory/tasks/task-v2/paper-provenance.json\nprintf done\n",
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
                "profile": "claude-code",
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
        environments={
            "claude-code": {
                "ANTHROPIC_BASE_URL": VOLC,
                "ANTHROPIC_AUTH_TOKEN": "fixture",
            }
        },
        executables={"claude-code": executable},
    )
    assert result["status"] == "semantic-complete-e1"
    return plan_path, root / "factory-run.json", work, work / "factory/tasks/task-v2"


def _build_cli_session_review(
    semantic_dir: Path,
    *,
    authoring_digest: str,
    static_digest: str,
    paper_digest: str,
    models,
    decision: str = "promising",
) -> Path:
    """Fabricate a valid CLI-agent-session review tree the finalizer can verify.

    Produces, per model, a strict review.json plus a self-consistent
    agent-session receipt with sealed stdout/stderr, and an aggregate binding
    each session by digest -- mirroring factory_review's real output.
    """

    from orbenchlab import agent_sessions, factory_review

    semantic_dir.mkdir(parents=True, exist_ok=True)
    reviewers = []
    bindings = []
    route_digest = "sha256:" + "a" * 64
    executable_digest = "sha256:" + "b" * 64
    for index, model in enumerate(models):
        slug = factory_review._model_slug(model)
        review_root = semantic_dir / slug
        review_root.mkdir(parents=True, exist_ok=True)
        verdict_doc = {
            "decision": decision,
            "shape_complete": True,
            "rubric_complete": True,
            "criteria": [
                {"name": name, "status": "pass", "evidence": "fixture " + name}
                for name in sorted(REQUIRED_REVIEW_CRITERIA)
            ],
        }
        from orbenchlab import agent_sessions

        (review_root / "review.json").write_text(json.dumps(verdict_doc), encoding="utf-8")
        normalized = factory_review._validate_review_document(verdict_doc)
        session_id = ("%032x" % (index + 1))[:32]
        session_dir = review_root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "stdout.bin").write_bytes(b"stdout-" + slug.encode())
        (session_dir / "stderr.bin").write_bytes(b"")
        result_tree = agent_sessions._tree_digest(review_root, exclude=review_root / "sessions")
        receipt = {
            "schema_version": "orbenchlab.agent-session.receipt.v1",
            "session_id": session_id,
            "identity": {
                "profile": "claude-code",
                "model": model,
                "stage": f"factory-review/{slug}/round-1",
                "route_digest": route_digest,
                "executable_digest": executable_digest,
                "max_budget_usd": 1.0,
            },
            "status": "completed",
            "result_tree_digest": result_tree,
            "stdout_digest": "sha256:" + hashlib.sha256((session_dir / "stdout.bin").read_bytes()).hexdigest(),
            "stderr_digest": "sha256:" + hashlib.sha256(b"").hexdigest(),
        }
        receipt["receipt_digest"] = agent_sessions._digest(
            {k: v for k, v in receipt.items() if k != "receipt_digest"}
        )
        (session_dir / "receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        reviewers.append({"model": model, "review": normalized})
        bindings.append(
            {
                "model": model,
                "session_id": session_id,
                "session_receipt_digest": factory_finalize._file_digest(session_dir / "receipt.json"),
                "route_digest": route_digest,
                "executable_digest": executable_digest,
                "verdict_digest": factory_finalize._value_digest(normalized),
                "status": "completed",
            }
        )
    payload = {
        "schema_version": "orbenchlab.volc-authoring-review.v1",
        "review_mechanism": "cli-agent-session",
        "round": 1,
        "task_tree_digest": authoring_digest,
        "static_receipt_digest": static_digest,
        "paper_digest": paper_digest,
        "models": [str(m) for m in models],
        "review_count": len(models),
        "aggregate_decision": (
            "promising-needs-harbor" if decision == "promising" else "needs-human"
        ),
        "reviewers": reviewers,
        "session_bindings": bindings,
    }
    payload["review_digest"] = factory_finalize._value_digest(payload)
    path = semantic_dir / "volc-authoring-review.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _evidence(tmp_path: Path, task: Path) -> tuple[Path, Path, Path, Path, Path]:
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
    trials = []
    for model, status in (("frontier", "pass"), ("weak", "fail")):
        for trial in range(1, 6):
            trials.append(
                {
                    "model": model,
                    "trial": trial,
                    "hint_level": 0,
                    "status": status,
                    "phase": "verifier",
                    "request_digest": "sha256:" + "1" * 64,
                    "response_digest": "sha256:" + "2" * 64,
                    "solver_digest": "sha256:" + "3" * 64,
                    "verifier": {"receipt_valid": True, "status": status},
                    **({"failure_mode": "verifier_failed"} if status == "fail" else {}),
                }
            )
    arms = volc_rollout._summarize_trials(trials)
    discrimination = volc_rollout._discrimination_summary(
        arms, ["frontier", "weak"], repetitions=5
    )
    calibration = _signed(
        {
            "schema_version": "orbenchlab.screening-report.v1",
            "task": task_id,
            "task_tree_digest": runtime_digest,
            "tasks": [{"task": task_id, "family": task_id, "arms": arms, "discrimination": discrimination, "discrimination_index_observed_gap": discrimination["observed_gap"], "decision": "review-promising", "evidence_level": "E3"}],
            "trials": trials,
            "run_contract": {"models": ["frontier", "weak"], "repetitions": 5, "hint_levels": [0], "max_tokens": 2400, "timeout_sec": 120, "test_image": "fixture:test"},
        },
        "report_digest",
    )
    static_path = _write(tmp_path / "static.json", static)
    semantic_path = _build_cli_session_review(
        tmp_path / "semantic",
        authoring_digest=authoring_digest,
        static_digest=static["receipt_digest"],
        paper_digest="sha256:" + "8" * 64,
        models=("review-a", "review-b"),
    )
    harbor_path = _write(tmp_path / "harbor.json", harbor)
    calibration_path = _write(tmp_path / "calibration.json", calibration)
    models = []
    for name, arm in sorted(arms.items()):
        models.append(
            {
                "model": name,
                "n_observed": arm["n"],
                "n_complete": arm["complete"],
                "metric_n": arm["metric_n"],
                "solve_n": arm["solve_n"],
                "quality_n": arm["quality_n"],
                "feasibility_n": arm["feasibility_n"],
                "solve_rate": arm["solve_rate"],
                "quality_pass_rate": arm["quality_pass_rate"],
                "mean_feasibility": arm["mean_feasibility"],
                "infra_exceptions": arm["infra_exceptions"],
                "failure_modes": arm["failure_modes"],
                "evidence_levels": ["E3"],
                "source_reports": [str(calibration_path)],
            }
        )
    card = {
        "task_card_schema_version": "orbenchlab.task-card.v1",
        "task_id": task_id,
        "family": task_id,
        "title": "Fixture task",
        "purpose": "Exercise a strict optimization task.",
        "source": {"status": "bound"},
        "difficulty": {
            "axes": [{"name": "instance_size", "levels": ["small", "large"]}],
            "interventions": {"hint_levels": [0, 1]},
            "declared": True,
        },
        "performance": {
            "models": models,
            "control_screenings": [],
            "observed_gaps": [discrimination["observed_gap"]],
        },
        "decision": "review-promising",
        "evidence": {
            "level": "E3",
            "task_genome_path": "fixture-genome.json",
            "task_genome_digest": "sha256:" + "9" * 64,
            "source_reports": [str(harbor_path), str(calibration_path)],
            "report_digests": [
                factory_finalize._file_digest(harbor_path),
                factory_finalize._file_digest(calibration_path),
            ],
        },
        "limitations": ["Fixture evidence."],
        "intake": {"path": None, "present": False},
        "summary_markdown": "# Fixture task\n",
    }
    summary_path = _write(
        tmp_path / "task-cards.json",
        {"pipeline_schema_version": "orbenchlab.pipeline.v1", "cards": [card]},
    )
    return static_path, semantic_path, harbor_path, calibration_path, summary_path


def test_finalize_promotes_only_complete_independent_evidence(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, semantic, harbor, calibration, summary = _evidence(tmp_path, task)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        semantic_review_path=semantic,
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
    static, semantic, harbor, calibration, summary = _evidence(tmp_path, task)
    harbor_doc = json.loads(harbor.read_text())
    harbor_doc["tasks"][0]["control_gates"]["nop"]["reward"] = 1.0
    _write(harbor, harbor_doc)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        semantic_review_path=semantic,
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
    static, semantic, harbor, calibration, summary = _evidence(tmp_path, task)
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
        semantic_review_path=semantic,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert next(g for g in receipt["gates"] if g["name"] == "final_summary")["status"] == "pass"


def test_calibration_summary_cannot_override_raw_trial_outcomes(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, semantic, harbor, calibration, summary = _evidence(tmp_path, task)
    calibration_doc = json.loads(calibration.read_text())
    calibration_doc["trials"][0]["status"] = "fail"
    calibration_doc["trials"][0]["failure_mode"] = "verifier_failed"
    calibration_doc["trials"][0]["verifier"]["status"] = "fail"
    calibration_doc["report_digest"] = factory_finalize._value_digest(
        {key: value for key, value in calibration_doc.items() if key != "report_digest"}
    )
    _write(calibration, calibration_doc)
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        semantic_review_path=semantic,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert next(g for g in receipt["gates"] if g["name"] == "model_calibration")["status"] == "fail"


def test_agent_written_all_good_summary_is_not_a_pipeline_task_card(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, semantic, harbor, calibration, _ = _evidence(tmp_path, task)
    summary = _write(tmp_path / "agent-summary.json", {"summary": "all gates passed"})
    receipt = factory_finalize.build_receipt(
        plan_path=plan,
        factory_run_path=run,
        workdir=work,
        task_dir=task,
        static_receipt_path=static,
        semantic_review_path=semantic,
        harbor_receipt_path=harbor,
        calibration_receipt_path=calibration,
        final_summary_path=summary,
    )
    assert receipt["promoted"] is False
    assert next(g for g in receipt["gates"] if g["name"] == "final_summary")["status"] == "fail"


def test_static_gate_requires_real_passing_provenance_rows(tmp_path: Path):
    plan, run, work, task = _factory(tmp_path)
    static, semantic, harbor, calibration, summary = _evidence(tmp_path, task)
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
        semantic_review_path=semantic,
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
    static, semantic, harbor, _, summary = _evidence(tmp_path, task)
    out = tmp_path / "final"
    code = main(
        [
            "agent-factory", "finalize", "--plan", str(plan), "--factory-run", str(run),
            "--workdir", str(work), "--task-dir", str(task), "--static-receipt", str(static),
            "--semantic-review", str(semantic),
            "--harbor-receipt", str(harbor), "--calibration-receipt", str(tmp_path / "missing.json"),
            "--final-summary", str(summary), "--out", str(out),
        ]
    )
    assert code == 8
    assert json.loads(capsys.readouterr().out)["promoted"] is False
    written = json.loads((out / "factory-finalization.json").read_text(encoding="utf-8"))
    assert written["decision"] == "not-promoted"


def test_mechanical_gate_accepts_real_validate_task_review_rows():
    task = Path("examples/tasks/alphaevolve-scheduling")
    receipt = task_authoring.validate_task(task, paper_provenance=task / "paper-provenance.json")
    assert receipt["decision"] in {"ready-for-human-review", "ready-for-harbor-validation"}
    assert any(row["status"] == "review" for row in receipt["implementation_criteria"])
    validator = factory_finalize._static_validator(task_authoring._task_tree_digest(task))
    assert validator(receipt)["receipt_digest"] == receipt["receipt_digest"]


def test_semantic_gate_rejects_forged_single_or_incomplete_review(tmp_path: Path):
    _, _, _, task = _factory(tmp_path)
    static, semantic, _, _, _ = _evidence(tmp_path, task)
    static_digest = json.loads(static.read_text())["receipt_digest"]
    semantic_root = semantic.parent
    validate = factory_finalize._semantic_validator(
        task_authoring._task_tree_digest(task),
        static_digest,
        "sha256:" + "8" * 64,
        semantic_root=semantic_root,
    )
    original = json.loads(semantic.read_text())
    # The genuine CLI-session review verifies.
    assert validate(original)["review_mechanism"] == "cli-agent-session"

    def resign(value):
        value["review_digest"] = factory_finalize._value_digest(
            {k: v for k, v in value.items() if k != "review_digest"}
        )
        return value

    for mutate in ("forged", "single", "incomplete"):
        value = resign(json.loads(json.dumps(original)))
        if mutate == "forged":
            value["aggregate_decision"] = "needs-human"
        elif mutate == "single":
            value["models"] = value["models"][:1]
            value["reviewers"] = value["reviewers"][:1]
            value["review_count"] = 1
            value["session_bindings"] = value["session_bindings"][:1]
        else:
            value["reviewers"][0]["review"]["criteria"].pop()
        resign(value)
        try:
            validate(value)
        except factory_finalize.FactoryFinalizeError:
            pass
        else:
            raise AssertionError(f"{mutate} semantic review accepted")

    # Defect A: a document that fakes a passing rubric with no CLI sessions
    # (or with the sessions deleted, or the verdict tampered) must fail.
    forged_no_sessions = resign(
        {
            **json.loads(json.dumps(original)),
            "review_mechanism": "forged-no-sessions",
            "session_bindings": [],
        }
    )
    with pytest.raises(factory_finalize.FactoryFinalizeError):
        validate(forged_no_sessions)

    deleted_receipts = resign(json.loads(json.dumps(original)))
    first_slug = __import__("orbenchlab.factory_review", fromlist=["_model_slug"])._model_slug(
        deleted_receipts["models"][0]
    )
    import shutil as _shutil

    _shutil.rmtree(semantic_root / first_slug / "sessions")
    with pytest.raises(factory_finalize.FactoryFinalizeError):
        validate(deleted_receipts)


def test_real_pipeline_card_is_review_promising_with_bound_genome(tmp_path: Path):
    _, _, _, task = _factory(tmp_path)
    _, _, harbor, calibration, _ = _evidence(tmp_path, task)
    genome = _write(tmp_path / "genome.json", {
        "family": volc_rollout._task_id(task), "title": "Fixture task",
        "difficulty_axes": {"instance_size": {"levels": ["small", "large"]}},
        "provenance": {"source": "fixture"},
    })
    out = tmp_path / "pipeline"
    pipeline.run(out=out, task_inputs=[genome], screening_inputs=[harbor, calibration])
    cards = json.loads((out / "task-cards.json").read_text())["cards"]
    assert len(cards) == 1
    assert cards[0]["task_id"] == volc_rollout._task_id(task)
    assert cards[0]["decision"] == "review-promising"


def _harbor_matrix_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    from orbenchlab import volc_rollout

    task = tmp_path / "matrix-task"
    task.mkdir()
    (task / "task.toml").write_text('[task]\nname = "demo_matrix"\n', encoding="utf-8")
    task_id = volc_rollout._task_id(task)
    tree = volc_rollout._task_tree_digest(task)
    trials = []
    for model, passed in (("frontier", 5), ("weak", 0)):
        for attempt in range(1, 6):
            status = "pass" if attempt <= passed else "fail"
            trials.append(
                {
                    "trial_id": factory_finalize._value_digest([model, attempt]),
                    "model_id": model,
                    "attempt": attempt,
                    "status": status,
                    "agent_failure_class": None,
                    "usage": {"cost_usd": 0.1},
                    "reward": 1.0 if status == "pass" else 0.0,
                    "trial_name": f"{task_id}__{attempt}",
                    "trial_result_digest": "sha256:" + "1" * 64,
                    "reward_digest": "sha256:" + "2" * 64,
                    "ctrf_digest": "sha256:" + "3" * 64,
                    "ctrf_summary": {
                        "tests": 2,
                        "passed": 2 if status == "pass" else 0,
                        "failed": 0 if status == "pass" else 2,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "trajectories": [
                        {
                            "path": f"{task_id}__{attempt}/agent/trajectory.json",
                            "digest": "sha256:" + "9" * 64,
                            "steps": 3,
                        }
                    ],
                }
            )
    matrix = {
        "schema_version": "orbenchlab.harbor-model-matrix.v1",
        "task": task_id,
        "task_tree_digest": tree,
        "models": ["frontier", "weak"],
        "repetitions": 5,
        "rectangular": True,
        "trials": trials,
        "jobs": [],
        "provider_route_digest": "sha256:" + "4" * 64,
        "preregistration_digest": None,
        "agent": {
            "name": "claude-code",
            "executable_digest": "sha256:" + "a" * 64,
            "max_budget_usd_per_trial": 1.0,
            "max_turns_per_trial": 40,
            "budget_enforcement": "claude-cli-max-budget-usd",
            "max_job_attempts_per_model": 2,
            "maximum_model_liability_usd": 20.0,
            "liability_accounting": "crash-safe whole-job reservation before subprocess launch",
        },
        "evidence_level": "E3",
        "checkpoint_capability": False,
        "limitations": ["Verifier-grounded Harbor trajectories; not TB-Science acceptance."],
    }
    matrix["receipt_digest"] = factory_finalize._value_digest(
        {key: value for key, value in matrix.items() if key != "receipt_digest"}
    )
    digests = {
        key: "sha256:" + char * 64
        for key, char in zip(
            (
                "job_result_digest",
                "trial_result_digest",
                "ctrf_digest",
                "reward_digest",
                "artifact_manifest_digest",
            ),
            "bcdef",
        )
    }
    gates = {
        "oracle": {
            "gate": "pass",
            "control": "oracle",
            "task_name": task_id,
            "reward": 1.0,
            "ctrf_summary": {"tests": 2, "passed": 2, "failed": 0, "skipped": 0, "pending": 0, "other": 0},
            **digests,
        },
        "nop": {
            "gate": "pass",
            "control": "nop",
            "task_name": task_id,
            "reward": 0.0,
            "ctrf_summary": {"tests": 2, "passed": 0, "failed": 2, "skipped": 0, "pending": 0, "other": 0},
            **digests,
        },
    }
    controls = {
        "schema_version": "orbenchlab.screening-report.v1",
        "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
        "task_tree_digest": tree,
        "authoring_task_tree_digest": tree,
        "executed_task_tree_digest": tree,
        "tasks": [
            {
                "task": task_id,
                "family": task_id,
                "arms": {},
                "control_gates": gates,
                "decision": "collect-more-evidence",
                "evidence_level": "E3",
            }
        ],
    }
    controls["report_digest"] = factory_finalize._value_digest(
        {key: value for key, value in controls.items() if key != "report_digest"}
    )
    return task, matrix, controls


def test_harbor_matrix_screening_is_accepted_as_calibration_evidence(tmp_path: Path):
    from orbenchlab import harbor_model_matrix, volc_rollout

    task, matrix, controls = _harbor_matrix_fixture(tmp_path)
    report = harbor_model_matrix.build_screening_report(
        matrix, harbor_controls=controls, out=tmp_path / "screening"
    )
    validator = factory_finalize._calibration_validator(
        volc_rollout._task_tree_digest(task)
    )
    details = validator(report)
    assert details["calibration_kind"] == "harbor-model-matrix"
    assert details["models"] == ["frontier", "weak"]
    assert details["minimum_repetitions"] == 5
    assert details["observed_gap"] == 1.0
    assert details["gap_95_lower_bound"] > 0


def test_harbor_matrix_calibration_rejects_unverified_trials(tmp_path: Path):
    from orbenchlab import harbor_model_matrix, volc_rollout

    task, matrix, controls = _harbor_matrix_fixture(tmp_path)
    report = harbor_model_matrix.build_screening_report(
        matrix, harbor_controls=controls, out=tmp_path / "screening"
    )
    forged = dict(report)
    forged["trials"] = [dict(row) for row in report["trials"]]
    forged["trials"][0]["verifier"] = {**forged["trials"][0]["verifier"], "receipt_valid": False}
    forged["report_digest"] = factory_finalize._value_digest(
        {key: value for key, value in forged.items() if key != "report_digest"}
    )
    validator = factory_finalize._calibration_validator(
        volc_rollout._task_tree_digest(task)
    )
    with pytest.raises(factory_finalize.FactoryFinalizeError):
        validator(forged)


def test_post_session_verdict_forgery_is_rejected(tmp_path: Path):
    # Reproduces the reviewer's exploit: two completed sessions whose real
    # verdict is 'revise' are rewritten to 'promising' post-session, with every
    # public sha256 in the aggregate recomputed. The result-tree drift is
    # detected, so promotion is refused.
    from orbenchlab import agent_sessions, factory_review

    semantic = tmp_path / "semantic"
    path = _build_cli_session_review(
        semantic,
        authoring_digest="sha256:" + "7" * 64,
        static_digest="sha256:" + "5" * 64,
        paper_digest="sha256:" + "8" * 64,
        models=("review-a", "review-b"),
        decision="revise",  # honest reviewers said revise
    )
    validate = factory_finalize._semantic_validator(
        "sha256:" + "7" * 64, "sha256:" + "5" * 64, "sha256:" + "8" * 64,
        semantic_root=semantic,
    )
    # The honest document is not promising -> rejected on the decision gate.
    with pytest.raises(factory_finalize.FactoryFinalizeError):
        validate(_load_json(path))

    # Forge: flip every reviewer's on-disk verdict + the aggregate to promising
    # and recompute all public digests, exactly as the exploit does.
    promising = {
        "decision": "promising", "shape_complete": True, "rubric_complete": True,
        "criteria": [
            {"name": n, "status": "pass", "evidence": "forged " + n}
            for n in sorted(REQUIRED_REVIEW_CRITERIA)
        ],
    }
    normalized = factory_review._validate_review_document(promising)
    doc = _load_json(path)
    for slug_model in doc["models"]:
        slug = factory_review._model_slug(slug_model)
        (semantic / slug / "review.json").write_text(json.dumps(promising), encoding="utf-8")
    for reviewer, binding in zip(doc["reviewers"], doc["session_bindings"]):
        reviewer["review"] = normalized
        binding["verdict_digest"] = factory_finalize._value_digest(normalized)
    doc["aggregate_decision"] = "promising-needs-harbor"
    doc["review_digest"] = factory_finalize._value_digest(
        {k: v for k, v in doc.items() if k != "review_digest"}
    )
    # The session receipts (with the sealed result_tree_digest) were NOT changed,
    # so the recomputed result tree drifts from the receipt: forgery rejected.
    with pytest.raises(factory_finalize.FactoryFinalizeError, match="drifted"):
        validate(doc)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
