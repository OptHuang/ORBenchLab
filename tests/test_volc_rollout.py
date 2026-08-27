from pathlib import Path

import json

from orbenchlab import volc_rollout
from orbenchlab.volc_review import VolcConfig
from orbenchlab.volc_rollout import (
    VolcRolloutError,
    _control_gates,
    _extract_solver,
    _hint,
    _discrimination_summary,
    _run_container,
    _summarize_trials,
    _task_id,
)


ROOT = Path(__file__).parents[1]


def test_task_identity_matches_task_genome_family():
    assert _task_id(ROOT / "examples/tasks/alphaevolve-scheduling") == "alphaevolve_scheduling"
    assert _task_id(ROOT / "examples/tasks/vrp-recovery") == "vrp_recovery"


def test_solver_extractor_accepts_bounded_file_aliases():
    assert _extract_solver({"solver_py": "print(1)"}) == "print(1)"
    assert _extract_solver({"files": {"submission/solver.py": "print(2)"}}) == "print(2)"


def test_hint_ladder_is_task_specific_and_explicit():
    alpha = _hint(ROOT / "examples/tasks/alphaevolve-scheduling", 2)
    vrp = _hint(ROOT / "examples/tasks/vrp-recovery", 1)
    assert "alphaevolve-scheduling.solution.v1" in alpha
    assert "initial_routes.json" in vrp


def test_trial_summary_separates_outcome_failures_from_infrastructure():
    trials = [
        {"model": "ark", "hint_level": 0, "status": "pass"},
        {"model": "ark", "hint_level": 0, "status": "fail", "failure_mode": "verifier_failed"},
        {"model": "ark", "hint_level": 0, "status": "invalid_output", "failure_mode": "missing_solver_py"},
        {"model": "ark", "hint_level": 0, "status": "infra_error", "error_type": "VolcReviewError"},
        {"model": "ark", "hint_level": 1, "status": "timeout", "error_type": "VerifierTimeout"},
    ]

    arms = _summarize_trials(trials)

    base = arms["ark@hint-0"]
    assert base["n"] == 4
    assert base["complete"] == 3
    assert base["metric_n"] == 3
    assert base["solve_rate"] == 1 / 3
    assert base["infra_exceptions"] == ["VolcReviewError"]
    assert base["failure_modes"] == ["missing_solver_py", "verifier_failed"]
    hinted = arms["ark@hint-1"]
    assert hinted["complete"] == 0
    assert hinted["metric_n"] == 0
    assert hinted["solve_rate"] is None
    assert hinted["infra_exceptions"] == ["VerifierTimeout"]
    assert hinted["failure_modes"] == ["verifier_timeout_inconclusive"]


def test_discrimination_requires_rectangular_repeated_positive_ci():
    arms = {
        "frontier@hint-0": {"solve_n": 5, "solve_rate": 1.0, "infra_exceptions": []},
        "open@hint-0": {"solve_n": 5, "solve_rate": 0.0, "infra_exceptions": []},
    }
    result = _discrimination_summary(arms, ["frontier", "open"], repetitions=5)
    assert result["rectangular"] is True
    assert result["observed_gap"] == 1.0
    assert result["gap_95_lower_bound"] > 0
    assert result["promising"] is True

    too_small = _discrimination_summary(arms, ["frontier", "open"], repetitions=4)
    assert too_small["rectangular"] is False
    assert too_small["promising"] is False


def test_suite_writes_one_pipeline_ready_report(monkeypatch, tmp_path):
    tasks = []
    for name in ("alpha", "vrp"):
        root = tmp_path / name
        root.mkdir()
        (root / "task.toml").write_text(f'[task]\nname = "terminal-bench-science/{name}"\n')
        tasks.append((root, f"{name}:tests"))

    def fake_run(task_dir, **kwargs):
        task = Path(task_dir).name
        return {
            "tasks": [{"task": task, "family": task, "arms": {}, "decision": "collect-more-evidence", "evidence_level": "E3"}],
            "trials": [{"model": "ark", "trial": 1, "hint_level": 0, "status": "pass"}],
            "report_digest": "sha256:" + ("a" if task == "alpha" else "b") * 64,
            "task_tree_digest": "sha256:" + ("c" if task == "alpha" else "d") * 64,
        }

    monkeypatch.setattr(volc_rollout, "run_rollout", fake_run)
    report = volc_rollout.run_suite(
        tasks,
        config=VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret", "ark"),
        models=["ark"],
        out=tmp_path / "out",
        repetitions=1,
        hint_levels=[0, 1],
    )

    assert [row["task"] for row in report["tasks"]] == ["alpha", "vrp"]
    assert all(row.get("task_tree_digest") for row in report["tasks"])
    assert {row["task"] for row in report["trials"]} == {"alpha", "vrp"}
    written = json.loads((tmp_path / "out/screening-report.json").read_text())
    assert written["suite_schema_version"] == "orbenchlab.volc-screening-suite.v1"


def test_control_gates_require_oracle_acceptance_and_nop_rejection():
    gates = _control_gates(
        [
            {"control": "oracle", "status": "pass", "verifier": {"receipt_valid": True}},
            {"control": "nop", "status": "fail", "verifier": {"receipt_valid": True}},
        ]
    )
    assert gates["oracle"]["gate"] == "pass"
    assert gates["nop"]["gate"] == "pass"
    timeout = _control_gates(
        [{"control": "nop", "status": "timeout", "verifier": {"receipt_valid": False}}]
    )
    assert timeout["nop"]["gate"] == "infrastructure-error"
    assert timeout["oracle"]["gate"] == "missing"
    accepted = _control_gates(
        [{"control": "nop", "status": "pass", "verifier": {"receipt_valid": True}}]
    )
    assert accepted["nop"]["gate"] == "fail"


def _mock_docker_run(monkeypatch, stage: Path, *, ctrf=None, reward=None, returncode=0):
    def fake_run(*args, **kwargs):
        logs = stage / "logs/verifier"
        logs.mkdir(parents=True, exist_ok=True)
        if ctrf is not None:
            (logs / "ctrf.json").write_text(json.dumps(ctrf))
        if reward is not None:
            (logs / "reward.txt").write_text(str(reward))
        return b"pytest output", b"", returncode, None

    monkeypatch.setattr(volc_rollout.agent_sessions, "_bounded_process", fake_run)


def test_verifier_receipt_fails_closed_when_ctrf_is_missing(monkeypatch, tmp_path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    _mock_docker_run(monkeypatch, stage, reward="1")

    result = _run_container(task, stage, test_image="tests:image", timeout_sec=10)

    assert result["status"] == "infra_error"
    assert result["error_type"] == "VerifierReceiptMissing"
    assert result["receipt_valid"] is False


def test_verifier_receipt_binds_reward_to_ctrf(monkeypatch, tmp_path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    ctrf = {
        "results": {
            "summary": {"tests": 2, "passed": 2, "failed": 0, "skipped": 0, "pending": 0, "other": 0}
        }
    }
    _mock_docker_run(monkeypatch, stage, ctrf=ctrf, reward="1")

    result = _run_container(task, stage, test_image="tests:image", timeout_sec=10)

    assert result["status"] == "pass"
    assert result["receipt_valid"] is True
    assert result["ctrf"]["summary"]["passed"] == 2


def test_verifier_receipt_rejects_test_count_drift(monkeypatch, tmp_path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    ctrf = {
        "results": {
            "summary": {"tests": 1, "passed": 1, "failed": 0, "skipped": 0, "pending": 0, "other": 0}
        }
    }
    _mock_docker_run(monkeypatch, stage, ctrf=ctrf, reward="1")

    result = _run_container(
        task, stage, test_image="tests:image", timeout_sec=10, expected_tests=2
    )

    assert result["status"] == "infra_error"
    assert result["error_type"] == "VerifierTestCountMismatch"


def test_rollout_rejects_partial_or_duplicate_control_contract(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('[task]\nname = "terminal-bench-science/task"\n')
    config = VolcConfig("https://ark.cn-beijing.volces.com/api/coding", "secret", "ark")
    for controls in (["oracle"], ["oracle", "oracle"], []):
        try:
            volc_rollout.run_rollout(
                task,
                config=config,
                models=["ark"],
                test_image="tests:image",
                out=tmp_path / "out",
                controls=controls,
            )
        except VolcRolloutError:
            pass
        else:
            raise AssertionError("partial control contract was accepted")
