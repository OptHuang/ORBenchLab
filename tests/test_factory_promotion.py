from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import (
    agentic_factory,
    factory_promotion,
    factory_supervisor,
    harbor_model_matrix,
    task_authoring,
    volc_rollout,
)
from orbenchlab.volc_review import REQUIRED_REVIEW_CRITERIA

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"
DIGEST = "sha256:" + "a" * 64
VOLC = "https://ark.cn-beijing.volces.com/api/coding"
PROVIDER = {"ANTHROPIC_BASE_URL": VOLC, "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}
SELECTED = "factory/tasks/task-v2/alphaevolve-scheduling"


def _value_digest(value) -> str:
    return factory_promotion._value_digest(value)


def _stage(stage_id, *, model, depends_on=(), outputs, postchecks=()):
    row = {
        "id": stage_id,
        "role": f"{stage_id} agent",
        "profile": "claude-code",
        "model": model,
        "prompt": f"Do the {stage_id} work.",
        "depends_on": list(depends_on),
        "timeout_sec": 60,
        "max_attempts": 2,
        "max_budget_usd": 0.25,
        "max_output_bytes": 1024 * 1024,
        "required_outputs": outputs,
    }
    if postchecks:
        row["postchecks"] = list(postchecks)
    return row


def _plan() -> dict:
    return agentic_factory.compile_plan(
        name="promotion factory",
        source_binding_digest=DIGEST,
        stages=[
            _stage(
                "task-repair-v2",
                model="author",
                outputs=[{"path": "factory/tasks/task-v2", "kind": "directory"}],
                postchecks=("tb-science-static-gate",),
            ),
            _stage(
                "task-review-science",
                model="rev-a",
                depends_on=("task-repair-v2",),
                outputs=[{"path": "factory/reviews/task-review-science.md", "kind": "text"}],
            ),
            _stage(
                "task-review-verifier",
                model="rev-b",
                depends_on=("task-repair-v2",),
                outputs=[{"path": "factory/reviews/task-review-verifier.md", "kind": "text"}],
            ),
            _stage(
                "final-synthesis",
                model="rev-a",
                depends_on=("task-review-science", "task-review-verifier"),
                outputs=[
                    {"path": "factory/final/task-review-summary.json", "kind": "json"},
                    {"path": "factory/final/task-genome.json", "kind": "json"},
                ],
            ),
        ],
    )


def _workspace(tmp_path: Path) -> Path:
    workdir = tmp_path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    shutil.copy2(
        GOOD_TASK / "paper-provenance.json",
        workdir / "factory-input" / "paper-provenance.json",
    )
    return workdir


def _factory_cli(tmp_path: Path, workdir: Path) -> Path:
    provenance = workdir / "factory-input" / "paper-provenance.json"
    provenance_digest = "sha256:" + hashlib.sha256(provenance.read_bytes()).hexdigest()
    genome = {
        "family": "alphaevolve_scheduling",
        "title": "AlphaEvolve scheduling benchmark",
        "design_goal": "Schedule jobs under paper-derived constraints.",
        "selected_task": SELECTED,
        "source": {
            "title": "AlphaEvolve",
            "url": "https://arxiv.org/abs/2506.13131",
            "paper_provenance_digest": provenance_digest,
        },
        "difficulty_axes": {
            "instance_scale": {
                "levels": [1, 2, 3],
                "meaning": "number of jobs",
                "expected_direction": "solve rate decreases",
            }
        },
    }
    summary = {
        "selected_task": SELECTED,
        "task_summary": "A scheduling task derived from the AlphaEvolve paper.",
        "evidence_level": "E1-agent-session-process",
        "limitations": ["Semantic completion only; gates own promotion."],
    }
    executable = tmp_path / "factory-agent"
    executable.write_text(
        f"""#!/bin/sh
payload=$(cat)
mkdir -p factory/tasks factory/reviews factory/final
case "$payload" in
  *'task-repair-v2 agent'*)
    rm -rf factory/tasks/task-v2
    mkdir -p factory/tasks/task-v2
    cp -r "{GOOD_TASK}" factory/tasks/task-v2/alphaevolve-scheduling
    ;;
  *'task-review-science agent'*)
    printf 'science review ok\\n' > factory/reviews/task-review-science.md
    ;;
  *'task-review-verifier agent'*)
    printf 'verifier review ok\\n' > factory/reviews/task-review-verifier.md
    ;;
  *'final-synthesis agent'*)
    cat > factory/final/task-genome.json <<'JSON'
{json.dumps(genome, indent=2)}
JSON
    cat > factory/final/task-review-summary.json <<'JSON'
{json.dumps(summary, indent=2)}
JSON
    ;;
esac
printf done
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _runtime_evidence(evidence_root: Path, task: Path) -> None:
    task_id = volc_rollout._task_id(task)
    tree = volc_rollout._task_tree_digest(task)
    trials = []
    for model, passed in (("frontier", 5), ("weak", 0)):
        for attempt in range(1, 6):
            status = "pass" if attempt <= passed else "fail"
            trials.append(
                {
                    "trial_id": _value_digest([model, attempt]),
                    "model_id": model,
                    "attempt": attempt,
                    "status": status,
                    "agent_failure_class": None,
                    "usage": {"cost_usd": 0.05},
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
                            "steps": 4,
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
            "max_budget_usd_per_trial": 0.5,
            "max_turns_per_trial": 40,
            "budget_enforcement": "claude-cli-max-budget-usd",
            "max_job_attempts_per_model": 2,
            "maximum_model_liability_usd": 10.0,
            "liability_accounting": "crash-safe whole-job reservation before subprocess launch",
        },
        "evidence_level": "E3",
        "checkpoint_capability": False,
        "limitations": ["Verifier-grounded Harbor trajectories; not TB-Science acceptance."],
    }
    matrix["receipt_digest"] = _value_digest(
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
    controls["report_digest"] = _value_digest(
        {key: value for key, value in controls.items() if key != "report_digest"}
    )
    controls_dir = evidence_root / "baseline" / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    (controls_dir / "harbor-control-screening.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True), encoding="utf-8"
    )
    harbor_model_matrix.build_screening_report(
        matrix, harbor_controls=controls, out=evidence_root / "baseline" / "matrix"
    )


def _fake_semantic_review(monkeypatch) -> None:
    def fake_process(target, args, *, timeout_sec):
        task, paper_path, static_json, output, _env, models, _timeout, _tokens = args
        static = json.loads(Path(static_json).read_text(encoding="utf-8"))
        paper = json.loads(
            (Path(task) / "paper-provenance.json").read_text(encoding="utf-8")
        )
        reviewers = [
            {
                "model": model,
                "review": {
                    "decision": "promising",
                    "shape_complete": True,
                    "rubric_complete": True,
                    "criteria": [
                        {"name": name, "status": "pass", "evidence": "inspected"}
                        for name in sorted(REQUIRED_REVIEW_CRITERIA)
                    ],
                },
            }
            for model in models
        ]
        review = {
            "schema_version": "orbenchlab.volc-authoring-review.v1",
            "task_tree_digest": task_authoring._task_tree_digest(Path(task)),
            "static_receipt_digest": static["receipt_digest"],
            "paper_digest": paper["source_content_digest"],
            "aggregate_decision": "promising-needs-harbor",
            "models": list(models),
            "review_count": len(models),
            "reviewers": reviewers,
        }
        review["review_digest"] = _value_digest(review)
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "volc-authoring-review.json").write_text(
            json.dumps(review, indent=2, sort_keys=True), encoding="utf-8"
        )
        return None

    monkeypatch.setattr(factory_supervisor, "_run_builtin_process", fake_process)


def _run_semantic_factory(tmp_path: Path) -> tuple[dict, Path, Path]:
    plan = _plan()
    workdir = _workspace(tmp_path)
    factory_out = tmp_path / "factory-run"
    executable = _factory_cli(tmp_path, workdir)
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=factory_out,
        environments={"claude-code": PROVIDER},
        executables={"claude-code": executable},
    )
    assert result["status"] == "semantic-complete-e1"
    return plan, workdir, factory_out


def test_promotion_runs_all_gates_and_writes_final_report(tmp_path: Path, monkeypatch):
    plan, workdir, factory_out = _run_semantic_factory(tmp_path)
    evidence_root = tmp_path / "autopilot"
    _runtime_evidence(evidence_root, workdir / SELECTED)
    _fake_semantic_review(monkeypatch)
    state = {
        "barriers": {
            "baseline": {"observed_usage": {"cost_usd": 1.25, "n_output_tokens": 100}}
        }
    }
    summary = factory_promotion.run_promotion(
        plan=plan,
        workdir=workdir,
        factory_out=factory_out,
        evidence_root=evidence_root,
        out=evidence_root / "promotion",
        provider_env=PROVIDER,
        state=state,
    )
    assert summary["promoted"] is True
    assert summary["decision"] == "eligible-for-human-release-review"
    assert summary["gates"]["static"]["status"] == "pass"
    assert summary["gates"]["runtime_evidence"]["status"] == "pass"
    assert summary["gates"]["finalize"]["status"] == "pass"
    final = json.loads(
        (evidence_root / "promotion" / "final" / "factory-finalization.json").read_text()
    )
    assert final["promoted"] is True
    assert final["evidence_level"] == "E3"
    calibration_gate = next(
        gate for gate in final["gates"] if gate["name"] == "model_calibration"
    )
    assert calibration_gate["calibration_kind"] == "harbor-model-matrix"
    report = (evidence_root / "promotion" / "final-report.md").read_text(encoding="utf-8")
    assert "frontier@hint-0" in report
    assert "任务是什么" in report
    assert "可复现命令" in report
    cards = json.loads(
        (evidence_root / "promotion" / "cards" / "task-cards.json").read_text()
    )
    assert cards["cards"][0]["decision"] == "review-promising"
    # Promotion is resumable: a second call reuses every receipt byte-for-byte.
    again = factory_promotion.run_promotion(
        plan=plan,
        workdir=workdir,
        factory_out=factory_out,
        evidence_root=evidence_root,
        out=evidence_root / "promotion",
        provider_env=PROVIDER,
        state=state,
    )
    assert again["promoted"] is True
    assert again["promotion_digest"] == summary["promotion_digest"]


def test_promotion_without_matching_runtime_evidence_blocks_explicitly(
    tmp_path: Path, monkeypatch
):
    plan, workdir, factory_out = _run_semantic_factory(tmp_path)
    evidence_root = tmp_path / "autopilot"
    evidence_root.mkdir()
    _fake_semantic_review(monkeypatch)
    summary = factory_promotion.run_promotion(
        plan=plan,
        workdir=workdir,
        factory_out=factory_out,
        evidence_root=evidence_root,
        out=evidence_root / "promotion",
        provider_env=PROVIDER,
        state={},
    )
    assert summary["promoted"] is False
    assert summary["gates"]["runtime_evidence"]["reason"] == "promotion_evidence_missing"
    assert (evidence_root / "promotion" / "final-report.md").is_file()


def test_promotion_never_promotes_a_blocked_semantic_review(tmp_path: Path, monkeypatch):
    plan, workdir, factory_out = _run_semantic_factory(tmp_path)
    evidence_root = tmp_path / "autopilot"
    _runtime_evidence(evidence_root, workdir / SELECTED)

    def blocked_process(target, args, *, timeout_sec):
        return "builtin_nonzero_exit"

    monkeypatch.setattr(factory_supervisor, "_run_builtin_process", blocked_process)
    summary = factory_promotion.run_promotion(
        plan=plan,
        workdir=workdir,
        factory_out=factory_out,
        evidence_root=evidence_root,
        out=evidence_root / "promotion",
        provider_env=PROVIDER,
        state={},
    )
    assert summary["promoted"] is False
    assert summary["gates"]["semantic_review"]["status"] == "blocked"
