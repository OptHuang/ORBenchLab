import json
from pathlib import Path

import pytest

from orbenchlab.harbor_controls import HarborControlError, build_receipt, write_receipt


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _job(root: Path, *, control: str, reward: float, passed: int, failed: int) -> Path:
    job = root / f"{control}-job"
    trial = job / f"task__{control}"
    _write_json(
        job / "result.json",
        {
            "id": f"{control}-id",
            "n_total_trials": 1,
            "stats": {"n_completed_trials": 1, "n_errored_trials": 0},
        },
    )
    _write_json(
        trial / "result.json",
        {
            "task_name": "terminal-bench-science/demo-task",
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": None,
        },
    )
    _write_json(
        trial / "verifier/ctrf.json",
        {
            "results": {
                "summary": {
                    "tests": passed + failed,
                    "passed": passed,
                    "failed": failed,
                    "skipped": 0,
                    "pending": 0,
                    "other": 0,
                }
            }
        },
    )
    (trial / "verifier/reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    _write_json(
        trial / "artifacts/manifest.json",
        [
            {
                "source": "/root/submission/solver.py",
                "status": "ok" if control == "oracle" else "failed",
            }
        ],
    )
    return job


def test_builds_pipeline_ready_harbor_control_receipt(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        '[task]\nname = "terminal-bench-science/demo-task"\n', encoding="utf-8"
    )
    oracle = _job(tmp_path, control="oracle", reward=1.0, passed=5, failed=0)
    nop = _job(tmp_path, control="nop", reward=0.0, passed=0, failed=5)

    receipt = build_receipt(task, executed_task_dir=task, oracle_job=oracle, nop_job=nop)
    paths = write_receipt(receipt, tmp_path / "out")

    row = receipt["tasks"][0]
    assert row["task"] == "demo_task"
    assert row["evidence_level"] == "E3"
    assert row["control_gates"]["oracle"]["gate"] == "pass"
    assert row["control_gates"]["nop"]["gate"] == "pass"
    assert paths["json"].is_file() and paths["markdown"].is_file()


def test_rejects_nop_timeout_or_false_acceptance_shape(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('[task]\nname = "terminal-bench-science/demo-task"\n')
    oracle = _job(tmp_path, control="oracle", reward=1.0, passed=1, failed=0)
    bad_nop = _job(tmp_path, control="nop", reward=0.0, passed=1, failed=0)

    with pytest.raises(HarborControlError, match="expected gate"):
        build_receipt(task, executed_task_dir=task, oracle_job=oracle, nop_job=bad_nop)


def test_rejects_executed_task_snapshot_drift(tmp_path):
    task = tmp_path / "task"
    executed = tmp_path / "executed"
    task.mkdir()
    executed.mkdir()
    config = '[task]\nname = "terminal-bench-science/demo-task"\n'
    (task / "task.toml").write_text(config)
    (executed / "task.toml").write_text(config)
    (task / "instruction.md").write_text("current")
    (executed / "instruction.md").write_text("executed")
    oracle = _job(tmp_path, control="oracle", reward=1.0, passed=1, failed=0)
    nop = _job(tmp_path, control="nop", reward=0.0, passed=0, failed=1)

    with pytest.raises(HarborControlError, match="snapshot differs"):
        build_receipt(task, executed_task_dir=executed, oracle_job=oracle, nop_job=nop)
