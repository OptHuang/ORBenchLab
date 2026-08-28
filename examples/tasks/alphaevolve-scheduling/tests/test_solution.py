from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
INSTANCE = ROOT / "instance.json"
REFERENCE_BOUNDS = ROOT / "data" / "reference-bounds.json"
SUBMISSION = ROOT / "submission" / "solution.json"
SOLVER = ROOT / "submission" / "solver.py"

# Runtime budget for the whole grading pass. The submitted solver is re-run
# once here; it must finish well within this bound (the outer Harbor verifier
# wall-clock is larger, but a correct solver for these frozen instances runs in
# well under a second). Documented in instruction.md so a slow-but-correct
# solver is a task-authoring signal, not a mysterious kill.
SOLVER_RUNTIME_BUDGET_SEC = 60


def _load() -> tuple[dict, dict]:
    assert INSTANCE.is_file(), "frozen instance.json is missing"
    assert REFERENCE_BOUNDS.is_file(), "reference-bound provenance is missing"
    assert SOLVER.is_file(), "submission/solver.py is missing"
    subprocess.run(
        [sys.executable, str(SOLVER), "--instance", str(INSTANCE), "--output", str(SUBMISSION)],
        cwd=ROOT,
        check=True,
        timeout=SOLVER_RUNTIME_BUDGET_SEC,
    )
    assert SUBMISSION.is_file(), "submission/solution.json is missing"
    instance = json.loads(INSTANCE.read_text(encoding="utf-8"))
    output = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    assert output.get("schema_version") == "alphaevolve-scheduling.solution.v1"
    bounds = json.loads(REFERENCE_BOUNDS.read_text(encoding="utf-8"))
    assert bounds.get("schema_version") == "alphaevolve-scheduling.reference-bounds.v1"
    assert bounds.get("provenance")
    for level in instance["levels"]:
        assert bounds["bounds"][level["id"]] == level["max_makespan"]
    return instance, output


def _check_level(level: dict, schedule: dict, seed: int) -> None:
    # Give a precise contract message when the per-(level, seed) entry is the
    # wrong shape (e.g. a bare list of rows) so the failure is diagnosable as a
    # schema violation, not an opaque AttributeError misread as solver quality.
    assert isinstance(schedule, dict), (
        "each schedules[level][seed] entry must be an object with 'operations' "
        f"and 'makespan', not {type(schedule).__name__}"
    )
    jobs = {job["id"]: job for job in level["jobs"]}
    rows = schedule.get("operations")
    assert isinstance(rows, list), "schedule['operations'] must be a list of operation rows"
    assert isinstance(schedule.get("makespan"), int) and not isinstance(
        schedule.get("makespan"), bool
    ), "schedule['makespan'] must be an integer"
    expected = {
        operation["id"]
        for job in level["jobs"]
        for operation in job["operations"]
    }
    actual = [row.get("operation_id") for row in rows]
    assert len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected
    by_job: dict[str, list[dict]] = {job_id: [] for job_id in jobs}
    by_machine: dict[str, list[dict]] = {}
    for row in rows:
        operation = next(
            op for job in level["jobs"] for op in job["operations"] if op["id"] == row["operation_id"]
        )
        expected_duration = operation["duration"] + ((seed + row["operation_index"]) % 2)
        assert isinstance(row.get("start"), int) and isinstance(row.get("end"), int)
        assert row["start"] >= 0 and row["end"] > row["start"]
        assert row["end"] - row["start"] == expected_duration
        by_job[row["job_id"]].append(row)
        by_machine.setdefault(row["machine"], []).append(row)
    for job in level["jobs"]:
        ordered = sorted(by_job[job["id"]], key=lambda row: row["operation_index"])
        assert [row["operation_id"] for row in ordered] == [op["id"] for op in job["operations"]]
        for previous, current in zip(ordered, ordered[1:]):
            assert previous["end"] <= current["start"]
    for machine_rows in by_machine.values():
        ordered = sorted(machine_rows, key=lambda row: (row["start"], row["end"]))
        for previous, current in zip(ordered, ordered[1:]):
            assert previous["end"] <= current["start"]
    makespan = max(row["end"] for row in rows)
    assert schedule.get("makespan") == makespan
    assert makespan <= level["max_makespan"]


def test_all_levels_are_feasible_and_objective_is_recomputed():
    instance, output = _load()
    schedules = output.get("schedules")
    assert isinstance(schedules, dict)
    levels = {level["id"]: level for level in instance["levels"]}
    assert set(schedules) == set(levels)
    for level_id, level in levels.items():
        assert isinstance(schedules[level_id], dict), (
            f"schedules['{level_id}'] must map str(seed) to a schedule object"
        )
        assert set(schedules[level_id]) == {str(seed) for seed in level["seeds"]}
        for seed in level["seeds"]:
            _check_level(level, schedules[level_id][str(seed)], seed)
