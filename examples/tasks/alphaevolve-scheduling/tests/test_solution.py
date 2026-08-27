from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
INSTANCE = ROOT / "instance.json"
SUBMISSION = ROOT / "submission" / "solution.json"


def _load() -> tuple[dict, dict]:
    assert INSTANCE.is_file(), "frozen instance.json is missing"
    assert SUBMISSION.is_file(), "submission/solution.json is missing"
    instance = json.loads(INSTANCE.read_text(encoding="utf-8"))
    output = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    assert output.get("schema_version") == "alphaevolve-scheduling.solution.v1"
    return instance, output


def _check_level(level: dict, schedule: dict, seed: int) -> None:
    jobs = {job["id"]: job for job in level["jobs"]}
    rows = schedule.get("operations")
    assert isinstance(rows, list)
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
        assert set(schedules[level_id]) == {str(seed) for seed in level["seeds"]}
        for seed in level["seeds"]:
            _check_level(level, schedules[level_id][str(seed)], seed)
