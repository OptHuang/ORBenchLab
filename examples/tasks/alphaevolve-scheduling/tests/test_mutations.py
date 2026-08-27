from __future__ import annotations

from copy import deepcopy

import pytest

from test_solution import _check_level, _load


def _tiny(instance: dict, output: dict) -> tuple[dict, dict, int]:
    seed = 1
    level = next(level for level in instance["levels"] if level["id"] == "tiny")
    return level, output["schedules"]["tiny"][str(seed)], seed


def test_duplicate_operation_is_rejected():
    instance, output = _load()
    level, schedule, seed = _tiny(instance, output)
    mutated = deepcopy(schedule)
    mutated["operations"].append(deepcopy(mutated["operations"][0]))
    with pytest.raises(AssertionError):
        _check_level(level, mutated, seed)


def test_machine_overlap_is_rejected():
    instance, output = _load()
    level, schedule, seed = _tiny(instance, output)
    mutated = deepcopy(schedule)
    mutated["operations"][1]["start"] = mutated["operations"][0]["start"]
    mutated["operations"][1]["end"] = mutated["operations"][0]["end"]
    with pytest.raises(AssertionError):
        _check_level(level, mutated, seed)


def test_fabricated_makespan_is_rejected():
    instance, output = _load()
    level, schedule, seed = _tiny(instance, output)
    mutated = deepcopy(schedule)
    mutated["makespan"] = 0
    with pytest.raises(AssertionError):
        _check_level(level, mutated, seed)


def test_precedence_violation_is_rejected():
    instance, output = _load()
    level, schedule, seed = _tiny(instance, output)
    mutated = deepcopy(schedule)
    first = next(row for row in mutated["operations"] if row["job_id"] == "J1" and row["operation_index"] == 0)
    second = next(row for row in mutated["operations"] if row["job_id"] == "J1" and row["operation_index"] == 1)
    second["start"] = first["start"]
    second["end"] = second["start"] + (second["end"] - second["start"])
    with pytest.raises(AssertionError):
        _check_level(level, mutated, seed)
