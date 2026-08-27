from __future__ import annotations

from copy import deepcopy

import pytest

from test_solution import _load, check_routes, route_cost


def _tiny():
    instance, events, initial, replans, audit = _load()
    level = next(x for x in instance["levels"] if x["id"] == "tiny")
    record = next(x for x in replans if x["level_id"] == "tiny" and x["seed"] == 1)
    return level, events[0], record


def test_duplicate_service_rejected():
    level, event, rec = _tiny()
    mutated = deepcopy(rec["routes"])
    mutated["1"] = ["0", "1", "0"]
    with pytest.raises(AssertionError):
        check_routes(level, mutated, 1)


def test_closed_edge_rejected():
    level, event, rec = _tiny()
    route = deepcopy(rec["routes"])["0"]
    route[-2] = "2"
    with pytest.raises(AssertionError):
        assert all((a, b) not in {("0", "2"), ("2", "0")} for a, b in zip(route, route[1:]))


def test_fabricated_objective_rejected():
    level, event, rec = _tiny()
    expected = sum(route_cost(route, level["nodes"], 1) for route in rec["routes"].values()) + 2
    with pytest.raises(AssertionError):
        assert rec["objective"] + 1 == expected
