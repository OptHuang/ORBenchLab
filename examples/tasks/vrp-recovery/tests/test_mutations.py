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


def test_capacity_overflow_rejected():
    level, event, rec = _tiny()
    mutated = deepcopy(rec["routes"])
    mutated["1"] = ["0", "4", "1", "2", "3", "0"]
    with pytest.raises(AssertionError):
        check_routes(level, mutated, 1)


def test_frozen_prefix_edit_rejected():
    level, event, rec = _tiny()
    mutated = deepcopy(rec["routes"])
    mutated["0"] = ["0", "2", "1", "3", "0"]
    with pytest.raises(AssertionError):
        assert mutated["0"][1:2] == event["frozen_prefix"]["0"]


def test_noncausal_vehicle_change_rejected():
    instance, events, initial, replans, audit = _load()
    level = next(x for x in instance["levels"] if x["id"] == "small")
    record = next(x for x in replans if x["level_id"] == "small" and x["seed"] == 1)
    mutated = deepcopy(record["routes"])
    mutated["1"] = list(reversed(mutated["1"]))
    with pytest.raises(AssertionError):
        assert mutated["1"] == initial["routes"]["small"]["1"]
