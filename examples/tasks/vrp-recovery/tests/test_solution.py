from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
INSTANCE = ROOT / "instance.json"
EVENTS = ROOT / "events.jsonl"
SUBMISSION = ROOT / "submission"


def _load() -> tuple[dict, list[dict], dict, list[dict], dict]:
    assert INSTANCE.is_file() and EVENTS.is_file()
    assert (SUBMISSION / "solver.py").is_file()
    initial = json.loads((SUBMISSION / "initial_routes.json").read_text())
    replans = [json.loads(x) for x in (SUBMISSION / "replans.jsonl").read_text().splitlines() if x.strip()]
    audit = json.loads((SUBMISSION / "event_audit.json").read_text())
    return json.loads(INSTANCE.read_text()), [json.loads(x) for x in EVENTS.read_text().splitlines() if x.strip()], initial, replans, audit


def edge_cost(a: str, b: str, nodes: dict[str, list[int]], seed: int) -> int:
    ax, ay = nodes[a]
    bx, by = nodes[b]
    return abs(ax - bx) + abs(ay - by) + ((seed + int(a) + int(b)) % 2)


def route_cost(route: list[str], nodes: dict[str, list[int]], seed: int) -> int:
    return sum(edge_cost(a, b, nodes, seed) for a, b in zip(route, route[1:]))


def check_routes(level: dict, routes: dict[str, list[str]], seed: int) -> None:
    vehicle_by_id = {v["id"]: v for v in level["vehicles"]}
    customers = set(level["demands"])
    seen: list[str] = []
    for vid, route in routes.items():
        assert vid in vehicle_by_id
        assert route[0] == route[-1] == "0"
        body = route[1:-1]
        assert all(node in customers for node in body)
        assert sum(level["demands"][node] for node in body) <= vehicle_by_id[vid]["capacity"]
        seen.extend(body)
    assert set(seen) == customers and len(seen) == len(customers)


def test_replay_is_feasible_and_objective_is_recomputed():
    instance, events, initial, replans, audit = _load()
    assert len(replans) == len(events) * sum(len(level["seeds"]) for level in instance["levels"])
    records = {(r["level_id"], r["seed"], r["event_id"]): r for r in replans}
    audit_records = {(r["level_id"], r["seed"], r["event_id"]): r for r in audit["records"]}
    assert len(records) == len(replans) == len(audit_records)
    for level in instance["levels"]:
        lid = level["id"]
        for seed in level["seeds"]:
            before = initial["routes"][lid][str(seed)]
            check_routes(level, before, seed)
            for event in events:
                rec = records[(lid, seed, event["event_id"])]
                after = rec["routes"]
                check_routes(level, after, seed)
                for vid, prefix in event["frozen_prefix"].items():
                    assert after[vid][1 : 1 + len(prefix)] == prefix
                for vid in before:
                    if vid not in event["affected_vehicle_ids"]:
                        assert after[vid] == before[vid]
                closed = {tuple(edge) for edge in event["closed_edges"]} | {tuple(reversed(edge)) for edge in event["closed_edges"]}
                assert all((a, b) not in closed for route in after.values() for a, b in zip(route, route[1:]))
                expected = sum(route_cost(route, level["nodes"], seed) for route in after.values()) + 2 * len(event["affected_vehicle_ids"])
                assert rec["objective"] == expected <= level["max_objective"]
                ar = audit_records[(lid, seed, event["event_id"])]
                assert ar["frozen_prefix_ok"] and ar["closure_ok"] and ar["objective_recomputed"]

