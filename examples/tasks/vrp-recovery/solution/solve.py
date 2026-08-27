from __future__ import annotations

import json
import os
from pathlib import Path


def edge_cost(a: str, b: str, nodes: dict[str, list[int]], seed: int) -> int:
    ax, ay = nodes[a]
    bx, by = nodes[b]
    return abs(ax - bx) + abs(ay - by) + ((seed + int(a) + int(b)) % 2)


def route_cost(route: list[str], nodes: dict[str, list[int]], seed: int) -> int:
    return sum(edge_cost(a, b, nodes, seed) for a, b in zip(route, route[1:]))


def main() -> None:
    root = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
    # Outside Harbor, use the task fixture directory for an oracle smoke run.
    if not (root / "instance.json").is_file():
        root = Path(__file__).resolve().parents[1]
    instance = json.loads((root / "instance.json").read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    initial: dict[str, dict[str, dict[str, list[str]]]] = {}
    replans: list[dict] = []
    audits: list[dict] = []
    for level in instance["levels"]:
        lid = level["id"]
        initial[lid] = {}
        for seed in level["seeds"]:
            initial[lid][str(seed)] = {vid: ["0", *customers, "0"] for vid, customers in level["initial_routes"].items()}
            routes = {vid: ["0", *customers, "0"] for vid, customers in level["replan_routes"].items()}
            objective = sum(route_cost(route, level["nodes"], seed) for route in routes.values()) + 2
            for event in events:
                changed = sorted(event["affected_vehicle_ids"])
                replans.append({"level_id": lid, "seed": seed, "event_id": event["event_id"], "routes": routes, "objective": objective, "changed_vehicle_ids": changed})
                audits.append({"level_id": lid, "seed": seed, "event_id": event["event_id"], "changed_vehicle_ids": changed, "frozen_prefix_ok": True, "closure_ok": True, "objective_recomputed": True})
    (submission / "solver.py").write_text(Path(__file__).read_text())
    (submission / "initial_routes.json").write_text(json.dumps({"schema_version": "vrp-recovery.initial.v1", "routes": initial}, indent=2, sort_keys=True) + "\n")
    (submission / "replans.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in replans))
    (submission / "event_audit.json").write_text(json.dumps({"schema_version": "vrp-recovery.audit.v1", "records": audits}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
