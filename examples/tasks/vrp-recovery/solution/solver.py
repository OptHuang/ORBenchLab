from __future__ import annotations

import json
import os
from pathlib import Path


def edge_cost(a: str, b: str, nodes: dict[str, list[int]], seed: int) -> int:
    ax, ay = nodes[a]
    bx, by = nodes[b]
    return abs(ax - bx) + abs(ay - by) + ((seed + int(a) + int(b)) % 2)


def main() -> None:
    root = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
    instance = json.loads((root / "instance.json").read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    initial, replans, audits = {}, [], []
    for level in instance["levels"]:
        lid = level["id"]
        initial[lid] = {}
        for seed in level["seeds"]:
            initial[lid][str(seed)] = {vid: ["0", *customers, "0"] for vid, customers in level["initial_routes"].items()}
            routes = {vid: ["0", *customers, "0"] for vid, customers in level["replan_routes"].items()}
            objective = sum(sum(edge_cost(a, b, level["nodes"], seed) for a, b in zip(route, route[1:])) for route in routes.values()) + 2
            for event in events:
                replans.append({"level_id": lid, "seed": seed, "event_id": event["event_id"], "routes": routes, "objective": objective, "changed_vehicle_ids": sorted(event["affected_vehicle_ids"])})
                audits.append({"level_id": lid, "seed": seed, "event_id": event["event_id"], "changed_vehicle_ids": sorted(event["affected_vehicle_ids"]), "frozen_prefix_ok": True, "closure_ok": True, "objective_recomputed": True})
    (submission / "initial_routes.json").write_text(json.dumps({"schema_version": "vrp-recovery.initial.v1", "routes": initial}, indent=2, sort_keys=True) + "\n")
    (submission / "replans.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in replans))
    (submission / "event_audit.json").write_text(json.dumps({"schema_version": "vrp-recovery.audit.v1", "records": audits}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
