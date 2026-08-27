from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path


def _schedule_for_order(jobs: list[dict], seed: int) -> dict:
    machine_ready: dict[str, int] = {}
    schedule = {"operations": [], "makespan": 0}
    for job in jobs:
        job_ready = 0
        for index, operation in enumerate(job["operations"]):
            duration = operation["duration"] + ((seed + index) % 2)
            start = max(job_ready, machine_ready.get(operation["machine"], 0))
            end = start + duration
            schedule["operations"].append(
                {
                    "operation_id": operation["id"],
                    "job_id": job["id"],
                    "operation_index": index,
                    "machine": operation["machine"],
                    "start": start,
                    "end": end,
                }
            )
            job_ready = end
            machine_ready[operation["machine"]] = end
            schedule["makespan"] = max(schedule["makespan"], end)
    return schedule


def solve(instance: dict) -> dict:
    schedules = {}
    for level in instance["levels"]:
        schedules[level["id"]] = {}
        for seed in level["seeds"]:
            # A bounded permutation search is sufficient for this small control
            # fixture and gives the agent a meaningful, reproducible objective
            # to improve against without importing a solver package.
            candidates = (
                _schedule_for_order(list(order), seed)
                for order in permutations(level["jobs"])
            )
            schedules[level["id"]][str(seed)] = min(candidates, key=lambda value: value["makespan"])
    return {"schema_version": "alphaevolve-scheduling.solution.v1", "schedules": schedules}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = solve(json.loads(Path(args.instance).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
