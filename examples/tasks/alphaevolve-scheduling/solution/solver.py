from __future__ import annotations

import json
import os
from itertools import permutations
from pathlib import Path


def schedule_for_order(jobs: list[dict], seed: int) -> dict:
    machine_ready: dict[str, int] = {}
    schedule = {"operations": [], "makespan": 0}
    for job in jobs:
        job_ready = 0
        for index, operation in enumerate(job["operations"]):
            duration = operation["duration"] + ((seed + index) % 2)
            start = max(job_ready, machine_ready.get(operation["machine"], 0))
            end = start + duration
            schedule["operations"].append({"operation_id": operation["id"], "job_id": job["id"], "operation_index": index, "machine": operation["machine"], "start": start, "end": end})
            job_ready = end
            machine_ready[operation["machine"]] = end
            schedule["makespan"] = max(schedule["makespan"], end)
    return schedule


def main() -> None:
    root = Path(os.environ.get("ORBENCH_TASK_ROOT", "/root"))
    instance = json.loads((root / "instance.json").read_text(encoding="utf-8"))
    schedules = {}
    for level in instance["levels"]:
        schedules[level["id"]] = {}
        for seed in level["seeds"]:
            candidates = (schedule_for_order(list(order), seed) for order in permutations(level["jobs"]))
            schedules[level["id"]][str(seed)] = min(candidates, key=lambda value: value["makespan"])
    out = root / "submission" / "solution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": "alphaevolve-scheduling.solution.v1", "schedules": schedules}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
