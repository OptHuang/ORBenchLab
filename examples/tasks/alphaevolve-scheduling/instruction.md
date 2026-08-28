# Improve the scheduling algorithm

You are given `instance.json`, which contains three frozen job-shop levels:
`tiny`, `small`, and `medium`. Each level must be solved for seeds 1, 2, and
3. The seed deterministically changes processing times; do not replace or
ignore a seed. Each job has an ordered list of operations; each operation names
one machine and a positive processing duration.

For a base duration `d`, the verifier uses the deterministic seeded duration
`d + ((seed + operation_index) % 2)`, where `operation_index` is zero-based
within the job. Use this rule exactly for every `(level, seed)` pair.

Create exactly these two files under `submission/`:

1. `submission/solver.py`: a self-contained Python 3 program that accepts
   `--instance /root/instance.json --output /root/submission/solution.json`.
2. `submission/solution.json`: the output produced by running that program on
   the supplied instance.

The JSON output must contain one schedule for every `(level, seed)` pair. Every
operation must occur exactly once, job precedence and machine non-overlap must
hold, and the reported makespan must equal the verifier's recomputed value. The
makespan for each seed must not exceed its level's published feasibility bound.

Use schema version `alphaevolve-scheduling.solution.v1`. The output is an
object with two top-level keys, `schema_version` and `schedules`. `schedules`
is indexed as `schedules[level_id][str(seed)]`, and **each `(level, seed)` entry
is itself an object with exactly two keys**: `operations` (a list of operation
rows) and `makespan` (the integer objective, equal to the maximum operation
`end`). It is **not** a bare list of rows. Each operation row must contain
`operation_id`, `job_id`, `operation_index`, `machine`, `start`, and `end`
(`start` and `end` are non-negative integers with `end > start`).

Minimal shape (one level `tiny`, one seed `1`, one operation shown):

```json
{
  "schema_version": "alphaevolve-scheduling.solution.v1",
  "schedules": {
    "tiny": {
      "1": {
        "operations": [
          {"operation_id": "j0-o0", "job_id": "j0", "operation_index": 0,
           "machine": "m0", "start": 0, "end": 3}
        ],
        "makespan": 3
      }
    }
  }
}
```

Keep the input files unchanged and do not download packages or contact the
network. The verifier re-runs your `solver.py` once and it must finish within
60 seconds on the frozen instances (a correct solver runs in well under a
second). Run a local smoke test before finishing.
