# Improve the scheduling algorithm

You are given `instance.json`, which contains three frozen job-shop levels:
`tiny`, `small`, and `medium`. Each level must be solved for seeds 1, 2, and
3. The seed deterministically changes processing times; do not replace or
ignore a seed. Each job has an ordered list of operations; each operation names
one machine and a positive processing duration.

Create exactly these two files under `submission/`:

1. `submission/solver.py`: a self-contained Python 3 program that accepts
   `--instance /root/instance.json --output /root/submission/solution.json`.
2. `submission/solution.json`: the output produced by running that program on
   the supplied instance.

The JSON output must contain one schedule for every `(level, seed)` pair. Every
operation must occur exactly once, job precedence and machine non-overlap must
hold, and the reported makespan must equal the verifier's recomputed value. The
makespan for each seed must not exceed its level's published feasibility bound.
Keep the input files unchanged and do not download packages or contact the
network. Run a local smoke test before finishing.
