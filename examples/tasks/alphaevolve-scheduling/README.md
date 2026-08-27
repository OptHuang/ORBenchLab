# AlphaEvolve-inspired scheduling task

This is a task-local operations-research benchmark inspired by the paper
*AlphaEvolve: A coding agent for scientific and algorithmic discovery*. It is
not a reproduction of the paper's proprietary experiments or source code. The
paper-to-task mapping and the non-redistribution boundary are recorded in
`data/paper-task-derivation.json`.

## Difficulty

The executable axis is `instance_scale`: `tiny`, `small`, and `medium` increase
the number of jobs, operations, machine contention, and the search space. Each
level has three explicit deterministic seeds for replication. Precedence depth
and machine coupling are properties of these frozen fixtures, not independently
measured arms. Difficulty is measured only after repeated seeds and independent
model systems, not assumed from level names.

## Reference solution

`solution/solve.py` and `solution/solver.py` are deterministic earliest-feasible-start
heuristics. They construct a schedule without external packages, recompute the
makespan, and write the exact artifact schema consumed by the verifier. The
verifier reruns `submission/solver.py`, so a hand-written `solution.json` is not
sufficient. Bounds are recorded in `data/reference-bounds.json` as a transparent
oracle-control contract, not an optimality claim.

## Verification

The verifier runs in a separate no-network environment. It reruns the declared
solver, then checks the output schema, operation coverage, job precedence,
machine non-overlap, makespan recomputation, and the bound for every difficulty
level. Tests inspect final artifacts rather than the commands used to create
them and emit a CTRF report at `/logs/verifier/ctrf.json`. Mutation controls
change one invariant at a time (duplicate operation, precedence violation,
overlap, or fabricated makespan) and must be rejected before model comparison.
