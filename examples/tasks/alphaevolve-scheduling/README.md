# AlphaEvolve-inspired scheduling task

This is a task-local operations-research benchmark inspired by the paper
*AlphaEvolve: A coding agent for scientific and algorithmic discovery*. It is
not a reproduction of the paper's proprietary experiments or source code. The
paper-to-task mapping and the non-redistribution boundary are recorded in
`data/paper-task-derivation.json`.

## Difficulty

The primary axis is `instance_scale`: `tiny`, `small`, and `medium` increase the
number of jobs, operations, machine contention, and the size of the search
space. Secondary axes are precedence depth and machine coupling. A future hint
ladder can expose no scaffold, an invariant checklist, or a generic scheduling
template; it must never reveal a schedule or objective. Difficulty is a
measured property only after repeated seeds and independent model systems, not
an assumption from the level names.

## Reference solution

`solution/solve.py` is a deterministic earliest-feasible-start heuristic. It
constructs a schedule without external packages, recomputes the makespan, and
writes the exact artifact schema consumed by the verifier. It is a solvability
control and not a claim that the heuristic is optimal. The task accepts the
published per-level bounds so agents can improve ordering and local search while
remaining objectively verifiable.

## Verification

The verifier runs in a separate no-network environment. It checks the output
schema, operation coverage, job precedence, machine non-overlap, makespan
recomputation, and the bound for every difficulty level. Tests inspect final
artifacts rather than the commands used to create them and emit a CTRF report
at `/logs/verifier/ctrf.json`. Mutation controls should change one invariant at
a time (duplicate operation, precedence violation, overlap, or fabricated
makespan) and must be rejected before any model comparison.
