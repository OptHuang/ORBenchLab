# Multi-agent VRP recovery

This clean-room task is conceptually inspired by *Multi-Agent Environments for
Vehicle Routing Problems*. It is not a reproduction of that paper, its data,
or its repository. The paper-to-task mapping and provenance boundary are
recorded in `data/paper-task-derivation.json`.

## Difficulty

`customer_count` scales from 4 to 8 customers, while `vehicle_count` and route
coupling increase at the same time. Every cell has seeds 1, 2, and 3. The event
axis adds a closed edge after a frozen prefix, forcing a local repair without
allowing a global rewrite. The hint axis is controlled by instruction variants:
L0 gives only the contract, L1 names the frozen-prefix and closure invariants,
and L2 names the affected event and vehicle set. Difficulty is not declared
monotone until two independent agent systems are measured on all cells.

## Reference solution

`solution/solve.py` is a deterministic nearest-neighbour construction followed
by a constrained local repair. It is a solvability control, not an optimality
claim. It writes the four allowed artifacts and recomputes the objective from
the route geometry.

## Verification

The separate no-network verifier checks assignment closure, depot and capacity
constraints, frozen prefixes, closed-edge avoidance, affected-vehicle scope,
objective recomputation, deterministic replay, and audit coverage. Mutation
tests independently reject duplicate service, capacity overflow, closed-edge
use, frozen-prefix edits, noncausal route changes, and fabricated objectives.
The test script emits `/logs/verifier/ctrf.json`.
