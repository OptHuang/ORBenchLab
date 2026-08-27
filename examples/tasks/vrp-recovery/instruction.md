# Vehicle-routing recovery task

Edit `/root/submission/solver.py` and run it to produce all four declared
artifacts. The frozen `/root/instance.json` contains three levels (`tiny`,
`small`, `medium`) and deterministic seeds 1, 2, and 3. For every level and
seed, emit an initial route plan, one replan record for each disruption in
`events.jsonl`, and an event audit.

Each vehicle route starts and ends at depot `0`, visits each customer exactly
once across the fleet, respects capacity, and uses only listed nodes. During a
replan, customer assignments and the completed prefix in `frozen_prefix` must
not change. A closed undirected edge may not appear in any route after its
event. Only vehicles named by `affected_vehicle_ids` may change. Recompute the
integer objective from the routes; never copy a claimed objective.

The verifier's deterministic edge cost is `abs(ax-bx) + abs(ay-by) +
((seed + int(a) + int(b)) % 2)`, using the coordinates in `instance.json`.
For each event, the objective is the sum of all route edge costs plus `2` for
each affected vehicle. Apply this formula when writing every replan record.

Write `initial_routes.json` as `{\"schema_version\":\"vrp-recovery.initial.v1\",\"routes\":{level_id:{str(seed):{vehicle_id:[nodes]}}}}`.
Write one JSON object per line to `replans.jsonl` with `level_id`, integer
`seed`, `event_id`, `routes`, integer `objective`, and sorted
`changed_vehicle_ids`. Write `event_audit.json` as
`{\"schema_version\":\"vrp-recovery.audit.v1\",\"records\":[...]}` with one
record per replan and boolean `frozen_prefix_ok`, `closure_ok`, and
`objective_recomputed` fields.

Do not modify `/root/instance.json`, `/root/events.jsonl`, or the verifier.
Do not use the network or read files outside `/root`. The output schemas are
illustrated by `solution/solve.py`; improve the route ordering or repair logic
while preserving every invariant.
