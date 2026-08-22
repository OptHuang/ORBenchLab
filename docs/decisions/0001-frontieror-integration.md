# 0001 — FrontierOR: wrap the official harness, defer Harbor conversion

**Status:** accepted
**Date:** 2026-08-22
**Upstream inspected:** `Minw913/FrontierOR` at `8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9`

## Decision

The initial FrontierOR integration **wraps the official external harness**. It
materialises no Harbor task packages, writes no adapter, and does not reproduce
the trusted checker, the security boundary or the scoring formula.

Harbor task conversion stays open as a **later, separately-validated adapter**
for a declared task subset, gated on the parity evidence described at the end of
this record.

## Context

FrontierOR is the opposite shape from ORAgentBench. ORAgentBench ships Harbor
task packages, so the correct integration is to consume them. FrontierOR ships a
complete trusted evaluation stack of its own, and the question is whether to use
it or to rebuild its semantics inside Harbor.

`orbench integration inspect frontieror` establishes the following from the
checkout itself. Everything here is verified by the inspector, not assumed:

**The official entry point is public and versioned.**
`python -m frontieror.infra` exposes `agent`, `submission`, `contract` and
`security-check`.

**The scorer is published machine-readably.** `python -m frontieror.infra contract`
returns, with no network access and no model call:

```json
{
  "contract_version": "staged-qte-v1",
  "scorer": "staged_qte",
  "aggregation": "arithmetic_mean_over_declared_instances",
  "runtime_measurement": "trusted_host_wall_clock",
  "candidate_reported_timestamps_trusted": false,
  "private_parameters": [
    "reference_objective", "reference_runtime",
    "checker_implementation", "final_instance_membership"
  ]
}
```

The instance score has two regimes: `max(0, 1-g)` on quality alone, and
`(1-g) + max(0, 1-t/tau)` when the quality gap is inside the stage boundary —
where `t` is trusted-host wall time and `tau` is the reference runtime.

**Ten artefacts are trusted-only**, including `reference_objective`,
`reference_runtime`, `feasibility_checker`, `final_instance_membership`,
`final_instance_provenance` and `private_trace`. Upstream states that official
final instances are unpublished server-only data, and that files distributed via
the public dataset are suitable for local integration tests but not as hidden
leaderboard tests.

**Upstream owns the security boundary**: a candidate container, an egress proxy,
a credential-isolating model proxy that gives the agent an ephemeral token
rather than the platform key, and a black-box probe (`security-check`) covering
host-file access, root writes, public networking, timeout escape and output
flooding.

## Options considered

**A. Materialise Harbor tasks now (a true adapter).**
Rejected. It requires shipping substitutes for data upstream deliberately
withholds, and re-expressing upstream's sandbox as `task.toml` declarations. Two
independent implementations of the same grading semantics diverge; the
divergence surfaces as a wrong number under the right name, which is the
failure mode hardest to notice and most damaging when it is noticed late.

**B. Wrap the official harness.** Chosen.

**C. Do both, with the adapter as an experiment.**
Rejected for the first release. Two integrations for one benchmark means two
sets of numbers, and no basis yet for saying which is authoritative. Revisit
once the parity fixture below exists — at that point the adapter has a
correctness criterion instead of an opinion.

## Why conversion is not parity-safe yet

Four blockers, recorded as `facts.harbor_conversion_blockers` in the inspection
report so CI can assert on them:

| Blocker | Detail |
| --- | --- |
| `trusted_host_timing` | The score includes a speed term measured as trusted-host wall clock, against a reference runtime measured on a single pinned core. A container on a shared runner yields a differently-defined number. Upstream also explicitly does not trust candidate-reported timestamps, so the measurement must stay on the trusted host. |
| `undistributed_reference_data` | Reference objectives, reference runtimes, the checker implementation and final-instance membership are trusted-only. A task package cannot contain them, and shipping substitutes makes it a different benchmark. |
| `upstream_security_boundary` | Candidate isolation, the egress proxy and the credential-isolating model proxy are upstream components. Re-expressing them as `network_mode` declarations is a re-implementation whose divergences appear as silent scoring differences. |
| `multi_candidate_inner_loop` | Test-time self-evolution runs many candidates per task inside the harness, with a stage-1 gate, a dev set and a held-out test set. Harbor's trial model would have to be mapped onto that structure before conversion means anything. |

## Consequences

**Good.**
* No second implementation of the scoring formula, so no divergence to chase.
* The trusted checker and reference data stay on the trusted side, where their
  hiddenness is what makes them useful.
* There is a genuinely zero-cost integration path — reading the contract — so CI
  can verify the integration on every pull request without a key, a licence or a
  container.
* Upstream improvements arrive for free; we are pinned to a commit, not a fork.

**Costs, stated plainly.**
* FrontierOR results do not flow through Harbor's job model, so they do not get
  Harbor's resume, retry or regrade for free.
* Any campaign requires a performance-isolated self-hosted runner, a Gurobi
  licence and an OpenRouter key. Campaign validation refuses anything less.
* Results produced on a GitHub-hosted runner are `exploratory` by construction
  and may never be published as official scores.
* The normalized slice for FrontierOR must be derived from the harness's own
  output format, which is a separate piece of work not yet done.

## Revisit when

Any of the following changes the calculus:

1. Upstream publishes a reference-runtime normalisation usable off the trusted
   host — this removes `trusted_host_timing`.
2. Upstream publishes a distributable checker and reference bundle for a
   declared task subset — this removes `undistributed_reference_data` for that
   subset.
3. A differential re-scoring fixture exists: frozen submissions including one
   violation per hard-constraint family and both sides of the tolerance
   boundary, where the official checker and a converted verifier agree key for
   key.

Condition 3 is required regardless. It is the cheapest possible parity test —
deterministic, no agent, no model — and it is where "we wrapped it and the
semantics changed" bugs actually live.
