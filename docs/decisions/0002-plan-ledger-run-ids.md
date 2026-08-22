# 0002 — Run identity lives in a plan ledger, not in the runner

**Status:** accepted
**Date:** 2026-08-22

## Decision

ORBenchLab derives its own `run_id` as a pure function of frozen campaign
inputs, writes a **plan ledger before anything executes**, and reconciles the
runner's trials back to ledger entries by `match_key` afterwards.

It does not attempt to push its identifiers into the runner.

## Context

Harbor names trials with `f"{task_name[:32]}__{ShortUUID().random(7)}"` — a
random suffix, generated at expansion time. Nothing content-addressed comes back
out. Two consequences follow: a rerun cannot be recognised as the same work, and
results cannot be matched to a plan except by guessing.

Passing a chosen name in would be the obvious fix, but the job-config path does
not expose one: the expansion over attempts, tasks and agents omits the trial
name, so it always falls through to the random default. Building on an entry
point we cannot demonstrate would put the whole identity scheme on an unverified
assumption.

## The scheme

```
task_uid   = "{bench}@{dataset_digest}/{task_name}#{task_checksum}"
agent_uid  = sha256(canon(scaffold, version, model, prompt_digest,
                          tool_policy, kwargs, sorted(env_KEY_NAMES)))
budget_uid = sha256(canon(budget))
cfg_digest = sha256(canon(spec minus job_name, debug, comments, description))

run_id     = sha256(canon([cfg_digest, task_uid, agent_uid,
                           budget_uid, seed, attempt]))[:16]
match_key  = sha256(canon([task_name, agent_uid, seed, attempt]))
shard      = int(run_id[:8], 16) % n_shards
```

Four properties fall out, each of which is a test in `tests/test_ids.py`:

* **Idempotent.** The same spec always yields the same run ids, so a rerun is
  recognisable rather than duplicative.
* **Shardable without coordination.** Sharding is a pure function of `run_id`,
  so every machine computes the same assignment with no service to ask.
* **Credential-safe.** Only environment variable *names* are hashed. Rotating a
  key changes nothing; renaming one changes the agent's identity, which is
  correct, because that is a real configuration change.
* **Seed-independent agent identity.** `seed` is a separate component, so the
  same agent keeps one identity across seeds.

## Reconciliation

Because a trial cannot carry our id, ingest matches on `match_key`, and the
compiler makes that lookup exact rather than heuristic by enforcing:

* one job carries exactly one `(agent_uid, seed, attempt)` combination;
* `n_attempts = 1`;
* seeds are expressed as separate jobs.

`(job_name, task_name)` then identifies one ledger entry. The rules for
everything else are deliberately unforgiving:

| Situation | Action |
| --- | --- |
| One ledger entry matches | Bind `run_id` to the trial |
| Several unconsumed entries match | Integrity error — **never guess** |
| No entry matches | Record as `orphan_trial`; excluded from the capability denominator, and its count is disclosed in the report |

Guessing would be the worst option available: it produces plausible data that
cannot be traced back to a plan.

## Consequences

**Good.** Idempotency, coordination-free sharding and de-duplication with no
dependency on an unverified runner entry point. The ledger is written first, so
an unplanned trial can never be silently adopted.

**Costs.** More jobs, since one job may carry only one `(agent, seed, attempt)`
combination — acceptable, because jobs are sharded anyway. Reconciliation is a
post-hoc step rather than an intrinsic property of the run. And
`task_checksum` is only known after execution, so at plan time `task_uid`
carries the placeholder `pending`; content addressing at plan time comes from
`dataset_digest` instead, which is computed over the sorted per-task
`task.toml` digests.

## Revisit when

The runner exposes a documented way to supply a trial name through the job path.
At that point the ledger becomes a cross-check rather than the primary
mechanism — but it should stay, because it is also what makes an unplanned trial
detectable.
