# ORBenchLab report — oragentbench-smoke-20260822-796d74b1

**Evidence label: EXPLORATORY**

Downgraded from `partial`:

- 'partial' requires at least one capability measurement repeated 3 or more times for the same (task, agent) configuration; every configuration here was measured once (R0)

## What this report may and may not say

Every figure below comes from an independent deterministic verifier (`E-V`) or from the trace (`E-T`). The strength of a statement is `min(evidence grade, repetition class)`.

Not every capability measurement reaches `R1`: the weakest (task, agent) configuration was run 1 time(s), and `R1` requires at least 3. `R0` evidence supports **case diagnosis only**, so this report draws no cross-agent or cross-model ranking conclusions, and the renderer rejects comparative wording while any capability measurement is `R0`.

## Campaign

| field | value |
| --- | --- |
| campaign_id | `oragentbench-smoke-20260822-796d74b1` |
| integration | `oragentbench` |
| site | `local-docker` |
| perf_isolated | `false` |
| load sampling | `none` |
| trials | 6 |
| trials in capability denominator | 6 |
| strict pass rule | strict pass = feasibility >= 1 and quality >= 1.0, where quality is the upstream ORAgentBench reward key (0-2, 1.0 means the reference objective was matched) |

## Task-health controls (zero model cost)

`oracle` and `nop` are Harbor built-in agents, so these controls cost nothing to run and are the cheapest evidence that a task set is well-formed.

| metric | value | n | strength | claim | definition |
| --- | --- | --- | --- | --- | --- |
| `oracle_pass_rate` | 1.000 | 3 | `E-V/R0` | `[C009]` | count(oracle strict pass) / count(oracle trials); a healthy task set is 1.0 |
| `nop_fail_rate` | 1.000 | 3 | `E-V/R0` | `[C010]` | count(nop non-pass) / count(nop trials); a task passable by doing nothing is broken |

## Agent measurements

### `nop`

| metric | value | n | strength | claim | definition |
| --- | --- | --- | --- | --- | --- |
| `feasibility_rate` | 0.000 | 3 | `E-V/R0` | `[C001]` | count(feasibility >= 1) / count(capability trials) |
| `strict_pass_rate` | 0.000 | 3 | `E-V/R0` | `[C002]` | strict pass = feasibility >= 1 and quality >= 1.0, where quality is the upstream ORAgentBench reward key (0-2, 1.0 means the reference objective was matched) |
| `mean_quality_when_feasible` | n/a — no feasible trial produced a quality score | 0 | `E-V/R0` | `[C003]` | mean(quality) over trials with feasibility >= 1 |
| `cost_per_strict_pass_usd` | n/a — no strict pass; denominator is zero | 3 | `E-V/R0` | `[C004]` | sum(cost_usd) / count(strict pass) |

### `oracle`

| metric | value | n | strength | claim | definition |
| --- | --- | --- | --- | --- | --- |
| `feasibility_rate` | 1.000 | 3 | `E-V/R0` | `[C005]` | count(feasibility >= 1) / count(capability trials) |
| `strict_pass_rate` | 1.000 | 3 | `E-V/R0` | `[C006]` | strict pass = feasibility >= 1 and quality >= 1.0, where quality is the upstream ORAgentBench reward key (0-2, 1.0 means the reference objective was matched) |
| `mean_quality_when_feasible` | 1.140 | 3 | `E-V/R0` | `[C007]` | mean(quality) over trials with feasibility >= 1 |
| `cost_per_strict_pass_usd` | 0.000 | 3 | `E-V/R0` | `[C008]` | sum(cost_usd) / count(strict pass) |

## Mandatory disclosures

These counts are always shown. `infra_suspect` and capability inclusion are separate fields: soft load evidence alone does not reassign blame, while hard or contradictory evidence may explicitly exclude a trial with an `exclusion_basis`.

| metric | value | n | strength | claim | definition |
| --- | --- | --- | --- | --- | --- |
| `infra_suspect_share` | 0.000 | 6 | `E-T/R0` | `[C011]` | count(infra_suspect) / count(trials); denominator inclusion is reported separately by counts_toward_capability and excluded_trial_share |
| `excluded_trial_share` | 0.000 | 6 | `E-T/R0` | `[C012]` | count(trials excluded from capability metrics) / count(trials); exclusion is explicit and each excluded trial carries an exclusion_basis |
| `orphan_trial_count` | 1 | 6 | `E-T/R0` | `[C013]` | count(trials with no plan-ledger match) |
| `no_load_sampling_share` | 1.000 | 6 | `E-T/R0` | `[C014]` | share of trials with load_source = 'none'; load-based suspicion is unavailable, but hard or contradictory evidence may still set infra_suspect |

## Evidence index

| claim | metric | scope | strength | runs |
| --- | --- | --- | --- | --- |
| `C001` | `feasibility_rate` | `agent:nop` | `E-V/R0` | 3 run id(s) |
| `C002` | `strict_pass_rate` | `agent:nop` | `E-V/R0` | 3 run id(s) |
| `C003` | `mean_quality_when_feasible` | `agent:nop` | `E-V/R0` | 3 run id(s) |
| `C004` | `cost_per_strict_pass_usd` | `agent:nop` | `E-V/R0` | 3 run id(s) |
| `C005` | `feasibility_rate` | `agent:oracle` | `E-V/R0` | 3 run id(s) |
| `C006` | `strict_pass_rate` | `agent:oracle` | `E-V/R0` | 3 run id(s) |
| `C007` | `mean_quality_when_feasible` | `agent:oracle` | `E-V/R0` | 3 run id(s) |
| `C008` | `cost_per_strict_pass_usd` | `agent:oracle` | `E-V/R0` | 3 run id(s) |
| `C009` | `oracle_pass_rate` | `control:oracle` | `E-V/R0` | 3 run id(s) |
| `C010` | `nop_fail_rate` | `control:nop` | `E-V/R0` | 3 run id(s) |
| `C011` | `infra_suspect_share` | `disclosure` | `E-T/R0` | 6 run id(s) |
| `C012` | `excluded_trial_share` | `disclosure` | `E-T/R0` | 6 run id(s) |
| `C013` | `orphan_trial_count` | `disclosure` | `E-T/R0` | n/a |
| `C014` | `no_load_sampling_share` | `disclosure` | `E-T/R0` | 6 run id(s) |

Full run-id lists are in `evidence_index.json`. Each claim resolves to the run ids it was computed from, so any figure here can be traced back to the underlying runs.

Generated by orbenchlab 0.1.0 — deterministic output, no wall-clock content.
