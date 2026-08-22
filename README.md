# ORBenchLab

[![CI](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml)
[![Integration contract](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml)

A control plane for running operations-research agent benchmarks and reporting
what they actually showed.

ORBenchLab does not execute benchmarks and does not reimplement them. It sits
either side of an upstream benchmark's own machinery:

* **before** — a campaign spec compiles into a plan, stable external run ids and
  the upstream runner's own job configs;
* **after** — normalized results render into a report whose claims are bounded
  by the evidence behind them.

The benchmark itself stays where it belongs. ORAgentBench's task packages,
verifiers and metrics are consumed unchanged; FrontierOR's trusted harness,
checkers and scoring formula are invoked, never copied.

## Why this exists

Benchmark tooling tends to fail in three ways, and each has a countermeasure
here:

| Failure | Countermeasure |
| --- | --- |
| A wrapper quietly forks the benchmark, drifts, and reports different numbers under the same name | Integrations are read-only inspectors; a test fails if any upstream artefact is vendored into this repository |
| Results cannot be matched back to what was planned, because the runner names runs randomly | Run ids are a pure function of frozen inputs, written to a plan ledger *before* anything executes |
| One rollout becomes "model A beats model B" | Claim strength is `min(evidence grade, repetition class)`, enforced by the renderer — single-rollout output is scanned for comparative wording and rejected |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11+. One runtime dependency (PyYAML).

## Use

```bash
# What is integrated, and what does each one need?
orbench integrations list

# Inspect an upstream checkout. Static read: no model calls, no execution.
git clone https://github.com/ORAgentBench/ORAgentBench /tmp/oragentbench
orbench integration inspect oragentbench --source /tmp/oragentbench --json out/inspection.json

# Validate and plan a campaign. Deterministic: same spec, byte-identical output.
orbench campaign validate campaigns/oragentbench-controls.yaml
orbench campaign plan     campaigns/oragentbench-controls.yaml --out out/plan

# Render a report from a normalized slice.
orbench report build --input fixtures/normalized/oragentbench-controls.json --out out/report

# Build the exact upstream command for an agent run — without running it.
python tools/run_benchmark_smoke.py oragentbench \
    --source /tmp/ORAgentBench --task additive_microfactory_order_planning \
    --scaffold claude-code --model <pinned-model> --date 2026-08-22 \
    --allow-missing-tooling
```

`orbench integration inspect` emits JSON with a `checks` array, a `facts` object
and a `decisions` object — CI asserts on all three.

## The two integrations

They are deliberately opposite shapes, because an integration layer that only
handles one shape is not an integration layer.

| | ORAgentBench | FrontierOR |
| --- | --- | --- |
| Form | `harbor-native` | `official-external-harness` |
| Adapter written | none — upstream already ships Harbor task packages | none — upstream ships its own harness |
| Verifier | upstream's `tests/test.sh` | upstream's trusted checkers, never distributed |
| Scoring | upstream reward keys (`feasibility`, `quality`) | upstream `staged_qte`, read from `python -m frontieror.infra contract` |
| Runtime in the score | no | **yes** — so campaigns are refused on any site not declared performance-isolated |
| Zero-cost path | `oracle` / `nop` controls, plus upstream's own `--dry-run` | contract inspection |
| Agent run invokes | upstream's Harbor wrapper (`run_harbor_prebuild.py` → `harbor run -c`) | `python -m frontieror.infra agent` |
| Needs a model key | only for agent campaigns | only for agent campaigns |
| Needs a solver licence | no | yes |

`docs/integrations.md` has the full comparison, the exact secrets and runner
labels, and the local smoke commands.

## GitHub Actions readiness

The zero-cost paths are live and observed on GitHub: the Python 3.11/3.13 CI
matrix, both pinned-upstream contract inspections, ORAgentBench and FrontierOR
`validate-only` smoke dispatches, and report rendering have all completed
successfully. See [Actions](https://github.com/OptHuang/ORBenchLab/actions).

Actual agent execution is intentionally not run on a shared GitHub-hosted
machine. Configure the repository secret/variable listed below and attach a
self-hosted runner with the required labels:

| Integration | Repository configuration | Runner |
| --- | --- | --- |
| ORAgentBench | secret `MODEL_API_KEY`; variable `MODEL_BASE_URL` | `self-hosted`, `orbench-exec`; Docker + Harbor |
| FrontierOR | secret `OPENROUTER_API_KEY` | the above plus `perf-isolated`, Gurobi licence, pinned cores and upstream images |

Then dispatch `benchmark-smoke` with `mode=agent`, a pinned model id, and the
literal acknowledgement `i-accept-model-costs`. The protected
`benchmark-agent` environment adds a separate approval click before model spend.

## Evidence labels

Every report carries one of three labels, and the renderer lowers it when its
preconditions are unmet rather than accepting it on trust:

* `exploratory` — a single rollout per configuration. Case diagnosis only.
* `partial` — at least one configuration repeated three or more times.
* `validated` — additionally requires verified durable replicas of the
  underlying evidence. **Durability verification is not implemented in this
  release**, so a `validated` campaign cannot currently be planned; validation
  says so instead of letting the label through.

A downgrade is always printed at the top of the report with its reason.

## What is not here

Stated plainly, because a missing feature that reads as present is worse than an
absent one:

* No execution *from `orbench`*. The CLI never invokes a benchmark. Running one
  is `tools/run_benchmark_smoke.py`'s job: it validates inputs against the
  upstream checkout, builds the command upstream documents for itself, and hands
  it over. See `docs/integrations.md`.
* No ingest of raw job bundles into a warehouse. Reports are built from a small
  normalized slice; producing that slice from raw bundles is not yet implemented.
* No durability verification, so no `validated` reports.
* No paid agent run has been performed from this repository. Hosted CI,
  integration inspection, zero-cost smoke preflight and report rendering have
  been observed; the self-hosted model-calling jobs remain deliberately
  unexecuted until credentials, runner capabilities and cost approval exist.

## Repository layout

```
src/orbenchlab/
  core/           identifiers, evidence rules, a small schema validator
  integrations/   the registry and the two first-class integrations
  campaign/       spec validation and the compiler
  report/         metrics and the renderer
  execution.py    upstream command construction, validation, receipts
  schemas/        published JSON schemas
tools/            run_benchmark_smoke.py — the one thing that starts an upstream run
campaigns/        example campaign specs
sites/            execution site declarations
fixtures/         normalized rollout slices used by the report tests
templates/        starting point for a third integration
docs/             integration guide, contribution notes, decision records
.github/scripts   clone-upstream.sh — clones the pinned commit for both workflows
.github/workflows ci, integration-contract, benchmark-smoke, report
```

## Contributing and security

See `CONTRIBUTING.md` and `SECURITY.md`. In short: no vendored upstream code, no
secrets in specs or plans, and no claim in a report that its evidence index
cannot back.

## Licence

Apache-2.0. See `LICENSE`.

Upstream benchmarks carry their own licences and are not redistributed here.
