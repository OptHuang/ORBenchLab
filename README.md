# ORBenchLab

[![CI](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml)
[![Integration contract](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml)

A control plane for running operations-research agent benchmarks and reporting
what they actually showed. ORAgentBench now has one lifecycle command:
**inspect → plan → preflight → execute → ingest → report** in one immutable
workspace.

ORBenchLab does not reimplement benchmark logic. It sits either side of an
upstream benchmark's own machinery:

* **before** — a campaign spec compiles into a plan, stable external run ids and
  the upstream runner's own job configs;
* **during** — an explicit `--execute` hands the compiled config to the
  upstream runner and records a sanitized receipt;
* **after** — raw Harbor results are reconciled exactly against the plan ledger,
  then normalized results render into a report whose claims are bounded
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

# One command prepares an auditable workspace. This is the safe default: it
# does not start Docker or contact a model provider.
orbench run oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent oracle --date 2026-08-24 \
  --workspace ./orbench-runs

# Check the actual execution host, then run the zero-model-cost oracle control.
orbench doctor oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning --agent oracle
orbench run oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent oracle --date 2026-08-24 \
  --workspace ./orbench-runs --execute
```

The checkout directory must be named exactly `ORAgentBench`, matching
upstream's dataset-path contract. Repeating the same command verifies and
resumes the same content-addressed workspace instead of scheduling a duplicate.

### Reuse a local Codex coding-plan login

Harbor 0.16+'s Codex adapter can inject the host's private
`~/.codex/auth.json`. This keeps credentials out of campaign specs, job YAML,
receipts and reports:

```bash
orbench doctor oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model gpt-5.5 --auth-mode codex-auth-json

orbench run oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model gpt-5.5 --auth-mode codex-auth-json \
  --date 2026-08-24 --wall-clock-sec 1200 --max-cost-usd 5 \
  --workspace ./orbench-runs --execute \
  --acknowledge-cost i-accept-model-costs
```

The provider base URL is discovered from `~/.codex/config.toml`; alternatively
set the non-secret `ORBENCH_MODEL_BASE_URL` or pass `--model-base-url`. The auth
file must exist with no group/world permissions. `doctor` reports only whether
the URL is configured and never prints it or reads auth-file contents.

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

* The one-command lifecycle currently executes and ingests **ORAgentBench**.
  FrontierOR remains on its official external harness path and still requires a
  licensed, performance-isolated runner before its runtime-bearing score is
  meaningful.
* Harbor ingestion is a local run-bundle transform, not a remote warehouse.
  Raw trajectories and verifier material stay on the execution host; only the
  normalized slice, report and sanitized receipt are intended for sharing.
* No durability verification, so no `validated` reports.
* A model run still requires an explicit cost acknowledgement. A successful
  prepare/doctor/control run is not evidence that a paid model rollout was
  performed.

## Repository layout

```
src/orbenchlab/
  core/           identifiers, evidence rules, a small schema validator
  integrations/   the registry and the two first-class integrations
  campaign/       spec validation and the compiler
  report/         metrics and the renderer
  ingest/         Harbor plan-ledger reconciliation and normalized slices
  execution.py    upstream command construction, validation, receipts
  workflow.py     one-command ORAgentBench lifecycle and immutable workspace
  schemas/        published JSON schemas
tools/            legacy checked command builder used by compatibility CI
scripts/          runner bootstrap
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
