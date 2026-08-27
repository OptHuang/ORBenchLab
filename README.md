# ORBenchLab

[![CI](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/ci.yml)
[![Integration contract](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml/badge.svg)](https://github.com/OptHuang/ORBenchLab/actions/workflows/integration-contract.yml)

A control plane for running operations-research agent benchmarks and reporting
what they actually showed. ORAgentBench now has one lifecycle command:
**inspect → plan → preflight → execute → ingest → report** in one
integrity-checked run workspace.

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

For the discovery loop, `orbench intake` provides a separate metadata-only
path: it collects configured public RSS/arXiv/GitHub feeds, records response
digests, de-duplicates entries, and writes a human-review queue. It makes no
model calls, reads no credentials, touches no `raw/` files, and cannot author or
publish a task. See [`docs/source-intake.md`](docs/source-intake.md).

## Unattended final task cards

Once a daily intake and one or more Harbor screening reports exist, one command
builds the final view that an operator needs to read:

```bash
orbench pipeline run \
  --intake-config intake/or-feeds.example.yaml \
  --out artifacts/pipeline/$(date -u +%F)
```

With no `--tasks` or `--screenings`, the command discovers
`docs/task-genomes/` and `artifacts/**/*screening*.json`. It de-duplicates
identical report copies by content digest, then writes `task-cards.md`,
`task-cards.json`, `pipeline-summary.json` and `pipeline-manifest.json`.
Each card contains the task purpose, declared difficulty axes and levels,
model/route solve and quality rates, completion and infrastructure exceptions,
the automatic decision (`keep`, `review-promising`, `collect-more-evidence`,
`revise-or-drop` or `quarantine`), and an evidence-bound limitations section.

This is an unattended *summarization and gating* stage. It does not silently
turn a paper into executable code, call a model, or promote a task whose source,
verifier or difficulty contract is missing. Such candidates are emitted as
`quarantine` cards with their observed evidence still visible. Harbor/Volc
execution remains the preceding campaign stage, and live same-checkpoint hint
injection is not inferred from this report.

## Paper to Terminal-Bench Science task

For a paper-backed candidate, run the authoring gate before packaging anything
for Harbor:

```bash
orbench task-author validate \
  --task-dir path/to/task \
  --paper-provenance path/to/paper-provenance.json \
  --round 1 \
  --out artifacts/task-authoring/round-1
```

The receipt covers the current TB-Science task-template fields and the 39
implementation-rubric names. Deterministic checks cover task.toml shape,
instruction/README/solution/tests presence, executable verification, CTRF
output, separate no-network verifier configuration, artifact allowlists,
resource bounds, task metadata, paper digest/license fields, and known security
hazards. Semantic criteria such as scientific grounding, genuine difficulty,
novelty, agentic scope, anti-cheat robustness, and test/instruction alignment
remain explicit `review` items. A receipt is therefore `blocked`,
`ready-for-human-review`, or `ready-for-harbor-validation`; it is never a
TB-Science acceptance or PR approval. Later rounds link the previous receipt
digest so an agent-tuning loop remains auditable. A paper digest without a
local `source_path` byte match and registry resolution remains a review item;
the gate does not trust a caller-provided hash as independent evidence. Each
receipt also records a path-independent `task_tree_digest`, so changes to any
task file are visible between rounds. The receipt separates the 39-item
implementation counts from additional paper/round provenance counts, so the
rubric count and status totals cannot contradict each other.

When the static gate has produced a receipt, the semantic authoring pass can
be delegated to Volcengine models on the execution host:

```bash
set -a; . ~/.config/claude/ark-volces.env; set +a
orbench task-author review \
  --task-dir path/to/task \
  --paper-provenance path/to/paper-provenance.json \
  --receipt artifacts/task-authoring/round-1/authoring-receipt.json \
  --models ark-code-latest,deepseek-v4-flash \
  --round 1 \
  --out artifacts/task-authoring/round-1-volc
```

The command refuses non-Volc HTTPS hosts, sends only bounded public evidence,
and writes a structured reviewer digest rather than raw prompts or model
responses. It never upgrades a statically blocked receipt; its output is a
revision proposal with task purpose, difficulty axes, rubric findings and
next edits. The official automation still requires `harbor check` and runtime
evidence before a task can be submitted.[TB-Science review automation](https://github.com/harbor-framework/terminal-bench-science/blob/main/TASK_REVIEW_AUTOMATION.md)

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

For an execution host, the bootstrap script creates the local environment and
also discovers Harbor installed by `uv tool` plus the `uv` executable required
by Harbor's `uv-script` metric. If a non-interactive shell cannot see them at
their usual user-local locations, the script links the discovered executables
into `.venv/bin` without overwriting existing commands:

```bash
ORBENCH_PYTHON="$HOME/.local/bin/python3.12" ./scripts/bootstrap-runner.sh
source .venv/bin/activate
orbench --version
harbor --version
```

For non-standard locations, set `ORBENCH_HARBOR_BIN_DIR` to the directory
containing `harbor` and `ORBENCH_UV_BIN_DIR` to the directory containing `uv`.
The script never assumes a user name.

## Use

```bash
# What is integrated, and what does each one need?
orbench integrations list

# Inspect an upstream checkout. Static read: no model calls, no execution.
git clone https://github.com/ORAgentBench/ORAgentBench /tmp/ORAgentBench
orbench integration inspect oragentbench --source /tmp/ORAgentBench --json out/inspection.json

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
Campaign identity also includes ORBenchLab's job-config contract version: when
a release changes execution semantics, the same user spec receives a new
campaign id and the older workspace remains byte-for-byte evidence rather than
being rewritten in place.
Preparation copies the inspected checkout into a content-addressed, run-local
source snapshot. The upstream command reads only that snapshot, whose digest is
checked immediately before and after execution. The original Git commit remains
provenance; later edits to the operator checkout do not change a prepared run.

After an exit-zero upstream process, ORBenchLab records Docker's actual image ID
and any RepoDigests for the built tag. If that identity cannot be inspected, the
campaign fails before result ingest. Because the image ID exists only after a
build, it is post-run evidence rather than a campaign-ID input: the campaign
binds the source/build recipe and exact scaffold version, while the manifest
and receipt bind the image Docker actually produced.

After a completed run, cross the host boundary only through the tested exporter:

```bash
orbench export \
  --run-root ./orbench-runs/<campaign-id> \
  --destination ./share/<campaign-id>
```

It re-verifies the complete local workspace, emits a fixed normalized/report
allowlist, derives path-free public metadata, and writes a new
`share-integrity.sha256`. The raw local manifest, receipt, integrity ledger,
jobs and logs remain on the execution host.

### Run a model agent with a scoped API credential

Treat benchmark tasks as untrusted code. ORBenchLab refuses direct execution
with `codex-auth-json`: a long-lived `~/.codex/auth.json` must not be mounted
into task containers. Create a short-lived, revocable provider key, set a
provider-side spend/rate limit, and expose it only for the duration of the run:

```bash
export MODEL_API_KEY='<ephemeral scoped key>'
export MODEL_BASE_URL='https://provider.example/v1'
export ORBENCH_MODEL_ID='provider/model-version-2026-08-01'
export ORBENCH_SCAFFOLD_VERSION='1.2.3'  # exact release; never latest

orbench doctor oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model "$ORBENCH_MODEL_ID" \
  --scaffold-version "$ORBENCH_SCAFFOLD_VERSION" --auth-mode api-key

orbench run oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model "$ORBENCH_MODEL_ID" \
  --scaffold-version "$ORBENCH_SCAFFOLD_VERSION" --auth-mode api-key \
  --date 2026-08-24 --wall-clock-sec 1200 --max-cost-usd 5 \
  --workspace ./orbench-runs --execute \
  --acknowledge-cost i-accept-model-costs

unset MODEL_API_KEY MODEL_BASE_URL ORBENCH_MODEL_ID ORBENCH_SCAFFOLD_VERSION
```

`MODEL_API_KEY` is secret. `MODEL_BASE_URL` is non-secret configuration, but it
is also the destination that receives that credential: `codex` and
`claude-code` therefore require it before preparation. ORBenchLab canonicalizes
the HTTPS route, stores only its `sha256:` digest in campaign identity and the
local/public manifest, and requires the runtime `MODEL_BASE_URL` to canonicalize
to the same route before Harbor starts. The URL itself belongs neither in a
campaign file nor on the command line. Use a protected repository environment
and an ephemeral secret when dispatching from GitHub Actions. The recorded
`mini-swe-agent` profile has no provider-base-URL variable and does not require
this route binding.

`--wall-clock-sec` is an enforced local timeout. `--max-cost-usd` is recorded as
an audit envelope but cannot stop an arbitrary provider after exactly that
amount; configure the real hard spend/rate limit on the short-lived provider
credential itself.

### Two Codex authentication routes (not interchangeable)

| Route | Credential boundary | Safe use in this repository | Current pinned ORAgentBench |
| --- | --- | --- | --- |
| `api-key` | A short-lived, revocable provider key injected only into the approved job | Paid benchmark rollout, including Internet-enabled tasks | **Supported.** This is the route for the current 107 public-network tasks |
| `codex-login` | A separate private-controller analysis runner is signed in to Codex with ChatGPT; no login export is stored in GitHub | Authentication/safety diagnosis now; future host-side analysis or broker-backed rollout | **Refused.** Direct benchmark execution is disabled and all 107 pinned tasks enable public Internet |

Installing the Codex CLI is not a login. For `codex-login`, the Unix user that
owns the isolated private-controller analysis runner must complete `codex
login` once, and `codex login status` must report a ChatGPT login rather than
an API-key login. The
route consumes the ChatGPT plan's Codex allowance and remains subject to plan
limits; it does not turn an API-key login into subscription usage. See the
[Codex authentication](https://learn.chatgpt.com/docs/auth) and
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
documentation.

The subscription route is deliberately not a shortcut around task isolation.
ORBenchLab never asks an operator to put a personal Codex login export in a
repository secret or Actions artifact. A task must opt into an audited safe
policy; an absent, unknown or public-network declaration is unsafe. An isolated
container also cannot call the model service merely because login exists on the
host. The current route is therefore an authentication and safety-diagnosis
entry point, not a working subscription-backed container rollout. Actual use
requires future host-side `/analyze` execution or a broker/model proxy that
keeps the personal login outside the task boundary. The pinned ORAgentBench
official agent rollout remains `api-key` only.

When a ChatGPT-plan run is ingested through Harbor, any dollar figure derived
from tokens is an **API-equivalent estimate** for comparison. It is not an
invoice, balance deduction or measurement of the actual ChatGPT-plan charge.

`codex-auth-json` remains a legacy plan/doctor transport for diagnosing an
existing local setup. Like `codex-login`, it is not executable until a broker
keeps the auth file outside every task and verifier boundary. Rollout must use a
scoped API credential.

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
machine. A self-hosted runner needs more than Codex: Python 3.12, Docker,
Harbor, a private durable `ORBENCH_RUNS_ROOT`, a protected GitHub Environment,
and a dedicated Unix service user that owns the benchmark runner and run root.
Do **not** add a personal Codex login to that Docker-capable account. A future
host-side analysis worker must use a separate private-controller runner identity
(preferably a separate disposable VM). Configure the route-specific settings:

| Integration | Repository configuration | Runner |
| --- | --- | --- |
| ORAgentBench (`api-key`) | secret `MODEL_API_KEY`; variables `MODEL_BASE_URL`, `ORBENCH_RUNS_ROOT` | `self-hosted`, `orbench-exec`; Python 3.12 + Docker + Harbor |
| Authentication/safety diagnosis; future broker-backed analysis (`codex-login`) | no model secret in GitHub; isolated analysis user pre-logged into ChatGPT Codex | separate private-controller runner label/account; never the benchmark Docker worker |
| FrontierOR | secret `OPENROUTER_API_KEY` | the above plus `perf-isolated`, Gurobi licence, pinned cores and upstream images |

Then dispatch `benchmark-smoke` with `mode=agent`, a pinned model id, an exact
released `scaffold_version` (floating labels such as `latest` are rejected),
and the literal acknowledgement `i-accept-model-costs`. The protected
`benchmark-agent` environment adds a separate approval click before model spend.
Use a revocable, provider-limited key for `MODEL_API_KEY`; never store a Codex
login export or another long-lived personal credential in repository secrets.

A long-lived runner attached directly to a public repository increases the
blast radius of a workflow or dependency compromise. Self-hosted modes are
therefore **disabled in this public repository**; it supports `validate-only`
dispatches. Execute from a **private mirror/controller repository** containing
the complete ORBenchLab tree at a reviewed commit, with protected environments
and reviewed workflows. Do not run an untrusted branch or pull-request ref.
Prefer an ephemeral runner for model jobs; if the runner is persistent, do not
let public pull requests or arbitrary refs schedule it. A runner containing a
ChatGPT login is supported only behind that private controller boundary. The detailed contract and
zero-model acceptance commands are in
[`docs/self-hosted-runner.md`](docs/self-hosted-runner.md).

`ORBENCH_RUNS_ROOT` is an absolute, runner-owned persistent directory outside
the checkout and `RUNNER_TEMP` (for example `/srv/orbench/runs`). The workflow
fails before creating a campaign if it is temporary, a symlink, group/world
writable, owned by another account, or has less than 20 GiB free. This durable
root lets a later dispatch verify and resume an interrupted campaign; its raw
jobs and logs remain on that host and are never uploaded.

Pinned ORAgentBench currently launches task images through one fixed Docker
base alias. ORBenchLab therefore holds a secure runner-account-wide,
non-blocking alias lock for the complete upstream execution and verifies that
the fixed alias and the campaign-specific image resolve to the same image ID.
Two campaigns cannot safely run against the same Docker daemon at once: run all
jobs for one daemon under the same Unix account, or use separate hosts/isolated
daemons for parallel rollout capacity. Different Unix accounts must not share a
daemon because their private lock directories cannot coordinate.
`ORBENCH_HOST_LOCK_DIR` may point at a dedicated absolute lock directory; the
default is below the runner account's private cache.

The workflow may publish only the exporter output: the normalized slice,
rendered report, path-free `public-manifest.json` / `public-receipt.json`, and
`share-integrity.sha256`. It must not upload the local manifest/receipt,
workspace integrity ledger, Harbor job bundle, agent trajectory, verifier
workspace, upstream stdout/stderr logs, or any other raw evidence; those stay
on the execution host.

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
  exporter-produced normalized slice, report, public metadata and share
  integrity ledger are intended for sharing. Raw Harbor job directories, local
  manifest/receipt and logs must not be uploaded as Actions artefacts.
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
  export.py       verified public evidence boundary and share integrity
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
