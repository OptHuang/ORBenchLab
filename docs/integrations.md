# Integration guide

What each integration needs, how to smoke it locally, and what a third
benchmark has to provide.

## Contents

* [Integration forms](#integration-forms)
* [ORAgentBench](#oragentbench)
* [FrontierOR](#frontieror)
* [Running an agent](#running-an-agent)
* [Secrets and runner labels](#secrets-and-runner-labels)
* [Local smoke runs](#local-smoke-runs)
* [Adding a third benchmark](#adding-a-third-benchmark)

## Integration forms

Three forms, chosen by what the upstream benchmark already owns. Picking the
wrong one is the most expensive mistake available here, so the comparison is
explicit.

| | **Harbor-native** | **Official external harness** | **True Harbor adapter** |
| --- | --- | --- | --- |
| Upstream ships | Harbor task packages | its own runner and grader | a dataset, no runner |
| We write | nothing | a thin invocation + preconditions | a build-time generator |
| Who grades | upstream's verifier, run by Harbor | upstream's trusted grader | our generated verifier |
| Who owns scoring | upstream | upstream | us, and we must prove it matches |
| Isolation boundary | Harbor `task.toml` | upstream's own sandbox | Harbor `task.toml` |
| Hidden data | stays in `tests/`, injected after the agent | stays on upstream's trusted side | we must obtain and host it |
| Parity risk | very low — same artefacts | low — same code path | **high** — a second implementation |
| Parity evidence needed | dataset digest | contract version + entry point | differential re-scoring, key for key |
| Cost to add | hours | days | weeks, plus ongoing drift |
| Choose when | tasks are already Harbor packages | scoring depends on things Harbor cannot reproduce | upstream has data but no runner, and conversion is provably faithful |
| Example | ORAgentBench | FrontierOR | *(none yet)* |

The middle column is not a lesser option. Wrapping is the correct answer
whenever upstream's grade depends on something a container cannot reproduce —
trusted-host timing, undistributed reference data, a bespoke security boundary.
Converting anyway produces a differently-defined number under the same name,
which is worse than not integrating at all.

### The test for "may I convert this to Harbor tasks?"

Convert only when **all** of these hold:

1. Everything needed to grade a submission can legitimately live inside a task
   package. If reference values or a checker are undistributed, stop.
2. The score does not depend on host-level properties a container cannot
   reproduce. Wall-clock timing against a reference runtime does depend on them.
3. The isolation the benchmark relies on can be expressed in the runner's own
   declarations, not re-implemented.
4. You can produce a differential re-scoring fixture — frozen submissions,
   including one violation per hard-constraint family and both sides of any
   tolerance boundary — where the upstream checker and the converted verifier
   agree key for key.

Point 4 is the one that catches real bugs. Most "we wrapped it and the semantics
changed" failures are verifier bugs, and verifier bugs need no agent to find.

## ORAgentBench

**Form:** Harbor-native. **Adapter: none.**

Upstream already ships everything Harbor consumes: `harbor_tasks/<task>/task.toml`
with `instruction.md`, `environment/`, `solution/` and `tests/`, plus
`metrics/per_dimension_reward.py`, `experiments/config/*.yaml`, `skills/` and
`difficulty.json`. Writing an adapter would create a second copy of the tasks
that drifts from upstream.

What ORBenchLab does: inspect the checkout, record the dataset digest and reward
keys, compile campaigns into Harbor job configs, and report.

What it never does: copy a task, patch a verifier, or reimplement the metric.

### Inspecting

```bash
git clone https://github.com/ORAgentBench/ORAgentBench /tmp/oragentbench
orbench integration inspect oragentbench --source /tmp/oragentbench
```

Findings at the pinned commit `c9eb952435a4352f33daa2a35efe0f8c76d31b28`
(107 tasks). Warnings are reported against upstream and are **not** patched
locally — a local patch would fork the benchmark:

| Check | Result |
| --- | --- |
| `harbor_native_task_packages` | pass — 107 packages |
| `task_name_declared` | pass — all namespaced `oragentbench/<task>` |
| `reward_channel` | pass — 107/107 write `/logs/verifier/reward.json` with `feasibility`, `quality` |
| `dataset_digest` | pass — content-addressed over sorted `task.toml` digests |
| `verifier_environment_mode` | **warn** — no task declares it, so all fall back to the runner default |
| `agent_network_isolation` | **warn** — 107/107 set `allow_internet = true` |
| `multi_step_tasks` | **warn** — 8/107 are multi-step, which the runner's regrade path does not support |
| `solver_license_grant` | pass — inferred `none`; no task declares `requires_gurobi` |
| `official_metric_script` | pass |
| `oracle_control_config` | pass — upstream's own oracle job config |
| `nop_control_available` | skip — a runner built-in; not verifiable from this repository |

The two warnings matter for what a campaign may claim. With
`allow_internet = true` and no declared verifier isolation, a campaign cannot
claim anti-speculation isolation on the strength of the upstream state alone.

### Controls

`oracle` and `nop` are runner built-ins, so the whole control suite costs
nothing in model spend. Expect oracle to pass everything and nop to pass
nothing; anything else means the task set is broken and should be quarantined
before any paid campaign runs against it.

## FrontierOR

**Form:** official external harness. **Adapter: none. Harbor conversion:
deferred.**

The decision and its evidence are in
[`decisions/0001-frontieror-integration.md`](decisions/0001-frontieror-integration.md).
Summary of what the inspector verifies rather than assumes:

* The official entry point is `python -m frontieror.infra`, exposing `agent`,
  `submission`, `contract` and `security-check`.
* `python -m frontieror.infra contract` publishes the scorer machine-readably:
  `staged_qte`, contract version `staged-qte-v1`, runtime measured as
  `trusted_host_wall_clock`, and `candidate_reported_timestamps_trusted: false`.
* Ten artefacts are trusted-only, including `reference_objective`,
  `reference_runtime`, `feasibility_checker` and `final_instance_membership`.
* Upstream owns the security boundary: candidate container, egress proxy,
  credential-isolating model proxy, and a black-box probe.

Because runtime enters the score, campaign validation **refuses** to plan a
FrontierOR campaign against a site that has not declared `perf_isolated: true`:

```console
$ orbench campaign validate campaigns/frontieror-contract-check.yaml
orbench: error: campaign spec is invalid (2 problem(s)):
  - integration 'frontieror' scores runtime, but site 'local-docker' declares
    perf_isolated: false. Performance-scored benchmarks may only run on a
    performance-isolated site with pinned cores; otherwise the number is not
    comparable to anything.
  - integration 'frontieror' requires a solver licence, but site 'local-docker'
    declares solver_license_slots: 0
```

That refusal is the feature. Anything measured on a shared GitHub-hosted runner
is `exploratory` by construction and must never be published as an official
score.

### Invoking the official harness

ORBenchLab gates and labels; upstream's commands do the work. From a FrontierOR
checkout:

```bash
# Zero cost, no network, no model: read the published scoring contract.
python -m frontieror.infra contract

# Verify a frozen submission bundle (needs the dataset and a Gurobi licence).
python -m frontieror.infra submission <submission_dir> --paper-id <id>

# Trusted agent evaluation (needs OPENROUTER_API_KEY, licence, built images).
python -m frontieror.infra agent --paper-id <id> --primary-model <route> ...

# Before trusting a runner image, probe the boundary.
python -m frontieror.infra security-check --candidate-image frontieror-candidate:1
```

Official final instances are unpublished server-only data. What is downloadable
from HuggingFace is suitable for local integration tests, **not** for
leaderboard scoring.

## Running a benchmark

`orbench run` is the primary lifecycle boundary. It validates the checkout,
compiles a deterministic plan, writes an integrity-protected workspace, invokes
the pinned upstream runner only with `--execute`, reconciles Harbor output, and
renders the report. Repeating the same command verifies and resumes the same
campaign. A paid path additionally needs the exact scaffold CLI version and
`--acknowledge-cost i-accept-model-costs`.

`tools/run_benchmark_smoke.py` remains a legacy zero-cost command-construction
and upstream-dry-run helper. It is not the evidence lifecycle used by the
self-hosted execution jobs.

### ORAgentBench

**Controls.** A fresh-host oracle or nop run delegates to upstream's own build
and control script, from the parent of the checkout:

```bash
bash <checkout>/experiments/scripts/run_oracle_all.sh \
    --config <compiled job config>
```

It builds the required base image rather than trusting a floating local cache.

**Paid agents.** The lifecycle invokes the pinned Python prebuild wrapper:

```bash
python3 <checkout>/source/scripts/run_harbor_prebuild.py \
    -c <compiled job config>
```

The compiled config requests a fresh base rebuild, a content-derived image tag,
and an exact scaffold version; floating `latest` versions are rejected. The
wrapper applies dynamic skills, swaps in upstream's prebuilt agent class, and
execs `harbor run -c <transformed config>`. A crash recovery adds upstream's
`--resume --cleanup-before-resume` and first binds the recovered Harbor config
and copied task/skills content back to the compiled campaign. ORBenchLab does
not use `--skip-build` for evidence runs.

**Driving it:**

```bash
export MODEL_API_KEY='<short-lived provider key>'
export MODEL_BASE_URL='https://provider.example/v1'
orbench doctor oragentbench \
    --source upstream/ORAgentBench \
    --task additive_microfactory_order_planning \
    --agent claude-code --model <pinned-model-id> \
    --scaffold-version <exact-cli-version>
orbench run oragentbench \
    --source upstream/ORAgentBench \
    --task additive_microfactory_order_planning \
    --agent claude-code --model <pinned-model-id> \
    --scaffold-version <exact-cli-version> \
    --date 2026-08-24 --workspace ./orbench-runs --execute \
    --acknowledge-cost i-accept-model-costs
unset MODEL_API_KEY MODEL_BASE_URL
```

**The checkout directory must be named `ORAgentBench`.** Upstream resolves the
relative dataset path `ORAgentBench/harbor_tasks` against the checkout's
*parent*, so any other name silently produces a job config whose dataset path
does not exist. `.github/scripts/clone-upstream.sh` gets this right; the script
refuses a wrongly-named checkout.

**The zero-cost check worth running first.** Upstream's wrapper has its own
`--dry-run`: it transforms the config, applies dynamic skills, prints the
`harbor run` command it would execute, and stops. It needs neither Harbor,
Docker nor a credential, so it proves the compiled config is genuinely
consumable for free:

```bash
python tools/run_benchmark_smoke.py oragentbench ... --upstream-dry-run --execute
```

`benchmark-smoke.yml` runs this on every dispatch, before any paid job starts.

**Agent profiles.** `--scaffold` selects a profile recorded from upstream's own
experiment configs — which environment variables the scaffold reads and which
kwargs upstream pins:

| Scaffold | Environment | Pinned kwargs |
| --- | --- | --- |
| `claude-code` | `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` ← `MODEL_API_KEY`; `ANTHROPIC_BASE_URL` ← `MODEL_BASE_URL`; `ANTHROPIC_MODEL` = the model id | `disallowed_tools: WebSearch,WebFetch` |
| `codex` | `OPENAI_API_KEY` ← `MODEL_API_KEY`; `OPENAI_BASE_URL` ← `MODEL_BASE_URL` | `reasoning_effort: high`, `web_search: disabled` |
| `mini-swe-agent` | `MSWEA_API_KEY` ← `MODEL_API_KEY` | — |

Only variable names appear in the plan. The compiled job config carries
`${MODEL_API_KEY}`; the runner substitutes the value. The normalized provider
route digest, not the route string, enters campaign identity, and execution
requires the runtime route to match it.

### FrontierOR

**The command that runs**, from the checkout:

```bash
python3 -m frontieror.infra agent \
    --paper-id <id> --primary-model <provider/model> \
    --stage1-instances tiny --dev-set large_1 --test-set large_2 \
    --coral-agent-count 1 --coral-attempts 10 --coral-max-steps 10 \
    --coral-max-seconds auto --cpus 1 --memory 128G --run-id <id>
```

**Driving it:**

```bash
python tools/run_benchmark_smoke.py frontieror \
    --source upstream/FrontierOR \
    --paper-id bierwirth2017 --primary-model openai/gpt-5.4 \
    --stage1-instances tiny --dev-set large_1 --test-set large_2 \
    --run-id orbench-agent-smoke \
    --receipt out/agent-receipt.json \
    --execute --acknowledge-cost i-accept-model-costs
```

**What is deliberately absent from that command.** Upstream fixes its
trusted-agent profile and appends it itself: `--framework coral`,
`--exec-mode docker`, `--stage2-scorer staged_qte`,
`--coral-agent-isolation docker`, `--coral-model-access proxy`,
`--coral-agent-image frontieror-coral-agent:0.1`, `--anti-hack`. Supplying any
of them from here could only weaken or contradict the profile, so the builder
**refuses them as inputs** — along with anything outside upstream's documented
agent interface. `--extra` takes one shell-quoted string so those flags reach
the validator instead of being swallowed by argparse.

**Validation before anything starts.** The paper id must appear in that
checkout's `paper_meta_info.json`; instance names must be `tiny` or `large_<N>`;
the held-out test set must not overlap stage-1 or dev; the model must be a full
provider/model route and not a floating alias.

**Machine setup you must do once**, per upstream's README:

```bash
bash test_time_self_evolution/coral/setup.sh
docker build -f frontieror/infra/docker/candidate.Dockerfile   -t frontieror-candidate:1 .
docker build -f frontieror/infra/docker/agent.Dockerfile       -t frontieror-coral-agent:0.1 .
docker build -f frontieror/infra/docker/model-proxy.Dockerfile -t frontieror-coral-model-proxy:0.1 .
python -m frontieror.infra security-check --candidate-image frontieror-candidate:1
```

Plus: the HuggingFace dataset downloaded locally, `GRB_LICENSE_FILE` pointing at
a readable Gurobi licence, `OPENROUTER_API_KEY` in the environment, and
`ORBENCH_PERF_ISOLATED=true` on a host whose cores are actually pinned. Each is
checked; a missing one stops the job.

**What such a run may claim.** Official final instances are unpublished
server-only data. A run against the public dataset is an integration test, not a
leaderboard score — and a run on any host that is not performance-isolated is
exploratory by construction, because the score contains a wall-clock term.

## Secrets and runner labels

### Repository secrets

| Secret | Needed by | When | Notes |
| --- | --- | --- | --- |
| *(none)* | `ci.yml`, `integration-contract.yml`, `report.yml` | always | These reference no secrets; a test enforces it |
| `MODEL_API_KEY` (secret) | `benchmark-smoke.yml`, `mode=agent`, ORAgentBench | agent campaigns only | Set a provider-side spend cap on the key. CI cannot interrupt spend |
| `MODEL_BASE_URL` (variable) | `benchmark-smoke.yml`, `mode=agent`, ORAgentBench | agent campaigns only | The provider base URL. Required, but not itself a credential — a repository *variable*, not a secret |
| `ORBENCH_RUNS_ROOT` (variable) | `benchmark-smoke.yml`, self-hosted ORAgentBench jobs | controls and agent campaigns | Persistent runner-owned workspace used for deterministic recovery; never point it at `RUNNER_TEMP` or the checkout |
| `OPENROUTER_API_KEY` (secret) | `benchmark-smoke.yml`, `mode=agent`, FrontierOR | agent campaigns only | FrontierOR routes model calls through OpenRouter and hands the agent container only an ephemeral proxy token |

### GitHub Environments

| Environment | Contains | Approval | Used by |
| --- | --- | --- | --- |
| `controls` | nothing paid; scopes runner access | no | `oracle-controls` job |
| `benchmark-agent` | model API keys | **required reviewers** | `agent` job |

Required reviewers on `benchmark-agent` are what actually stop an unattended
spend. The workflow only refuses to start without the preconditions.

### Runner labels

| Labels | Capability | Required by |
| --- | --- | --- |
| `self-hosted, orbench-exec` | Docker, the `harbor` CLI, Python 3.11+, git | `oracle-controls` and `agent-oragentbench` |
| `self-hosted, orbench-exec, perf-isolated` | the above, plus pinned cores, no co-tenancy, a Gurobi licence and the three FrontierOR images | `agent-frontieror` |

Runner-side environment variables the smoke workflow checks:

| Variable | Meaning |
| --- | --- |
| `GRB_LICENSE_FILE` | path to a readable Gurobi licence (FrontierOR) |
| `ORBENCH_PERF_ISOLATED` | `true` only on a machine whose cores are actually pinned (FrontierOR) |
| `ORBENCH_RUNS_ROOT` | absolute, current-user-owned persistent workspace outside the checkout and runner temp directory |
| `ORBENCH_HOST_LOCK_DIR` | optional private absolute directory for the runner-account-wide ORAgentBench Docker alias lock; do not share one daemon across Unix accounts |

Upstream checkouts are cloned by the workflow itself, at the pinned commit, via
`.github/scripts/clone-upstream.sh` — so a runner does not need one prepared in
advance.

Each is checked for real — a missing one fails the job rather than degrading it.

## Local smoke runs

All zero cost. None calls a model or executes a benchmark.

```bash
# 0. install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. what is integrated
orbench integrations list

# 2. clone both upstreams at their pinned commits and inspect them.
#    The directory names matter: ORAgentBench resolves its dataset path
#    against the checkout's parent.
bash .github/scripts/clone-upstream.sh oragentbench /tmp
bash .github/scripts/clone-upstream.sh frontieror   /tmp
orbench integration inspect oragentbench --source /tmp/ORAgentBench --json out/oab.json
orbench integration inspect frontieror   --source /tmp/FrontierOR   --json out/fo.json

# 3. read FrontierOR's own scoring contract (upstream command, zero cost)
(cd /tmp/FrontierOR && python -m frontieror.infra contract)

# 4. validate campaigns — the second one must be rejected
orbench campaign validate campaigns/oragentbench-controls.yaml
orbench campaign validate campaigns/frontieror-contract-check.yaml || echo "rejected as designed"

# 5. plan, and confirm the plan is deterministic
orbench campaign plan campaigns/oragentbench-controls.yaml --out out/plan-a
orbench campaign plan campaigns/oragentbench-controls.yaml --out out/plan-b
diff -r out/plan-a out/plan-b && echo "byte-identical"

# 6. render a report from fixtures
orbench report build --input fixtures/normalized/oragentbench-controls.json --out out/report
cat out/report/summary.md

# 7. confirm the evidence gate refuses to over-label a single-rollout slice
orbench report build --input fixtures/normalized/oragentbench-smoke-r0.json \
  --out out/report-gate --require-label validated || echo "refused as designed"

# 8. prepare the one-command ORAgentBench lifecycle without executing Docker.
orbench run oragentbench \
  --source /tmp/ORAgentBench --task additive_microfactory_order_planning \
  --agent oracle --date 2026-08-24 --workspace /tmp/orbench-runs

# 9. the full test suite
python -m pytest -q
```

Step 8 is safe by default: it writes the inspected source snapshot, compiled
plan, manifest, receipt and integrity ledger but does not start Docker or call a
model. Add `orbench doctor ...` before any real execution.

### Running an actual ORAgentBench control campaign

This one does execute containers, on your machine, with no model spend. It needs
Docker and the Harbor CLI, which ORBenchLab does not install:

```bash
orbench doctor oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning --agent oracle
orbench run oragentbench \
  --source /tmp/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent oracle --date 2026-08-24 \
  --workspace /srv/orbench/runs --execute
orbench export \
  --run-root /srv/orbench/runs/<campaign-id> \
  --destination ./share/<campaign-id>
```

The lifecycle invokes upstream, ingests and reconciles the resulting Harbor
trial, renders the report, and marks the campaign complete only after Docker
image identity and evidence integrity checks pass. `orbench export` is the only
supported host-to-share boundary; raw jobs, trajectories and logs remain local.

## Adding a third benchmark

Full walkthrough in [`adding-a-benchmark.md`](adding-a-benchmark.md); the
template is in `templates/integration_template/`.

A third benchmark must provide, at minimum:

1. **A module with `NAME`, `KIND`, `describe()` and `inspect()`** registered in
   `src/orbenchlab/integrations/registry.py`.
2. **A pinned upstream commit** in the module, cloned by
   `integration-contract.yml`.
3. **A content-addressed dataset digest** emitted as
   `facts["dataset_digest"]`, so run ids are pinned to a dataset state.
4. **An explicit form decision** in `decisions`, with `adapter_required` and —
   where relevant — `harbor_conversion_parity_safe` and the blockers behind it.
5. **Declared preconditions**: every secret, licence, dataset, image and runner
   capability, each `fail_closed`.
6. **A zero-cost path** — a contract read, a static validation, or built-in
   controls — so the integration can be exercised in CI without spend.
7. **A decision record** in `docs/decisions/` if the form is anything other than
   the obvious one.
8. **Checks that fail closed.** A check that cannot be performed is `skip` with
   a reason, never `pass`.

If the form is `harbor-adapter`, add the differential re-scoring fixture
described above before the integration is trusted for anything but exploration.
