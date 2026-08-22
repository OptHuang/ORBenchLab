# Build report — pre-publication handoff snapshot

> Historical note: this document records the Opus 5 implementation handoff
> before the repository was transferred to the local workspace and published.
> Absolute paths, test counts and GitHub-status statements below describe that
> build host at that moment; the current authoritative status is the repository
> README plus the live GitHub Actions checks.

**Repository target:** `OptHuang/ORBenchLab` (built on the collaborator host at `/root/projects/orbenchlab-build`)
**Date:** 2026-08-22 (initial build; revised the same day to close the agent-execution gap)
**Environment:** Linux 6.8.0-63-generic, Python 3.12.3, PyYAML 6.0.x, pytest 9.1.1

---

## 1. The key integration decision

Two benchmarks, deliberately opposite forms. The forms are not a stylistic
choice — each follows from what upstream already owns.

### ORAgentBench → Harbor-native, no adapter

Inspecting `ORAgentBench/ORAgentBench@c9eb952435a4352f33daa2a35efe0f8c76d31b28`
shows the benchmark is already Harbor-native: **107 task packages** under
`harbor_tasks/`, every one declaring `[task] name = "oragentbench/<task>"`, all
107 writing `/logs/verifier/reward.json` with keys `feasibility` and `quality`,
plus `metrics/per_dimension_reward.py`, `experiments/config/run_oracle_all.yaml`,
`skills/` and `difficulty.json`.

So ORBenchLab writes **no adapter** and **materialises no second copy of the
tasks**. It records the dataset digest and reward keys, compiles campaigns into
Harbor job configs, and reports. Upstream's verifier, reference metrics and
aggregation script are consumed as they are.

The inspector also reports three things about upstream that constrain what a
campaign may claim — reported as warnings, **not patched locally**, because a
local patch would fork the benchmark:

| Warning | Finding | Consequence |
| --- | --- | --- |
| `verifier_environment_mode` | 0/107 tasks declare it | Separate-mode verifiers are what make hidden grading and official regrade possible; all 107 fall back to the Harbor default |
| `agent_network_isolation` | 107/107 set `allow_internet = true` | A campaign cannot claim anti-speculation isolation from this upstream state |
| `multi_step_tasks` | 8/107 are multi-step | Harbor's regrade path does not support multi-step tasks, so these cannot be re-scored officially |

`solver_license_grant` is inferred as `none`: no task declares
`requires_gurobi`.

### FrontierOR → official external harness, Harbor conversion deferred

Inspecting `Minw913/FrontierOR@8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9`, the
integration **runs upstream's own contract command** —
`python -m frontieror.infra contract`, zero cost, no network, no model — and
reads back:

```json
{ "scorer": "staged_qte",
  "contract_version": "staged-qte-v1",
  "runtime_measurement": "trusted_host_wall_clock",
  "candidate_reported_timestamps_trusted": false,
  "private_parameters": ["reference_objective", "reference_runtime",
                         "checker_implementation", "final_instance_membership"] }
```

Ten artefacts are marked trusted-only. Upstream owns the security boundary:
`frontieror/infra/security_check.py` plus three container definitions, an egress
proxy and a credential-isolating model proxy.

**Decision: wrap the official harness. Do not materialise Harbor tasks yet.**
Recorded in `docs/decisions/0001-frontieror-integration.md`, with four blockers
emitted machine-readably as `facts.harbor_conversion_blockers`:

1. `trusted_host_timing` — the score includes a speed term measured as
   trusted-host wall clock against a reference runtime on a pinned core. A
   container on a shared runner produces a differently-defined number.
2. `undistributed_reference_data` — reference objectives, reference runtimes,
   the checker and final-instance membership are trusted-only. A task package
   cannot contain them; shipping substitutes makes it a different benchmark.
3. `upstream_security_boundary` — re-expressing candidate isolation and the
   proxies as `task.toml` declarations is a re-implementation whose divergences
   surface as silent scoring differences.
4. `multi_candidate_inner_loop` — test-time self-evolution runs many candidates
   per task inside the harness; Harbor's trial model would have to be mapped
   onto that first.

The consequence is enforced in code, not documentation. Because runtime enters
the score, campaign validation refuses a FrontierOR campaign on any site not
declared performance-isolated:

```console
$ orbench campaign validate campaigns/frontieror-contract-check.yaml
orbench: error: campaign spec is invalid (2 problem(s)):
  - integration 'frontieror' scores runtime, but site 'local-docker' declares
    perf_isolated: false. Performance-scored benchmarks may only run on a
    performance-isolated site with pinned cores; otherwise the number is not
    comparable to anything.
  - integration 'frontieror' requires a solver licence, but site 'local-docker'
    declares solver_license_slots: 0
$ echo $?
2
```

Neither integration vendors a single line of upstream code, and a test proves
it (`test_this_repository_vendors_no_upstream_benchmark_code`).

---

## 2. The agent execution paths

The first build left `benchmark-smoke.yml` with a placeholder that always
exited. That has been replaced with two real paths, each invoking the command
its upstream project documents for itself. Both were read from the pinned
checkouts, not recalled; the provenance travels with each command.

### ORAgentBench — upstream's own Harbor wrapper

```bash
# cwd: the parent of the checkout
python3 <checkout>/source/scripts/run_harbor_prebuild.py \
    -c <compiled job config> --skip-build
```

That is exactly what upstream runs. `experiments/scripts/run_claude_code.sh` is
`bash scripts/run_harbor_prebuild.sh -c <config> --skip-build`; that shell
script execs `python source/scripts/run_harbor_prebuild.py "$@"`; and the
wrapper's own last line is `harbor run -c <transformed config>` executed with
`cwd=BENCH_ROOT.parent`.

Two facts from `source/scripts/run_harbor_prebuild.py` shaped the
implementation and would have been easy to get wrong:

* `resolve_repo_path` resolves a relative dataset path against
  `BENCH_ROOT.parent`, so the checkout **must be named `ORAgentBench`** or the
  dataset path in the job config resolves to nothing. `clone-upstream.sh`
  enforces the name, and the builder refuses a wrongly-named checkout.
* `selected_dataset_tasks` filters on task **directory** names and raises on an
  unknown one, so a task input is validated against `harbor_tasks/<name>/task.toml`
  before anything starts.

**Verified for real, at zero cost.** Upstream's wrapper has its own `--dry-run`
that transforms the config and prints the harbor command without running it,
needing neither Harbor, Docker nor a credential. Run against our compiled
config it exits 0, and its transformed output shows upstream consuming our
agent block correctly:

```console
$ python tools/run_benchmark_smoke.py oragentbench \
    --source upstream/ORAgentBench --task additive_microfactory_order_planning \
    --scaffold claude-code --model deepseek-v4-pro --date 2026-08-22 \
    --upstream-dry-run --execute

# Transformed Harbor config: /tmp/oragentbench-prebuild-.../oab-...--claude-code-deepseek-v4-pro--s1--a1--sh0.yaml
agents:
- name: null
  import_path: ORAgentBench.harbor_agents.prebuilt_agents:PrebuiltClaudeCode
  model_name: deepseek-v4-pro
  override_setup_timeout_sec: 420
  env:
    ANTHROPIC_AUTH_TOKEN: ${MODEL_API_KEY}
    ANTHROPIC_API_KEY: ${MODEL_API_KEY}
    ANTHROPIC_BASE_URL: ${MODEL_BASE_URL}
    ANTHROPIC_MODEL: deepseek-v4-pro
    CLAUDE_CODE_ATTRIBUTION_HEADER: '0'
  kwargs: {version: null, disallowed_tools: WebSearch,WebFetch, orbench_seed: 1}
datasets:
- path: /tmp/oragentbench-skills-.../harbor_tasks
  task_names: [additive_microfactory_order_planning]

+ harbor run -c /tmp/oragentbench-prebuild-.../oab-...yaml
# upstream exited 0
```

Upstream rewrote `name: claude-code` into its own prebuilt agent class, injected
`CLAUDE_CODE_ATTRIBUTION_HEADER`, copied the selected task into a temporary
dataset with dynamic skills, and emitted the `harbor run` command. Credentials
stayed `${NAME}` placeholders throughout. This is the strongest evidence
available without spending money: upstream itself accepted the configuration.

Making that possible required the compiler to emit upstream's actual agent
shape, so three fields were added to the campaign spec — `env_from_secret`
(variable name → secret name), `env_literals` (non-secret literals such as a
model id, validated to reject credential-shaped values), and `import_path` /
`setup_timeout_sec`. Agent profiles for `claude-code`, `codex` and
`mini-swe-agent` were recorded from upstream's own experiment configs.

### FrontierOR — the official trusted-agent entry point

```bash
# cwd: the checkout
python3 -m frontieror.infra agent \
    --paper-id <id> --primary-model <provider/model> \
    --stage1-instances tiny --dev-set large_1 --test-set large_2 \
    --coral-agent-count 1 --coral-attempts 10 --coral-max-steps 10 \
    --coral-max-seconds auto --cpus 1 --memory 128G --run-id <id>
```

Flag order follows upstream's README example, and every flag was checked against
`frontieror/infra/cli.py::_agent_parser`.

**What the command deliberately omits.**
`frontieror/infra/policy.py::hardened_agent_argv` *appends* the non-overridable
trusted profile — `--modes self_evolve --framework coral --exec-mode docker
--stage2-scorer staged_qte --coral-agent-isolation docker --coral-model-access
proxy --coral-agent-image frontieror-coral-agent:0.1 --anti-hack` — and raises
on a conflicting value. Supplying any of them from here could only weaken or
contradict the profile, so the builder refuses them as inputs rather than
relying on upstream to catch it. It also refuses anything outside upstream's
documented agent interface.

Observed refusals, each exiting non-zero before any process starts:

```console
$ ... --extra "--exec-mode bare"
error: refusing to pass ['--exec-mode']: upstream fixes its trusted-agent profile
(coral framework, docker isolation, staged_qte scoring, proxied model access,
anti-hack) and appends those flags itself.                                   # exit 2

$ ... --extra "--totally-made-up 1"
error: refusing to pass ['--totally-made-up']: not part of upstream's documented
'python -m frontieror.infra agent' interface                                 # exit 2

$ ... --dev-set large_2 --test-set large_2
error: the held-out test set overlaps the stage1/dev instances the agent can
see: ['large_2'].                                                            # exit 2

$ ... --paper-id notapaper2020
error: paper id 'notapaper2020' is not in this checkout's paper_meta_info.json
(179 known ids)                                                              # exit 2

$ ... --primary-model openai/gpt-latest
error: model route 'openai/gpt-latest' is a floating alias; pin an exact model  # exit 2
```

The job runs only on `[self-hosted, orbench-exec, perf-isolated]`, and fails
closed without `OPENROUTER_API_KEY`, a readable `GRB_LICENSE_FILE`,
`ORBENCH_PERF_ISOLATED=true` and all three upstream container images.

### Where the code lives

| Piece | Purpose |
| --- | --- |
| `src/orbenchlab/execution.py` | Pure command construction, input validation, preconditions, receipt sanitization. No I/O beyond reading the checkout |
| `tools/run_benchmark_smoke.py` | The only thing that starts an upstream process. Dry run by default; `--execute` plus a cost acknowledgement for paid paths |
| `.github/scripts/clone-upstream.sh` | Clones the pinned commit into the directory name each project requires; reads the pin from the integration registry so YAML and library cannot disagree |

`orbench` itself still executes nothing.

## 3. Test evidence

### 3.1 Full suite

```console
$ python -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
$ .venv/bin/python -m pytest
266 passed, 2 skipped in 1.14s
```

The 2 skips are the opt-in tests that need real upstream checkouts. With them:

```console
$ git clone https://github.com/ORAgentBench/ORAgentBench /tmp/.../oragentbench
$ git clone https://github.com/Minw913/FrontierOR       /tmp/.../frontieror
$ ORBENCH_ORAGENTBENCH_SOURCE=/tmp/.../oragentbench \
  ORBENCH_FRONTIEROR_SOURCE=/tmp/.../frontieror \
  .venv/bin/python -m pytest
268 passed in 1.29s
```

| File | Tests | Covers |
| --- | ---: | --- |
| `tests/test_execution.py` | 76 | **Exact upstream argv** for both integrations; task/paper/instance validation; path traversal; trusted-profile refusals; leaky-split refusal; preconditions; receipt sanitization; the script's own exit codes |
| `tests/test_report.py` | 34 | Golden reports; both downgrade paths; the comparative-claim guard; the unbacked-claim guard; metrics |
| `tests/test_campaign.py` | 32 | 17 validation-policy rejections; byte-identical compilation; ledger schema; no secrets in job configs |
| `tests/test_workflows.py` | 31 | YAML validity, sha pinning, permissions, timeouts, secrets, fail-closed guards, agent-job shape, `bash -n`, embedded-python compilation |
| `tests/test_integrations.py` | 29 | Registry; both inspectors against miniature trees; fail-closed paths; no vendoring; 2 opt-in real-checkout tests |
| `tests/test_cli.py` | 23 | Every subcommand and every exit code |
| `tests/test_no_reimplementation.py` | 16 | AST guard against reimplementing execution; dependency minimality; required files; credential scan |
| `tests/test_ids.py` | 14 | Determinism, seed isolation, credential-safety, sharding, match keys, re-score ids |
| `tests/test_schemas.py` | 13 | Schema loading; the validator subset; refusal of unimplemented keywords |
| **Total** | **268** | |

### 3.2 Clean-room install

Verified from an empty virtual environment, not only from the development one:

```console
$ python3 -m venv cleanroom
$ cleanroom/bin/python -m pip install -e ".[dev]"
Successfully installed PyYAML-6.0.3 iniconfig-2.3.0 orbenchlab-0.1.0 packaging-26.3 pluggy-1.6.0 pygments-2.21.0 pytest-9.1.1
$ cleanroom/bin/python -m pytest -q
266 passed, 2 skipped
$ cleanroom/bin/orbench --version
orbench 0.1.0
```

One runtime dependency (PyYAML), enforced by
`test_runtime_dependencies_stay_minimal`.

### 3.3 Integration inspection against real upstream heads

```console
$ orbench integration inspect oragentbench --source <ORAgentBench checkout>
status: degraded  counts: {'fail': 0, 'pass': 9, 'skip': 1, 'warn': 3}
  [pass] harbor_native_task_packages: found 107 Harbor task packages under harbor_tasks/
  [pass] task_name_declared: all 107 tasks declare [task].name, 107 namespaced as <org>/<task>
  [pass] dataset_digest: sha256:ff55ae31f9a742ca7cc2b6fc48eb29c98d09c0992dfa8a40d1194b85dde5fe63
  [pass] reward_channel: 107/107 tasks write /logs/verifier/reward.json with keys ['feasibility', 'quality']
  [warn] verifier_environment_mode: no task declares [verifier].environment_mode
  [warn] agent_network_isolation: 107/107 tasks set [environment].allow_internet = true
  [warn] multi_step_tasks: 8/107 tasks are multi-step ([[steps]])
  [pass] solver_license_grant: inferred 'none' (0 tasks declare requires_gurobi = true)
  [pass] official_metric_script: metrics/per_dimension_reward.py found
  [pass] oracle_control_config: experiments/config/run_oracle_all.yaml present; declares the built-in oracle agent
  [skip] nop_control_available: a Harbor built-in; not verifiable from this upstream repository
  [pass] difficulty_metadata: difficulty.json present
  [pass] no_vendored_task_copy: ORBenchLab contains no vendored harbor_tasks/ copy

$ orbench integration inspect frontieror --source <FrontierOR checkout>
status: degraded  counts: {'fail': 0, 'pass': 9, 'skip': 0, 'warn': 1}
  [pass] official_harness_entrypoint: exposes ['agent', 'submission', 'contract', 'security-check']
  [pass] public_scoring_contract: scorer='staged_qte' version='staged-qte-v1'
  [pass] performance_isolation_required: runtime enters the score (trusted_host_wall_clock)
  [pass] hidden_artifact_visibility: 10 artefacts trusted-only, incl. reference_objective, reference_runtime,
                                     feasibility_checker, final_instance_membership
  [pass] trusted_checkers_present: 2 sample checkers present upstream; referenced, never copied
  [pass] security_boundary_owned_upstream: security_check.py plus 3 container definitions
  [pass] submission_verifier_entrypoint: zero-model-call verification path available
  [pass] external_preconditions_declared: 5 preconditions, all fail-closed
  [pass] no_vendored_upstream_copy: no FrontierOR checker, harness or scoring code vendored
  [warn] harbor_conversion_parity_safe: not parity-safe yet (4 blockers)
```

Both reports carry
`execution: {model_calls: 0, benchmark_executed: false, network_access: false, reads_credentials: false}`,
which CI asserts on.

### 3.4 Determinism

```console
$ orbench campaign plan campaigns/oragentbench-controls.yaml --out /tmp/plan-a
$ orbench campaign plan campaigns/oragentbench-controls.yaml --out /tmp/plan-b
$ diff -r /tmp/plan-a /tmp/plan-b && echo "byte-identical"
byte-identical
```

18 runs across 6 jobs, campaign id `oragentbench-controls-20260822-7a6d8ee1`.
One job per `(agent, seed, attempt)` with `n_attempts: 1`, which is what makes
`(job_name, task_name)` an exact ledger lookup rather than a guess.

### 3.5 Report goldens and the evidence gate

```console
$ orbench report build --input fixtures/normalized/oragentbench-controls.json --out /tmp/report-controls
{ "intended_label": "validated", "effective_label": "partial",
  "downgrade_reasons": ["'validated' requires every referenced run to have at least 2 verified
     content-addressed replicas; the slice reports min_replica_count=1 and verified=False"],
  "comparative_claims_allowed": false, "claims": 13 }
$ diff /tmp/report-controls/summary.md tests/golden/oragentbench-controls.summary.md   # no output

$ orbench report build --input fixtures/normalized/oragentbench-smoke-r0.json --out /tmp/report-smoke
{ "intended_label": "partial", "effective_label": "exploratory",
  "downgrade_reasons": ["'partial' requires at least one capability measurement repeated 3 or more
     times for the same (task, agent) configuration; every configuration here was measured once (R0)"] }
$ diff /tmp/report-smoke/summary.md tests/golden/oragentbench-smoke-r0.summary.md      # no output

$ orbench report build --input fixtures/normalized/oragentbench-smoke-r0.json \
    --out /tmp/report-gate --require-label validated
orbench: effective evidence label 'exploratory' is weaker than the required 'validated'
$ echo $?
4
```

Both downgrade paths — durability and repetition — fire on real fixtures. The
comparative-claim guard is tested by injecting a comparison into the renderer
and confirming it is rejected at `R0` and accepted once repetition supports it
(`test_a_comparative_claim_in_r0_output_is_rejected`,
`test_the_same_claim_is_allowed_once_repetition_supports_it`).

---

## 4. Workflow verification

**No hosted GitHub Actions run has been observed.** The repository was never
created on GitHub and nothing was pushed. What follows was verified locally, by
parsing each workflow and executing its `run` blocks in per-step subshells —
which is how Actions runs them, but is not the same as watching Actions run
them.

### 4.1 `ci.yml` — every step executed locally, exit 0

Extracted with PyYAML and run in order (skipping only `pip install`, already
satisfied by the venv):

| Step | Result |
| --- | --- |
| Unit, schema and golden tests | `183 passed, 2 skipped` |
| CLI smoke — registry and schemas | both integrations listed; 3 schemas listed |
| Schema — shipped fixtures validate | both fixtures valid |
| Campaign — shipped specs validate or fail for a stated reason | controls accepted; FrontierOR spec **rejected as required** |
| Plan determinism | `diff -r` empty; ledger schema-valid |
| Report golden | both diffs empty |
| Evidence gate | R0 slice **refused** as `validated` (exit 4) |
| Workflow and repository guards | `40 passed` |

### 4.2 `integration-contract.yml` — executed locally against fresh clones, exit 0

Both matrix entries, cloning by exact commit into empty directories:

```console
########## matrix: oragentbench ##########
resolved ref=c9eb952435a4352f33daa2a35efe0f8c76d31b28
checked out c9eb952435a4352f33daa2a35efe0f8c76d31b28
status: degraded  counts: {'fail': 0, 'pass': 9, 'skip': 1, 'warn': 3}
oragentbench: integration decisions unchanged

########## matrix: frontieror ##########
resolved ref=8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9
checked out 8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9
status: degraded  counts: {'fail': 0, 'pass': 9, 'skip': 0, 'warn': 1}
frontieror: integration decisions unchanged
```

The three ORAgentBench warnings and the one FrontierOR warning surfaced as
`::warning::` annotations. No secrets, no model calls, no containers.

**One real defect was found and fixed by running this.** The clone step
originally used `cd upstream/<integration>` and then bare `git` commands. Every
Actions step gets a fresh shell, so that only worked by accident of where the
step happened to start. It now uses explicit `git -C "$DEST"` paths.

### 4.3 `report.yml` — executed locally, both directions

Fixtures path, exit 0: guard passed, both slices schema-validated, both reports
built, labels printed with their downgrade reasons.

Negative test of the raw-bundle guard — a slice was doctored to contain
`trajectory.json` and `reference_objective`:

```console
=== Guard — reject raw bundle content in the input ===
::error::incoming/leaky.json contains raw bundle or hidden verifier content
$ echo $?
1
```

The guard operates on uploaded bytes, so it does not depend on the uploader
having been careful.

### 4.4 `benchmark-smoke.yml` — preflight executed locally, all paths exercised

The workflow now has four jobs: `preflight` (GitHub-hosted, zero cost),
`oracle-controls`, `agent-oragentbench` and `agent-frontieror`.

**Preflight, executed locally against fresh clones at the pinned commits:**

| Dispatch | Result |
| --- | --- |
| `integration=oragentbench, mode=validate-only` | exit 0 — cloned, task validated, plan compiled, upstream command built |
| `integration=frontieror, mode=validate-only` | exit 0 — cloned, paper id and split validated, official command built |
| the conditional upstream-wrapper dry-run step | exit 0 — upstream printed `+ harbor run -c ...` and stopped |
| `mode=agent` without acknowledgement | exit 1 — `::error::mode=agent spends money; re-dispatch with acknowledge_cost set to i-accept-model-costs` |
| `mode=agent` with acknowledgement | exit 0 — gate sets `run_agent_oab=true` |
| `mode=oracle-controls, integration=frontieror` | exit 1 — `::error::oracle/nop controls are Harbor built-ins ...` |

**The clone script**, shared with `integration-contract.yml` so the pin cannot
drift between workflows:

```console
$ bash .github/scripts/clone-upstream.sh oragentbench upstream
cloned oragentbench -> upstream/ORAgentBench at c9eb952435a4352f33daa2a35efe0f8c76d31b28
$ bash .github/scripts/clone-upstream.sh frontieror upstream
cloned frontieror -> upstream/FrontierOR at 8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9
$ bash .github/scripts/clone-upstream.sh nope upstream
error: unknown integration 'nope'                                  # exit 2
```

It reads the pin from the integration registry, checks out that exact commit,
verifies the resulting HEAD matches, and uses the directory name each project
requires.

**The three self-hosted jobs were not executed.** They need a self-hosted
runner, Docker, the Harbor CLI, secrets and — for FrontierOR — a Gurobi licence,
three container images and a performance-isolated host. None exists here, and
running the agent paths would have meant paid model calls. Their command
construction, input validation and refusals are asserted by the 76 tests in
`tests/test_execution.py`, and the ORAgentBench path was additionally verified
end to end through upstream's own dry run (§2).

### 4.5 Static workflow guarantees (tested)

* All four workflows parse; every job declares `timeout-minutes`, `permissions:
  contents: read` and a `concurrency` group.
* All 20 `uses:` references (4 distinct actions) are pinned to full
  40-character shas with version comments, resolved from GitHub rather than
  recalled:
  `actions/checkout@08c6903c…` (v5.0.0), `actions/setup-python@e797f83b…`
  (v6.0.0), `actions/upload-artifact@330a01c4…` (v5.0.0),
  `actions/download-artifact@018cc2cf…` (v6.0.0).
* `ci.yml`, `integration-contract.yml` and `report.yml` reference **no**
  repository secrets. Only `benchmark-smoke.yml` does, and only
  `MODEL_API_KEY` / `OPENROUTER_API_KEY`.
* `benchmark-smoke.yml` is `workflow_dispatch`-only. No workflow uses
  `pull_request_target`. Preflight refuses to run on a fork. Self-hosted
  runners appear only in that workflow, and only the FrontierOR agent job
  requires the `perf-isolated` label.
* Both agent jobs upload the sanitized receipt and nothing else — asserted on
  the parsed workflow, not on a comment.
* Every `run` block passes `bash -n`; every heredoc Python body compiles.

---

## 5. What was built

```
src/orbenchlab/
  core/                  ids, evidence rules, errors, a small schema validator
  integrations/          base contract, registry, oragentbench, frontieror
  campaign/              spec validation, compiler
  report/                metrics, renderer
  execution.py           upstream command construction, validation, receipts
  schemas/               3 published JSON schemas
tools/                   regenerate_fixtures.py, run_benchmark_smoke.py
.github/scripts/         clone-upstream.sh
tests/                   268 tests
campaigns/               2 specs (one designed to be rejected)
sites/                   2 declarations
fixtures/normalized/     2 slices, generated from real planner output
tests/golden/            3 golden files
templates/               integration template + README
docs/                    integrations, adding-a-benchmark, 2 decision records
.github/workflows/       ci, integration-contract, benchmark-smoke, report
.github/dependabot.yml   monthly action-pin and dependency updates
```

CLI surface: `orbench integrations list`, `integration inspect|describe`,
`campaign validate|plan`, `report build`, `schema list|validate`.

Exit codes: `2` spec, `3` integration, `4` evidence, `6` schema, `7`
schema-feature.

---

## 6. Scope compliance

| Constraint | Status |
| --- | --- |
| Do not create GitHub repositories or push | No repository created, nothing pushed, no `git init` in the build directory |
| Do not run paid model calls or a full benchmark | Zero model calls. Two upstream subprocesses were executed, both free and both non-executing: `python -m frontieror.infra contract` (prints a JSON contract) and ORAgentBench's wrapper with `--dry-run` (prints the transformed config and the `harbor run` command, then stops). The agent paths were built and tested, never run |
| Do not read or emit credentials | No credential read. `agent_uid` hashes variable *names* only; job configs carry `${NAME}` placeholders; the inspector subprocess runs with a minimal environment; a test scans the repository for credential markers |
| Do not alter existing tmux sessions or repositories | No `tmux` command run. Upstream clones are read-only, in the scratchpad, outside the build directory |
| Do not claim hosted Actions or benchmark runs succeeded | Stated in §4, in §7, in `README.md` and in `SECURITY.md`: no hosted run and no paid agent run observed |
| Placeholders fail closed and are labelled | There is no longer a placeholder agent path — it was replaced with the real one (§2). What remains fail-closed: `mode=agent` without acknowledgement, missing secrets/licence/images/isolation, `validated` campaigns at validation time, FrontierOR on a shared site |

---

## 7. Known gaps

Stated plainly, because a gap that reads as a feature is worse than an open one.

1. **No ingest.** Producing a normalized slice from raw Harbor job directories
   is not implemented. The report path is exercised with fixtures generated
   from real planner output; their *scores* are synthetic and each fixture says
   so in its `_note` field.
2. **No durability verification**, so no `validated` reports. Validation
   refuses the label rather than accepting it on trust.
3. **No paid agent run has been observed.** The agent paths are implemented,
   their exact commands are asserted by tests, and upstream's own dry run
   accepted the compiled configuration — but no model call has been made from
   here, and no self-hosted job has run. What remains unproven is everything
   downstream of `harbor run` actually starting: image build, container
   execution, verifier results.
4. **No differential re-scoring fixture** for FrontierOR. That is the gate on
   any future Harbor conversion (`docs/decisions/0001`, condition 3), and it is
   the cheapest place to find "we wrapped it and the semantics changed" bugs.
5. **No hosted CI run observed.**
6. **`perf_isolated` is an assertion, not a measurement.** A site file claiming
   it is taken at face value, as is `ORBENCH_PERF_ISOLATED=true` on a runner.
7. **The budget is a compile-time gate, not a circuit breaker.** Job plugins
   expose start and end hooks only and are not called mid-job. Set a
   provider-side spend cap on the key; nothing here can interrupt spend.
8. **The FrontierOR agent path has no upstream dry run to lean on.** Unlike
   ORAgentBench, its `agent` subcommand has no no-op mode, so its command is
   verified by construction and by upstream's own argument parser — not by
   having watched upstream accept it.

## 8. Suggested next steps

1. Run the ORAgentBench oracle/nop controls on a real machine. Zero model cost,
   and it is the cheapest evidence that the task set is well-formed — and the
   first thing that exercises `harbor run` for real.
2. Ingest: Harbor job directory → normalized slice, reconciling by `match_key`
   and recording orphans. This closes the loop from execution to report.
3. Run one paid ORAgentBench agent smoke on a configured runner, cheapest model
   first, and confirm the receipt and the resulting bundle line up with the plan
   ledger.
4. Build the FrontierOR differential re-scoring fixture, then revisit the
   conversion decision with evidence instead of judgement.
5. Push, open a pull request, and watch the workflows actually run — the one
   thing this report cannot substitute for.
