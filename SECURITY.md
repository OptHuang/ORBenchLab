# Security notes

## Reporting a vulnerability

Open a [security advisory](https://github.com/OptHuang/ORBenchLab/security/advisories/new)
rather than a public issue. Include what you did, what happened and what you
expected. We will acknowledge and give an assessment timeline.

If the issue is in ORAgentBench, FrontierOR or Harbor rather than in
ORBenchLab, please report it to that project — ORBenchLab does not own their
code and cannot fix it.

## Threat model

ORBenchLab is a control plane. It compiles configuration, renders reports, and —
through one checked script — hands a validated command to an upstream benchmark's
own execution path. It does not implement execution, does not run untrusted code
itself, and holds no long-lived credentials. The security-relevant surface is
therefore narrow, and the rules below exist to keep it that way.

### Handing work to an upstream harness

`tools/run_benchmark_smoke.py` is the only thing here that starts an upstream
process, and it is deliberately narrow:

* **Inputs are resolved against the checkout, not trusted.** An ORAgentBench task
  must exist as a directory with a `task.toml`; a FrontierOR paper id must appear
  in that checkout's `paper_meta_info.json`; instance names must match upstream's
  own rule. Path separators and traversal are rejected outright.
* **Upstream's security profile cannot be weakened from here.** FrontierOR fixes
  its trusted-agent profile — coral framework, Docker isolation, `staged_qte`
  scoring, proxied model access, anti-hack — and appends those flags itself. The
  builder refuses every one of them as an input, and refuses anything outside
  upstream's documented agent interface, so a workflow input cannot smuggle an
  argument past it.
* **A leaky evaluation split is refused before the process starts**, because a
  test set the agent has already seen produces a number that means nothing.
* **Dry run is the default.** Execution requires `--execute`, and a paid path
  additionally requires an explicit cost acknowledgement.
* **Credentials never reach the argv.** They travel as `${NAME}` placeholders in
  the job config, or as process environment the script never reads.

### Credentials

* **Values never enter the repository or its artefacts.** A campaign spec
  declares environment variable *names* (`env_keys`); the compiler emits
  `${NAME}` placeholders into job configs and the runner supplies the value.
  Validation rejects an `env_keys` entry containing `=`.
* **Values never enter an identifier.** `agent_uid` hashes variable names only,
  so rotating a credential leaves every run id unchanged while *renaming* one
  changes the agent's identity — which is correct, because that is a real
  configuration change.
* **Inspection reads no credentials.** `orbench integration inspect` runs with a
  minimal environment and reports `reads_credentials: false`, which CI asserts.
* Never paste a key into an issue, a spec, a fixture or a test.

### CI

* `ci.yml`, `integration-contract.yml` and `report.yml` reference no repository
  secrets at all — a test enforces this.
* `benchmark-smoke.yml` is `workflow_dispatch` only. It has no `pull_request`
  trigger and does not use `pull_request_target`, so fork code cannot cause it
  to run.
* Its preflight refuses to proceed on a fork, and paid modes require both an
  explicit acknowledgement input and a GitHub Environment that can carry
  required reviewers.
* Self-hosted runners are used by that workflow only. Untrusted fork code never
  reaches them.
* Every third-party action is pinned to a full commit sha. A tag can be
  repointed; a sha cannot.
* Workflow permissions are `contents: read`; no workflow requests write scope.

### Artefacts

Raw evidence bundles — agent trajectories, verifier stdout, hidden reference
data, container images — must never be uploaded as CI artefacts. They are large,
and some of them are the answers.

Run receipts are sanitized before they leave the execution host: fields whose
*name* suggests a credential are redacted, inline `NAME=value` assignments are
rewritten, environment variables are reduced to `<set>`/`<unset>`, and a receipt
naming any raw-evidence artefact is refused outright. The agent jobs upload the
receipt and nothing else.

Reports are built from a *normalized slice*: run ids, attributions and numeric
reward keys. `report.yml` scans its input for raw-bundle markers and oversized
files and fails the run rather than publishing them, so the guard applies to
uploaded bytes and not only to good intentions.

### Hidden verifiers and reference data

A benchmark's grading data is only meaningful while it stays hidden.

* ORAgentBench keeps reference metrics in each task's `tests/` directory, which
  the runner injects after the agent has finished. ORBenchLab does not read,
  copy or republish it.
* FrontierOR marks reference objectives, reference runtimes, the checker
  implementation and final-instance membership as trusted-only, and does not
  distribute them. ORBenchLab wraps the official harness precisely so that this
  data stays on the trusted side. Its inspector records digests and paths, never
  contents.

If you are adding an integration, treat every upstream artefact marked hidden,
private or trusted-only as untouchable — including in fixtures and test data.

### Solver licences

Licences are granted per phase, never to both phases for convenience. A task
declares whether the agent, the verifier, both or neither may use the solver,
and the compiler injects the credential only into the phase that declared it. An
audit finding is "a licence appeared in a phase that did not declare it" — not
merely "a licence appeared on the agent side", because some tasks legitimately
require the agent to call a solver.

## Known limitations

Stated so nobody relies on a protection that does not exist:

* **Cost cannot be interrupted mid-run.** The budget field is a compile-time
  gate. Job plugins expose start and end hooks and are not called during a job,
  so nothing here is a circuit breaker. The only real spend limit is a
  provider-side cap on the key itself. Set one.
* **Durability is not verified.** `validated` reports require verified
  content-addressed replicas of the underlying evidence; that verification is
  unimplemented, so validation refuses `validated` campaigns rather than
  accepting the label on trust.
* **`perf_isolated` is an assertion, not a measurement.** A site file claiming
  performance isolation is taken at face value. Only set it for a machine whose
  cores you have actually pinned.
* **No hosted CI run has been observed.** The workflows parse and their guards
  are unit-tested, and their steps have been executed locally; that is not the
  same as having watched them run on GitHub.
* **No paid agent run has been observed.** The agent paths are implemented and
  their commands are asserted by tests, and upstream's own dry run has accepted
  the compiled configuration — but no model call has been made from here.
