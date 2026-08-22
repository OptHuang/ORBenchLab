# Contributing

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

The suite is offline and takes seconds. Tests that need a real upstream checkout
are opt-in:

```bash
git clone https://github.com/ORAgentBench/ORAgentBench /tmp/oragentbench
git clone https://github.com/Minw913/FrontierOR       /tmp/frontieror
ORBENCH_ORAGENTBENCH_SOURCE=/tmp/oragentbench \
ORBENCH_FRONTIEROR_SOURCE=/tmp/frontieror \
  python -m pytest -q
```

## Four rules

These are enforced by tests, not by review alone. If you find yourself working
around one, the design is probably wrong rather than the rule.

**1. Do not vendor upstream code.** No task packages, no verifiers, no checkers,
no scoring formulas copied into this repository. An integration reads an
upstream checkout and records what it found. A copy drifts, and a drifted copy
is a different benchmark wearing the same name.
*Enforced by* `tests/test_integrations.py::test_this_repository_vendors_no_upstream_benchmark_code`.

**2. Do not reimplement execution.** Scheduling, retries, resume, re-grading,
sandboxing and verifier execution belong to the upstream runner. ORBenchLab
compiles inputs, builds the command upstream documents for itself, and reads
outputs. A second implementation of an execution concern becomes a second source
of truth about what ran.

Building an upstream command line is delegation, not reimplementation — but keep
it that way. If you add to `orbenchlab.execution`, it should be validation,
argv construction or a receipt; never a loop that decides what to run next.
*Enforced by* `tests/test_no_reimplementation.py` and `tests/test_execution.py`.

**3. Do not let a claim outrun its evidence.** A number is only as strong as
`min(evidence grade, repetition class)`. Single-rollout output supports case
diagnosis, not ranking. If you add a metric, give it an honest grade and a real
repetition requirement, and return `None` with a reason when the requirement is
unmet.
*Enforced by* `tests/test_report.py`.

**4. Fail closed.** A missing secret, licence, runner or upstream source stops
the work. Never substitute a default, a mock or a zero that reads like a
measurement. Placeholders must exit non-zero and say what is missing.

## Adding an integration

See `docs/adding-a-benchmark.md` and start from
`templates/integration_template/`. The short version: write `describe()` and
`inspect()`, register the module, and add the integration to
`.github/workflows/integration-contract.yml`.

## Changing the report

The golden files in `tests/golden/` are the contract. After an intentional
change:

```bash
python tools/regenerate_fixtures.py
git diff -- fixtures tests/golden      # read this diff before committing
```

If the diff contains something you did not intend, that is the test working.

## Changing an upstream command

The exact argv is asserted in `tests/test_execution.py`. If upstream changes how
it runs things, update the builder and the test together, and put the new
provenance in `UpstreamCommand.provenance` — a file and a line a reviewer can
check, not a recollection.

Never add a flag that weakens an upstream security profile. FrontierOR's
trusted-agent profile is fixed by upstream and appended by it; every flag in it
is on a refusal list here, and so is anything outside upstream's documented
interface.

## Changing a workflow

Every action must be pinned to a full 40-character commit sha with a version
comment:

```yaml
- uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
```

Resolve one with:

```bash
git ls-remote https://github.com/actions/checkout 'refs/tags/v5.0.0^{}'
```

`tests/test_workflows.py` checks the pins, the permission scopes, the timeouts,
the secret handling and the fail-closed guards. Note what it cannot check:
whether the workflow succeeds when GitHub runs it. Nobody has observed a hosted
run of these workflows — do not describe them as passing on the strength of the
local suite.

## Style

Match the surrounding code. A few conventions worth stating:

* Comments explain *why*, especially where a rule exists to prevent a specific
  failure. The rules above are load-bearing and their rationale should survive
  in the code.
* Errors name the fix. `SpecError` lists every problem at once so one run of
  `orbench campaign validate` is enough to fix a spec.
* No wall-clock content in deterministic output. Plans, ledgers and reports must
  be byte-stable, which is what makes determinism testable at all.

## Commits and pull requests

Explain what changed and why. If a change relaxes a guard, say which one and
what replaces it. If a change alters what a report may claim, show the golden
diff in the pull request body.
