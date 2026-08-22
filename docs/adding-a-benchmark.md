# Adding a benchmark

A third integration is roughly a day's work if the upstream benchmark is
Harbor-native or ships its own harness, and considerably more if you have to
write a true adapter. Most of that day is deciding *which* of those it is —
so start there.

## Step 1 — decide the form

Answer these against the upstream repository, not against its README's
aspirations:

**Does upstream already ship task packages the runner consumes directly?**
→ `harbor-native`. Consume them in place. Write no adapter. (ORAgentBench.)

**Does upstream own a runner and a grader of its own?**
→ `official-external-harness`, unless you can satisfy every condition in the
conversion test below. (FrontierOR.)

**Does upstream ship data but no runner?**
→ `harbor-adapter`, and only once you can produce the parity fixture.

### The conversion test

You may convert to Harbor task packages only when **all four** hold:

1. Everything needed to grade a submission can legitimately live inside a task
   package. If reference values or a checker are undistributed, stop.
2. The score does not depend on host-level properties a container cannot
   reproduce. Wall-clock timing against a reference runtime does depend on them.
3. The isolation the benchmark relies on can be expressed in the runner's own
   declarations rather than re-implemented.
4. You can produce a **differential re-scoring fixture**: frozen submissions —
   one violating each hard-constraint family, plus both sides of any tolerance
   boundary — where the upstream checker and your converted verifier agree key
   for key.

If any fails, wrap the official harness and record why in `docs/decisions/`. That
is not a lesser outcome. A converted task that reports a differently-defined
number under the same name is worse than no conversion at all, because nobody
notices until a decision has already been made on it.

## Step 2 — write the module

```bash
cp templates/integration_template/integration.py src/orbenchlab/integrations/yourbench.py
```

Fill in:

| Item | Notes |
| --- | --- |
| `NAME`, `KIND` | Registry key and form |
| `UPSTREAM_REPO`, `PINNED_COMMIT` | The exact commit you inspected. CI clones it |
| `describe()` | Secrets, licences, runner labels, `performance_scored` |
| `inspect()` | Checks, facts, decisions |
| `FORBIDDEN_VENDORED_NAMES` | Upstream artefacts that must never appear here |

### What `inspect()` must produce

**`facts["dataset_digest"]`** — content-addressed over the upstream data, so run
ids are pinned to a dataset state. Without it, a silent upstream data change
contaminates an existing campaign instead of producing new ids.

**`facts["preconditions"]`** — every secret, licence, dataset, image and runner
capability, each with `fail_closed: true`. The smoke workflow gates on these.

**`decisions`** — at minimum `integration_form`, `adapter_required` and a
`rationale`. For an external-harness integration also
`harbor_tasks_materialized` and `harbor_conversion_parity_safe` with the
blockers behind it.

**Checks that fail closed.** `pass` means you looked and it was fine. `warn`
means you looked and found something that limits what campaigns may claim.
`skip` means you could not look — always with a note on how it *could* be
verified. `fail` means the checkout is unusable. Never report `pass` for
something you did not verify; the whole report is worthless if one check lies.

### Recording upstream problems

Report them; do not fix them locally. If upstream leaves the agent phase
network-enabled, or does not declare verifier isolation, that is a `warn` with
an `impact` note — and it constrains what campaigns may claim. Patching it here
would fork the benchmark, which is the failure mode this whole layer exists to
prevent. Send a pull request upstream instead.

## Step 3 — register it

```python
# src/orbenchlab/integrations/registry.py
from . import frontieror, oragentbench, yourbench

_MODULES = (oragentbench, frontieror, yourbench)
```

## Step 4 — test it offline

Build a miniature stand-in so the suite needs no network:

```
tests/fixtures/upstream/yourbench_min/
├── README.md          # say plainly that this is a stand-in, not a copy
└── ...                # the minimum shape inspect() reads
```

A stand-in reproduces *shape*, not content. Never copy real upstream verifier,
checker or reference data into it — that is vendoring with extra steps.

Then add tests covering: the happy path, an empty directory (must fail closed),
the decisions, and the preconditions. Model them on
`tests/test_integrations.py`.

## Step 5 — wire it into CI

In `.github/workflows/integration-contract.yml`, add to the matrix:

```yaml
- integration: yourbench
  repo: https://github.com/example/yourbench
```

and add its expected decisions to the assertion step, so a change of form fails
CI until somebody updates the assertion deliberately.

## Step 6 — add a campaign and a site

A campaign spec (`campaigns/yourbench-*.yaml`) demonstrating the zero-cost path.
If the integration is performance-scored, also confirm that validation *refuses*
a non-isolated site — and consider shipping that refusal as an example, the way
`campaigns/frontieror-contract-check.yaml` does. A guard nobody has seen fire is
a guard nobody trusts.

## Step 7 — decision record

If the form is anything other than the obvious one, write
`docs/decisions/000N-yourbench-integration.md`: the decision, the evidence,
the options rejected and why, the consequences including the costs, and the
conditions under which the decision should be revisited. `0001` is the worked
example.

## Checklist

* [ ] Form chosen against the conversion test, not by convenience
* [ ] Module written; every template TODO removed
* [ ] Registered in `registry.py`
* [ ] `dataset_digest` is content-addressed
* [ ] Preconditions declared and fail-closed
* [ ] Decisions recorded machine-readably
* [ ] Offline fixture tree and tests added
* [ ] Added to `integration-contract.yml`, including the decision assertion
* [ ] A campaign spec exercising the zero-cost path
* [ ] Decision record written if the form is non-obvious
* [ ] Nothing vendored — `python -m pytest tests/test_integrations.py -q` proves it
