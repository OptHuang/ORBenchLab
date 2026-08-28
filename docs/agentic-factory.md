# Autonomous paper-to-benchmark factory

ORBenchLab treats Codex and Claude Code as autonomous semantic workers. The
Python harness does not try to understand a paper, invent a verifier, diagnose
a trajectory, or choose a hint. It assigns those jobs to independent agent
sessions and enforces the parts that should not depend on model judgment:

- an immutable, checksummed stage DAG;
- a paper, deterministic page-marked text extraction, provenance and seed-task workspace binding;
- fixed agent profiles, provider route, model and prompt identity;
- whole-process time and output limits;
- bounded `stdout.live` / `stderr.live` traces while each session is running;
- one atomic receipt per attempt and safe successful-session reuse;
- required output paths and content digests;
- crash recovery, factory-level locking and fail-closed quarantine;
- OS-enforced read-only factory inputs (Bubblewrap on Linux, sandbox-exec on macOS);
- explicit evidence levels and no promotion from agent opinion.

## Default DAG

`orbench agent-factory prepare-paper` compiles a 19-session plan:

1. raw paper derivation, schema normalization and independent evidence criticism;
2. two independent task designs and autonomous selection;
3. strict task implementation;
4. independent scientific and verifier reviews;

Independence is enforced by the session filesystem view, not by prompt wording.
Each semantic stage can read `factory-input/` plus outputs owned by its transitive
dependency ancestors. Already-completed outputs from non-ancestor stages are
still protected from writes and are additionally masked from reads (Bubblewrap
mount masks on Linux, sandbox-exec read denials on macOS). The visible and hidden
path digests, together with the completed-stage snapshot, are persisted in the
session and attempt receipts so resume replays the same DAG visibility boundary.
5. task repair;
6. semantic indexing of trusted Harbor Oracle/NOP controls;
7. semantic indexing of a trusted equal-budget frontier/weak Harbor model matrix;
8. verifier-grounded trajectory diagnosis;
9. same-checkpoint intervention study when real checkpoints exist;
10. difficulty-lattice design and variant authoring;
11. semantic recomputation of a trusted repeated multi-model calibration receipt;
12. final human review summary.

Each semantic node is a real Claude Code CLI session. Independent nodes may use
different models. Codex plans can be compiled for inspection, but unattended
factory execution and promotion fail closed because the current Codex CLI does
not expose a hard provider-spend flag. A stage may retry under a new attempt
identity, but it may not overwrite a historical attempt receipt.

### Deterministic in-loop gates (repair loop)

Task-producing stages declare harness-owned postchecks that run immediately
after their outputs validate:

- `tb-science-static-gate` reruns the complete deterministic TB-Science
  authoring gate over the stage's task tree (`task-author-v1`,
  `task-repair-v2`);
- `variant-conformance` additionally validates the variant manifest and
  rejects rename-only lattices: every variant must pass the static gate and no
  variant may be byte-identical to the base task or to another variant
  (`variant-author`).

A failed postcheck fails the attempt with `deterministic_gate_failed`, keeps
the outputs in place, and writes the exact findings to
`factory/gate/<stage>-postcheck.json`; the next bounded attempt reads them and
repairs in place. Exhausting `max_attempts` quarantines the run with the gate
findings attached — the repair loop is the existing attempt machinery, not a
separate mutable workflow. Stage output directories follow a slug convention:
the strict task lives at `factory/tasks/task-v2/<task-slug>` where the slug
equals the `task.toml` name basename, because the gate requires the task
directory name to match its slug.

The autopilot repeats the static gate immediately before every Harbor barrier
(baseline task and each difficulty variant); a blocked tree quarantines the
run with machine-readable `static-gate-blocked` before any model spend.

Runtime nodes do not ask a semantic agent to assert that it ran Harbor. The
trusted harness must first write control/model/difficulty receipts under the
read-only `factory-input/trusted/` boundary; the following agent session may
analyze them, but cannot create the evidence that unlocks a gate.

## Evidence boundary

A fully traversed agent DAG ends as `semantic-complete-e1`. This means the
agents ran and their declared files are present, hash-bound and unchanged. It
does **not** mean the task passed static checks, a verifier, Harbor, a causal E4
intervention or the final promotion policy.

The trusted path remains:

```text
agent output (E1)
  -> deterministic task-authoring gate
  -> task verifier / Harbor Oracle and NOP (E3)
  -> repeated equal-budget model rollouts
  -> optional real same-checkpoint interventions (E4)
  -> conservative task-card promotion
```

Restarting a model with a hint is not a same-checkpoint intervention. If the
runtime cannot resume exact state, the intervention stage must record E4 as
unavailable.

The session runner exposes live trace bytes for monitoring and seals them as
digest-bound logs on completion. The plain factory runner reports
`hint_injection_supported: false`, and Harbor trials remain independent full
restarts — a monitor that merely starts a fresh CLI process with a hint stays
E3.

A separate, capability-gated intervention channel now exists for agent
sessions: `claude --print --input-format stream-json` accepts additional user
messages on the open stdin of the same running session.
`orbench agent-session capability` reports the machine-readable capability per
profile/runtime (Codex and Harbor trials are `unsupported`, fail-closed);
`orbench agent-session intervene` runs one monitored session that fires a
fixed predeclared policy hint into the same session, recording event arrival
times, the injection instant, pre/post-checkpoint event counts, and whether
the runtime *confirmed* a post-hint event; and `orbench intervention-study`
pairs treatment sessions with identically instrumented no-injection controls,
grades every trial with the caller's verifier, and emits
`E4-controlled-same-session-intervention` only when the capability is
supported, every treatment injection was confirmed, and both arms have at
least three verifier-graded trials. Anything less is labelled
`E3-underpowered` or `E1-incomplete`; an unsupported profile receives an `E0`
receipt instead of a silently downgraded run.

## Commands

Prepare a bound workspace and plan:

```bash
orbench agent-factory prepare-paper \
  --paper-file paper.pdf \
  --paper-provenance paper-provenance.json \
  --seed-task examples/tasks/alphaevolve-scheduling \
  --workdir out/factory-workspace \
  --plan-out out/factory-plan.json \
  --author-model ark-code-latest \
  --reviewer-model reviewer-a \
  --reviewer-model reviewer-b \
  --frontier-model frontier-model \
  --weak-model weak-model
```

Run or resume it:

```bash
orbench agent-factory run \
  --plan out/factory-plan.json \
  --workdir out/factory-workspace \
  --out out/factory-run
```

For the unattended semantic/runtime loop, use `autopilot`. It advances one
agent stage at a time, pauses at `runtime-controls` and `calibration`, launches
real Harbor Oracle/NOP and repeated Claude Code trials, installs only validated
receipts and sanitized ATIF bundles under `factory-input/trusted/`, then resumes
the DAG:

```bash
orbench agent-factory autopilot \
  --plan out/factory-plan.json \
  --workdir out/factory-workspace \
  --factory-out out/factory-run \
  --out out/autopilot \
  --harbor-executable /absolute/path/to/harbor \
  --claude-executable /absolute/path/to/claude \
  --frontier-model doubao-seed-2.0-pro \
  --weak-model doubao-seed-2.0-lite \
  --repetitions 5 \
  --max-budget-usd 0.5 \
  --max-variants 6 \
  --max-job-attempts 2 \
  --max-harbor-liability-usd 100
```

The autopilot no longer stops at `semantic-complete-e1`. After the semantic
DAG and both trusted barriers finish, a deterministic promotion phase runs
unattended: static gate over the selected task, digest-matched reuse of the
Harbor control/calibration receipts the barriers already produced (a selected
variant reuses its own validated matrix; a missing binding blocks with
`promotion_evidence_missing` instead of relaunching paid jobs), the
independent two-model Volc semantic review, the deterministic pipeline task
card, the fail-closed finalizer, and one operator-facing
`promotion/final-report.md` binding every receipt (task purpose, provenance,
per-gate outcomes, frontier/weak Wilson intervals, per-variant difficulty
calibration, graded trajectory/intervention evidence, observed costs, open
items and reproduction commands). Terminal states are `promoted` or
`promotion-blocked`, both resumable; `--stop-after-semantic` restores the old
boundary. The real Harbor model-matrix screening report is first-class
calibration evidence for `finalize`.

The state file is content-bound and resumable. The worst-case Harbor model
liability includes the baseline plus every allowed variant and every crash-safe
job attempt before the first trial starts. Each whole model job is atomically
charged before Harbor launches, so a crashed job cannot restore budget. Variant
promotion requires at least three ordered levels, clean
Oracle/NOP controls, a complete equal-budget frontier/weak rectangle with at
least five repetitions per cell, recomputed Wilson intervals, monotonicity and
a conservative separation level. Exploratory selection remains labelled
exploratory; `--held-out` is accepted only when the manifest predeclares and
digest-binds selection evidence.

`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` or the corresponding OpenAI
profile variables are read only at the session boundary. Receipts contain a
route digest and never the credential.

`prepare-paper` runs bounded `pdftotext` once and binds the extractor binary,
version, arguments, page count and output digest into the workspace manifest.
Agents reason over the read-only `paper.txt` snapshot and use the original PDF
only for difficult anchor checks. This avoids paying every session to rediscover
the same PDF text.

After independent static, Harbor and repeated-model commands have produced
their receipts, run the fail-closed finalizer:

```bash
orbench agent-factory finalize \
  --plan out/factory-plan.json \
  --factory-run out/factory-run/factory-run.json \
  --workdir out/factory-workspace \
  --task-dir out/factory-workspace/factory/tasks/task-v2 \
  --static-receipt out/static/authoring-receipt.json \
  --harbor-receipt out/harbor/harbor-control-screening.json \
  --calibration-receipt out/calibration/screening-report.json \
  --final-summary out/cards/task-cards.json \
  --out out/final
```

The finalizer recomputes calibration arms and conservative model separation
from raw per-trial outcomes. It also validates the deterministic task card and
binds it to the exact Harbor and calibration artifact bytes. Agent-written
“all passed” summaries cannot unlock promotion. The strongest result is
`eligible-for-human-release-review` at E3, not publication or TB-Science
acceptance.

The finalizer consumes independent receipts; it does not itself launch Docker,
Harbor or the repeated model campaign. `agent-factory autopilot` now owns the
semantic/runtime loop and produces its final agent review packet plus trusted
E3 runtime receipts. Formal release promotion remains a separate deterministic
`agent-factory supervise`/`finalize` step, so semantic completion cannot silently
publish a task.

For a real repeated Harbor coding-agent matrix with verifier outcomes and ATIF
trajectories, use the bounded launcher. The Claude executable is mounted
read-only into each task container, avoiding an untrusted runtime download:

```bash
orbench harbor-model-matrix \
  --task-dir out/factory-workspace/factory/tasks/task-v2 \
  --harbor-executable /absolute/path/to/harbor \
  --claude-executable /absolute/path/to/claude \
  --model doubao-seed-2.0-pro \
  --model doubao-seed-2.0-lite \
  --repetitions 5 \
  --max-budget-usd 1 \
  --max-turns 40 \
  --out out/harbor-model-matrix
```

The receipt requires every trial to have a complete verifier result and a
non-empty ATIF trace. Agent budget exhaustion is recorded as a model outcome
when the verifier still completed; setup/network failures without verifier
evidence are rejected. This remains E3 because every arm is a fresh restart.

## Batch orchestration

`orbench agent-factory batch` consumes a candidate queue from a pluggable
provider (`explicit-list`, or `paper-binding-dir` over the output of
`orbench intake bind-paper`), screens each candidate deterministically
(provenance binding digest against the actual paper bytes, strict seed task —
no model calls), refuses to start when the admitted set's worst-case provider
liability exceeds `--max-total-liability-usd`, and drives one isolated
autopilot per candidate (own workspace, budgets, receipts, resume). A failed
or quarantined candidate never blocks the rest, and re-running the same batch
resumes every non-terminal candidate; daily scheduling belongs to external
automation invoking this one idempotent CLI.

Difficulty genomes are machine-readable end to end: the variant manifest may
declare bounded `secondary_axes`, every per-variant `axis_levels` key must be
a declared axis, and the trusted difficulty receipt embeds a
`difficulty_genome` (axes plus per-variant levels and frozen task-tree
digests) with pairwise-distinct variant trees that also differ from the base
task.

## Current security boundary

The CLI runner restricts environment variables, customizations, MCP servers,
time and captured output. Paper-factory sessions disable Bash entirely; agents
use Read/Glob/Grep/Edit/Write while deterministic and Harbor gates own command
execution. This prevents a paper-prompt-injected shell from reading or
exfiltrating the provider credential. Factory sessions additionally require an inherited
OS policy that makes the complete `factory-input/` tree immutable: Bubblewrap
on Linux mounts the host root read-only, exposes only the workspace as writable,
then remounts trusted inputs and completed outputs read-only. macOS uses a
default-deny-write sandbox with equivalent workspace/protected-path rules. A
platform without one of these mechanisms fails closed. This protects trusted
receipts from the agent, but it is not a complete host or network sandbox:
Claude's Bash tool can still access other paths allowed to the host account.
Production unattended workers should therefore still run in a disposable
VM/container or dedicated low-privilege account with an explicit egress policy.
Separate factory roots can run concurrently; one factory state chain is
serialized to prevent receipt races.
