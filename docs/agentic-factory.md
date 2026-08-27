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
digest-bound logs on completion. It deliberately reports
`hint_injection_supported: false`: true E4 injection still requires a runtime
that can pause and continue the same checkpoint. A monitor that merely starts
a fresh CLI process with a hint must remain E3.

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
