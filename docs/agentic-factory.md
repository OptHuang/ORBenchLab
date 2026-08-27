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
- explicit evidence levels and no promotion from agent opinion.

## Default DAG

`orbench agent-factory prepare-paper` compiles an 18-session plan:

1. primary paper derivation and independent evidence criticism;
2. two independent task designs and autonomous selection;
3. strict task implementation;
4. independent scientific and verifier reviews;
5. task repair;
6. Harbor Oracle/NOP controls;
7. equal-budget frontier and weak-model pilots;
8. verifier-grounded trajectory diagnosis;
9. same-checkpoint intervention study when real checkpoints exist;
10. difficulty-lattice design and variant authoring;
11. repeated multi-model calibration;
12. final human review summary.

Each semantic node is a real Codex or Claude Code CLI session. Independent
nodes may use different models. A stage may retry under a new attempt identity,
but it may not overwrite a historical attempt receipt.

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
Harbor or the repeated model campaign. A higher-level unattended controller
must still execute those commands and pass their exact output paths here.

## Current security boundary

The CLI runner restricts environment variables, customizations, MCP servers,
time and captured output. Its working directory is an execution and evidence
boundary, not an operating-system filesystem or network sandbox. Claude's Bash
tool and Codex inherit the host account's filesystem permissions. Production
unattended workers should therefore run in a disposable VM/container or a
dedicated low-privilege account with read-only inputs and an explicit egress
policy. Separate factory roots can run concurrently; one factory state chain is
serialized to prevent receipt races.
