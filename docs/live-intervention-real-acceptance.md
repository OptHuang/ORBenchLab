# P0-B live-intervention — real acceptance evidence (syu-ubuntu)

Harbor-native, same-session live intervention validated end-to-end on
`syu-ubuntu` with the real Volc API, real Docker, and the real Claude CLI
2.1.183. This is verifier-grounded E4 evidence, not a fixture.

## Stack under test
- Persistent Claude `stream-json` session on the host, model routed through the
  P0-A credential relay (real ARK token stays only in the host relay).
- Agent tools confined to the container MCP proxy (`harbor_container_proxy`);
  built-in Bash/Read/Write/Edit/Web disabled.
- Task runs in a Docker container started `--network none` with an empty env.
- Grading by the SEPARATE frozen verifier inside the same container.

## Run: `~/orbench-fable5-real/live-smoke-run4`
Trivial verifier-grounded task: write a word to `/work/answer.txt`; the verifier
rewards `HINTED`. Baseline is instructed to write `BASELINE`; the L1 hint
corrects it to `HINTED` mid-session via the interrupt/hint protocol.

| arm | interrupt sent/acked | interrupted boundary | hint sent/replayed | verifier `got` | reward |
|-----|----------------------|----------------------|--------------------|----------------|--------|
| baseline | no / no | — | no / no | `BASELINE` | 0.0 |
| L1 | yes / yes (event 17) | event 20 | yes / yes (step 63) | `HINTED` | 1.0 |

- `hint_recovered = [0.0, 1.0]` — the same-session interrupt+hint causally
  flipped the outcome.
- `quiescent_snapshot.no_in_flight_tool = true` — the hint was sent only after a
  quiescent interrupted boundary.
- `single_session = true` for both arms (distinct real Claude session ids).
- `secret_leaked_paths = []` — the provider credential appears in no journal,
  ATIF, artifact, container env, or `/proc`.
- ATIF (`orbenchlab.live-intervention.atif.v1`) preserves the interrupt-ack step
  and an independent `user-hint` step carrying the `intervention_id`.

## Container isolation (free smoke, real Docker)
`scripts/live_container_isolation_smoke.py` → all true: canary absent from the
container env and `/proc`, no network egress, MCP proxy runs inside the
container, verifier reward/CTRF read path works.

## Critical relay fix surfaced by this run
The first real Claude-CLI-through-relay call returned `API Error`: Claude posts
to `/api/coding/v1/messages?beta=true`, and the P0-A relay (only ever fake-
upstream tested) rejected any query string. The relay now allows the
known-safe `?beta=true` query. This affected every relay-transport session,
including the factory's default `credential_relay=True` path.

## Reproduce
```
python3 scripts/live_container_isolation_smoke.py            # free
python3 scripts/live_intervention_real_smoke.py <out_dir>    # ~2 paid arms
```

## Still pending
- Full repeated-causal study (≥5 baseline + ≥5 each L1/L2/L3) on a real authored
  task, and wiring the live study into autopilot promotion to replace the
  honest E0/E1 "not-run" capability receipt.
