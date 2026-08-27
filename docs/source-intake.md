# Daily OR source intake (metadata-only)

`orbench intake` is the first, deliberately conservative stage of the discovery
loop.  It collects public RSS/Atom, arXiv, and GitHub metadata, records response
digests, de-duplicates entries across feeds and days, and emits a queue for a
human reviewer.  It does **not** call a model, read credentials, copy source
documents into `raw/`, author a task, or publish anything.

## Configure feeds

Start from [`intake/or-feeds.example.yaml`](../intake/or-feeds.example.yaml):

```yaml
version: 1
feeds:
  - id: arxiv-math-oc
    kind: arxiv
    url: "https://export.arxiv.org/api/query?search_query=cat:math.OC&max_results=50"
    tags: [or, paper]
```

Only HTTPS URLs are accepted.  Userinfo, fragments, and credential-like query
parameters (`token`, `api_key`, and similar) are rejected.  The collector uses
the public endpoint with a bounded response size and a metadata-only user
agent.  It does not consult environment variables for authentication.

## Run a collection

```bash
orbench intake validate --config intake/or-feeds.yaml
orbench intake collect \
  --config intake/or-feeds.yaml \
  --out artifacts/source-intake/2026-08-27 \
  --previous artifacts/source-intake/2026-08-26
```

The output directory contains:

* `intake.json` — feed status, response digests, normalized items, a
  `review_queue_digest` binding to the adjacent JSONL, and the explicit
  zero-model/zero-credential policy;
* `review_queue.jsonl` — only `new` or `updated` items, each marked
  `state: pending` and carrying review dimensions (`or_relevance`, `novelty`,
  `task_potential`, `reproducibility`);
* `intake-manifest.json` — SHA-256 digests of the two artifacts (including the
  exact queue rows) and a statement that raw response bodies and task
  publication are disabled.

`--previous` may point to either a prior `intake.json` or its bundle directory.
An unchanged identity/content digest is labelled `duplicate`; a changed record
with the same identity is labelled `updated`.  Duplicate occurrences in the
same snapshot are merged and counted.  A failed feed remains visible in
`intake.json`; the command writes the partial bundle and exits with code 8 so a
daily runner can alert instead of silently treating it as complete.

Bundles are idempotent: writing a different payload over an existing artifact
fails closed.  Keep them under `artifacts/` (or another review workspace), not
under `raw/`.  Human review is the gate before any future task-genome proposal;
this prototype intentionally has no task-authoring or publishing command.

For offline tests, call `orbenchlab.source_intake.collect` with an injected
fetcher returning `FetchResponse`.  This exercises exactly the same parser and
digest path without network access.

## Daily scheduling (operator-controlled)

The collector is safe to schedule because it has no model or credential path,
but scheduling remains an explicit operator choice.  For example, a cron entry
can invoke a small wrapper that sets `RUN_DATE` and passes the previous bundle:

```cron
17 7 * * * cd /absolute/path/to/ORBenchLab && RUN_DATE="$(date -u +\%F)" .venv/bin/orbench intake collect --config intake/or-feeds.yaml --out "artifacts/source-intake/${RUN_DATE}" --previous "artifacts/source-intake/$(date -u -d 'yesterday' +\%F)" >> artifacts/source-intake/intake.log 2>&1
```

On macOS, replace the GNU `date -d 'yesterday'` expression with an explicit
wrapper or launchd job.  Exit code `8` means the snapshot was written but at
least one feed failed; alert on it and review the partial bundle instead of
promoting it automatically.
