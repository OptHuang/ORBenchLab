# Self-hosted runner contract

This document answers one narrow question: what has to be true before a
self-hosted GitHub Actions runner is allowed to execute an ORBenchLab job?
Installing Codex is only one optional line in that contract.

## The two routes

| Route | Where the credential lives | Intended use | Pinned ORAgentBench status |
| --- | --- | --- | --- |
| `api-key` | Protected GitHub Environment supplies a short-lived provider key to the approved job | Internet-enabled benchmark rollout | **Usable.** This is the formal rollout route for the current dataset |
| `codex-login` | A separate private-controller analysis runner is signed in to ChatGPT Codex | Authentication/safety diagnosis now; future broker-backed rollout or host-side analysis | **Blocked.** Direct benchmark execution is disabled and the pinned 107/107 tasks enable public Internet |

The routes never fall back into one another. An API-key login consumes API
billing even if the same account also has a ChatGPT plan. A ChatGPT login uses
the plan's included Codex allowance and plan limits; it is not free unlimited
API capacity.

## Recommended topology

Do not attach a model-capable self-hosted runner directly to this public
repository. Its workflow enforces `validate-only`; self-hosted modes require a
private repository. The supported layout is:

1. Keep ORBenchLab public and reviewable.
2. Create a private mirror/controller repository containing the complete
   ORBenchLab tree at a full, reviewed commit SHA. The current workflow installs
   and invokes files from its own checkout; a workflow-only controller is not
   sufficient.
3. Protect that default branch, and update the mirror only by explicitly
   reviewing and importing a pinned public commit. Do not accept a pull-request
   ref, branch name or user-supplied repository URL at dispatch time.
4. Put model jobs behind a protected GitHub Environment with required
   reviewers. Keep the API key, when used, in that environment rather than in
   the public repository.
5. Prefer an ephemeral runner for each model job. If the runner is persistent,
   dedicate it to this controller and disable all pull-request triggers.

A public workflow file, dependency or composite action is executable code on a
self-hosted machine. The private controller reduces the set of people and refs
that can change that code; it does not replace code review or least privilege.

## Host contract

The execution host needs all of the following:

* a dedicated Unix service user for the API-key benchmark runner;
* Python 3.12 and this package's virtual environment;
* Docker, with the service user able to run the benchmark containers;
* Harbor and `uv` visible to the non-interactive runner shell;
* a private, persistent `ORBENCH_RUNS_ROOT` outside the checkout and
  `RUNNER_TEMP`, owned by that user, mode `0700`, with at least 20 GiB free;
* the `controls` and `benchmark-agent` protected GitHub Environments;
* the labels `self-hosted, orbench-exec`;
* for future `codex-login` analysis only, a pinned Codex CLI on a **separate**
  private-controller runner identity, preferably a disposable VM.

Docker access is effectively host-root access. Do not share the benchmark
account or Docker daemon with unrelated jobs, and do not place a personal
ChatGPT login in that account: every workflow step executed as the account can
read it. Within each runner pool, its process and runner-local state must belong
to that pool's service user; logging in from an administrator's interactive
account does not configure the service.

One possible persistent root is:

```bash
sudo install -d -o orbench -g orbench -m 0700 /srv/orbench/runs
```

Set the controller repository variable `ORBENCH_RUNS_ROOT=/srv/orbench/runs`.
Do not use a temporary directory or a symlink.

## Route A: short-lived API key

Use this for the current ORAgentBench rollout.

1. Create a revocable provider key with a server-side spend and rate limit.
2. Store it as `MODEL_API_KEY` in the protected `benchmark-agent`
   Environment. Store `MODEL_BASE_URL` as a non-secret **repository variable**:
   the GitHub-hosted preflight needs it before any protected environment is
   released.
3. Require a reviewer before the environment is released.
4. Dispatch only a pinned model id and exact scaffold version.
5. Revoke or rotate the key after the campaign.

The following checks do not call a model. Run them in the same non-interactive
shell and as the same Unix user as Actions:

```bash
python3.12 --version
docker version
docker info
harbor --version
uv --version
orbench doctor oragentbench \
  --source /srv/orbench/upstream/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model provider/model-version \
  --scaffold-version 0.138.0 --auth-mode api-key
```

`doctor` verifies configuration and preconditions; it does not perform a model
completion. A successful doctor is not evidence that a rollout ran.

## Route B: runner-local ChatGPT Codex login

The current implementation is an authentication and safety-diagnosis entry
point. It is not a way to run the current Internet-enabled ORAgentBench tasks
without an API key, and it does not make a no-network Harbor container capable
of reaching the model service.

On the isolated private-controller analysis runner (not the Docker benchmark
worker), complete the normal interactive login once as its service user:

```bash
codex login
```

The zero-model acceptance check is:

```bash
codex --version
codex login status
```

The status must explicitly report a ChatGPT login. If it reports an API-key
login, this route is not configured and calls would still use API billing. Do
not inspect, copy, print or upload the CLI credential store; the workflow needs
only the status result.

For the pinned ORAgentBench checkout, the following doctor command is expected
to fail before model execution because the selected task enables public
Internet:

```bash
orbench doctor oragentbench \
  --source /srv/orbench/upstream/ORAgentBench \
  --task additive_microfactory_order_planning \
  --agent codex --model gpt-5.6-sol \
  --scaffold-version 0.138.0 --auth-mode codex-login
```

That refusal is the acceptance result for the current dataset: it proves the
credential boundary fails closed. The doctor still makes no model call. A
future rollout needs either host-side execution or a broker/model proxy with
explicitly allowlisted model egress and no personal login material inside the
task. A `no-network` declaration by itself is not an executable solution.

## Cost and evidence semantics

For `api-key`, provider billing is authoritative; `--max-cost-usd` is only an
ORBenchLab audit envelope unless the provider also enforces the cap.

For `codex-login`, ChatGPT plan usage is authoritative. Harbor may calculate a
dollar value from tokens and API prices so runs can be compared uniformly. The
report must label that value **API-equivalent estimate**. It is not the amount
charged to the ChatGPT account and must not be presented as actual spend.

## No-model acceptance checklist

* [ ] The workflow runs only from the private controller's reviewed default
      branch and protected environment.
* [ ] The private mirror contains the complete ORBenchLab tree at a full pinned
      public commit SHA.
* [ ] Python 3.12, Docker, Harbor and `uv` are visible to the runner service
      user.
* [ ] `ORBENCH_RUNS_ROOT` is persistent, private, owned by that user and has
      sufficient free space.
* [ ] `orbench doctor ... --auth-mode api-key` performs configuration checks
      without a model call.
* [ ] For `codex-login`, a separate private-controller runner identity reports
      ChatGPT; the Docker benchmark worker has no personal login material.
* [ ] For the pinned ORAgentBench dataset, `codex-login` doctor is rejected as
      designed; nobody claims a subscription-backed rollout succeeded.
* [ ] No documentation or report claims that a no-network task can call a model
      without a broker or host-side execution path.
* [ ] No runner-local login material, raw trajectories or verifier workspaces
      are uploaded as Actions artifacts.

See also the official [Codex authentication](https://learn.chatgpt.com/docs/auth),
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) and
[pricing](https://learn.chatgpt.com/docs/pricing) documentation.
