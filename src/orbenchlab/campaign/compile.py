"""Campaign compiler: spec -> plan + plan ledger + Harbor job configs.

The compiler is a pure function of the spec. Same spec in, byte-identical
output — no clock, no randomness, no machine-dependent ordering. That is what
makes run ids shardable, resumable and de-duplicable without any coordination
service.

What it deliberately does not do: schedule, execute, retry, resume or grade.
Those are Harbor's, and reimplementing them here would create a second source of
truth for what actually ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..core import ids
from ..integrations import registry
from .spec import AgentSpec, CampaignSpec

PLAN_SCHEMA_VERSION = "1.0"
LEDGER_SCHEMA_VERSION = "1.0"

#: Label written onto job environments so cleanup can be scoped to one campaign.
#: A global docker prune on a shared runner destroys other campaigns' work.
CAMPAIGN_LABEL_KEY = "orbench.campaign"


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    task_name: str
    task_uid: str
    agent_id: str
    agent_uid: str
    budget_uid: str
    seed: int
    attempt: int
    shard: int
    job_name: str
    match_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "task_uid": self.task_uid,
            "agent_id": self.agent_id,
            "agent_uid": self.agent_uid,
            "budget_uid": self.budget_uid,
            "seed": self.seed,
            "attempt": self.attempt,
            "shard": self.shard,
            "job_name": self.job_name,
            "match_key": self.match_key,
        }


@dataclass(frozen=True)
class PlannedJob:
    job_name: str
    agent_id: str
    seed: int
    attempt: int
    shard: int
    task_names: tuple[str, ...]
    config_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "agent_id": self.agent_id,
            "seed": self.seed,
            "attempt": self.attempt,
            "shard": self.shard,
            "n_tasks": len(self.task_names),
            "task_names": list(self.task_names),
            "config_path": self.config_path,
        }


@dataclass(frozen=True)
class CompiledCampaign:
    campaign_id: str
    cfg_digest: str
    spec: CampaignSpec
    runs: tuple[PlannedRun, ...]
    jobs: tuple[PlannedJob, ...]
    job_configs: dict[str, dict[str, Any]]

    def plan_dict(self) -> dict[str, Any]:
        integration_desc = registry.describe(self.spec.integration)
        zero_cost_runs = sum(
            1
            for run in self.runs
            if _agent_by_id(self.spec, run.agent_id).is_zero_cost
        )
        return {
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_cfg_digest": f"sha256:{self.cfg_digest}",
            "date": self.spec.date,
            "integration": self.spec.integration,
            "integration_kind": integration_desc["kind"],
            "site": self.spec.site,
            "evidence_intent": self.spec.evidence_intent.value,
            "dataset": {"path": self.spec.dataset_path, "digest": self.spec.dataset_digest},
            "counts": {
                "tasks": len(self.spec.tasks),
                "agents": len(self.spec.agents),
                "seeds": len(self.spec.seeds),
                "attempts": self.spec.attempts,
                "shards": self.spec.shards,
                "runs": len(self.runs),
                "jobs": len(self.jobs),
            },
            "cost": {
                "zero_cost_runs": zero_cost_runs,
                "model_calling_runs": len(self.runs) - zero_cost_runs,
                "max_cost_usd": self.spec.budget.get("max_cost_usd"),
                "note": (
                    "compile-time gate only. Job plugins expose on_job_start/on_job_end "
                    "and are not called mid-job, so nothing here is a runtime circuit "
                    "breaker; the only real spend limit is the provider-side key cap."
                ),
            },
            "jobs": [job.to_dict() for job in self.jobs],
            "runs": [run.to_dict() for run in self.runs],
        }

    def ledger_dict(self) -> dict[str, Any]:
        """The plan ledger, written before anything executes.

        Harbor names trials randomly, so a trial cannot carry our run id. Ingest
        reconciles by ``match_key``; a trial matching no entry is an orphan and
        is recorded as such rather than being quietly adopted.
        """
        return {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_cfg_digest": f"sha256:{self.cfg_digest}",
            "written_before_execution": True,
            "match_key_definition": "sha256(canon([task_name, agent_uid, seed, attempt]))",
            "reconciliation_rules": {
                "multiple_ledger_hits": "integrity error — never guess",
                "no_ledger_hit": "orphan_trial — recorded, excluded from the capability denominator",
                "one_job_per": "(agent_uid, seed, attempt) with n_attempts = 1",
            },
            "entries": [
                {
                    "run_id": run.run_id,
                    "shard": f"{run.shard}/{self.spec.shards}",
                    "task_uid": run.task_uid,
                    "task_name": run.task_name,
                    "agent_uid": run.agent_uid,
                    "budget_uid": run.budget_uid,
                    "seed": run.seed,
                    "attempt": run.attempt,
                    "job_name": run.job_name,
                    "match_key": run.match_key,
                }
                for run in self.runs
            ],
        }


def compile_campaign(spec: CampaignSpec) -> CompiledCampaign:
    """Expand a spec into runs, jobs and Harbor job configs."""
    cfg_digest = ids.campaign_cfg_digest(spec.raw)
    campaign_id = ids.campaign_id(spec.slug, spec.date.replace("-", ""), cfg_digest)
    budget_uid = ids.budget_uid(spec.budget)

    agent_uids = {agent.id: _agent_uid(agent) for agent in spec.agents}

    runs: list[PlannedRun] = []
    for agent in spec.agents:
        for seed in sorted(spec.seeds):
            for attempt in range(1, spec.attempts + 1):
                for task_name in sorted(spec.tasks):
                    task_uid = ids.task_uid(
                        spec.integration, spec.dataset_digest, task_name, "pending"
                    )
                    run_id = ids.run_id(
                        cfg_digest=cfg_digest,
                        task_uid_value=task_uid,
                        agent_uid_value=agent_uids[agent.id],
                        budget_uid_value=budget_uid,
                        seed=seed,
                        attempt=attempt,
                    )
                    shard = ids.shard_of(run_id, spec.shards)
                    runs.append(
                        PlannedRun(
                            run_id=run_id,
                            task_name=task_name,
                            task_uid=task_uid,
                            agent_id=agent.id,
                            agent_uid=agent_uids[agent.id],
                            budget_uid=budget_uid,
                            seed=seed,
                            attempt=attempt,
                            shard=shard,
                            job_name=_job_name(campaign_id, agent.id, seed, attempt, shard),
                            match_key=ids.match_key(
                                task_name=task_name,
                                agent_uid_value=agent_uids[agent.id],
                                seed=seed,
                                attempt=attempt,
                            ),
                        )
                    )

    _assert_unique_run_ids(runs)
    jobs, job_configs = _build_jobs(spec, campaign_id, runs)
    return CompiledCampaign(
        campaign_id=campaign_id,
        cfg_digest=cfg_digest,
        spec=spec,
        runs=tuple(runs),
        jobs=tuple(jobs),
        job_configs=job_configs,
    )


def _agent_uid(agent: AgentSpec) -> str:
    return ids.agent_uid(
        scaffold=agent.scaffold,
        scaffold_version=agent.scaffold_version,
        model_id=agent.model,
        prompt_digest=agent.prompt_digest,
        tool_policy=agent.tool_policy,
        agent_kwargs=agent.agent_kwargs,
        env_keys=agent.env_keys,
        env_from_secret=agent.env_from_secret,
        env_literals=agent.env_literals,
        import_path=agent.import_path,
        setup_timeout_sec=agent.setup_timeout_sec,
    )


def _agent_by_id(spec: CampaignSpec, agent_id: str) -> AgentSpec:
    for agent in spec.agents:
        if agent.id == agent_id:
            return agent
    raise KeyError(agent_id)  # pragma: no cover - compiler invariant


def _job_name(campaign_id: str, agent_id: str, seed: int, attempt: int, shard: int) -> str:
    return f"{campaign_id}--{agent_id}--s{seed}--a{attempt}--sh{shard}"


def _assert_unique_run_ids(runs: list[PlannedRun]) -> None:
    seen: dict[str, PlannedRun] = {}
    for run in runs:
        previous = seen.get(run.run_id)
        if previous is not None:
            raise AssertionError(
                "run id collision between "
                f"{previous.task_name}/{previous.agent_id}/seed={previous.seed} and "
                f"{run.task_name}/{run.agent_id}/seed={run.seed}"
            )
        seen[run.run_id] = run


def _build_jobs(
    spec: CampaignSpec, campaign_id: str, runs: list[PlannedRun]
) -> tuple[list[PlannedJob], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, int], list[PlannedRun]] = {}
    for run in runs:
        grouped.setdefault((run.agent_id, run.seed, run.attempt, run.shard), []).append(run)

    jobs: list[PlannedJob] = []
    configs: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        agent_id, seed, attempt, shard = key
        members = grouped[key]
        job_name = members[0].job_name
        task_names = tuple(sorted(run.task_name for run in members))
        config_path = f"jobs/{job_name}.yaml"
        jobs.append(
            PlannedJob(
                job_name=job_name,
                agent_id=agent_id,
                seed=seed,
                attempt=attempt,
                shard=shard,
                task_names=task_names,
                config_path=config_path,
            )
        )
        configs[job_name] = _job_config(
            spec, campaign_id, job_name, _agent_by_id(spec, agent_id), seed, task_names
        )
    return jobs, configs


def _job_config(
    spec: CampaignSpec,
    campaign_id: str,
    job_name: str,
    agent: AgentSpec,
    seed: int,
    task_names: tuple[str, ...],
) -> dict[str, Any]:
    """Emit a Harbor job config.

    Field names and shape follow upstream's own experiment configs
    (`ORAgentBench/experiments/config/run_claude_code.yaml` and `run_codex.yaml`)
    so that the result is consumable by `harbor run -c` and by upstream's
    prebuild wrapper without translation.

    Secret values never appear: ``env`` carries ``${NAME}`` placeholders and the
    runner substitutes them.
    """
    harbor = spec.harbor
    agent_kwargs = dict(agent.agent_kwargs)
    # Seed travels through agent kwargs so it lands in Harbor's lock file and is
    # recoverable from the bundle rather than only from our plan.
    agent_kwargs.setdefault("orbench_seed", seed)

    env: dict[str, str] = {key: f"${{{key}}}" for key in agent.env_keys}
    # Variable name -> secret name, for scaffolds whose variable differs from
    # the secret supplying it.
    env.update({variable: f"${{{secret}}}" for variable, secret in agent.env_from_secret.items()})
    # Non-secret literals last: a model name is not a credential and the job
    # config is more readable with it inline, as upstream writes it.
    env.update(agent.env_literals)

    agent_entry: dict[str, Any] = {
        "name": agent.scaffold,
        "import_path": agent.import_path,
        "model_name": agent.model,
        "override_setup_timeout_sec": agent.setup_timeout_sec,
        "env": env,
        "kwargs": agent_kwargs,
    }
    if agent.tool_policy:
        agent_entry["kwargs"] = {**agent_kwargs, "tool_policy": dict(agent.tool_policy)}

    config: dict[str, Any] = {
        "job_name": job_name,
        "jobs_dir": harbor.get("jobs_dir", "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": harbor.get("n_concurrent_trials", 4),
        "timeout_multiplier": harbor.get("timeout_multiplier", 1.0),
        "metrics": [
            {"type": metric.get("type", "uv-script"),
             "kwargs": {k: v for k, v in metric.items() if k != "type"}}
            for metric in spec.metrics
        ],
        "retry": {
            "max_retries": int(spec.retry.get("max_retries", 0)),
            "include_exceptions": list(spec.retry.get("include_exceptions") or []) or None,
            "exclude_exceptions": list(spec.retry.get("exclude_exceptions") or []) or None,
        },
        "environment": {
            "type": harbor.get("environment_type", "docker"),
            "force_build": bool(harbor.get("force_build", False)),
            "delete": bool(harbor.get("delete_environment", True)),
            # Campaign-scoped label: cleanup filters on this instead of pruning
            # everything on a shared runner.
            "kwargs": {"labels": {CAMPAIGN_LABEL_KEY: campaign_id}},
        },
        "agents": [agent_entry],
        "datasets": [
            {
                "path": spec.dataset_path,
                "task_names": list(task_names),
                "exclude_task_names": None,
            }
        ],
        "tasks": [],
    }
    # Upstream's prebuild wrapper reads this block to bake the agent CLI into
    # the base image; Harbor itself ignores it. Emitted only when the spec asks
    # for it, so a control campaign's config stays byte-identical.
    pre_build = harbor.get("pre_build")
    if pre_build:
        config["pre_build"] = dict(pre_build)
    return config


def write_plan(compiled: CompiledCampaign, out_dir: str | Path) -> dict[str, Path]:
    """Write plan.json, plan_ledger.json and one job config per job.

    Returns a mapping of artefact name to path. Output is byte-stable for a
    given spec, which the determinism test relies on.
    """
    out = Path(out_dir)
    (out / "jobs").mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    plan_path = out / "plan.json"
    _write_json(plan_path, compiled.plan_dict())
    written["plan"] = plan_path

    ledger_path = out / "plan_ledger.json"
    _write_json(ledger_path, compiled.ledger_dict())
    written["plan_ledger"] = ledger_path

    for job in compiled.jobs:
        job_path = out / "jobs" / f"{job.job_name}.yaml"
        header = (
            "# Generated by orbench campaign plan. Do not edit by hand.\n"
            f"# campaign_id: {compiled.campaign_id}\n"
            f"# Secrets appear as ${{NAME}} placeholders only; values are supplied by the runner.\n"
        )
        body = yaml.safe_dump(
            compiled.job_configs[job.job_name], sort_keys=False, default_flow_style=False
        )
        job_path.write_text(header + body, encoding="utf-8")
        written[f"job:{job.job_name}"] = job_path
    return written


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
