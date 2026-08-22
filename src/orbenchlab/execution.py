"""Construction and validation of the *upstream* commands that execute a benchmark.

ORBenchLab still does not execute benchmarks. What this module does is build the
exact command line an upstream project documents for its own execution path,
validate every input against that checkout, and refuse to build a command it
cannot justify. Running it is `tools/run_benchmark_smoke.py`'s job, and all it
does is hand the argv to the upstream process.

The distinction matters and is worth stating precisely, because it is the line
between an integration and a fork:

* **Delegation** (what happens here) — we assemble `harbor run -c <config>` or
  `python -m frontieror.infra agent ...` and let upstream do the work. If
  upstream changes how it runs things, our command stops matching and a
  contract test fails loudly.
* **Reimplementation** (what does not happen here) — scheduling trials,
  retrying, resuming, grading, sandboxing. A second implementation of any of
  those becomes a second source of truth about what actually ran.

Everything below the CLI layer is a pure function: given inputs and a checkout,
it returns an argv, a working directory and the environment variable *names*
involved. That is what makes the commands unit-testable without a model key, a
licence, Docker, or a single paid call.

Commands recorded here were read from the pinned upstream checkouts, not
recalled. Their provenance is in `UpstreamCommand.provenance`.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core.errors import PreconditionError, SpecError

# --------------------------------------------------------------------------- #
# ORAgentBench — upstream execution surface
# --------------------------------------------------------------------------- #

#: ORAgentBench resolves relative dataset paths (`ORAgentBench/harbor_tasks`)
#: against the *parent* of its own checkout, and runs Harbor from there. So the
#: checkout directory has to carry this exact name or the dataset path in the
#: job config resolves to nothing. Verified in
#: `source/scripts/run_harbor_prebuild.py::resolve_repo_path`.
ORAGENTBENCH_CHECKOUT_DIRNAME = "ORAgentBench"

#: Upstream's own Harbor wrapper, used by `experiments/scripts/run_claude_code.sh`
#: for agent runs. It pre-builds the agent CLI into the base image, applies
#: dynamic skills, then execs `harbor run -c <transformed config>`.
ORAGENTBENCH_PREBUILD_WRAPPER = "source/scripts/run_harbor_prebuild.py"

#: The dataset path that appears in every upstream job config.
ORAGENTBENCH_DATASET_PATH = f"{ORAGENTBENCH_CHECKOUT_DIRNAME}/harbor_tasks"

#: Agent configuration shapes taken from upstream's own experiment configs
#: (`experiments/config/run_claude_code.yaml`, `run_codex.yaml`,
#: `experiments/scripts/run_test.sh`). These record the *interface* — which
#: environment variables an agent scaffold reads and which kwargs upstream
#: pins — not any benchmark logic.
#:
#: `env_from_secret` maps an environment variable the scaffold reads to the
#: name of the repository secret supplying it. Values never appear here; the
#: compiler emits `${SECRET_NAME}` placeholders and the runner substitutes.
AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "claude-code": {
        "scaffold": "claude-code",
        "provider_style": "anthropic-compatible",
        "env_from_secret": {
            "ANTHROPIC_AUTH_TOKEN": "MODEL_API_KEY",
            "ANTHROPIC_API_KEY": "MODEL_API_KEY",
            "ANTHROPIC_BASE_URL": "MODEL_BASE_URL",
        },
        # MODEL_BASE_URL is required but is not itself a credential; it may be a
        # repository *variable* rather than a secret.
        "non_secret_env": ("MODEL_BASE_URL",),
        # ANTHROPIC_MODEL is a literal, not a credential: upstream sets it to
        # the same string as model_name.
        "env_literal_from_model": "ANTHROPIC_MODEL",
        "kwargs": {"version": None, "disallowed_tools": "WebSearch,WebFetch"},
        "setup_timeout_sec": 420,
        "prebuilt": True,
    },
    "codex": {
        "scaffold": "codex",
        "provider_style": "openai-compatible",
        "env_from_secret": {
            "OPENAI_API_KEY": "MODEL_API_KEY",
            "OPENAI_BASE_URL": "MODEL_BASE_URL",
        },
        "non_secret_env": ("MODEL_BASE_URL",),
        "env_literal_from_model": None,
        "kwargs": {
            "version": None,
            "reasoning_effort": "high",
            "reasoning_summary": "auto",
            "web_search": "disabled",
        },
        "setup_timeout_sec": 420,
        "prebuilt": True,
    },
    "mini-swe-agent": {
        "scaffold": "mini-swe-agent",
        "provider_style": "mini-swe-agent",
        "env_from_secret": {"MSWEA_API_KEY": "MODEL_API_KEY"},
        "non_secret_env": (),
        "env_literal_from_model": None,
        "kwargs": {"version": None},
        "setup_timeout_sec": 420,
        "prebuilt": True,
    },
}

#: Zero-model-cost controls. Harbor built-ins; they take no credentials at all.
CONTROL_SCAFFOLDS = ("oracle", "nop")

# --------------------------------------------------------------------------- #
# FrontierOR — upstream execution surface
# --------------------------------------------------------------------------- #

#: The official trusted-evaluation entry point.
FRONTIEROR_AGENT_ENTRYPOINT = ("-m", "frontieror.infra", "agent")

#: Flags upstream's `_agent_parser()` accepts. Anything outside this set is
#: refused before the process starts, so a workflow input cannot smuggle an
#: argument past the official interface.
FRONTIEROR_ALLOWED_AGENT_FLAGS = frozenset(
    {
        "--paper-id",
        "--primary-model",
        "--secondary-model",
        "--stage1-instances",
        "--dev-set",
        "--test-set",
        "--stage1-time-limit",
        "--stage2-time-limit",
        "--test-time-limit",
        "--stage1-gap-threshold",
        "--stage2-stage-boundary",
        "--stage2-time-policy",
        "--stage2-time-buffer",
        "--test-time-policy",
        "--test-time-buffer",
        "--coral-attempts",
        "--coral-max-seconds",
        "--coral-attempts-budget-multiplier",
        "--coral-agent-count",
        "--coral-agent-model",
        "--coral-max-steps",
        "--coral-max-turns",
        "--coral-heartbeat-reflect-every",
        "--coral-heartbeat-pivot-every",
        "--coral-heartbeat-consolidate-every",
        "--paper-workers",
        "--dev-instance-workers",
        "--test-instance-workers",
        "--wls-egress",
        "--cpus",
        "--memory",
        "--t_max",
        "--run-id",
    }
)

#: Upstream *appends* its non-overridable trusted profile (`--framework coral`,
#: `--exec-mode docker`, `--stage2-scorer staged_qte`,
#: `--coral-agent-isolation docker`, `--coral-model-access proxy`,
#: `--coral-agent-image ...`, `--anti-hack`) and raises if a caller supplies a
#: conflicting value. Passing any of them from here is at best redundant and at
#: worst an attempted downgrade, so we refuse them outright rather than relying
#: on upstream to catch it.
FRONTIEROR_FORBIDDEN_AGENT_FLAGS = frozenset(
    {
        "--framework",
        "--exec-mode",
        "--stage2-scorer",
        "--coral-agent-isolation",
        "--coral-model-access",
        "--coral-agent-image",
        "--modes",
        "--anti-hack",
        "--no-anti-hack",
        "--coral-gateway",
    }
)

#: Container images upstream's trusted profile requires, built from
#: `frontieror/infra/docker/*.Dockerfile` per its README.
FRONTIEROR_REQUIRED_IMAGES = (
    "frontieror-candidate:1",
    "frontieror-coral-agent:0.1",
    "frontieror-coral-model-proxy:0.1",
)

#: Proxy mode needs a full provider/model route (e.g. `openai/gpt-5.4`); the
#: agent container only ever sees a short name and an ephemeral proxy token.
_ROUTE_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.\-:]+$")

#: Mirrors upstream's `is_valid_instance_name`: "tiny", or "large_<N>" with
#: N >= 1. This is argument validation so the job fails before starting, not a
#: copy of any checker or scorer.
_INSTANCE_RE = re.compile(r"^(tiny|large_[1-9][0-9]*)$")

_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class UpstreamCommand:
    """A command belonging to an upstream project, ready to be handed over."""

    argv: tuple[str, ...]
    cwd: str
    #: Environment variable names the command needs. Names only — a value in
    #: this structure would be a leaked credential.
    required_env: tuple[str, ...] = ()
    #: Where this command was read from, so a reviewer can check it.
    provenance: str = ""
    description: str = ""
    #: Whether this invocation makes model calls, and therefore costs money.
    makes_model_calls: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "required_env": list(self.required_env),
            "provenance": self.provenance,
            "description": self.description,
            "makes_model_calls": self.makes_model_calls,
        }

    def shell(self) -> str:
        """Human-readable form for logs. Never used to execute anything."""
        import shlex

        return shlex.join(self.argv)


@dataclass
class PreconditionReport:
    """What is present, what is missing, and therefore whether we may proceed."""

    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def require(self, condition: bool, description: str) -> None:
        (self.satisfied if condition else self.missing).append(description)

    def raise_if_unmet(self, context: str) -> None:
        if self.missing:
            bullets = "\n".join(f"  - {item}" for item in self.missing)
            raise PreconditionError(
                f"{context}: {len(self.missing)} precondition(s) unmet:\n{bullets}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "satisfied": self.satisfied, "missing": self.missing}


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def validate_oragentbench_source(source: Path) -> Path:
    """Check a checkout is usable as a Harbor dataset root.

    The directory-name check is not pedantry. Upstream resolves the relative
    dataset path against the parent of its checkout, so a clone named anything
    else silently produces a job config whose dataset path does not exist.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise PreconditionError(f"ORAgentBench source is not a directory: {source}")
    if source.name != ORAGENTBENCH_CHECKOUT_DIRNAME:
        raise PreconditionError(
            f"ORAgentBench checkout must be named {ORAGENTBENCH_CHECKOUT_DIRNAME!r} "
            f"(got {source.name!r}). Upstream resolves the dataset path "
            f"{ORAGENTBENCH_DATASET_PATH!r} against the checkout's parent directory, so any "
            "other name resolves to a path that does not exist."
        )
    if not (source / "harbor_tasks").is_dir():
        raise PreconditionError(f"{source}/harbor_tasks not found; not an ORAgentBench checkout")
    return source


def validate_oragentbench_task(source: Path, task_name: str) -> str:
    """Resolve a task name against the checkout, refusing anything else.

    Upstream filters datasets by *directory* name and raises on an unknown one.
    Validating here means a typo fails in seconds instead of after the image
    build.
    """
    if not _TASK_NAME_RE.match(task_name or ""):
        raise SpecError(
            f"task name {task_name!r} is not a plain task directory name; "
            "path separators and traversal are rejected"
        )
    task_dir = Path(source) / "harbor_tasks" / task_name
    if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
        available = sorted(
            child.name
            for child in (Path(source) / "harbor_tasks").iterdir()
            if (child / "task.toml").is_file()
        )
        hint = ", ".join(available[:5])
        raise SpecError(
            f"task {task_name!r} is not a validated task in this checkout "
            f"({len(available)} available, e.g. {hint})"
        )
    return task_name


def validate_frontieror_source(source: Path) -> Path:
    source = Path(source).resolve()
    if not source.is_dir():
        raise PreconditionError(f"FrontierOR source is not a directory: {source}")
    if not (source / "frontieror" / "infra" / "cli.py").is_file():
        raise PreconditionError(
            f"{source} does not expose frontieror/infra/cli.py; there is no official "
            "trusted-evaluation entry point here to invoke"
        )
    return source


def validate_frontieror_paper_id(source: Path, paper_id: str) -> str:
    """Resolve a paper id against upstream's own metadata file."""
    if not _TASK_NAME_RE.match(paper_id or ""):
        raise SpecError(f"paper id {paper_id!r} is not a plain identifier")
    meta = Path(source) / "paper_meta_info.json"
    if not meta.is_file():
        raise PreconditionError(
            f"{meta} not found; cannot validate the paper id against upstream metadata"
        )
    import json

    entries = json.loads(meta.read_text(encoding="utf-8"))
    known = {entry.get("paper_id") for entry in entries if isinstance(entry, dict)}
    if paper_id not in known:
        raise SpecError(
            f"paper id {paper_id!r} is not in this checkout's paper_meta_info.json "
            f"({len(known)} known ids)"
        )
    return paper_id


def validate_frontieror_instances(names: Sequence[str], *, role: str) -> tuple[str, ...]:
    """Apply upstream's instance-name rule before the process starts."""
    if not names:
        raise SpecError(f"{role} must list at least one instance")
    bad = [name for name in names if not _INSTANCE_RE.match(name)]
    if bad:
        raise SpecError(
            f"{role} contains invalid instance name(s) {bad}; upstream accepts 'tiny' "
            "or 'large_<N>' with N >= 1"
        )
    if len(set(names)) != len(names):
        raise SpecError(f"{role} contains duplicate instances")
    return tuple(names)


def validate_frontieror_split(
    *, stage1: Sequence[str], dev: Sequence[str], test: Sequence[str]
) -> None:
    """Refuse a leaky split before upstream has to.

    Upstream enforces dev/final isolation itself; failing here means the
    operator learns about it immediately rather than after image builds.
    """
    public = set(stage1) | set(dev)
    overlap = sorted(public & set(test))
    if overlap:
        raise SpecError(
            "the held-out test set overlaps the stage1/dev instances the agent can "
            f"see: {overlap}. Upstream requires disjoint sets, and a leaky split makes "
            "the score meaningless."
        )


def validate_model_route(route: str) -> str:
    """Proxy mode needs a full provider/model route, and never a floating one."""
    if not route:
        raise SpecError("a model route is required (for example 'openai/gpt-5.4')")
    if not _ROUTE_RE.match(route):
        raise SpecError(
            f"model route {route!r} must be a full provider/model route such as "
            "'openai/gpt-5.4'; upstream's proxy mode requires the provider prefix"
        )
    if route.endswith("latest") or "*" in route:
        raise SpecError(f"model route {route!r} is a floating alias; pin an exact model")
    return route


def validate_pinned_model(model: str) -> str:
    if not model:
        raise SpecError("a pinned model id is required")
    if model.endswith("latest") or "*" in model or model == "auto":
        raise SpecError(f"model {model!r} is a floating alias; pin an exact model id")
    return model


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id or ""):
        raise SpecError(
            f"run id {run_id!r} must be lowercase alphanumeric with hyphens; it becomes "
            "a directory name on the execution host"
        )
    return run_id


def validate_extra_flags(extra: Sequence[str]) -> tuple[str, ...]:
    """Refuse anything outside upstream's documented agent interface."""
    flags = [token for token in extra if token.startswith("--")]
    forbidden = sorted(set(flags) & FRONTIEROR_FORBIDDEN_AGENT_FLAGS)
    if forbidden:
        raise SpecError(
            f"refusing to pass {forbidden}: upstream fixes its trusted-agent profile "
            "(coral framework, docker isolation, staged_qte scoring, proxied model "
            "access, anti-hack) and appends those flags itself. Supplying them here "
            "could only weaken the profile or contradict it."
        )
    unknown = sorted(set(flags) - FRONTIEROR_ALLOWED_AGENT_FLAGS)
    if unknown:
        raise SpecError(
            f"refusing to pass {unknown}: not part of upstream's documented "
            "'python -m frontieror.infra agent' interface"
        )
    return tuple(extra)


# --------------------------------------------------------------------------- #
# command builders
# --------------------------------------------------------------------------- #


def oragentbench_controls_command(
    *, source: Path, job_config: Path, python: str = "python3"
) -> UpstreamCommand:
    """`harbor run -c <config>`, the path upstream's run_oracle_all.sh takes.

    Controls use Harbor's built-in oracle and nop agents, so this makes no model
    calls at all.
    """
    source = validate_oragentbench_source(source)
    return UpstreamCommand(
        argv=("harbor", "run", "-c", str(Path(job_config).resolve())),
        cwd=str(source.parent),
        required_env=(),
        provenance=(
            "ORAgentBench experiments/scripts/run_oracle_all.sh: "
            "`command=(harbor run -c \"${CONFIG}\" \"${args[@]}\")` executed from the "
            "checkout's parent directory"
        ),
        description="Harbor job over ORAgentBench tasks using built-in control agents",
        makes_model_calls=False,
    )


def oragentbench_agent_command(
    *,
    source: Path,
    job_config: Path,
    python: str = "python3",
    skip_build: bool = True,
    dry_run: bool = False,
    resume: bool = False,
    required_env: Sequence[str] = (),
) -> UpstreamCommand:
    """Upstream's own Harbor wrapper, the path run_claude_code.sh takes.

    `experiments/scripts/run_claude_code.sh` is literally
    `bash scripts/run_harbor_prebuild.sh -c <config> --skip-build`, and that
    shell script execs `python source/scripts/run_harbor_prebuild.py "$@"`. The
    wrapper pre-builds the agent CLI into the base image, applies dynamic
    skills, and then execs `harbor run -c <transformed config>`.

    We call the Python entry point directly so the interpreter is explicit and
    the command is exactly reproducible from a workflow log.
    """
    source = validate_oragentbench_source(source)
    wrapper = source / ORAGENTBENCH_PREBUILD_WRAPPER
    if not wrapper.is_file():
        raise PreconditionError(
            f"{wrapper} not found; this checkout does not ship the upstream Harbor wrapper"
        )
    argv = [python, str(wrapper), "-c", str(Path(job_config).resolve())]
    if skip_build:
        argv.append("--skip-build")
    if resume:
        argv.append("--resume")
    if dry_run:
        argv.append("--dry-run")
    return UpstreamCommand(
        argv=tuple(argv),
        cwd=str(source.parent),
        # The credentials live in the job config as ${NAME} placeholders, so the
        # names travel with the command even though none is on its argv.
        required_env=tuple(sorted(set(required_env))),
        provenance=(
            "ORAgentBench experiments/scripts/run_claude_code.sh -> "
            "scripts/run_harbor_prebuild.sh -> source/scripts/run_harbor_prebuild.py, "
            "which execs `harbor run -c <transformed config>`"
        ),
        description=(
            "ORAgentBench Harbor wrapper (dry run: prints the transformed config and the "
            "harbor command, runs nothing)"
            if dry_run
            else "ORAgentBench Harbor wrapper: pre-build agent CLI, then harbor run"
        ),
        # A dry run stops before Harbor starts, so nothing is spent.
        makes_model_calls=not dry_run,
    )


def frontieror_agent_command(
    *,
    source: Path,
    paper_id: str,
    primary_model: str,
    stage1_instances: Sequence[str],
    dev_instances: Sequence[str],
    test_instances: Sequence[str],
    run_id: str,
    cpus: int = 1,
    memory: str = "128G",
    coral_agent_count: int = 1,
    coral_attempts: int = 10,
    coral_max_steps: int = 10,
    coral_max_seconds: str = "auto",
    extra: Sequence[str] = (),
    python: str = "python3",
) -> UpstreamCommand:
    """`python -m frontieror.infra agent ...` — the official trusted entry point.

    Flag order follows upstream's README example. Notably absent: every flag in
    the fixed trusted profile. Upstream appends those itself and raises on a
    conflicting value, so this command cannot weaken isolation, swap the
    scorer, or hand the platform key to the agent container.
    """
    source = validate_frontieror_source(source)
    paper_id = validate_frontieror_paper_id(source, paper_id)
    primary_model = validate_model_route(primary_model)
    stage1 = validate_frontieror_instances(stage1_instances, role="stage1 instances")
    dev = validate_frontieror_instances(dev_instances, role="dev set")
    test = validate_frontieror_instances(test_instances, role="test set")
    validate_frontieror_split(stage1=stage1, dev=dev, test=test)
    run_id = validate_run_id(run_id)
    extra = validate_extra_flags(extra)

    if cpus < 1:
        raise SpecError("cpus must be >= 1")
    if not re.match(r"^\d+[KMGT]?$", memory):
        raise SpecError(f"memory {memory!r} must look like '128G'")

    argv = [
        python,
        *FRONTIEROR_AGENT_ENTRYPOINT,
        "--paper-id",
        paper_id,
        "--primary-model",
        primary_model,
        "--stage1-instances",
        *stage1,
        "--dev-set",
        *dev,
        "--test-set",
        *test,
        "--coral-agent-count",
        str(coral_agent_count),
        "--coral-attempts",
        str(coral_attempts),
        "--coral-max-steps",
        str(coral_max_steps),
        "--coral-max-seconds",
        str(coral_max_seconds),
        "--cpus",
        str(cpus),
        "--memory",
        memory,
        "--run-id",
        run_id,
        *extra,
    ]
    return UpstreamCommand(
        argv=tuple(argv),
        cwd=str(source),
        required_env=("OPENROUTER_API_KEY", "GRB_LICENSE_FILE"),
        provenance=(
            "FrontierOR README 'Trusted Agent Evaluation' example, validated against "
            "frontieror/infra/cli.py::_agent_parser; the fixed trusted profile is "
            "appended by frontieror/infra/policy.py::hardened_agent_argv"
        ),
        description="FrontierOR official trusted-agent evaluation",
        makes_model_calls=True,
    )


def frontieror_contract_command(
    *, source: Path, python: str = "python3"
) -> UpstreamCommand:
    """The zero-cost path: print the published scoring contract."""
    source = validate_frontieror_source(source)
    return UpstreamCommand(
        argv=(python, "-m", "frontieror.infra", "contract"),
        cwd=str(source),
        required_env=(),
        provenance="FrontierOR frontieror/infra/cli.py, `contract` subcommand",
        description="Print the published staged_qte scoring contract",
        makes_model_calls=False,
    )


def frontieror_security_check_command(
    *, source: Path, candidate_image: str = "frontieror-candidate:1", python: str = "python3"
) -> UpstreamCommand:
    """Upstream's black-box probe of the candidate container boundary."""
    source = validate_frontieror_source(source)
    return UpstreamCommand(
        argv=(python, "-m", "frontieror.infra", "security-check", "--candidate-image", candidate_image),
        cwd=str(source),
        required_env=(),
        provenance="FrontierOR README, run before releasing a runner image",
        description="Probe host-file, environment, network and timeout escapes",
        makes_model_calls=False,
    )


# --------------------------------------------------------------------------- #
# campaign spec construction for a validated agent run
# --------------------------------------------------------------------------- #


def oragentbench_agent_campaign_spec(
    *,
    slug: str,
    date: str,
    dataset_digest: str,
    task_name: str,
    scaffold: str,
    model: str,
    seeds: Sequence[int] = (1,),
    wall_clock_sec: int = 2700,
    max_cost_usd: float = 25.0,
    site: str = "local-docker",
    jobs_dir: str = "jobs",
    n_concurrent_trials: int = 1,
) -> dict[str, Any]:
    """Build a campaign spec for one validated ORAgentBench agent run.

    The agent block is filled from the recorded upstream profile, so the
    compiled job config carries the environment variables that scaffold
    actually reads and the kwargs upstream pins — rather than a plausible
    guess that Harbor would accept and the agent would ignore.
    """
    if scaffold not in AGENT_PROFILES:
        raise SpecError(
            f"scaffold {scaffold!r} has no recorded upstream agent profile; "
            f"known: {sorted(AGENT_PROFILES)}"
        )
    profile = AGENT_PROFILES[scaffold]
    validate_pinned_model(model)

    env_literals: dict[str, str] = {}
    literal_var = profile.get("env_literal_from_model")
    if literal_var:
        env_literals[literal_var] = model

    agent: dict[str, Any] = {
        "id": f"{scaffold}-{model}".lower().replace("/", "-").replace(".", "-"),
        "scaffold": scaffold,
        "model": model,
        "env_from_secret": dict(profile["env_from_secret"]),
        "env_literals": env_literals,
        "agent_kwargs": dict(profile["kwargs"]),
        "setup_timeout_sec": profile.get("setup_timeout_sec"),
    }

    return {
        "schema_version": "1.0",
        "slug": slug,
        "date": date,
        "integration": "oragentbench",
        "site": site,
        # One rollout of one task supports case diagnosis, nothing more.
        "evidence_intent": "exploratory",
        "dataset": {"path": ORAGENTBENCH_DATASET_PATH, "digest": dataset_digest},
        "tasks": [task_name],
        "agents": [agent],
        "budget": {"wall_clock_sec": wall_clock_sec, "max_cost_usd": max_cost_usd},
        "seeds": list(seeds),
        "attempts": 1,
        "shards": 1,
        "harbor": {
            "jobs_dir": jobs_dir,
            "n_concurrent_trials": n_concurrent_trials,
            "environment_type": "docker",
            # Consumed by upstream's prebuild wrapper; Harbor ignores it.
            "pre_build": {"enabled": True, "agent": scaffold, "rebuild_base": False},
        },
        "retry": {"max_retries": 0, "include_exceptions": []},
        "metrics": [
            {"type": "uv-script", "script_path": f"{ORAGENTBENCH_CHECKOUT_DIRNAME}/metrics/per_dimension_reward.py"}
        ],
    }


# --------------------------------------------------------------------------- #
# preconditions
# --------------------------------------------------------------------------- #


def oragentbench_preconditions(
    *,
    source: Path,
    task_name: str,
    scaffold: str,
    model: str,
    environ: Mapping[str, str] | None = None,
    require_docker: bool = True,
    require_harbor: bool = True,
    require_secrets: bool = True,
) -> PreconditionReport:
    """Everything that must hold before an ORAgentBench agent run may start."""
    environ = os.environ if environ is None else environ
    report = PreconditionReport()

    try:
        resolved = validate_oragentbench_source(source)
        report.require(True, f"ORAgentBench checkout present at {resolved}")
    except PreconditionError as exc:
        report.require(False, str(exc))
        return report

    try:
        validate_oragentbench_task(resolved, task_name)
        report.require(True, f"task {task_name!r} exists in harbor_tasks/ with a task.toml")
    except SpecError as exc:
        report.require(False, str(exc))

    if scaffold in CONTROL_SCAFFOLDS:
        report.require(True, f"scaffold {scaffold!r} is a Harbor built-in control (no cost)")
    elif scaffold in AGENT_PROFILES:
        report.require(True, f"scaffold {scaffold!r} has a recorded upstream agent profile")
        try:
            validate_pinned_model(model)
            report.require(True, f"model {model!r} is pinned")
        except SpecError as exc:
            report.require(False, str(exc))
        non_secret = set(AGENT_PROFILES[scaffold].get("non_secret_env") or ())
        names = sorted(set(AGENT_PROFILES[scaffold]["env_from_secret"].values()))
        if not require_secrets:
            report.require(
                True,
                "credentials not required: upstream's wrapper dry run transforms the "
                f"config and prints the harbor command without contacting a provider "
                f"(would otherwise need {names})",
            )
            names = []
        for name in names:
            # Emptiness only. The value is never read, logged or hashed.
            kind = "variable" if name in non_secret else "secret"
            report.require(
                bool(environ.get(name)),
                f"{kind} {name} is set (value never read or logged)",
            )
    else:
        report.require(
            False,
            f"scaffold {scaffold!r} has no recorded upstream agent profile; known: "
            f"{sorted(AGENT_PROFILES)} plus controls {list(CONTROL_SCAFFOLDS)}",
        )

    if require_harbor:
        report.require(
            shutil.which("harbor") is not None, "the harbor CLI is on PATH on this host"
        )
    if require_docker:
        report.require(shutil.which("docker") is not None, "docker is on PATH on this host")
    return report


def frontieror_preconditions(
    *,
    source: Path,
    environ: Mapping[str, str] | None = None,
    require_docker: bool = True,
    require_perf_isolation: bool = True,
    require_images: bool = True,
    available_images: Iterable[str] | None = None,
) -> PreconditionReport:
    """Everything that must hold before a FrontierOR agent run may start.

    Performance isolation is included because FrontierOR's score contains a
    speed term measured as trusted-host wall clock. A run on a co-tenanted
    machine does not produce a weaker number; it produces a differently-defined
    one.
    """
    environ = os.environ if environ is None else environ
    report = PreconditionReport()

    try:
        resolved = validate_frontieror_source(source)
        report.require(True, f"FrontierOR checkout present at {resolved}")
    except PreconditionError as exc:
        report.require(False, str(exc))
        return report

    report.require(
        bool(environ.get("OPENROUTER_API_KEY")),
        "environment variable OPENROUTER_API_KEY is set (value never read or logged)",
    )
    licence = environ.get("GRB_LICENSE_FILE")
    report.require(
        bool(licence) and Path(licence).is_file(),
        "GRB_LICENSE_FILE points at a readable Gurobi licence file",
    )
    if require_docker:
        report.require(shutil.which("docker") is not None, "docker is on PATH on this host")
    if require_perf_isolation:
        report.require(
            environ.get("ORBENCH_PERF_ISOLATED") == "true",
            "ORBENCH_PERF_ISOLATED=true — this host has pinned cores and no co-tenancy, "
            "so a wall-clock measurement here is comparable to the reference runtime",
        )
    if require_images:
        present = set(available_images or ())
        for image in FRONTIEROR_REQUIRED_IMAGES:
            report.require(image in present, f"container image {image} has been built")
    return report


def docker_images_present(*, images: Iterable[str] | None = None) -> list[str]:
    """List locally available image tags, or an empty list if docker is absent."""
    if images is not None:
        return list(images)
    if shutil.which("docker") is None:
        return []
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - host dependent
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# receipts
# --------------------------------------------------------------------------- #

#: Redacted in any receipt. Matching is on the *name*, so a value never has to
#: be recognised to be withheld.
_SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "LICENSE", "LICENCE")

#: Filenames that indicate raw evidence. A receipt naming one of these would
#: mean the sanitizer had been bypassed.
RAW_BUNDLE_MARKERS = (
    "trajectory.json",
    "test-stdout",
    "test-stderr",
    "reward.txt",
    "reference_metrics",
    "reference_objective",
    "reference_runtime",
)


def sanitize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Strip anything a receipt must not carry off the execution host.

    Two classes of thing: credential values, and raw evidence. Environment
    variables are reduced to `{name: "<set>"|"<unset>"}` — enough to debug a
    misconfiguration, never enough to leak one.
    """
    return _sanitize(dict(receipt))


#: Markers that describe a variable rather than reveal it. Redacting these
#: would hide the one thing a receipt is for — whether the host was configured.
_PRESENCE_MARKERS = ("<set>", "<unset>", "<redacted>")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if _looks_secret(str(key)) and not (
                isinstance(item, str) and item in _PRESENCE_MARKERS
            ):
                clean[key] = "<redacted>"
            else:
                clean[key] = _sanitize(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _redact_env_assignments(value)
    return value


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_NAME_HINTS)


_ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:" + "|".join(_SECRET_NAME_HINTS) + r")[A-Z0-9_]*)=(\S+)"
)


def _redact_env_assignments(text: str) -> str:
    return _ENV_ASSIGNMENT_RE.sub(r"\1=<redacted>", text)


def env_presence(names: Iterable[str], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Report which variables are set without revealing any value."""
    environ = os.environ if environ is None else environ
    return {name: ("<set>" if environ.get(name) else "<unset>") for name in sorted(set(names))}


def build_receipt(
    *,
    integration: str,
    mode: str,
    command: UpstreamCommand,
    campaign_id: str | None,
    preconditions: PreconditionReport,
    exit_code: int | None,
    evidence_label: str,
    output_root: str | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """A small, sanitized record of what was run — never the bundle itself."""
    receipt = {
        "receipt_schema_version": "1.0",
        "integration": integration,
        "mode": mode,
        "campaign_id": campaign_id,
        "evidence_label": evidence_label,
        "upstream_command": {
            "argv": list(command.argv),
            "cwd": command.cwd,
            "provenance": command.provenance,
            "makes_model_calls": command.makes_model_calls,
        },
        "environment": env_presence(command.required_env),
        "preconditions": preconditions.to_dict(),
        "exit_code": exit_code,
        "executed": exit_code is not None,
        "output_root": output_root,
        "raw_bundle_uploaded": False,
        "notes": list(notes),
    }
    return sanitize_receipt(receipt)


def assert_receipt_is_shareable(receipt: Mapping[str, Any]) -> None:
    """Last line of defence before a receipt leaves the execution host."""
    import json

    blob = json.dumps(receipt)
    hits = [marker for marker in RAW_BUNDLE_MARKERS if marker in blob]
    if hits:
        raise PreconditionError(
            f"receipt references raw evidence {hits}; raw bundles stay on the execution host"
        )
    if receipt.get("raw_bundle_uploaded"):
        raise PreconditionError("receipt claims a raw bundle was uploaded")
