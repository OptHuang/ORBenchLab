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

import json
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core.errors import PreconditionError, SpecError
from .core.urls import provider_route_digest, validate_https_base_url

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

#: Pinned upstream entry point for zero-model-cost controls.  Unlike calling
#: ``harbor run`` ourselves, this script first builds the shared base image on
#: a fresh host and then delegates to Harbor with the supplied config.
ORAGENTBENCH_ORACLE_RUNNER = "experiments/scripts/run_oracle_all.sh"

#: The same upstream build step used by the oracle runner.  Recovery uses it
#: only when the immutable image recorded for an interrupted job is no longer
#: available under its ID in the local Docker daemon.
ORAGENTBENCH_BASE_IMAGE_BUILDER = "scripts/build_base_image.sh"

#: Both upstream control and prebuilt-agent paths build through this fixed
#: alias before Harbor starts.  ORBenchLab serializes all use of it per host.
ORAGENTBENCH_FIXED_BASE_IMAGE = "oragentbench-base:py311-scip"

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

# These upstream adapters send their API credential to a configurable base
# URL. The destination is therefore a required identity input, not merely an
# optional runtime variable. mini-swe-agent's recorded profile has no base-URL
# variable and remains route-free.
ROUTE_PINNED_SCAFFOLDS = frozenset({"claude-code", "codex"})

#: The first Harbor release whose Codex adapter supports the auth/runtime path
#: documented by ORBenchLab.  Keep the compatibility floor explicit so a CLI
#: from an unrelated or stale Harbor installation cannot pass ``doctor`` just
#: because an executable with the same name happens to be on PATH.
MINIMUM_HARBOR_VERSION = (0, 16, 0)

#: Runner probes must never make ``doctor`` hang behind an unhealthy Docker
#: socket or a broken wrapper installation.
RUNNER_PROBE_TIMEOUT_SECONDS = 5.0

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def discover_codex_base_url(environ: Mapping[str, str] | None = None) -> str:
    """Return Codex's configured provider URL without inspecting auth data.

    Environment overrides are useful on CI/self-hosted runners.  Otherwise we
    read only ``config.toml`` (never ``auth.json``) and resolve the selected
    provider table used by Codex itself.
    """
    environ = os.environ if environ is None else environ
    direct = environ.get("ORBENCH_MODEL_BASE_URL") or environ.get("OPENAI_BASE_URL")
    if direct:
        return direct.strip()
    explicit = environ.get("CODEX_CONFIG_PATH")
    if explicit:
        path = Path(explicit).expanduser()
    else:
        codex_home = Path(environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        path = codex_home / "config.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ""
    top_level = data.get("openai_base_url")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()
    provider = data.get("model_provider")
    providers = data.get("model_providers")
    if isinstance(provider, str) and isinstance(providers, Mapping):
        selected = providers.get(provider)
        if isinstance(selected, Mapping):
            value = selected.get("base_url")
            if isinstance(value, str):
                return value.strip()
    return ""

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
    #: Non-secret process-environment overrides required by upstream.  Values
    #: are recorded in receipts, so credentials must never be placed here.
    env_overrides: dict[str, str] = field(default_factory=dict)
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
            "env_overrides": dict(self.env_overrides),
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


def oragentbench_task_allows_internet(source: Path, task_name: str) -> bool:
    """Read the selected task's declared agent-network policy."""
    validate_oragentbench_task(source, task_name)
    path = Path(source) / "harbor_tasks" / task_name / "task.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SpecError(f"cannot read the selected task policy from task.toml: {exc}") from None
    return (data.get("environment") or {}).get("allow_internet") is True


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


def validate_pinned_scaffold_version(version: str) -> str:
    """Require the CLI baked into a paid image to have an exact version."""
    value = str(version or "").strip()
    if not value:
        raise SpecError("a pinned scaffold CLI version is required for a model agent")
    if value.lower() in {"latest", "auto", "stable", "main", "master"} or "*" in value:
        raise SpecError(
            f"scaffold version {value!r} is floating; pin an exact released CLI version"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}", value):
        raise SpecError("scaffold version contains unsupported characters")
    return value


def _oragentbench_prebuilt_image_tag(
    *, scaffold: str, scaffold_version: str, dataset_digest: str
) -> str:
    payload = json.dumps(
        {
            "dataset_digest": dataset_digest,
            "scaffold": scaffold,
            "scaffold_version": scaffold_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"orbench-oab-{scaffold}:{digest}"


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
    *, source: Path, job_config: Path
) -> UpstreamCommand:
    """Delegate controls to upstream's pinned oracle runner.

    The script accepts any Harbor config, builds the shared base image unless
    explicitly told not to, and then execs ``harbor run -c``.  We deliberately
    do not pass ``--skip-build``: a one-click run must also work on a fresh
    execution host instead of depending on an unrecorded local image cache.
    """
    source = validate_oragentbench_source(source)
    runner = source / ORAGENTBENCH_ORACLE_RUNNER
    if not runner.is_file():
        raise PreconditionError(
            "the pinned ORAgentBench checkout does not ship its oracle runner"
        )
    return UpstreamCommand(
        argv=("bash", str(runner), "--config", str(Path(job_config).resolve())),
        cwd=str(source.parent),
        required_env=(),
        provenance=(
            "ORAgentBench experiments/scripts/run_oracle_all.sh --config: builds the "
            "shared base image, then execs `harbor run -c <config>` from the "
            "checkout parent"
        ),
        description="Harbor job over ORAgentBench tasks using built-in control agents",
        makes_model_calls=False,
    )


def oragentbench_controls_resume_command(
    *, source: Path, job_dir: Path, expected_image_id: str | None
) -> UpstreamCommand:
    """Restore the exact control base image, then resume one persisted Harbor job.

    Harbor owns the job lock and recovery semantics.  Once ``config.json``
    exists, starting the oracle runner again would invoke a new ``harbor run``
    against the same directory and conflict with that lock.  The only safe
    recovery path is Harbor's documented ``job resume --job-path`` command.
    If trials already exist, the recorded immutable image ID is retagged to the
    fixed alias when local, or the pinned builder must recreate that exact ID.
    If an early interruption created no trial, there is no prior image to bind
    and the pinned builder establishes it. Harbor is not started on a recorded
    image path unless the alias resolves to that ID exactly.
    """
    source = validate_oragentbench_source(source)
    builder = source / ORAGENTBENCH_BASE_IMAGE_BUILDER
    if not builder.is_file():
        raise PreconditionError(
            "the pinned ORAgentBench checkout does not ship its base image builder"
        )
    job_dir = Path(job_dir).resolve()
    config = job_dir / "config.json"
    if config.is_symlink() or not config.is_file():
        raise PreconditionError("the interrupted Harbor job has no safe config.json")
    if expected_image_id is not None and re.fullmatch(
        r"sha256:[0-9a-f]{64}", expected_image_id
    ) is None:
        raise PreconditionError(
            "the interrupted Harbor job has no valid immutable Docker image ID"
        )
    shell = (
        'set -euo pipefail\n'
        'if [ -z "$3" ]; then\n'
        '  bash "$1" "$2"\n'
        'elif docker image inspect "$3" >/dev/null 2>&1; then\n'
        '  docker image tag "$3" "$2"\n'
        'else\n'
        '  bash "$1" "$2"\n'
        'fi\n'
        'actual_id="$(docker image inspect --format \'{{.Id}}\' "$2")"\n'
        'if ! printf \'%s\\n\' "$actual_id" | grep -Eq \'^sha256:[0-9a-f]{64}$\'; then\n'
        '  echo "fixed base image has no valid immutable ID" >&2\n'
        '  exit 1\n'
        'fi\n'
        'if [ -n "$3" ] && [ "$actual_id" != "$3" ]; then\n'
        '  echo "fixed base image identity mismatch" >&2\n'
        '  exit 1\n'
        'fi\n'
        'exec harbor job resume --job-path "$4"'
    )
    return UpstreamCommand(
        argv=(
            "bash",
            "-c",
            shell,
            "orbench-control-resume",
            str(builder),
            ORAGENTBENCH_FIXED_BASE_IMAGE,
            expected_image_id or "",
            str(job_dir),
        ),
        cwd=str(source.parent),
        required_env=(),
        provenance=(
            "Restore the exact recorded ORAgentBench runtime base image "
            "(immutable-ID retag, "
            "or scripts/build_base_image.sh fallback), or build when no trial "
            "existed; verify the fixed alias before official "
            "`harbor job resume --job-path <jobdir>` recovery"
        ),
        description="Resume an interrupted ORAgentBench control job through Harbor",
        makes_model_calls=False,
    )


def oragentbench_agent_command(
    *,
    source: Path,
    job_config: Path,
    python: str = "python3",
    skip_build: bool = False,
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
        # The pinned wrapper's cleanup is campaign-scoped and refuses to run
        # while another Harbor process is active.  A bare --resume can inherit
        # orphan containers or incomplete trial directories after SIGKILL.
        argv.extend(("--resume", "--cleanup-before-resume"))
    if dry_run:
        argv.append("--dry-run")
    return UpstreamCommand(
        argv=tuple(argv),
        cwd=str(source.parent),
        # The credentials live in the job config as ${NAME} placeholders, so the
        # names travel with the command even though none is on its argv.
        required_env=tuple(sorted(set(required_env))),
        # The prebuild wrapper rewrites the agent to an import path under the
        # repository namespace. Harbor is a console script and does not put its
        # current working directory on sys.path, so the checkout parent must be
        # explicit for ``ORAgentBench.harbor_agents`` to resolve.
        env_overrides={"PYTHONPATH": str(source.parent)},
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
    scaffold_version: str,
    model: str,
    seeds: Sequence[int] = (1,),
    wall_clock_sec: int = 2700,
    max_cost_usd: float = 25.0,
    site: str = "local-docker",
    jobs_dir: str = "jobs",
    n_concurrent_trials: int = 1,
    auth_mode: str = "api-key",
    model_base_url: str = "",
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
    validate_pinned_scaffold_version(scaffold_version)

    if auth_mode not in {"api-key", "codex-auth-json"}:
        raise SpecError(
            f"auth_mode {auth_mode!r} must be 'api-key' or 'codex-auth-json'"
        )
    if auth_mode == "codex-auth-json" and scaffold != "codex":
        raise SpecError("codex-auth-json is only valid for the codex scaffold")
    if scaffold in ROUTE_PINNED_SCAFFOLDS:
        # Fail before a campaign workspace is created. The normalized URL is
        # retained only in process memory; its digest is the persisted identity
        # input for API-key campaigns.
        if not model_base_url:
            raise SpecError(
                f"scaffold {scaffold!r} requires a pinned HTTPS provider route"
            )
        model_base_url = validate_https_base_url(model_base_url)
    elif model_base_url:
        model_base_url = validate_https_base_url(model_base_url)

    env_literals: dict[str, str] = {}
    env_from_secret = dict(profile["env_from_secret"])
    if auth_mode == "codex-auth-json":
        model_base_url = validate_https_base_url(model_base_url)
        env_from_secret = {}
        env_literals = {
            "CODEX_FORCE_AUTH_JSON": "true",
            "OPENAI_BASE_URL": model_base_url.strip(),
        }
    else:
        literal_var = profile.get("env_literal_from_model")
        if literal_var:
            env_literals[literal_var] = model

    agent: dict[str, Any] = {
        "id": f"{scaffold}-{model}".lower().replace("/", "-").replace(".", "-"),
        "scaffold": scaffold,
        "model": model,
        "env_from_secret": env_from_secret,
        "env_literals": env_literals,
        "scaffold_version": scaffold_version,
        "agent_kwargs": {**dict(profile["kwargs"]), "version": scaffold_version},
        "setup_timeout_sec": profile.get("setup_timeout_sec"),
    }
    if model_base_url:
        agent["provider_route_digest"] = provider_route_digest(model_base_url)
    if auth_mode != "api-key":
        agent["auth_mode"] = auth_mode

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
            "pre_build": {
                "enabled": True,
                "agent": scaffold,
                "rebuild_base": True,
                # This tag is a pure function of every source/build identity
                # input plus the exact scaffold CLI version.  The post-run
                # receipt additionally records Docker's actual image ID.
                "image_tag": _oragentbench_prebuilt_image_tag(
                    scaffold=scaffold,
                    scaffold_version=scaffold_version,
                    dataset_digest=dataset_digest,
                ),
            },
        },
        "retry": {"max_retries": 0, "include_exceptions": []},
        "metrics": [
            {"type": "uv-script", "script_path": f"{ORAGENTBENCH_CHECKOUT_DIRNAME}/metrics/per_dimension_reward.py"}
        ],
    }


# --------------------------------------------------------------------------- #
# preconditions
# --------------------------------------------------------------------------- #


_HARBOR_BARE_VERSION_RE = re.compile(
    r"(?i)^\s*v?(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][A-Za-z0-9_.-]+)?\s*$"
)
_HARBOR_LABELED_VERSION_RE = re.compile(
    r"(?i)\bharbor\b(?:\s+cli)?\s*,?\s*(?:version\s+)?"
    r"v?(\d+)\.(\d+)(?:\.(\d+))?"
)


def _probe_docker_daemon(
    executable: str,
    *,
    command_runner: CommandRunner | None = None,
) -> tuple[bool, str]:
    """Check that Docker's client can reach a daemon, without logging output."""
    runner = subprocess.run if command_runner is None else command_runner
    argv = [executable, "info"]
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=RUNNER_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"docker info timed out after {RUNNER_PROBE_TIMEOUT_SECONDS:g}s; "
            "the Docker daemon is not confirmed reachable"
        )
    except (OSError, subprocess.SubprocessError):
        return False, "docker info could not run; the Docker daemon is not confirmed reachable"
    if result.returncode != 0:
        return False, "docker info failed; the Docker daemon is not reachable by this runner"
    return True, "docker info succeeded; the Docker daemon is reachable by this runner"


def _parse_harbor_version(output: str) -> tuple[int, int, int] | None:
    match = _HARBOR_BARE_VERSION_RE.fullmatch(output)
    if match is None:
        match = _HARBOR_LABELED_VERSION_RE.search(output)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def _probe_harbor_version(
    executable: str,
    *,
    command_runner: CommandRunner | None = None,
) -> tuple[bool, str]:
    """Run the real CLI and enforce the documented Harbor compatibility floor."""
    runner = subprocess.run if command_runner is None else command_runner
    argv = [executable, "--version"]
    minimum = _format_version(MINIMUM_HARBOR_VERSION)
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=RUNNER_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"harbor --version timed out after {RUNNER_PROBE_TIMEOUT_SECONDS:g}s; "
            f"require Harbor >= {minimum}"
        )
    except (OSError, subprocess.SubprocessError):
        return False, f"harbor --version could not run; require Harbor >= {minimum}"
    if result.returncode != 0:
        return False, f"harbor --version failed; require Harbor >= {minimum}"
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    version = _parse_harbor_version(output)
    if version is None:
        return False, f"Harbor CLI version is not parseable; require Harbor >= {minimum}"
    rendered = _format_version(version)
    if version < MINIMUM_HARBOR_VERSION:
        return False, (
            f"Harbor CLI {rendered} is below the supported minimum Harbor {minimum}"
        )
    return True, f"Harbor CLI {rendered} is compatible (minimum Harbor {minimum})"


def _probe_uv_cli(
    executable: str,
    *,
    command_runner: CommandRunner | None = None,
) -> tuple[bool, str]:
    """Check the executable required by Harbor's compiled uv-script metric."""
    runner = subprocess.run if command_runner is None else command_runner
    argv = [executable, "--version"]
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=RUNNER_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"uv --version timed out after {RUNNER_PROBE_TIMEOUT_SECONDS:g}s; "
            "the uv-script metric cannot run"
        )
    except (OSError, subprocess.SubprocessError):
        return False, "uv --version could not run; the uv-script metric cannot run"
    if result.returncode != 0:
        return False, "uv --version failed; the uv-script metric cannot run"
    return True, "uv CLI is runnable for Harbor's uv-script metric"


def _codex_auth_json_has_basic_structure(path: Path) -> bool:
    """Validate only the non-secret container shape; never return its contents."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload) and all(
        isinstance(key, str) and bool(key) for key in payload
    )


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
    auth_mode: str = "api-key",
    model_base_url: str = "",
    command_runner: CommandRunner | None = None,
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
        normalized_pinned_route = ""
        if scaffold in ROUTE_PINNED_SCAFFOLDS:
            try:
                normalized_pinned_route = validate_https_base_url(model_base_url)
                report.require(True, "a credential-safe HTTPS provider route is pinned")
            except SpecError as exc:
                report.require(False, str(exc))
        try:
            validate_pinned_model(model)
            report.require(True, f"model {model!r} is pinned")
        except SpecError as exc:
            report.require(False, str(exc))
        if auth_mode == "codex-auth-json":
            report.require(
                scaffold == "codex",
                "codex-auth-json is only valid for the codex scaffold",
            )
            # The shared provider-route check above covers this transport too.
            try:
                internet_enabled = oragentbench_task_allows_internet(resolved, task_name)
                report.require(
                    not internet_enabled,
                    "codex-auth-json refuses to expose a long-lived Codex login to an "
                    "internet-enabled benchmark task; use an ephemeral scoped API credential",
                )
            except SpecError as exc:
                report.require(False, str(exc))
            if require_secrets:
                explicit = environ.get("CODEX_AUTH_JSON_PATH")
                auth_path = (
                    Path(explicit).expanduser()
                    if explicit
                    else Path.home() / ".codex" / "auth.json"
                )
                report.require(
                    auth_path.is_file(),
                    f"Codex auth file exists at {auth_path}",
                )
                if auth_path.is_file():
                    report.require(
                        _codex_auth_json_has_basic_structure(auth_path),
                        "Codex auth file is valid JSON with a non-empty object root "
                        "(parsed locally; contents never logged)",
                    )
                    try:
                        mode = stat.S_IMODE(auth_path.stat().st_mode)
                    except OSError:
                        mode = 0o777
                    report.require(
                        mode & 0o077 == 0,
                        f"Codex auth file is private (mode {mode:04o}; require no group/world access)",
                    )
            else:
                report.require(
                    True,
                    "Codex auth file not required for a prepare-only run",
                )
            names: list[str] = []
        else:
            non_secret = set(AGENT_PROFILES[scaffold].get("non_secret_env") or ())
            names = sorted(set(AGENT_PROFILES[scaffold]["env_from_secret"].values()))
        if not require_secrets and auth_mode != "codex-auth-json":
            report.require(
                True,
                "credentials not required: upstream's wrapper dry run transforms the "
                f"config and prints the harbor command without contacting a provider "
                f"(would otherwise need {names})",
            )
            names = []
        for name in names:
            kind = "variable" if name in non_secret else "secret"
            configured = bool(environ.get(name))
            report.require(
                configured,
                f"{kind} {name} is set (value never logged or persisted)",
            )
            if configured and name == "MODEL_BASE_URL":
                try:
                    normalized_runtime_route = validate_https_base_url(environ[name])
                    report.require(
                        True,
                        "provider endpoint uses a credential-safe HTTPS URL",
                    )
                    if (
                        normalized_pinned_route
                        and normalized_runtime_route == normalized_pinned_route
                    ):
                        report.require(
                            True,
                            "runtime provider route matches the pinned campaign route",
                        )
                    else:
                        report.require(
                            False,
                            "runtime provider route does not match the pinned campaign route",
                        )
                except SpecError as exc:
                    # The validator deliberately reports only the violated
                    # policy.  Never echo the URL: it decides where the API
                    # credential will be sent and may itself contain userinfo.
                    report.require(False, str(exc))
    else:
        report.require(
            False,
            f"scaffold {scaffold!r} has no recorded upstream agent profile; known: "
            f"{sorted(AGENT_PROFILES)} plus controls {list(CONTROL_SCAFFOLDS)}",
        )

    if require_harbor:
        harbor = shutil.which("harbor")
        if harbor is None:
            report.require(False, "the Harbor CLI is on PATH on this host")
        else:
            ok, description = _probe_harbor_version(
                harbor, command_runner=command_runner
            )
            report.require(ok, description)
        uv = shutil.which("uv")
        if uv is None:
            report.require(False, "the uv CLI is on PATH for the uv-script metric")
        else:
            ok, description = _probe_uv_cli(uv, command_runner=command_runner)
            report.require(ok, description)
    if require_docker:
        docker = shutil.which("docker")
        if docker is None:
            report.require(False, "docker is on PATH on this host")
        else:
            ok, description = _probe_docker_daemon(
                docker, command_runner=command_runner
            )
            report.require(ok, description)
    return report


def frontieror_preconditions(
    *,
    source: Path,
    environ: Mapping[str, str] | None = None,
    require_docker: bool = True,
    require_perf_isolation: bool = True,
    require_images: bool = True,
    available_images: Iterable[str] | None = None,
    command_runner: CommandRunner | None = None,
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
        docker = shutil.which("docker")
        if docker is None:
            report.require(False, "docker is on PATH on this host")
        else:
            ok, description = _probe_docker_daemon(
                docker, command_runner=command_runner
            )
            report.require(ok, description)
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


def docker_image_fingerprint(
    image: str, *, command_runner: CommandRunner | None = None
) -> dict[str, Any] | None:
    """Read Docker's actual immutable identity for one local image tag."""
    runner = subprocess.run if command_runner is None else command_runner
    try:
        result = runner(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=RUNNER_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        return None
    row = payload[0]
    image_id = row.get("Id")
    repo_digests = row.get("RepoDigests") or []
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        return None
    if not isinstance(repo_digests, list) or not all(
        isinstance(value, str) for value in repo_digests
    ):
        return None
    return {
        "requested_tag": image,
        "image_id": image_id,
        "repo_digests": sorted(set(repo_digests)),
    }


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
        return _redact_sensitive_text(value)
    return value


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_NAME_HINTS)


_ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:" + "|".join(_SECRET_NAME_HINTS) + r")[A-Z0-9_]*)=(\S+)"
)


def _redact_env_assignments(text: str) -> str:
    return _ENV_ASSIGNMENT_RE.sub(r"\1=<redacted>", text)


# Provider clients frequently include an Authorization header or a request URL
# in exception strings.  Mapping-key redaction cannot see those values, so a
# receipt that simply sanitizes dictionaries is not safe enough to publish.
_AUTH_HEADER_RE = re.compile(
    r"(?i)(\bauthorization\s*:\s*(?:bearer|basic)\s+)([^\s,;]+)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|credential)\s*=\s*)([^&\s,;]+)"
)


def _redact_sensitive_text(text: str) -> str:
    """Redact common credential shapes embedded inside log/error strings."""
    text = _redact_env_assignments(text)
    text = _AUTH_HEADER_RE.sub(r"\1<redacted>", text)
    return _SENSITIVE_ASSIGNMENT_RE.sub(r"\1<redacted>", text)


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
