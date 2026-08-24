"""Campaign specs and execution sites.

A campaign spec is a *thin superset* of a Harbor job config, not a second
scheduling language. It adds only what Harbor has no opinion about: which
integration is being exercised, which site may run it, what evidence the
campaign is allowed to claim, and the identity inputs that make run ids stable.
Everything else is compiled straight through to Harbor.

Validation is where the campaign's safety properties live, so it is strict and
it fails closed. A spec that cannot be shown to satisfy a rule is rejected
rather than run with a warning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core.errors import SpecError
from ..core.urls import provider_route_digest, validate_https_base_url
from ..core.evidence import REQUIRED_REPLICAS_FOR_VALIDATED, EvidenceLabel
from ..integrations import registry

SPEC_SCHEMA_VERSION = "1.0"
SUPPORTED_SPEC_VERSIONS = (SPEC_SCHEMA_VERSION,)

#: Agents that make no model calls. Harbor provides both as built-ins.
ZERO_COST_SCAFFOLDS = frozenset({"oracle", "nop"})

#: Only infrastructure and provider failures may be retried. Retrying a
#: capability failure inflates the pass rate, which is a measurement error, not
#: a robustness feature.
RETRYABLE_EXCEPTIONS = frozenset(
    {
        "EnvironmentSetupError",
        "EnvironmentStartError",
        "DockerBuildError",
        "RateLimitError",
        "ApiUsageLimitError",
        "ProviderError",
        "ConnectionError",
        "TimeoutError",
    }
)

#: Model identifiers that float. A campaign pinned to one of these is not
#: reproducible, and its run ids would silently mean different things over time.
FLOATING_MODEL_PATTERNS = (r"latest$", r"^latest", r"\*", r"^auto$")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ORAGENTBENCH_ROUTE_BOUND_SCAFFOLDS = frozenset({"claude-code", "codex"})


@dataclass(frozen=True)
class AgentSpec:
    """Declarative agent identity. Compiles to a Harbor ``AgentConfig``.

    Three separate ways to reach the agent's environment, deliberately kept
    apart so that what is secret is obvious by construction:

    ``env_keys``
        Variable names supplied by an identically-named secret. ``[FOO]``
        compiles to ``FOO: ${FOO}``.
    ``env_from_secret``
        Variable name to *secret name*, for scaffolds whose variable differs
        from the secret. ``{ANTHROPIC_AUTH_TOKEN: MODEL_API_KEY}`` compiles to
        ``ANTHROPIC_AUTH_TOKEN: ${MODEL_API_KEY}``.
    ``env_literals``
        Non-secret literals such as a model name. Validated to reject anything
        that looks like a credential.

    In every case a value that is actually secret appears only as a
    ``${NAME}`` placeholder; the runner substitutes it.
    """

    id: str
    scaffold: str
    scaffold_version: str | None = None
    model: str | None = None
    prompt_digest: str | None = None
    tool_policy: dict[str, Any] = field(default_factory=dict)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    env_keys: tuple[str, ...] = ()
    env_from_secret: dict[str, str] = field(default_factory=dict)
    env_literals: dict[str, str] = field(default_factory=dict)
    #: Explicit non-default credential transport.  ``None`` preserves the
    #: original API-key behavior and its stable identifiers.
    auth_mode: str | None = None
    #: Digest of the normalized provider endpoint. The endpoint itself is
    #: runtime configuration and must not be copied into API-key campaign
    #: artifacts, but changing its credential destination changes identity.
    provider_route_digest: str | None = None
    #: Harbor ``AgentConfig.import_path`` — used by upstream's prebuilt agent
    #: classes, which turn CLI installation into a no-op.
    import_path: str | None = None
    #: Harbor ``AgentConfig.override_setup_timeout_sec``.
    setup_timeout_sec: int | None = None

    @property
    def is_zero_cost(self) -> bool:
        return self.scaffold in ZERO_COST_SCAFFOLDS

    @property
    def secret_names(self) -> tuple[str, ...]:
        """Every secret this agent needs, by name."""
        return tuple(sorted(set(self.env_keys) | set(self.env_from_secret.values())))


@dataclass(frozen=True)
class Site:
    """An execution site. Sites are declarations, not a scheduler."""

    name: str
    perf_isolated: bool = False
    cpus: int | None = None
    memory_mb: int | None = None
    solver_license_slots: int = 0
    runner_labels: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_file(cls, path: Path) -> "Site":
        data = _load_yaml_mapping(path, what="site")
        try:
            return cls(
                name=str(data["name"]),
                perf_isolated=bool(data.get("perf_isolated", False)),
                cpus=_opt_int(data.get("cpus")),
                memory_mb=_opt_int(data.get("memory_mb")),
                solver_license_slots=_as_int(
                    data.get("solver_license_slots", 0),
                    field=f"{path}: solver_license_slots",
                ),
                runner_labels=tuple(str(x) for x in data.get("runner_labels", [])),
                notes=str(data.get("notes", "")),
            )
        except KeyError as exc:
            raise SpecError(f"{path}: site is missing required key {exc}") from None


@dataclass(frozen=True)
class CampaignSpec:
    """A validated campaign spec."""

    slug: str
    date: str
    integration: str
    site: str
    evidence_intent: EvidenceLabel
    dataset_path: str
    dataset_digest: str
    tasks: tuple[str, ...]
    agents: tuple[AgentSpec, ...]
    budget: dict[str, Any]
    seeds: tuple[int, ...]
    attempts: int
    shards: int
    harbor: dict[str, Any]
    retry: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    durability: dict[str, Any]
    raw: dict[str, Any]
    source_path: str | None = None

    @property
    def makes_model_calls(self) -> bool:
        return any(not agent.is_zero_cost for agent in self.agents)

    @property
    def n_planned_runs(self) -> int:
        return len(self.tasks) * len(self.agents) * len(self.seeds) * self.attempts


def load_spec(path: str | Path) -> dict[str, Any]:
    return _load_yaml_mapping(Path(path), what="campaign spec")


def _load_yaml_mapping(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpecError(f"{what} not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise SpecError(f"{path}: {what} must be a YAML mapping, got {type(data).__name__}")
    return data


def _opt_int(value: Any) -> int | None:
    return None if value is None else _as_int(value, field="site cpus/memory_mb")


def _as_int(value: Any, *, field: str) -> int:
    """Coerce a YAML scalar to an integer with an actionable error."""
    if isinstance(value, bool):
        raise SpecError(f"{field} must be an integer, got a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise SpecError(f"{field} must be an integer, got a non-integral number")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise SpecError(f"{field} must be an integer") from None


def _as_float(value: Any, *, field: str) -> float:
    """Coerce a YAML scalar to a finite number with an actionable error."""
    if isinstance(value, bool):
        raise SpecError(f"{field} must be a number, got a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise SpecError(f"{field} must be a number") from None
    if not math.isfinite(parsed):
        raise SpecError(f"{field} must be a finite number")
    return parsed


def validate(
    raw: Mapping[str, Any],
    *,
    sites_dir: str | Path = "sites",
    source_path: str | Path | None = None,
) -> CampaignSpec:
    """Validate a raw spec mapping and return a ``CampaignSpec``.

    Raises :class:`SpecError` with a single actionable message on the first
    violated rule.
    """
    problems: list[str] = []

    version = str(raw.get("schema_version", ""))
    if version not in SUPPORTED_SPEC_VERSIONS:
        raise SpecError(
            f"unsupported schema_version {version!r}; supported: {list(SUPPORTED_SPEC_VERSIONS)}"
        )

    slug = str(raw.get("slug", ""))
    if not _SLUG_RE.match(slug):
        problems.append(
            f"slug {slug!r} must match {_SLUG_RE.pattern} (lowercase, digits, hyphens)"
        )

    date = str(raw.get("date", ""))
    if not _DATE_RE.match(date):
        problems.append(
            "date must be an explicit YYYY-MM-DD string; campaign ids are derived from it "
            "and must not depend on the wall clock at compile time"
        )

    integration_name = str(raw.get("integration", ""))
    integration_desc: dict[str, Any] = {}
    if integration_name not in registry.names():
        problems.append(
            f"unknown integration {integration_name!r}; known: {registry.names()}"
        )
    else:
        integration_desc = registry.describe(integration_name)

    evidence_intent_raw = str(raw.get("evidence_intent", "exploratory"))
    try:
        evidence_intent = EvidenceLabel(evidence_intent_raw)
    except ValueError:
        problems.append(
            f"evidence_intent {evidence_intent_raw!r} must be one of "
            f"{[label.value for label in EvidenceLabel]}"
        )
        evidence_intent = EvidenceLabel.EXPLORATORY

    site_name = str(raw.get("site", ""))
    site = _resolve_site(site_name, sites_dir, problems)

    dataset = raw.get("dataset") or {}
    dataset_path = str(dataset.get("path", ""))
    dataset_digest = str(dataset.get("digest", ""))
    if not dataset_path:
        problems.append("dataset.path is required")
    if not _SHA256_RE.match(dataset_digest):
        problems.append(
            "dataset.digest must be 'sha256:<64 hex>' as reported by "
            "'orbench integration inspect'; run ids are pinned to it"
        )

    tasks = tuple(str(t) for t in (raw.get("tasks") or []))
    if not tasks:
        problems.append("tasks must list at least one task")
    if len(set(tasks)) != len(tasks):
        problems.append("tasks contains duplicates")

    agents = _parse_agents(
        raw.get("agents") or [], problems, integration_name=integration_name
    )

    raw_seeds = raw.get("seeds")
    if raw_seeds is not None and not isinstance(raw_seeds, (list, tuple)):
        raise SpecError("seeds must be a list of integers")
    if isinstance(raw_seeds, (list, tuple)) and not raw_seeds:
        problems.append("seeds must list at least one seed")
    seeds = tuple(
        _as_int(seed, field=f"seeds[{index}]")
        for index, seed in enumerate(raw_seeds or [1])
    )
    if len(set(seeds)) != len(seeds):
        problems.append("seeds contains duplicates")

    attempts = _as_int(raw.get("attempts", 1), field="attempts")
    if attempts != 1:
        problems.append(
            "attempts must be 1: a Harbor job carries exactly one (agent, seed, attempt) "
            "combination so that (job_name, task_name) identifies a ledger entry exactly. "
            "Express repetitions as additional seeds."
        )

    shards = _as_int(raw.get("shards", 1), field="shards")
    if shards < 1:
        problems.append("shards must be >= 1")

    budget = dict(raw.get("budget") or {})
    _validate_budget(budget, agents, problems)

    retry = dict(raw.get("retry") or {"max_retries": 0})
    _validate_retry(retry, problems)

    durability = dict(raw.get("durability") or {})
    _validate_evidence_intent(evidence_intent, durability, problems)

    if site is not None and integration_desc:
        _validate_site_capability(site, integration_desc, problems)

    metrics = tuple(dict(m) for m in (raw.get("metrics") or []))
    harbor = dict(raw.get("harbor") or {})
    if _as_int(harbor.get("n_attempts", 1), field="harbor.n_attempts") != 1:
        problems.append("harbor.n_attempts must be 1 (see attempts)")

    if problems:
        bullet_list = "\n".join(f"  - {problem}" for problem in problems)
        raise SpecError(f"campaign spec is invalid ({len(problems)} problem(s)):\n{bullet_list}")

    return CampaignSpec(
        slug=slug,
        date=date,
        integration=integration_name,
        site=site_name,
        evidence_intent=evidence_intent,
        dataset_path=dataset_path,
        dataset_digest=dataset_digest,
        tasks=tasks,
        agents=agents,
        budget=budget,
        seeds=seeds,
        attempts=attempts,
        shards=shards,
        harbor=harbor,
        retry=retry,
        metrics=metrics,
        durability=durability,
        raw=dict(raw),
        source_path=str(source_path) if source_path else None,
    )


def _resolve_site(site_name: str, sites_dir: str | Path, problems: list[str]) -> Site | None:
    if not site_name:
        problems.append("site is required")
        return None
    site_path = Path(sites_dir) / f"{site_name}.yaml"
    if not site_path.is_file():
        problems.append(
            f"site {site_name!r} not declared: {site_path} does not exist. "
            "Sites must be declared before a campaign may target them."
        )
        return None
    try:
        return Site.from_file(site_path)
    except SpecError as exc:
        problems.append(str(exc))
        return None


def _parse_agents(
    raw_agents: Any, problems: list[str], *, integration_name: str
) -> tuple[AgentSpec, ...]:
    if not raw_agents:
        problems.append("agents must list at least one agent")
        return ()
    agents: list[AgentSpec] = []
    seen_ids: set[str] = set()
    for index, raw_agent in enumerate(raw_agents):
        if not isinstance(raw_agent, Mapping):
            problems.append(f"agents[{index}] must be a mapping")
            continue
        agent_id = str(raw_agent.get("id", ""))
        scaffold = str(raw_agent.get("scaffold", ""))
        if not agent_id:
            problems.append(f"agents[{index}].id is required")
        elif not _AGENT_ID_RE.match(agent_id):
            problems.append(
                f"agents[{index}].id {agent_id!r} must match {_AGENT_ID_RE.pattern}; "
                "it becomes part of a job filename, so traversal, separators and "
                "whitespace are rejected"
            )
        if agent_id in seen_ids:
            problems.append(f"agents[{index}].id {agent_id!r} is duplicated")
        seen_ids.add(agent_id)
        if not scaffold:
            problems.append(f"agents[{index}].scaffold is required")

        model = raw_agent.get("model")
        scaffold_version_value = raw_agent.get("scaffold_version")
        scaffold_version = (
            str(scaffold_version_value).strip()
            if scaffold_version_value is not None
            else None
        )
        raw_auth_mode = raw_agent.get("auth_mode")
        auth_mode = str(raw_auth_mode) if raw_auth_mode is not None else None
        if auth_mode not in (None, "api-key", "codex-auth-json", "codex-login"):
            problems.append(
                f"agents[{index}].auth_mode {auth_mode!r} must be 'api-key', "
                "'codex-auth-json', or 'codex-login'"
            )
        if scaffold not in ZERO_COST_SCAFFOLDS:
            if not model:
                problems.append(
                    f"agents[{index}] ({agent_id!r}) uses scaffold {scaffold!r}, which makes "
                    "model calls, so model must be pinned to an exact identifier"
                )
            elif any(re.search(pattern, str(model)) for pattern in FLOATING_MODEL_PATTERNS):
                problems.append(
                    f"agents[{index}].model {model!r} is a floating alias; pin an exact "
                    "model identifier so run ids keep meaning the same thing"
                )
            if not (
                raw_agent.get("env_keys")
                or raw_agent.get("env_from_secret")
                or auth_mode in {"codex-auth-json", "codex-login"}
            ):
                problems.append(
                    f"agents[{index}] ({agent_id!r}) must declare env_keys or env_from_secret "
                    "(names only) so the required secrets are auditable before the campaign runs"
                )
            if integration_name == "oragentbench":
                if not scaffold_version:
                    problems.append(
                        f"agents[{index}].scaffold_version is required for ORAgentBench "
                        "model agents so the baked CLI is an identity input"
                    )
                elif any(
                    re.search(pattern, scaffold_version)
                    for pattern in FLOATING_MODEL_PATTERNS
                ) or scaffold_version.lower() in {"stable", "main", "master"}:
                    problems.append(
                        f"agents[{index}].scaffold_version {scaffold_version!r} is floating; "
                        "pin an exact released CLI version"
                    )

        env_keys = tuple(str(k) for k in (raw_agent.get("env_keys") or []))
        if any("=" in key for key in env_keys):
            problems.append(
                f"agents[{index}].env_keys must contain variable names only, never values"
            )
        malformed_keys = [key for key in env_keys if not _ENV_NAME_RE.match(key)]
        if malformed_keys:
            problems.append(
                f"agents[{index}].env_keys contains invalid environment names {malformed_keys}"
            )

        env_from_secret = {
            str(k): str(v) for k, v in (raw_agent.get("env_from_secret") or {}).items()
        }
        for variable, secret in env_from_secret.items():
            if not _ENV_NAME_RE.match(variable) or not _ENV_NAME_RE.match(secret):
                problems.append(
                    f"agents[{index}].env_from_secret maps {variable!r} -> {secret!r}; both "
                    "sides must be environment variable NAMES, never values"
                )

        env_literals = {str(k): str(v) for k, v in (raw_agent.get("env_literals") or {}).items()}
        for variable, value in env_literals.items():
            if not _ENV_NAME_RE.match(variable):
                problems.append(f"agents[{index}].env_literals key {variable!r} is not a variable name")
            if _looks_like_credential(variable, value):
                problems.append(
                    f"agents[{index}].env_literals[{variable!r}] looks like a credential. "
                    "Literals are written into the job config verbatim; route secrets "
                    "through env_keys or env_from_secret instead."
                )
        shadowed = sorted(set(env_literals) & (set(env_keys) | set(env_from_secret)))
        if shadowed:
            problems.append(
                f"agents[{index}].env_literals {shadowed} would overwrite a declared "
                "secret placeholder; declare each variable in exactly one place"
            )

        route_digest_value = raw_agent.get("provider_route_digest")
        route_digest = str(route_digest_value) if route_digest_value is not None else None
        if route_digest is not None and not _SHA256_RE.match(route_digest):
            problems.append(
                f"agents[{index}].provider_route_digest must be 'sha256:<64 hex>'"
            )

        if auth_mode in {"codex-auth-json", "codex-login"}:
            if scaffold != "codex":
                problems.append(
                    f"agents[{index}].auth_mode {auth_mode!r} is only valid for the "
                    "codex scaffold"
                )
            if env_keys or env_from_secret:
                problems.append(
                    f"agents[{index}] uses {auth_mode} and must not also declare "
                    "API-key secrets"
                )
            if env_literals.get("CODEX_FORCE_AUTH_JSON") != "true":
                problems.append(
                    f"agents[{index}] uses {auth_mode}, so env_literals must set "
                    "CODEX_FORCE_AUTH_JSON to the literal 'true'"
                )
            if auth_mode == "codex-auth-json" and not env_literals.get(
                "OPENAI_BASE_URL"
            ):
                problems.append(
                    f"agents[{index}] uses codex-auth-json, so env_literals must pin "
                    "OPENAI_BASE_URL"
                )
            elif auth_mode == "codex-login" and (
                env_literals.get("OPENAI_BASE_URL") or route_digest is not None
            ):
                problems.append(
                    f"agents[{index}] uses codex-login's native ChatGPT route and "
                    "must not declare OPENAI_BASE_URL or provider_route_digest"
                )
            elif env_literals.get("OPENAI_BASE_URL"):
                try:
                    validated_url = validate_https_base_url(env_literals["OPENAI_BASE_URL"])
                    derived_digest = provider_route_digest(validated_url)
                    if route_digest is None:
                        route_digest = derived_digest
                    elif route_digest != derived_digest:
                        problems.append(
                            f"agents[{index}].provider_route_digest does not match "
                            "the normalized OPENAI_BASE_URL"
                        )
                except SpecError as exc:
                    problems.append(f"agents[{index}].OPENAI_BASE_URL: {exc}")
        elif scaffold in ZERO_COST_SCAFFOLDS and auth_mode is not None:
            problems.append(
                f"agents[{index}] is a zero-cost control and may not declare auth_mode"
            )

        if (
            integration_name == "oragentbench"
            and scaffold in _ORAGENTBENCH_ROUTE_BOUND_SCAFFOLDS
            and auth_mode != "codex-login"
            and route_digest is None
        ):
            problems.append(
                f"agents[{index}] ({agent_id!r}) must pin provider_route_digest; "
                "the credential destination is part of ORAgentBench agent identity"
            )

        agents.append(
            AgentSpec(
                id=agent_id,
                scaffold=scaffold,
                scaffold_version=scaffold_version,
                model=str(model) if model else None,
                prompt_digest=(
                    str(raw_agent["prompt_digest"]) if raw_agent.get("prompt_digest") else None
                ),
                tool_policy=dict(raw_agent.get("tool_policy") or {}),
                agent_kwargs=dict(raw_agent.get("agent_kwargs") or {}),
                env_keys=env_keys,
                env_from_secret=env_from_secret,
                env_literals=env_literals,
                auth_mode=auth_mode,
                provider_route_digest=route_digest,
                import_path=(
                    str(raw_agent["import_path"]) if raw_agent.get("import_path") else None
                ),
                setup_timeout_sec=(
                    _as_int(
                        raw_agent["setup_timeout_sec"],
                        field=f"agents[{index}].setup_timeout_sec",
                    )
                    if raw_agent.get("setup_timeout_sec") is not None
                    else None
                ),
            )
        )
    return tuple(agents)


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Variable names whose *literal* value would almost certainly be a credential.
_CREDENTIAL_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

#: Value shapes that are credentials regardless of the variable's name.
_CREDENTIAL_VALUE_RE = re.compile(r"^(sk-|ghp_|gho_|github_pat_|xox[abp]-|AKIA[0-9A-Z]{16})")


def _looks_like_credential(variable: str, value: str) -> bool:
    """Reject a literal that is really a secret.

    A `${NAME}` placeholder is fine — that is a reference, not a value.
    """
    if value.startswith("${") and value.endswith("}"):
        return False
    if _CREDENTIAL_VALUE_RE.match(value):
        return True
    upper = variable.upper()
    return any(hint in upper for hint in _CREDENTIAL_NAME_HINTS)


def _validate_budget(
    budget: dict[str, Any], agents: tuple[AgentSpec, ...], problems: list[str]
) -> None:
    if "wall_clock_sec" not in budget:
        problems.append("budget.wall_clock_sec is required")
    else:
        wall_clock = _as_int(budget["wall_clock_sec"], field="budget.wall_clock_sec")
        if wall_clock < 1:
            problems.append("budget.wall_clock_sec must be >= 1")
    max_cost = budget.get("max_cost_usd")
    if max_cost is None:
        problems.append("budget.max_cost_usd is required (use 0 for zero-cost control campaigns)")
        return
    max_cost = _as_float(max_cost, field="budget.max_cost_usd")
    if max_cost < 0:
        problems.append("budget.max_cost_usd must be >= 0")
    makes_calls = any(not agent.is_zero_cost for agent in agents)
    if makes_calls and max_cost == 0:
        problems.append(
            "budget.max_cost_usd is 0 but the campaign includes model-calling agents; "
            "set a real ceiling. Note that this is a compile-time estimate gate, not a "
            "runtime circuit breaker — only the provider-side key limit can stop spend."
        )
    if not makes_calls and max_cost > 0:
        problems.append(
            "budget.max_cost_usd > 0 but every agent is a zero-cost control "
            "(oracle/nop); set it to 0 so the intent is unambiguous"
        )


def _validate_retry(retry: Mapping[str, Any], problems: list[str]) -> None:
    def exception_set(field: str) -> list[str]:
        if field not in retry:
            return []
        value = retry[field]
        if not isinstance(value, list):
            problems.append(f"retry.{field} must be a list")
            return []
        if not all(isinstance(item, str) for item in value):
            problems.append(f"retry.{field} must contain only strings")
            return []
        normalized = list(value)
        if len(set(normalized)) != len(normalized):
            problems.append(f"retry.{field} must not contain duplicates")
        return normalized

    include = exception_set("include_exceptions")
    exception_set("exclude_exceptions")
    disallowed = sorted(set(include) - RETRYABLE_EXCEPTIONS)
    if disallowed and all(isinstance(value, str) for value in include):
        problems.append(
            f"retry.include_exceptions contains capability-class exceptions {disallowed}; "
            f"only infrastructure and provider failures may be retried "
            f"({sorted(RETRYABLE_EXCEPTIONS)}). Retrying a capability failure inflates "
            "the pass rate."
        )
    max_retries = _as_int(retry.get("max_retries", 0), field="retry.max_retries")
    if max_retries > 0 and not include:
        problems.append(
            "retry.max_retries > 0 requires an explicit retry.include_exceptions allowlist; "
            "blanket retries silently change what is being measured"
        )


def _validate_evidence_intent(
    intent: EvidenceLabel, durability: Mapping[str, Any], problems: list[str]
) -> None:
    if intent is not EvidenceLabel.VALIDATED:
        return
    replicas = _as_int(durability.get("replicas", 0), field="durability.replicas")
    if replicas < REQUIRED_REPLICAS_FOR_VALIDATED:
        problems.append(
            f"evidence_intent 'validated' requires durability.replicas >= "
            f"{REQUIRED_REPLICAS_FOR_VALIDATED} with verified content-addressed copies. "
            "Durability verification is not implemented in this release, so a validated "
            "campaign cannot be planned yet — plan it as 'partial'."
        )
        return
    problems.append(
        "evidence_intent 'validated' additionally requires 'orbench durability verify', "
        "which is not implemented in this release; plan the campaign as 'partial'."
    )


def _validate_site_capability(
    site: Site, integration_desc: Mapping[str, Any], problems: list[str]
) -> None:
    if integration_desc.get("performance_scored") and not site.perf_isolated:
        problems.append(
            f"integration {integration_desc['name']!r} scores runtime, but site "
            f"{site.name!r} declares perf_isolated: false. Performance-scored benchmarks "
            "may only run on a performance-isolated site with pinned cores; otherwise the "
            "number is not comparable to anything."
        )
    requires = integration_desc.get("requires") or {}
    if requires.get("solver_license") and site.solver_license_slots < 1:
        problems.append(
            f"integration {integration_desc['name']!r} requires a solver licence, but site "
            f"{site.name!r} declares solver_license_slots: 0"
        )
