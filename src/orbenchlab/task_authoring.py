"""Fail-closed checks for turning a paper-backed idea into a TB-Science task.

This module is intentionally an *authoring gate*, not an LLM reviewer.  It
checks the parts that can be checked locally (task.toml shape, required task
files, verifier isolation, CTRF wiring, security hazards, provenance and
resource bounds) and records the remaining semantic rubric criteria as
``review``.  A task can therefore be generated and iterated without ever being
mistaken for a task that TB-Science has approved.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .core.errors import ORBenchError


class TaskAuthoringError(ORBenchError):
    """Invalid authoring input."""

    exit_code = 8


AUTHORING_SCHEMA_VERSION = "orbenchlab.tbscience-authoring.v1"
RUBRIC_COMMIT = "9e22cdb7f3201eb2a3f4f06164938d9a8bb39df1"
RUBRIC_SOURCE = f"https://github.com/harbor-framework/terminal-bench-science/blob/{RUBRIC_COMMIT}/rubrics/task-implementation.toml"
TASK_TEMPLATE_SOURCE = f"https://github.com/harbor-framework/terminal-bench-science/blob/{RUBRIC_COMMIT}/task-template.toml"

# The names are copied from the current TB-Science implementation rubric.  The
# list is kept here so a receipt makes rubric coverage explicit even when a
# criterion needs a human or rubric-agent decision.
IMPLEMENTATION_CRITERIA = (
    "verifiable",
    "well_specified",
    "solvable",
    "difficult",
    "scientifically_grounded",
    "scope",
    "outcome_verified",
    "anti_cheat_robustness",
    "task_security",
    "functional_verification",
    "ctrf_reporting",
    "ground_truth_provenance",
    "graded_instances_discriminate",
    "deterministic_reproducible",
    "essential_difficulty",
    "test_instruction_alignment",
    "do_not_modify_enforced",
    "novel",
    "agentic",
    "reviewable",
    "instruction_clarity",
    "solution_quality",
    "separate_verifier_configured",
    "verifier_execution_isolation",
    "artifact_efficiency",
    "environment_hygiene",
    "structured_data_schema",
    "typos",
    "difficulty_explanation_quality",
    "solution_explanation_quality",
    "verification_explanation_quality",
    "category_and_tags",
    "task_name",
    "resource_configuration",
    "task_readme",
    "task_authoring_dir",
    "expert_time_estimate",
    "task_toml_schema",
    "no_extraneous_files",
)

PROPOSAL_CRITERIA = (
    "verifiable",
    "well_specified",
    "solvable",
    "difficult",
    "scientifically_grounded",
    "scope",
    "outcome_verified",
)

_ALLOWED_ROOT_FILES = frozenset(
    {
        "task.toml",
        "instruction.md",
        "README.md",
        "Dockerfile",
        ".gitignore",
        "rubric-receipt.json",
        "paper-provenance.json",
    }
)
_ALLOWED_ROOT_DIRS = frozenset({"environment", "solution", "tests", "data", "assets"})
_ALLOWED_TOML_KEYS = {
    "": {"schema_version", "artifacts"},
    "task": {"name", "description", "authors", "keywords", "version"},
    "metadata": {
        "author_name", "author_email", "author_organization", "author_profile",
        "research_advisor", "referred_by", "domain", "field", "subfield", "tags",
        "expert_time_estimate_hours", "relevant_experience", "conflicts_of_interest",
    },
    "verifier": {"timeout_sec", "environment_mode", "environment"},
    "verifier.environment": {"network_mode", "allowed_hosts"},
    "agent": {"timeout_sec"},
    "environment": {"build_timeout_sec", "cpus", "memory_mb", "storage_mb", "gpus", "gpu_types", "validate_env", "network_mode"},
}
_TOML_SECTION_NAMES = frozenset({"task", "metadata", "verifier", "agent", "environment"})
_DANGEROUS_PATTERNS = (
    (re.compile(r"~/(?:\.ssh|\.aws|\.config/gh|ssh|aws|config/gh)|/root/\.ssh|/\.aws", re.I), "credential-store access"),
    (re.compile(r"/var/run/docker\.sock|docker\s+run\s+--privileged", re.I), "host/container escape"),
    (re.compile(r"rm\s+-rf\s+(?:/|/root|/home|\$HOME)", re.I), "destructive filesystem operation"),
    (re.compile(r"(?:base64\s+-d|eval\s*\(|exec\s*\()", re.I), "obfuscated or dynamic execution"),
    (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions|exfiltrat", re.I), "prompt injection or exfiltration"),
)


@dataclass(frozen=True)
class Criterion:
    name: str
    status: str
    reason: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _load_document(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise TaskAuthoringError(f"authoring input not found: {path}")
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise TaskAuthoringError(f"unsupported authoring input format: {path}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TaskAuthoringError(f"cannot read authoring input {path}: {type(exc).__name__}") from None
    if not isinstance(value, Mapping):
        raise TaskAuthoringError(f"authoring input must be an object: {path}")
    return value


def _criterion(name: str, status: str, reason: str, *evidence: str) -> Criterion:
    return Criterion(name=name, status=status, reason=reason, evidence=tuple(evidence))


def _section_text(readme: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", readme, re.I | re.M)
    if not match:
        return ""
    tail = readme[match.end() :]
    next_heading = re.search(r"^##\s+", tail, re.M)
    return (tail[: next_heading.start()] if next_heading else tail).strip()


def _files_under(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _task_tree_digest(root: Path) -> str:
    """Hash task contents using relative paths, never checkout-specific paths."""

    entries: list[dict[str, Any]] = []
    for path in _files_under(root):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "content_digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return _digest(entries)


def _paper_provenance(path: Path | None) -> tuple[dict[str, Any], list[Criterion]]:
    if path is None:
        return {}, [_criterion("paper_provenance", "fail", "paper provenance input is missing")]
    try:
        doc = dict(_load_document(path))
    except TaskAuthoringError as exc:
        return {}, [_criterion("paper_provenance", "fail", str(exc))]
    required = ("title", "url", "source_content_digest", "license_status")
    missing = [key for key in required if not str(doc.get(key, "")).strip()]
    if missing:
        return doc, [_criterion("paper_provenance", "fail", f"missing {', '.join(missing)}", path.name)]
    if not str(doc["url"]).startswith(("https://", "http://")):
        return doc, [_criterion("paper_provenance", "fail", "paper url must be http(s)", path.name)]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(doc["source_content_digest"])):
        return doc, [_criterion("paper_provenance", "fail", "source_content_digest must be sha256:<64 lowercase hex>", path.name)]
    source_binding = False
    source_path = doc.get("source_path")
    if source_path:
        source = Path(str(source_path))
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file() or source.is_symlink():
            return doc, [_criterion("paper_provenance", "fail", "source_path does not resolve to a regular local file", path.name)]
        actual = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != str(doc["source_content_digest"]):
            return doc, [_criterion("paper_provenance", "fail", "source_content_digest does not match source_path bytes", path.name)]
        source_binding = True
        # Do not put a machine-specific absolute path into the receipt digest.
        doc["source_path"] = source.name
    status = str(doc["license_status"]).lower().replace("-", "_")
    if status != "registry_resolved" or not source_binding:
        return doc, [_criterion("paper_provenance", "review", "paper/license provenance is caller- or human-asserted; local source binding and registry resolution are still required", path.name)]
    return doc, [_criterion("paper_provenance", "pass", "paper title, URL, content digest and license status are recorded", path.name)]


def _previous_receipt(
    path: Path | None, *, task_dir: Path, round_number: int
) -> tuple[str | None, list[Criterion]]:
    """Validate the optional prior-round receipt before linking it."""

    if path is None:
        if round_number > 1:
            return None, [
                _criterion(
                    "previous_receipt",
                    "fail",
                    "rounds after 1 must provide --previous",
                )
            ]
        return None, []
    if not path.is_file() or path.is_symlink():
        return None, [_criterion("previous_receipt", "fail", "previous receipt is not a regular file", path.name)]
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, [_criterion("previous_receipt", "fail", "previous receipt is not valid UTF-8 JSON", path.name)]
    if not isinstance(previous, Mapping):
        return None, [_criterion("previous_receipt", "fail", "previous receipt must be a JSON object", path.name)]
    problems: list[str] = []
    if previous.get("authoring_schema_version") != AUTHORING_SCHEMA_VERSION:
        problems.append("schema version mismatch")
    if previous.get("task_dir") != task_dir.name:
        problems.append("task directory mismatch")
    previous_round = previous.get("round")
    if not isinstance(previous_round, int) or isinstance(previous_round, bool) or previous_round >= round_number:
        problems.append("previous round must be an integer smaller than the current round")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(previous.get("task_tree_digest", ""))):
        problems.append("previous task_tree_digest is missing or malformed")
    recorded_digest = str(previous.get("receipt_digest", ""))
    unsigned = {key: value for key, value in previous.items() if key != "receipt_digest"}
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", recorded_digest) or _digest(unsigned) != recorded_digest:
        problems.append("previous receipt digest does not match its contents")
    if problems:
        return None, [_criterion("previous_receipt", "fail", "; ".join(problems), path.name)]
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), [
        _criterion("previous_receipt", "pass", "previous receipt is schema-, task-, round- and digest-valid", path.name)
    ]


def _security_criteria(task_dir: Path) -> list[Criterion]:
    findings: list[str] = []
    for path in _files_under(task_dir):
        if path.stat().st_size > 2_000_000:
            continue
        text = _read_text(path)
        for pattern, label in _DANGEROUS_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(task_dir)}: {label}")
    if findings:
        return [_criterion("task_security", "fail", "; ".join(sorted(set(findings))))]
    return [_criterion("task_security", "pass", "no known credential, escape, destructive or injection pattern found")]


def _mechanical_checks(task_dir: Path, config: Mapping[str, Any], readme: str) -> list[Criterion]:
    task = config.get("task") if isinstance(config.get("task"), Mapping) else {}
    metadata = config.get("metadata") if isinstance(config.get("metadata"), Mapping) else {}
    verifier = config.get("verifier") if isinstance(config.get("verifier"), Mapping) else {}
    agent = config.get("agent") if isinstance(config.get("agent"), Mapping) else {}
    environment = config.get("environment") if isinstance(config.get("environment"), Mapping) else {}
    task_name = str(task.get("name", ""))
    description = str(task.get("description", ""))
    slug = task_name.removeprefix("terminal-bench-science/")
    slug_ok = bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug)) and len(slug.split("-")) <= 3
    required_sections = all(isinstance(config.get(key), Mapping) for key in ("task", "metadata", "verifier", "agent", "environment"))
    test_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    environment_dir = task_dir / "environment"
    instruction = task_dir / "instruction.md"
    readme_path = task_dir / "README.md"
    test_script = test_dir / "test.sh"
    tests_dockerfile = test_dir / "Dockerfile"
    env_dockerfile = environment_dir / "Dockerfile"
    test_text = _read_text(test_script)
    instruction_text = _read_text(instruction)

    checks: list[Criterion] = []
    unknown_keys: list[str] = []
    for section, allowed in _ALLOWED_TOML_KEYS.items():
        value: Mapping[str, Any] = (
            {key: value for key, value in config.items() if key not in _TOML_SECTION_NAMES}
            if section == ""
            else config.get(section.split(".")[0], {})
        )
        if section == "verifier.environment":
            verifier_value = config.get("verifier", {})
            value = verifier_value.get("environment", {}) if isinstance(verifier_value, Mapping) else {}
        if isinstance(value, Mapping):
            unknown_keys.extend(
                f"{section or 'root'}.{key}" for key in value if key not in allowed
            )
    schema_ok = required_sections and bool(config.get("schema_version")) and not unknown_keys
    schema_reason = (
        "required task.toml sections, schema_version and only template keys are present"
        if schema_ok
        else "task.toml is missing required sections/schema_version or contains unknown keys: "
        + ", ".join(sorted(unknown_keys))
    )
    checks.append(_criterion("task_toml_schema", "pass" if schema_ok else "fail", schema_reason, "task.toml"))
    name_ok = task_name.startswith("terminal-bench-science/") and slug_ok and task_dir.name == slug
    checks.append(_criterion("task_name", "pass" if name_ok else "fail", "task name uses the TB-Science namespace, matches the directory and has <=3 kebab-case words" if name_ok else "task.name must be terminal-bench-science/<kebab-slug> with at most 3 words and match the task directory name", "task.toml"))
    checks.append(_criterion("well_specified", "pass" if description.strip() and instruction_text.strip() else "fail", "description and instruction.md are present" if description.strip() and instruction_text.strip() else "task description and instruction.md are required", "task.toml", "instruction.md"))
    checks.append(_criterion("task_readme", "pass" if readme_path.is_file() else "review", "README.md is present" if readme_path.is_file() else "README.md is optional in TB-Science; reviewer should confirm the task has sufficient development context", "README.md"))
    solution_ok = solution_dir.is_dir() and any(solution_dir.iterdir())
    checks.append(_criterion("solution_quality", "pass" if solution_ok else "fail", "solution directory is non-empty" if solution_ok else "solution/ must contain a reference solution", "solution"))
    checks.append(_criterion("verifiable", "pass" if test_script.is_file() and tests_dockerfile.is_file() else "fail", "tests/test.sh and tests/Dockerfile are present" if test_script.is_file() and tests_dockerfile.is_file() else "tests/test.sh and tests/Dockerfile are required", "tests/test.sh", "tests/Dockerfile"))
    executes_tests = bool(re.search(r"(?m)^\s*(?:pytest|python(?:3)?\b|bash\b)\s+", test_text))
    checks.append(_criterion("functional_verification", "pass" if executes_tests else "fail", "test script executes a program/test runner" if executes_tests else "test.sh appears not to execute a test or verifier", "tests/test.sh"))
    has_pytest = "pytest" in test_text
    ctrf_ok = bool(re.search(r"--ctrf(?:\s+|=).*(?:/logs/verifier/ctrf\.json|ctrf\.json)", test_text))
    ctrf_status = "pass" if (has_pytest and ctrf_ok) else "fail" if has_pytest else "review"
    ctrf_reason = (
        "pytest verifier emits /logs/verifier/ctrf.json"
        if has_pytest and ctrf_ok
        else "pytest verifier must emit /logs/verifier/ctrf.json"
        if has_pytest
        else "no pytest invocation; reviewer must confirm CTRF is genuinely N/A for a one-shot verifier"
    )
    checks.append(_criterion("ctrf_reporting", ctrf_status, ctrf_reason, "tests/test.sh"))
    separate = str(verifier.get("environment_mode", "")) == "separate"
    checks.append(_criterion("separate_verifier_configured", "pass" if separate else "fail", "verifier.environment_mode=separate" if separate else "verifier.environment_mode must be separate", "task.toml"))
    no_network = isinstance(verifier.get("environment"), Mapping) and str(verifier["environment"].get("network_mode", "")) == "no-network"
    checks.append(_criterion("verifier_execution_isolation", "pass" if no_network else "fail", "verifier environment has no network" if no_network else "verifier.environment.network_mode must be no-network", "task.toml"))
    # Harbor's task template keeps the artifact allowlist at the TOML root,
    # alongside schema_version (not inside [task]).
    artifacts = config.get("artifacts")
    artifact_ok = isinstance(artifacts, list) and all(isinstance(value, str) and value.startswith("/") for value in artifacts)
    checks.append(_criterion("artifact_efficiency", "pass" if artifact_ok else "fail", "artifacts is an explicit absolute-path allowlist" if artifact_ok else "task.artifacts must be an explicit list of absolute paths", "task.toml"))
    resources_ok = all(isinstance(environment.get(key), (int, float)) and environment[key] > 0 for key in ("build_timeout_sec", "cpus", "memory_mb", "storage_mb"))
    verifier_timeout = verifier.get("timeout_sec")
    agent_timeout = agent.get("timeout_sec")
    resources_ok = resources_ok and isinstance(verifier_timeout, (int, float)) and 0 < verifier_timeout <= 600 and isinstance(agent_timeout, (int, float)) and 0 < agent_timeout <= 18000
    checks.append(_criterion("resource_configuration", "pass" if resources_ok else "fail", "timeouts and resource values are positive and within TB-Science bounds" if resources_ok else "resource/timeouts are missing or outside bounds", "task.toml"))
    tags = metadata.get("tags") or task.get("keywords")
    domain_ok = all(str(metadata.get(key, "")).strip() for key in ("author_name", "author_email", "author_organization", "domain", "field", "subfield")) and isinstance(tags, list) and bool(tags)
    checks.append(_criterion("category_and_tags", "pass" if domain_ok else "fail", "domain/field/subfield, author metadata and keywords are populated" if domain_ok else "domain, field, subfield, author metadata and keywords must be populated", "task.toml"))
    estimate = metadata.get("expert_time_estimate_hours", 0)
    checks.append(_criterion("expert_time_estimate", "pass" if isinstance(estimate, (int, float)) and estimate > 0 else "fail", "expert time estimate is positive" if isinstance(estimate, (int, float)) and estimate > 0 else "metadata.expert_time_estimate_hours must be positive", "task.toml"))
    headings = {
        "difficulty_explanation_quality": "Difficulty",
        "solution_explanation_quality": "Reference solution",
        "verification_explanation_quality": "Verification",
    }
    for name, heading in headings.items():
        body = _section_text(readme, heading)
        checks.append(_criterion(name, "pass" if len(body) >= 80 else "fail", f"README has a substantive ## {heading} section" if len(body) >= 80 else f"README needs a substantive ## {heading} section", "README.md"))
    agent_network = str(environment.get("network_mode", "")) in {"public", "no-network", "allowlist"}
    environment_ok = env_dockerfile.is_file() and agent_network
    checks.append(_criterion("environment_hygiene", "pass" if environment_ok else "fail", "environment/Dockerfile and explicit agent network policy are present" if environment_ok else "environment/Dockerfile and environment.network_mode are required", "environment/Dockerfile", "task.toml"))
    extras = sorted(
        path.name
        for path in task_dir.iterdir()
        if path.name not in _ALLOWED_ROOT_FILES and path.name not in _ALLOWED_ROOT_DIRS
    )
    junk_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".DS_Store"}
    nested_junk = sorted(
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.name in junk_names or path.suffix == ".pyc"
    )
    all_extras = extras + nested_junk
    checks.append(_criterion("no_extraneous_files", "pass" if not all_extras else "fail", "task tree has no unreviewed root entries or generated caches" if not all_extras else "unexpected task-tree entries: " + ", ".join(all_extras), "task directory"))
    checks.append(_criterion("structured_data_schema", "pass" if any(path.suffix in {".json", ".csv", ".jsonl"} for path in _files_under(task_dir)) else "review", "task contains structured data artifacts" if any(path.suffix in {".json", ".csv", ".jsonl"} for path in _files_under(task_dir)) else "structured output schema needs reviewer confirmation"))
    checks.append(_criterion("do_not_modify_enforced", "pass" if re.search(r"do not modify|must not modify|leave .*unchanged", instruction_text, re.I) else "review", "instruction states immutable inputs" if re.search(r"do not modify|must not modify|leave .*unchanged", instruction_text, re.I) else "reviewer must confirm whether immutable inputs need an explicit contract", "instruction.md"))
    return checks


def _semantic_review_criteria() -> list[Criterion]:
    reasons = {
        "solvable": "requires a passing reference solution and reproducibility run",
        "difficult": "requires domain-expert difficulty judgment",
        "scientifically_grounded": "requires paper/workflow provenance judgment",
        "scope": "requires domain/field scope judgment",
        "outcome_verified": "requires review that tests grade final outcomes rather than process",
        "anti_cheat_robustness": "requires adversarial shortcut review",
        "ground_truth_provenance": "requires reference-value provenance review",
        "graded_instances_discriminate": "requires per-instance null/solution discrimination test",
        "deterministic_reproducible": "requires repeated verifier execution",
        "essential_difficulty": "requires checking difficulty is substantive rather than clerical",
        "test_instruction_alignment": "requires comparing every test obligation with instruction text",
        "novel": "requires novelty/memorization review",
        "agentic": "requires checking the task needs multi-step terminal work",
        "reviewable": "requires reviewer-facing artifacts and explanations",
        "instruction_clarity": "requires independent expert reading",
        "typos": "requires human/editorial review",
        "task_authoring_dir": "requires repository layout check against the target TB-Science checkout",
    }
    return [_criterion(name, "review", reasons.get(name, "requires the TB-Science rubric reviewer")) for name in reasons]


def validate_task(
    task_dir: str | Path,
    *,
    paper_provenance: str | Path | None = None,
    round_number: int = 1,
    previous_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic authoring receipt for one candidate task."""

    root = Path(task_dir)
    if not root.is_dir() or root.is_symlink():
        raise TaskAuthoringError(f"task directory must be a real directory: {root}")
    task_toml = root / "task.toml"
    if not task_toml.is_file():
        config: Mapping[str, Any] = {}
        parse_error = "task.toml is missing"
    else:
        try:
            config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
            parse_error = ""
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            config = {}
            parse_error = f"task.toml cannot be parsed: {type(exc).__name__}"
    readme = _read_text(root / "README.md")
    criteria: list[Criterion] = []
    if parse_error:
        criteria.append(_criterion("task_toml_schema", "fail", parse_error, "task.toml"))
    else:
        criteria.extend(_mechanical_checks(root, config, readme))
    criteria.extend(_security_criteria(root))
    criteria.extend(_semantic_review_criteria())
    paper_doc, paper_checks = _paper_provenance(Path(paper_provenance) if paper_provenance else None)
    # Every receipt must enumerate all official implementation criteria.  A
    # missing task.toml therefore produces explicit failures rather than a
    # misleadingly short checklist.
    present_names = {item.name for item in criteria}
    semantic_names = {item.name for item in _semantic_review_criteria()}
    for name in IMPLEMENTATION_CRITERIA:
        if name in present_names:
            continue
        status = "review" if name in semantic_names or name == "task_readme" else "fail"
        criteria.append(
            _criterion(
                name,
                status,
                "not evaluated because task.toml or required task files are unavailable",
            )
        )
    # Proposal-level criteria are explicitly represented; this is useful when
    # the same receipt is attached to a paper review before implementation.
    proposal = [{"name": name, "status": "review", "reason": "proposal rubric requires TB-Science reviewer"} for name in PROPOSAL_CRITERIA]
    all_checks = [*criteria, *paper_checks]
    fail_count = sum(item.status == "fail" for item in all_checks)
    review_count = sum(item.status == "review" for item in all_checks)
    implementation_counts = {
        "pass": sum(item.status == "pass" for item in criteria),
        "fail": sum(item.status == "fail" for item in criteria),
        "review": sum(item.status == "review" for item in criteria),
    }
    overall_counts = {
        "pass": sum(item.status == "pass" for item in all_checks),
        "fail": fail_count,
        "review": review_count,
    }
    if fail_count:
        decision = "blocked"
    elif review_count:
        decision = "ready-for-human-review"
    else:
        decision = "ready-for-harbor-validation"
    previous_digest, previous_checks = _previous_receipt(
        Path(previous_receipt) if previous_receipt else None,
        task_dir=root,
        round_number=int(round_number),
    )
    all_checks.extend(previous_checks)
    payload = {
        "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
        # Keep the receipt digest independent of the checkout's absolute path.
        "task_dir": root.name,
        "task_tree_digest": _task_tree_digest(root),
        "round": int(round_number),
        "rubric": {
            "commit": RUBRIC_COMMIT,
            "implementation_source": RUBRIC_SOURCE,
            "task_template_source": TASK_TEMPLATE_SOURCE,
            "implementation_criteria_count": len(IMPLEMENTATION_CRITERIA),
            "implementation_criteria": list(IMPLEMENTATION_CRITERIA),
        },
        "paper": paper_doc,
        "proposal_criteria": proposal,
        "implementation_criteria": [item.as_dict() for item in sorted(criteria, key=lambda item: item.name)],
        "provenance_checks": [item.as_dict() for item in [*paper_checks, *previous_checks]],
        # ``counts`` covers exactly the 39 implementation-rubric entries.
        # Provenance/chain checks are additional and are reported separately so
        # a 39-item rubric cannot silently acquire a 40th status.
        "counts": implementation_counts,
        "overall_counts": overall_counts,
        "decision": decision,
        "previous_receipt_digest": previous_digest,
        "limitations": [
            "No rubric LLM or model API was called by this gate.",
            "A ready-for-human-review receipt is not TB-Science pre-approval or PR acceptance.",
            "Harbor execution and repeated verifier runs remain required before publication.",
        ],
    }
    payload["receipt_digest"] = _digest(payload)
    return payload


def write_receipt(receipt: Mapping[str, Any], out: str | Path) -> dict[str, Path]:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "authoring-receipt.json"
    markdown_path = output / "authoring-receipt.md"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# TB-Science task authoring receipt",
        "",
        f"- Decision: **{receipt['decision']}**",
        f"- Round: `{receipt['round']}`",
        f"- Implementation counts (39 criteria): `{receipt['counts']}`",
        f"- Overall counts (including provenance): `{receipt.get('overall_counts', receipt['counts'])}`",
        f"- Rubric: `{receipt['rubric']['implementation_criteria_count']}` criteria from {receipt['rubric']['implementation_source']}",
        f"- Receipt digest: `{receipt['receipt_digest']}`",
        "",
        "## Implementation rubric",
        "",
        "| Criterion | Status | Reason |",
        "| --- | --- | --- |",
    ]
    for item in receipt["implementation_criteria"]:
        lines.append(f"| `{item['name']}` | `{item['status']}` | {item['reason']} |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in receipt["limitations"])
    lines.extend(["", "## Paper provenance", ""])
    for item in receipt.get("provenance_checks", []):
        lines.append(f"- `{item['status']}` {item['reason']}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "IMPLEMENTATION_CRITERIA",
    "PROPOSAL_CRITERIA",
    "RUBRIC_COMMIT",
    "RUBRIC_SOURCE",
    "TASK_TEMPLATE_SOURCE",
    "TaskAuthoringError",
    "validate_task",
    "write_receipt",
]
