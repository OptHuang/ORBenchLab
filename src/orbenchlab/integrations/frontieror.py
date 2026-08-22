"""FrontierOR integration — thin wrapper over the official external harness.

FrontierOR is the opposite shape from ORAgentBench. It owns a complete trusted
evaluation stack of its own: a versioned scoring contract (``staged_qte``), a
role-based artifact visibility policy, trusted feasibility checkers, a candidate
Docker boundary with an egress proxy and a credential-isolating model proxy, and
a black-box security probe. Its official entry point is
``python -m frontieror.infra {contract,submission,agent,security-check}``.

**Decision: wrap the official harness; do not materialise Harbor tasks yet.**
See ``docs/decisions/0001-frontieror-integration.md``. The short version, all of
which this module verifies rather than asserts:

1. The score includes a *speed* term measured as trusted-host wall clock. A
   Harbor task package executed on a shared runner cannot reproduce that number
   comparably, so a converted task would report a differently-defined score
   under the same name.
2. Reference objectives, reference runtimes, the checker implementation and
   final-instance membership are ``TRUSTED_ONLY`` and are not distributed. A
   Harbor task package would have to ship a substitute — that is a different
   benchmark, not an adaptation.
3. The security boundary is upstream's. Re-expressing it as Harbor
   ``network_mode`` declarations would be a re-implementation whose divergences
   surface as silent scoring differences.

So this integration reads FrontierOR's own machine-readable contract and
declares the preconditions for invoking its harness. Harbor task conversion
stays on the table as a later, separately-validated adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.errors import IntegrationError
from .base import (
    IntegrationKind,
    InspectionReport,
    find_repo_root,
    find_vendored,
    read_source_revision,
)

NAME = "frontieror"
KIND = IntegrationKind.OFFICIAL_EXTERNAL_HARNESS

UPSTREAM_REPO = "https://github.com/Minw913/FrontierOR"
#: Commit inspected while building this integration. CI clones exactly this.
PINNED_COMMIT = "8e95db622dfcfb7abb9dc9d45ceec8364d6a9be9"

INFRA_PKG = "frontieror/infra"
CONTRACTS_MODULE = "frontieror/infra/contracts.py"
SECURITY_CHECK_MODULE = "frontieror/infra/security_check.py"
TRUSTED_CHECKERS_DIR = "trusted_checkers"
SUBMISSION_PKG = "frontieror/infra/submission"

#: Official CLI verbs, discovered from the upstream CLI source rather than assumed.
EXPECTED_COMMANDS = ("agent", "submission", "contract", "security-check")

#: Upstream artefacts ORBenchLab must never copy into this repository.
FORBIDDEN_VENDORED_NAMES = (
    "trusted_checkers",
    "one_shot_eval.py",
    "feasibility_check.py",
)

CONTRACT_TIMEOUT_SEC = 120


def describe() -> dict[str, Any]:
    return {
        "name": NAME,
        "kind": KIND.value,
        "upstream_repo": UPSTREAM_REPO,
        "pinned_commit": PINNED_COMMIT,
        "summary": (
            "FrontierOR ships its own trusted harness, scoring contract and security "
            "boundary. ORBenchLab invokes the official entry point and records evidence; "
            "it does not reimplement the checker, the boundary or the scoring formula."
        ),
        "we_do_not_own": [
            "scoring formula (staged_qte, frontieror/infra/contracts.py)",
            "trusted feasibility checkers (trusted_checkers/**)",
            "candidate/agent security boundary (frontieror/infra/docker/**, egress + model proxy)",
            "reference objectives and reference runtimes (never distributed)",
            "final instance membership (server-only data)",
        ],
        "we_own": [
            "preconditions and fail-closed gating before invoking the harness",
            "evidence labelling of whatever the harness returns",
            "run identity and the plan ledger",
        ],
        "official_entrypoints": {
            "contract": "python -m frontieror.infra contract",
            "submission": "python -m frontieror.infra submission <dir> --paper-id <id>",
            "agent": "python -m frontieror.infra agent ...",
            "security_check": "python -m frontieror.infra security-check --candidate-image <img>",
        },
        "requires": {
            "secrets": ["OPENROUTER_API_KEY"],
            "model_api_key": True,
            "solver_license": True,
            "solver_license_env": "GRB_LICENSE_FILE",
            "dataset": "huggingface: SmartOR/FrontierOR (not vendored in the code repo)",
            "docker_images": [
                "frontieror-candidate:1",
                "frontieror-coral-agent:0.1",
                "frontieror-coral-model-proxy:0.1",
            ],
            "self_hosted_runner": True,
            "self_hosted_reason": (
                "the score contains a speed term measured as trusted-host wall clock; "
                "shared GitHub-hosted runners are not performance-comparable"
            ),
            "runner_labels": ["self-hosted", "orbench-exec", "perf-isolated"],
        },
        "performance_scored": True,
        "harbor_conversion": "deferred — see docs/decisions/0001-frontieror-integration.md",
    }


def inspect(source: Path) -> InspectionReport:
    """Statically inspect a FrontierOR checkout and read its official contract."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise IntegrationError(f"source is not a directory: {source}")

    report = InspectionReport(
        integration=NAME,
        integration_kind=KIND,
        source=str(source),
        source_revision=read_source_revision(source),
    )

    _check_official_entrypoint(report, source)
    contract = _check_scoring_contract(report, source)
    _check_runtime_scoring(report, contract)
    _check_visibility(report, contract)
    _check_trusted_checkers(report, source)
    _check_security_boundary(report, source)
    _check_external_preconditions(report, source)
    _check_no_vendored_copy(report)
    _check_conversion_parity(report, contract)
    _record_decisions(report, contract)
    return report


def _check_official_entrypoint(report: InspectionReport, source: Path) -> None:
    cli_path = source / INFRA_PKG / "cli.py"
    main_path = source / INFRA_PKG / "__main__.py"
    if not (cli_path.is_file() and main_path.is_file()):
        report.add(
            "official_harness_entrypoint",
            "fail",
            f"{INFRA_PKG}/cli.py or __main__.py missing; this checkout exposes no official "
            "trusted-evaluation entry point to wrap",
            expected=[f"{INFRA_PKG}/cli.py", f"{INFRA_PKG}/__main__.py"],
        )
        return
    text = cli_path.read_text(encoding="utf-8", errors="replace")
    found = tuple(cmd for cmd in EXPECTED_COMMANDS if f'"{cmd}"' in text)
    missing = [cmd for cmd in EXPECTED_COMMANDS if cmd not in found]
    report.facts["official_commands"] = list(found)
    report.add(
        "official_harness_entrypoint",
        "pass" if not missing else "warn",
        f"python -m frontieror.infra exposes {list(found)}",
        entrypoint="python -m frontieror.infra",
        missing_commands=missing,
    )


def _check_scoring_contract(report: InspectionReport, source: Path) -> dict[str, Any]:
    """Read the official contract by running upstream's own command.

    This is the whole reason a wrapper is viable: FrontierOR publishes its
    scorer definition machine-readably. Running it costs nothing, calls no
    model and touches no network. If it cannot be run, we fall back to reading
    the constants out of the source and mark the check degraded — we never
    substitute our own idea of the formula.
    """
    contracts_path = source / CONTRACTS_MODULE
    if not contracts_path.is_file():
        report.add(
            "public_scoring_contract",
            "fail",
            f"{CONTRACTS_MODULE} not found; no published scoring contract to bind to",
        )
        return {}

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "frontieror.infra", "contract"],
            cwd=str(source),
            capture_output=True,
            text=True,
            timeout=CONTRACT_TIMEOUT_SEC,
            env=env,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return _contract_static_fallback(report, contracts_path, f"invocation failed: {exc}")

    if completed.returncode != 0:
        reason = (completed.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
        return _contract_static_fallback(
            report, contracts_path, f"exit {completed.returncode}: {reason[0]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _contract_static_fallback(report, contracts_path, f"unparsable output: {exc}")

    scoring = payload.get("scoring", {})
    report.facts["scoring_contract"] = {
        "contract_version": scoring.get("contract_version"),
        "scorer": scoring.get("scorer"),
        "aggregation": scoring.get("aggregation"),
        "runtime_measurement": scoring.get("runtime_measurement"),
        "private_parameters": scoring.get("private_parameters", []),
        "source": "python -m frontieror.infra contract",
    }
    report.add(
        "public_scoring_contract",
        "pass",
        f"official contract read from upstream: scorer={scoring.get('scorer')!r} "
        f"version={scoring.get('contract_version')!r}",
        method="python -m frontieror.infra contract",
        contract_version=scoring.get("contract_version"),
        scorer=scoring.get("scorer"),
        contract_digest=f"sha256:{hashlib.sha256(completed.stdout.encode()).hexdigest()}",
    )
    return payload


def _contract_static_fallback(
    report: InspectionReport, contracts_path: Path, reason: str
) -> dict[str, Any]:
    """Recover the contract identity from source when the command cannot run."""
    text = contracts_path.read_text(encoding="utf-8", errors="replace")
    version = _extract_literal(text, "SCORING_CONTRACT_VERSION")
    report.facts["scoring_contract"] = {
        "contract_version": version,
        "scorer": "staged_qte" if version and "qte" in version else None,
        "source": f"static parse of {CONTRACTS_MODULE}",
    }
    report.add(
        "public_scoring_contract",
        "warn",
        f"could not run 'python -m frontieror.infra contract' ({reason}); "
        f"fell back to reading {CONTRACTS_MODULE} statically",
        contract_version=version,
        impact=(
            "the contract identity is known but not confirmed executable in this "
            "environment; do not treat a run from this environment as official"
        ),
    )
    return {}


def _extract_literal(text: str, name: str) -> str | None:
    match = re.search(rf'^{re.escape(name)}\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def _check_runtime_scoring(report: InspectionReport, contract: dict[str, Any]) -> None:
    """Does runtime enter the score? If so, performance isolation is mandatory."""
    scoring = contract.get("scoring") or {}
    measurement = scoring.get("runtime_measurement")
    instance_score = scoring.get("instance_score") or {}
    speed_term = instance_score.get("quality_and_speed")

    if not scoring:
        report.add(
            "performance_isolation_required",
            "warn",
            "scoring contract unavailable in this environment; assuming performance "
            "scoring is required (fail-closed default)",
            performance_scored=True,
            assumed=True,
        )
        report.facts["performance_scored"] = True
        report.facts["requires_perf_isolated_site"] = True
        return

    scored = bool(speed_term) or measurement == "trusted_host_wall_clock"
    report.facts["performance_scored"] = scored
    report.facts["requires_perf_isolated_site"] = scored
    report.add(
        "performance_isolation_required",
        "pass",
        (
            f"runtime enters the score (measurement={measurement!r}); campaigns must target a "
            "perf_isolated site with pinned cores"
            if scored
            else "runtime does not enter the score"
        ),
        runtime_measurement=measurement,
        speed_term=speed_term,
        candidate_reported_timestamps_trusted=scoring.get("candidate_reported_timestamps_trusted"),
        consequence=(
            "GitHub-hosted runners are shared virtual machines; any FrontierOR number "
            "produced there is exploratory and must never be published as an official score"
        ),
    )


def _check_visibility(report: InspectionReport, contract: dict[str, Any]) -> None:
    """Confirm the hidden artefacts that make local Harbor conversion unsafe."""
    visibility = (contract.get("visibility") or {}).get("artifacts") or {}
    if not visibility:
        report.add(
            "hidden_artifact_visibility",
            "skip",
            "visibility contract unavailable in this environment; "
            "hidden-artefact analysis not performed",
        )
        return
    trusted_only = sorted(k for k, v in visibility.items() if v == "trusted_only")
    report.facts["trusted_only_artifacts"] = trusted_only
    blocking = [
        artefact
        for artefact in (
            "reference_objective",
            "reference_runtime",
            "feasibility_checker",
            "final_instance_membership",
        )
        if artefact in trusted_only
    ]
    report.add(
        "hidden_artifact_visibility",
        "pass",
        f"{len(trusted_only)} artefacts are trusted-only, including {blocking}",
        trusted_only=trusted_only,
        blocking_for_harbor_conversion=blocking,
        consequence=(
            "a materialised Harbor task cannot contain these, so it could not reproduce "
            "the official score; it would need substitutes and would be a different benchmark"
        ),
    )


def _check_trusted_checkers(report: InspectionReport, source: Path) -> None:
    checkers_dir = source / TRUSTED_CHECKERS_DIR
    if not checkers_dir.is_dir():
        report.add(
            "trusted_checkers_present",
            "warn",
            f"{TRUSTED_CHECKERS_DIR}/ not found in this checkout; sample checkers are normally "
            "shipped for local integration tests",
        )
        return
    checkers = sorted(p for p in checkers_dir.rglob("feasibility_check.py"))
    report.facts["sample_trusted_checkers"] = [
        str(p.relative_to(source)) for p in checkers
    ]
    report.add(
        "trusted_checkers_present",
        "pass",
        f"{len(checkers)} sample trusted checker(s) present upstream; referenced, never copied",
        paths=[str(p.relative_to(source)) for p in checkers],
        note=(
            "the official checkers used for leaderboard scoring are trusted-only and are "
            "not in this repository; these samples exist for local integration tests"
        ),
    )


def _check_security_boundary(report: InspectionReport, source: Path) -> None:
    security_check = source / SECURITY_CHECK_MODULE
    dockerfiles = sorted((source / INFRA_PKG / "docker").glob("*.Dockerfile"))
    present = security_check.is_file() and bool(dockerfiles)
    report.facts["security_boundary_owner"] = "upstream"
    report.add(
        "security_boundary_owned_upstream",
        "pass" if present else "warn",
        (
            f"upstream owns the boundary: {SECURITY_CHECK_MODULE} plus "
            f"{len(dockerfiles)} container definition(s)"
            if present
            else "upstream security boundary artefacts not fully present in this checkout"
        ),
        security_check=SECURITY_CHECK_MODULE if security_check.is_file() else None,
        dockerfiles=[str(p.relative_to(source)) for p in dockerfiles],
        verification_command=(
            "python -m frontieror.infra security-check --candidate-image frontieror-candidate:1"
        ),
        note=(
            "ORBenchLab does not re-express this boundary as Harbor network_mode "
            "declarations; divergence there would show up as silent scoring differences"
        ),
    )


def _check_external_preconditions(report: InspectionReport, source: Path) -> None:
    """Everything the harness needs that this repository cannot supply.

    Recorded as preconditions, not as failures: a checkout is not broken merely
    because the dataset and licences live elsewhere. What matters is that any
    workflow invoking the harness gates on them and fails closed.
    """
    dataset_present = any((source / candidate).exists() for candidate in ("frontier-or", "data"))
    submission_pkg = (source / SUBMISSION_PKG).is_dir()

    report.add(
        "submission_verifier_entrypoint",
        "pass" if submission_pkg else "warn",
        (
            "official submission verifier available "
            "('python -m frontieror.infra submission'); this is the zero-model-call "
            "verification path"
            if submission_pkg
            else f"{SUBMISSION_PKG} not found; no official submission verifier to wrap"
        ),
        path=SUBMISSION_PKG,
    )

    preconditions = [
        {
            "id": "dataset",
            "requirement": "huggingface dataset SmartOR/FrontierOR downloaded locally",
            "satisfied_in_checkout": dataset_present,
            "fail_closed": True,
            "note": (
                "official final instances are unpublished server-only data; files in the "
                "public dataset are suitable for integration tests, not for leaderboard scoring"
            ),
        },
        {
            "id": "solver_license",
            "requirement": "Gurobi licence at GRB_LICENSE_FILE",
            "satisfied_in_checkout": False,
            "fail_closed": True,
        },
        {
            "id": "model_api_key",
            "requirement": "OPENROUTER_API_KEY (agent paths only; contract and submission paths need none)",
            "satisfied_in_checkout": False,
            "fail_closed": True,
        },
        {
            "id": "container_images",
            "requirement": "frontieror-candidate:1, frontieror-coral-agent:0.1, frontieror-coral-model-proxy:0.1",
            "satisfied_in_checkout": False,
            "fail_closed": True,
        },
        {
            "id": "perf_isolated_runner",
            "requirement": "self-hosted runner with pinned cores and no co-tenancy",
            "satisfied_in_checkout": False,
            "fail_closed": True,
        },
    ]
    report.facts["preconditions"] = preconditions
    report.add(
        "external_preconditions_declared",
        "pass",
        f"{len(preconditions)} external preconditions declared, all fail-closed",
        preconditions=[p["id"] for p in preconditions],
        dataset_present_in_checkout=dataset_present,
    )


def _check_no_vendored_copy(report: InspectionReport) -> None:
    repo_root = find_repo_root(Path(__file__))
    if repo_root is None:
        report.add(
            "no_vendored_upstream_copy",
            "skip",
            "ORBenchLab repository root not locatable from the installed package",
        )
        return
    vendored = find_vendored(repo_root, FORBIDDEN_VENDORED_NAMES)
    report.add(
        "no_vendored_upstream_copy",
        "pass" if not vendored else "fail",
        (
            "ORBenchLab vendors no FrontierOR checker, harness or scoring code"
            if not vendored
            else f"ORBenchLab vendors upstream artefacts at {vendored}"
        ),
        forbidden_names=list(FORBIDDEN_VENDORED_NAMES),
        vendored_paths=vendored,
    )


def _check_conversion_parity(report: InspectionReport, contract: dict[str, Any]) -> None:
    """State, machine-readably, why Harbor conversion is not parity-safe yet."""
    scoring = contract.get("scoring") or {}
    visibility = (contract.get("visibility") or {}).get("artifacts") or {}
    blockers = []

    if scoring.get("runtime_measurement") == "trusted_host_wall_clock" or not scoring:
        blockers.append(
            {
                "id": "trusted_host_timing",
                "detail": (
                    "score includes a speed term measured on the trusted host; a Harbor "
                    "container on a shared runner produces a differently-defined number"
                ),
            }
        )
    hidden = [
        a
        for a in ("reference_objective", "reference_runtime", "feasibility_checker",
                  "final_instance_membership")
        if visibility.get(a) == "trusted_only"
    ]
    if hidden or not visibility:
        blockers.append(
            {
                "id": "undistributed_reference_data",
                "detail": (
                    "reference objectives/runtimes, checker implementation and final instance "
                    f"membership are trusted-only ({hidden or 'contract unavailable'}); a task "
                    "package cannot contain them"
                ),
            }
        )
    blockers.append(
        {
            "id": "upstream_security_boundary",
            "detail": (
                "candidate isolation, egress proxy and credential-isolating model proxy are "
                "upstream components; re-expressing them in task.toml is a re-implementation"
            ),
        }
    )
    blockers.append(
        {
            "id": "multi_candidate_inner_loop",
            "detail": (
                "test-time self-evolution runs many candidates per task inside the harness; "
                "Harbor's trial model would have to be mapped onto that before conversion"
            ),
        }
    )

    report.facts["harbor_conversion_blockers"] = blockers
    report.add(
        "harbor_conversion_parity_safe",
        "warn",
        f"Harbor task conversion is not parity-safe yet ({len(blockers)} blockers); "
        "the initial integration wraps the official harness",
        parity_safe=False,
        blockers=[b["id"] for b in blockers],
        decision_record="docs/decisions/0001-frontieror-integration.md",
        revisit_when=(
            "upstream publishes a reference-runtime normalisation usable off the trusted host, "
            "or a distributable checker + reference bundle for a declared task subset"
        ),
    )


def _record_decisions(report: InspectionReport, contract: dict[str, Any]) -> None:
    report.decisions = {
        "integration_form": "official-external-harness",
        "adapter_required": False,
        "harbor_tasks_materialized": False,
        "harbor_conversion_parity_safe": False,
        "rationale": (
            "trusted-host timing in the score, undistributed reference data and an "
            "upstream-owned security boundary make a converted Harbor task a different "
            "measurement under the same name"
        ),
        "wrapped_entrypoints": [
            "python -m frontieror.infra contract",
            "python -m frontieror.infra submission",
            "python -m frontieror.infra agent",
            "python -m frontieror.infra security-check",
        ],
        "zero_cost_paths": ["contract"],
        "requires_model_calls": ["agent"],
        "requires_solver_license": ["submission", "agent"],
        "later_optional_work": (
            "a true Harbor adapter for a declared subset, gated on a differential "
            "re-scoring fixture that matches the official checker key-for-key"
        ),
        "scoring_contract_version": (report.facts.get("scoring_contract") or {}).get(
            "contract_version"
        ),
    }
