"""The integration contract.

An *integration* is how ORBenchLab connects to one upstream benchmark. It is a
read-only inspector plus a declaration of what running that benchmark requires.
It is deliberately not a re-implementation of the benchmark: an integration
never copies an upstream verifier, checker, scoring formula or security
boundary, and ``orbench integration inspect`` is a static read that makes no
model calls and executes no benchmark.

Three integration kinds exist, in increasing order of how much we own:

``harbor-native``
    The upstream repository already ships Harbor task packages. We consume
    them as they are. No adapter, no second copy of the tasks.

``official-external-harness``
    Upstream owns an official harness that we invoke through its public,
    versioned entry point. Correct when upstream's scoring depends on things
    we cannot faithfully reproduce inside Harbor (hidden reference data,
    trusted-host timing, a bespoke sandbox).

``harbor-adapter``
    A build-time generator that materialises Harbor task packages from an
    upstream dataset. Only appropriate once conversion is demonstrably
    parity-safe.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

INSPECTION_SCHEMA_VERSION = "1.0"

CheckStatus = Literal["pass", "fail", "warn", "skip"]
ReportStatus = Literal["ok", "degraded", "failed"]


class IntegrationKind(str, Enum):
    HARBOR_NATIVE = "harbor-native"
    OFFICIAL_EXTERNAL_HARNESS = "official-external-harness"
    HARBOR_ADAPTER = "harbor-adapter"


@dataclass(frozen=True)
class Check:
    """One machine-checkable assertion about an upstream source tree."""

    id: str
    status: CheckStatus
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionReport:
    """Machine-readable result of inspecting an upstream checkout.

    Serialised verbatim by ``orbench integration inspect --json``; CI asserts
    on ``status`` and on individual check ids.
    """

    integration: str
    integration_kind: IntegrationKind
    source: str
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    source_revision: dict[str, Any] = field(default_factory=dict)
    schema_version: str = INSPECTION_SCHEMA_VERSION

    def add(
        self,
        check_id: str,
        status: CheckStatus,
        detail: str,
        **evidence: Any,
    ) -> Check:
        check = Check(id=check_id, status=status, detail=detail, evidence=evidence)
        self.checks.append(check)
        return check

    @property
    def status(self) -> ReportStatus:
        """``failed`` if anything failed, ``degraded`` on warnings, else ``ok``.

        A skipped check never upgrades the status: skipping means we could not
        look, which is not the same as looking and finding nothing wrong.
        """
        statuses = {check.status for check in self.checks}
        if "fail" in statuses:
            return "failed"
        if "warn" in statuses:
            return "degraded"
        return "ok"

    def counts(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
        for check in self.checks:
            counts[check.status] += 1
        return counts

    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "integration": self.integration,
            "integration_kind": self.integration_kind.value,
            "source": self.source,
            "source_revision": self.source_revision,
            "status": self.status,
            "counts": self.counts(),
            # Stated explicitly so a reader never has to infer it: inspection is
            # a static read of a checkout.
            "execution": {
                "model_calls": 0,
                "benchmark_executed": False,
                "network_access": False,
                "reads_credentials": False,
            },
            "checks": [check.to_dict() for check in self.checks],
            "facts": self.facts,
            "decisions": self.decisions,
        }


@runtime_checkable
class Integration(Protocol):
    """What every integration must provide."""

    name: str
    kind: IntegrationKind

    def describe(self) -> dict[str, Any]:
        """Static declaration: upstream, pinned commit, requirements, decision."""

    def inspect(self, source: Path) -> InspectionReport:
        """Read ``source`` (an upstream checkout) and report machine-readably."""


#: Paths exempt from the "no vendored upstream copy" checks. Test fixtures are
#: deliberately miniature stand-ins for upstream trees, and a scratch clone under
#: ``upstream/`` is a working copy, not a vendored fork.
VENDORING_SCAN_EXCLUDED_PREFIXES = (
    ".git",
    ".venv",
    "upstream",
    "out",
    "tests/fixtures",
)


def find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the directory holding ``pyproject.toml``."""
    for parent in Path(start).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def find_vendored(repo_root: Path, names: tuple[str, ...]) -> list[str]:
    """Find upstream artefacts copied into this repository.

    Vendoring an upstream verifier, checker or task set is the failure mode the
    integrations exist to prevent: the copy drifts, and the benchmark quietly
    becomes a different benchmark under the same name.
    """
    hits: list[str] = []
    for name in names:
        for path in repo_root.rglob(name):
            relative = path.relative_to(repo_root)
            posix = relative.as_posix()
            if any(
                posix == prefix or posix.startswith(f"{prefix}/")
                for prefix in VENDORING_SCAN_EXCLUDED_PREFIXES
            ):
                continue
            hits.append(posix)
    return sorted(hits)


def read_source_revision(source: Path) -> dict[str, Any]:
    """Best-effort git revision of an upstream checkout.

    Returns ``{"is_git": False}`` rather than failing when the source is a plain
    directory: inspection must work on an unpacked tarball too.
    """
    git_dir = source / ".git"
    if not git_dir.exists():
        return {"is_git": False}
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        described = subprocess.run(
            ["git", "-C", str(source), "log", "-1", "--format=%cI"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover - env dependent
        return {"is_git": True, "error": str(exc)}
    return {"is_git": True, "commit": commit, "committed_at": described or None}
