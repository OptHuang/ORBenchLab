"""Template for a new benchmark integration.

Copy this file to ``src/orbenchlab/integrations/<yourbench>.py``, work through
the TODOs, and register the module in ``registry.py``. Then delete every TODO
comment — a template comment left in a real integration is a lie about what has
been checked.

Two rules dominate everything below:

* **Read, do not copy.** An integration inspects an upstream checkout and
  records what it found. Copying an upstream verifier, checker, task set or
  scoring formula into this repository creates a fork that drifts, and a drifted
  copy is a different benchmark under the same name. A test enforces this.
* **A check you could not perform is ``skip``, never ``pass``.** ``pass`` means
  "I looked and it was fine". If a property is unverifiable from the checkout —
  because it belongs to the runner, or to data upstream withholds — say ``skip``
  and record how it *could* be verified.

Worked examples: ``oragentbench.py`` (Harbor-native) and ``frontieror.py``
(official external harness). They are deliberately different shapes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from orbenchlab.core.errors import IntegrationError
from orbenchlab.integrations.base import (
    IntegrationKind,
    InspectionReport,
    find_repo_root,
    find_vendored,
    read_source_revision,
)

# TODO: the registry key. Lowercase, no spaces.
NAME = "yourbench"

# TODO: pick the form. See docs/integrations.md for the comparison table and the
# test for "may I convert this to Harbor tasks?".
#   HARBOR_NATIVE             upstream already ships Harbor task packages
#   OFFICIAL_EXTERNAL_HARNESS upstream owns its runner and grader
#   HARBOR_ADAPTER            you generate task packages — needs parity evidence
KIND = IntegrationKind.HARBOR_NATIVE

# TODO: upstream repository and the exact commit you inspected. CI clones this
# commit; it is not a "roughly current" marker.
UPSTREAM_REPO = "https://github.com/example/yourbench"
PINNED_COMMIT = "0" * 40

# TODO: upstream artefacts that must never appear in this repository.
FORBIDDEN_VENDORED_NAMES = ("upstream_tasks", "official_checker.py")


def describe() -> dict[str, Any]:
    """Static declaration. Must be answerable without touching a checkout.

    ``orbench integrations list`` renders this, and campaign validation reads
    ``performance_scored`` and ``requires`` to decide which sites may run it.
    """
    return {
        "name": NAME,
        "kind": KIND.value,
        "upstream_repo": UPSTREAM_REPO,
        "pinned_commit": PINNED_COMMIT,
        # TODO: one sentence on what upstream owns and what we do.
        "summary": "TODO",
        # TODO: be specific. This list is what stops a future contributor from
        # "helpfully" reimplementing something.
        "we_do_not_own": ["TODO: verifier", "TODO: scoring", "TODO: task data"],
        "we_own": [
            "campaign compilation and run identity",
            "evidence labelling and reporting",
        ],
        "requires": {
            # TODO: names only. A value here would be a leaked credential.
            "secrets": [],
            "model_api_key": False,
            "solver_license": False,
            "self_hosted_runner": False,
            "runner_labels": [],
        },
        # TODO: does runtime, memory or any host property enter the score? If
        # yes, campaign validation will refuse every site that is not declared
        # performance-isolated — which is the point.
        "performance_scored": False,
    }


def inspect(source: Path) -> InspectionReport:
    """Statically inspect an upstream checkout.

    Constraints, asserted by CI against ``report.execution``: no model calls, no
    benchmark execution, no network access, no credential reads. Inspection must
    be free and safe to run on every pull request, or it will not be run.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise IntegrationError(f"source is not a directory: {source}")

    report = InspectionReport(
        integration=NAME,
        integration_kind=KIND,
        source=str(source),
        source_revision=read_source_revision(source),
    )

    # ---------------------------------------------------------------- #
    # 1. Is this the benchmark we think it is? Fail closed if not.
    # ---------------------------------------------------------------- #
    marker = source / "TODO-a-file-that-must-exist"
    if not marker.exists():
        report.add(
            "upstream_shape",
            "fail",
            f"{marker.name} not found; this checkout is not a {NAME} tree",
            expected_path=marker.name,
        )
        _record_decisions(report)
        return report
    report.add("upstream_shape", "pass", f"{marker.name} found")

    # ---------------------------------------------------------------- #
    # 2. Content-address the dataset. Run ids are pinned to this digest, so a
    #    silent upstream data change produces different ids rather than
    #    contaminating an existing campaign.
    # ---------------------------------------------------------------- #
    digest = hashlib.sha256()
    for path in sorted((source / "TODO-dataset-dir").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    report.facts["dataset_digest"] = f"sha256:{digest.hexdigest()}"
    report.add(
        "dataset_digest",
        "pass",
        "dataset digest computed over sorted (relative path, file sha256) pairs",
        dataset_digest=report.facts["dataset_digest"],
    )

    # ---------------------------------------------------------------- #
    # 3. Discover the grading contract — do not assume it.
    #    Read the reward keys, the scoring contract version, whatever upstream
    #    actually publishes. Where upstream exposes a machine-readable contract
    #    command, prefer running it over parsing source; where it does not,
    #    parse and say so.
    # ---------------------------------------------------------------- #
    report.add("grading_contract", "skip", "TODO: discover and record the grading contract")

    # ---------------------------------------------------------------- #
    # 4. Declare every external precondition, each fail-closed. Secrets,
    #    licences, datasets, images, runner capabilities. Workflows gate on this.
    # ---------------------------------------------------------------- #
    report.facts["preconditions"] = [
        {
            "id": "TODO",
            "requirement": "TODO",
            "satisfied_in_checkout": False,
            "fail_closed": True,
        },
    ]
    report.add(
        "external_preconditions_declared",
        "pass",
        "preconditions declared, all fail-closed",
        preconditions=[p["id"] for p in report.facts["preconditions"]],
    )

    # ---------------------------------------------------------------- #
    # 5. Prove we vendored nothing.
    # ---------------------------------------------------------------- #
    repo_root = find_repo_root(Path(__file__))
    if repo_root is None:
        report.add(
            "no_vendored_upstream_copy",
            "skip",
            "repository root not locatable from the installed package",
        )
    else:
        vendored = find_vendored(repo_root, FORBIDDEN_VENDORED_NAMES)
        report.add(
            "no_vendored_upstream_copy",
            "pass" if not vendored else "fail",
            "no upstream artefacts vendored" if not vendored else f"vendored: {vendored}",
            vendored_paths=vendored,
        )

    _record_decisions(report)
    return report


def _record_decisions(report: InspectionReport) -> None:
    """Machine-readable integration decisions.

    ``integration-contract.yml`` asserts on these, so a change of form cannot
    happen quietly: flipping ``adapter_required`` fails CI until someone updates
    the assertion, which is exactly the conversation that change deserves.
    """
    report.decisions = {
        "integration_form": KIND.value,
        "adapter_required": False,
        # For an official-external-harness integration, also record:
        #   "harbor_tasks_materialized": False,
        #   "harbor_conversion_parity_safe": False,
        #   "rationale": "...",
        # and write a decision record under docs/decisions/.
        "rationale": "TODO: why this form, in one or two sentences",
        "zero_cost_paths": ["TODO: how CI exercises this without spending money"],
    }


# ---------------------------------------------------------------------- #
# Checklist before opening the pull request
# ---------------------------------------------------------------------- #
#
#  [ ] Registered in src/orbenchlab/integrations/registry.py
#  [ ] Added to the matrix in .github/workflows/integration-contract.yml
#  [ ] Decision assertion added to that workflow's "decisions" step
#  [ ] Tests added, using a miniature fixture tree under
#      tests/fixtures/upstream/ so the suite stays offline
#  [ ] Every TODO above removed
#  [ ] A zero-cost path exists and is exercised in CI
#  [ ] Decision record written if the form is not the obvious one
#  [ ] No upstream code vendored (the test will tell you)
