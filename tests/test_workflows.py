"""GitHub Actions workflows are code, so they get tested like code.

These checks run locally and in CI. They cannot prove a workflow *succeeds* on
GitHub — only an actual run can do that, and none has been observed for this
repository — but they do prove the files parse, that every action is pinned to
an immutable commit, and that the secret-handling and fail-closed rules hold.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(".github/workflows")
EXPECTED_WORKFLOWS = {
    "ci.yml",
    "integration-contract.yml",
    "benchmark-smoke.yml",
    "report.yml",
}

SHA_PINNED = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w.-]+)*@[0-9a-f]{40}$")
_USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)", re.MULTILINE)
_SECRET_REF = re.compile(r"\$\{\{\s*secrets\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _workflow_paths(repo_root: Path) -> list[Path]:
    return sorted((repo_root / WORKFLOW_DIR).glob("*.yml"))


@pytest.fixture(scope="module")
def workflows(repo_root: Path) -> dict[str, dict]:
    """Parse every workflow.

    ``on:`` is YAML 1.1's boolean ``True``, which is a well-known wart; it is
    normalised here so the assertions can read naturally.
    """
    parsed: dict[str, dict] = {}
    for path in _workflow_paths(repo_root):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if True in data:
            data["on"] = data.pop(True)
        parsed[path.name] = data
    return parsed


@pytest.fixture(scope="module")
def raw_workflows(repo_root: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in _workflow_paths(repo_root)}


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_all_expected_workflows_exist(workflows):
    assert set(workflows) == EXPECTED_WORKFLOWS


def test_every_workflow_parses_and_has_jobs(workflows):
    for name, data in workflows.items():
        assert data.get("name"), name
        assert data.get("on"), name
        assert data.get("jobs"), name
        for job_name, job in data["jobs"].items():
            assert job.get("runs-on"), f"{name}:{job_name}"
            assert job.get("steps"), f"{name}:{job_name}"


def test_every_job_has_a_timeout(workflows):
    """An unbounded job on a self-hosted runner blocks the machine indefinitely."""
    for name, data in workflows.items():
        for job_name, job in data["jobs"].items():
            assert "timeout-minutes" in job, f"{name}:{job_name} has no timeout"


def test_every_workflow_declares_least_privilege_permissions(workflows):
    for name, data in workflows.items():
        permissions = data.get("permissions")
        assert permissions, f"{name} does not declare permissions"
        assert permissions.get("contents") == "read", name
        assert "write" not in str(permissions), f"{name} requests write permissions"


def test_every_workflow_sets_concurrency(workflows):
    for name, data in workflows.items():
        assert data.get("concurrency"), f"{name} has no concurrency group"


# --------------------------------------------------------------------------- #
# action pinning
# --------------------------------------------------------------------------- #


def test_every_action_is_pinned_to_a_full_commit_sha(raw_workflows):
    """A tag can be repointed; a commit sha cannot.

    An unpinned ``uses:`` is a remotely-rewritable execution entry point into
    CI, which matters more here than usual because one of these workflows runs
    on a self-hosted machine.
    """
    problems = []
    for name, text in raw_workflows.items():
        for reference in _USES.findall(text):
            if reference.startswith("./") or reference.startswith("docker://"):
                continue
            if not SHA_PINNED.match(reference):
                problems.append(f"{name}: {reference}")
    assert not problems, "unpinned actions: " + ", ".join(problems)


def test_pinned_actions_carry_a_human_readable_version_comment(raw_workflows):
    for name, text in raw_workflows.items():
        for line in text.splitlines():
            if "uses:" in line and "@" in line and "./" not in line:
                assert "#" in line, f"{name}: no version comment on: {line.strip()}"


# --------------------------------------------------------------------------- #
# secrets and cost
# --------------------------------------------------------------------------- #


def test_zero_cost_workflows_reference_no_secrets(raw_workflows):
    """ci, integration-contract and report must run without any credential."""
    for name in ("ci.yml", "integration-contract.yml", "report.yml"):
        referenced = set(_SECRET_REF.findall(raw_workflows[name]))
        # github.token is not a repository secret; it is the built-in token.
        assert not referenced, f"{name} references secrets {sorted(referenced)}"


def test_only_the_smoke_workflow_may_reference_repository_secrets(raw_workflows):
    referenced = set(_SECRET_REF.findall(raw_workflows["benchmark-smoke.yml"]))
    assert referenced <= {"MODEL_API_KEY", "OPENROUTER_API_KEY"}, sorted(referenced)


def test_secrets_are_never_declared_at_workflow_level(workflows):
    """A top-level env with a secret exposes it to every step, including
    third-party actions that have no business seeing it."""
    for name, data in workflows.items():
        assert "secrets." not in str(data.get("env", {})), name


def test_the_paid_path_requires_manual_dispatch_and_acknowledgement(workflows, raw_workflows):
    smoke = workflows["benchmark-smoke.yml"]
    assert set(smoke["on"]) == {"workflow_dispatch"}, "the paid path must not auto-trigger"
    assert "i-accept-model-costs" in raw_workflows["benchmark-smoke.yml"]


def test_the_paid_path_runs_in_an_approvable_environment(workflows):
    smoke = workflows["benchmark-smoke.yml"]
    for job in ("agent-oragentbench", "agent-frontieror"):
        assert smoke["jobs"][job].get("environment") == "benchmark-agent", job
    assert smoke["jobs"]["oracle-controls"].get("environment") == "controls"


def test_fork_pull_requests_cannot_reach_the_self_hosted_runner(workflows, raw_workflows):
    smoke = workflows["benchmark-smoke.yml"]
    # Manual dispatch only, plus an explicit fork refusal in preflight.
    assert "pull_request" not in smoke["on"]
    assert "pull_request_target" not in smoke["on"]
    assert "github.event.repository.fork" in raw_workflows["benchmark-smoke.yml"]


def test_no_workflow_uses_pull_request_target(raw_workflows):
    """pull_request_target runs fork code with access to secrets."""
    for name, text in raw_workflows.items():
        assert "pull_request_target" not in text or name == "benchmark-smoke.yml", name


def test_self_hosted_jobs_are_confined_to_the_smoke_workflow(workflows):
    for name, data in workflows.items():
        for job_name, job in data["jobs"].items():
            runs_on = job["runs-on"]
            is_self_hosted = "self-hosted" in (
                runs_on if isinstance(runs_on, list) else [runs_on]
            )
            if is_self_hosted:
                assert name == "benchmark-smoke.yml", f"{name}:{job_name}"


# --------------------------------------------------------------------------- #
# fail-closed behaviour
# --------------------------------------------------------------------------- #


def test_the_smoke_workflow_fails_closed_on_missing_prerequisites(raw_workflows):
    text = raw_workflows["benchmark-smoke.yml"]
    for needle in (
        "OPENROUTER_API_KEY is not configured",
        "GRB_LICENSE_FILE must point at a readable Gurobi licence",
        "docker is not available on this runner",
        "the harbor CLI is not installed on this runner",
    ):
        assert needle in text, needle


def test_frontieror_on_a_non_isolated_runner_is_refused(raw_workflows):
    text = raw_workflows["benchmark-smoke.yml"]
    assert "ORBENCH_PERF_ISOLATED" in text
    assert "would not be an official score" in text


def test_github_hosted_frontieror_runs_are_labelled_exploratory(raw_workflows):
    text = raw_workflows["benchmark-smoke.yml"]
    assert "evidence_label=exploratory" in text
    assert "exploratory by construction" in text


def test_both_agent_paths_invoke_the_checked_script(raw_workflows, workflows):
    """The agent paths are real: each hands off to a tested command builder."""
    smoke = workflows["benchmark-smoke.yml"]
    text = raw_workflows["benchmark-smoke.yml"]
    for job in ("agent-oragentbench", "agent-frontieror"):
        assert job in smoke["jobs"], job
    assert text.count("tools/run_benchmark_smoke.py") >= 3
    assert "--execute --acknowledge-cost" in text
    # And no trace of the placeholder that used to sit here.
    assert "has no execution path in this release" not in text


def test_the_frontieror_agent_job_requires_a_performance_isolated_runner(workflows):
    runs_on = workflows["benchmark-smoke.yml"]["jobs"]["agent-frontieror"]["runs-on"]
    assert set(runs_on) == {"self-hosted", "orbench-exec", "perf-isolated"}


def test_the_oragentbench_agent_job_does_not_claim_performance_isolation(workflows):
    """Only the benchmark that scores runtime needs that label."""
    runs_on = workflows["benchmark-smoke.yml"]["jobs"]["agent-oragentbench"]["runs-on"]
    assert "perf-isolated" not in runs_on


def test_the_preflight_runs_a_zero_cost_upstream_check(raw_workflows):
    """Upstream's own wrapper validates the compiled config on every dispatch."""
    text = raw_workflows["benchmark-smoke.yml"]
    assert "--upstream-dry-run --execute" in text
    # Normalise line wrapping and comment markers: the claim matters, its
    # position in the YAML comment does not.
    prose = " ".join(text.replace("#", " ").split())
    assert "neither Harbor, Docker nor a credential" in prose


def test_agent_jobs_upload_only_a_receipt(workflows):
    smoke = workflows["benchmark-smoke.yml"]
    for job in ("agent-oragentbench", "agent-frontieror"):
        uploads = [
            step for step in smoke["jobs"][job]["steps"]
            if str(step.get("uses", "")).startswith("actions/upload-artifact")
        ]
        assert uploads, job
        for step in uploads:
            assert step["with"]["path"].strip() == "out/agent-receipt.json", job


def test_report_workflow_guards_against_raw_bundle_uploads(raw_workflows):
    text = raw_workflows["report.yml"]
    assert "trajectory.json" in text
    assert "reference_objective" in text
    assert "PRIVATE KEY" in text


def test_integration_contract_asserts_the_static_read_property(raw_workflows):
    text = raw_workflows["integration-contract.yml"]
    assert '"model_calls": 0' in text
    assert '"benchmark_executed": False' in text


def test_workflows_clone_through_the_shared_pinned_script(raw_workflows, repo_root):
    """One clone path, so the pin cannot drift between workflows."""
    script = repo_root / ".github/scripts/clone-upstream.sh"
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    # The pin comes from the integration registry, not from a literal in YAML.
    assert "registry.describe" in body
    assert "--depth 1 origin" in body
    for name in ("integration-contract.yml", "benchmark-smoke.yml"):
        assert ".github/scripts/clone-upstream.sh" in raw_workflows[name], name


def test_the_clone_script_uses_the_directory_names_upstream_requires(repo_root):
    """ORAgentBench resolves its dataset path against the checkout's parent."""
    body = (repo_root / ".github/scripts/clone-upstream.sh").read_text(encoding="utf-8")
    assert 'DIRNAME="ORAgentBench"' in body
    assert 'DIRNAME="FrontierOR"' in body


def test_ci_verifies_plan_determinism_and_the_report_golden(raw_workflows):
    text = raw_workflows["ci.yml"]
    assert "diff -r /tmp/plan-a /tmp/plan-b" in text
    assert "tests/golden/oragentbench-controls.summary.md" in text


# --------------------------------------------------------------------------- #
# the shell and python embedded in workflows
# --------------------------------------------------------------------------- #


def _run_blocks(workflows: dict[str, dict]):
    for name, data in workflows.items():
        for job_name, job in data["jobs"].items():
            for index, step in enumerate(job["steps"]):
                script = step.get("run")
                if script:
                    yield f"{name}:{job_name}[{index}]", script


def test_every_run_block_is_valid_bash(workflows):
    """A shell typo in a workflow is only discovered when the workflow runs."""
    import subprocess

    problems = []
    for location, script in _run_blocks(workflows):
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        if result.returncode != 0:
            problems.append(f"{location}: {result.stderr.strip()}")
    assert not problems, "\n".join(problems)


def test_embedded_python_is_not_indented(workflows):
    """``python -c`` and ``python - <<`` bodies must start at column zero.

    YAML block scalars strip the common indentation, so this holds today — but
    it holds by accident of formatting, and an editor that re-indents one line
    turns it into an IndentationError that only appears on GitHub.
    """
    problems = []
    for location, script in _run_blocks(workflows):
        lines = script.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            starts_python = stripped.startswith("python -c") or stripped.startswith("python - <<")
            starts_python = starts_python or ("=$(python -c" in stripped)
            if not starts_python or index + 1 >= len(lines):
                continue
            body = lines[index + 1]
            if body.strip() and body != body.lstrip():
                problems.append(f"{location} line {index + 2}: indented python body {body!r}")
    assert not problems, "\n".join(problems)


def test_embedded_python_compiles(workflows):
    """Compile heredoc python bodies so a syntax error fails here, not on GitHub."""
    problems = []
    for location, script in _run_blocks(workflows):
        lines = script.splitlines()
        index = 0
        while index < len(lines):
            if lines[index].strip().endswith("<<'PY'"):
                end = index + 1
                while end < len(lines) and lines[end].strip() != "PY":
                    end += 1
                body = "\n".join(lines[index + 1 : end])
                try:
                    compile(body, f"{location}", "exec")
                except SyntaxError as exc:
                    problems.append(f"{location}: {exc}")
                index = end
            index += 1
    assert not problems, "\n".join(problems)
