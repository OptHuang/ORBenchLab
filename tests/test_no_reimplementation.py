"""Guard against reimplementing what upstream already owns.

ORBenchLab compiles campaigns, tracks run identity and renders reports. It does
not schedule, execute, retry, resume or re-grade — Harbor does. It does not
score FrontierOR submissions — FrontierOR's trusted harness does.

The temptation to add "just a small runner here" is the failure mode these tests
exist to catch, because a second implementation of an execution concern becomes
a second source of truth about what actually ran.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path("src/orbenchlab")

#: Symbol names that would indicate a reimplementation of an execution concern.
#: Matched against defined class and function names, not against prose.
FORBIDDEN_SYMBOL_SUBSTRINGS = (
    "jobrunner",
    "trialrunner",
    "resume_job",
    "regrade",
    "rewardjudge",
    "judgeframework",
    "retrypolicy",
    "scheduler",
    "dispatchloop",
    "verifierengine",
    "sandbox",
)

#: Modules whose presence would mean we had started executing benchmarks.
FORBIDDEN_IMPORTS = ("docker", "kubernetes", "openai", "anthropic", "litellm")


def _python_sources(repo_root: Path) -> list[Path]:
    return sorted((repo_root / SRC).rglob("*.py"))


def _definitions(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.name, node.lineno))
    return found


def test_no_execution_concern_is_reimplemented(repo_root):
    """Building an upstream command line is delegation, not reimplementation.

    `orbenchlab.execution` assembles `harbor run -c ...` and
    `python -m frontieror.infra agent ...`. What it must never grow is a
    scheduler, a retry loop, a grader or a sandbox of its own — those are the
    symbol names below.
    """
    problems = []
    for path in _python_sources(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, lineno in _definitions(tree):
            lowered = name.lower().replace("_", "")
            for forbidden in FORBIDDEN_SYMBOL_SUBSTRINGS:
                if forbidden.replace("_", "") in lowered:
                    problems.append(
                        f"{path.relative_to(repo_root)}:{lineno}: {name!r} looks like a "
                        f"reimplementation of {forbidden!r}, which upstream owns"
                    )
    assert not problems, "\n".join(problems)


def test_no_execution_or_provider_libraries_are_imported(repo_root):
    problems = []
    for path in _python_sources(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                root = module.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    problems.append(
                        f"{path.relative_to(repo_root)}:{node.lineno}: imports {module!r}"
                    )
    assert not problems, "\n".join(problems)


def test_runtime_dependencies_stay_minimal(repo_root):
    """Every added runtime dependency is a thing that can drift or break."""
    import tomllib

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    names = {dep.split(">")[0].split("=")[0].split("[")[0].strip().lower() for dep in dependencies}
    assert names == {"pyyaml"}, f"unexpected runtime dependencies: {sorted(names)}"


def test_the_package_declares_apache_two(repo_root):
    import tomllib

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"]["text"] == "Apache-2.0"
    license_text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "LICENSE",
        "docs/integrations.md",
        "docs/adding-a-benchmark.md",
        "docs/decisions/0001-frontieror-integration.md",
        "docs/decisions/0002-plan-ledger-run-ids.md",
        "templates/integration_template/integration.py",
        "tools/run_benchmark_smoke.py",
        ".github/scripts/clone-upstream.sh",
    ],
)
def test_required_repository_files_exist(repo_root, path):
    assert (repo_root / path).is_file(), f"{path} is missing"


def test_no_credentials_are_committed(repo_root):
    """A crude but useful check for the obvious mistakes."""
    markers = ("BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY", "sk-ant-", "sk-or-v1-")
    skip_dirs = {".git", ".venv", "__pycache__", "out", "upstream", ".pytest_cache"}
    problems = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or set(path.parts) & skip_dirs:
            continue
        if path.suffix in {".png", ".jpg", ".gz", ".whl", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in markers:
            # The literal appears in this file and in the report guard, as the
            # pattern being searched for rather than as a credential.
            if marker in text and path.name not in {"test_no_reimplementation.py", "report.yml"}:
                problems.append(f"{path.relative_to(repo_root)}: {marker}")
    assert not problems, "\n".join(problems)
