from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep tests away from host locks and real provider credentials."""
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CODEX_AUTH_JSON_PATH",
        "MODEL_API_KEY",
        "MODEL_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ORBENCH_MODEL_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ORBENCH_HOST_LOCK_DIR", str(tmp_path / "host-locks"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sites_dir(repo_root: Path) -> Path:
    return repo_root / "sites"


@pytest.fixture(scope="session")
def campaigns_dir(repo_root: Path) -> Path:
    return repo_root / "campaigns"


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "fixtures"


@pytest.fixture(scope="session")
def golden_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "golden"


@pytest.fixture(scope="session")
def upstream_fixtures(repo_root: Path) -> Path:
    """Miniature stand-ins for upstream checkouts, so tests need no network."""
    return repo_root / "tests" / "fixtures" / "upstream"
