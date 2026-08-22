from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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
