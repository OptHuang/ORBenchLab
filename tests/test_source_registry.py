from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import source_registry as sr


def test_license_authority_allows_only_the_allowlist():
    auth = sr.LicenseAuthority()
    assert auth.decide(source_id="s1", declared_license="MIT", authority="spdx")["decision"] == "allowed"
    assert auth.decide(source_id="s1", declared_license="Apache-2.0", authority="spdx")["allowed_for_downstream"] is True
    # Unknown / pending -> deferred, never allowed.
    for lic in (None, "", "unknown", "pending-human", "CC-BY-NC-4.0", "GPL-3.0", "arxiv-nonexclusive"):
        d = auth.decide(source_id="s1", declared_license=lic, authority="spdx")
        assert d["allowed_for_downstream"] is False
        assert d["decision"] in {"deferred-license", "denied"}


def test_canonical_identity_unifies_channels():
    # arXiv abs, arXiv pdf, and an explicit arxiv_id map to one identity.
    assert sr.canonical_identity({"url": "https://arxiv.org/abs/2401.01234"}) == "arxiv_id:2401.01234"
    assert sr.canonical_identity({"url": "https://arxiv.org/pdf/2401.01234"}) == "arxiv_id:2401.01234"
    assert sr.canonical_identity({"arxiv_id": "2401.01234"}) == "arxiv_id:2401.01234"
    assert sr.canonical_identity({"url": "https://github.com/org/repo.git"}) == "repo:github.com/org/repo"


def test_registry_dedupes_same_work_from_multiple_channels(tmp_path: Path):
    # Acceptance 4 (harness half): same licensed OR work from arXiv + RSS + repo
    # becomes ONE canonical entry (aliases), not three candidates.
    reg = sr.SourceRegistry(tmp_path / "registry.json")
    digest = "sha256:" + "a" * 64
    r1 = reg.resolve(source={"channel": "arxiv", "url": "https://arxiv.org/abs/2401.01234"}, content_digest=digest)
    assert r1["status"] == "new" and r1["version"] == 1
    # Same bytes again (exact duplicate) -> duplicate-exact, no new entry.
    r2 = reg.resolve(source={"channel": "rss", "url": "https://arxiv.org/abs/2401.01234"}, content_digest=digest)
    assert r2["status"] == "duplicate-exact"
    assert len(reg.entries) == 1
    # Same canonical work, different bytes from a repo mirror -> new version + alias.
    r3 = reg.resolve(
        source={"channel": "repo", "url": "https://arxiv.org/abs/2401.01234", "arxiv_id": "2401.01234"},
        content_digest="sha256:" + "b" * 64,
    )
    assert r3["status"] == "new-version" and r3["version"] == 2
    assert len(reg.entries) == 1  # still one canonical work


def test_registry_cross_day_resume_and_new_version(tmp_path: Path):
    # Acceptance 8: v1 recorded; alias deduped; a new candidate admitted; and a
    # v2 recorded — all surviving a reload (new process) without duplication.
    path = tmp_path / "registry.json"
    reg = sr.SourceRegistry(path)
    reg.resolve(source={"channel": "arxiv", "arxiv_id": "2401.01234"}, content_digest="sha256:" + "1" * 64)
    reg.resolve(source={"channel": "arxiv", "arxiv_id": "9999.99999"}, content_digest="sha256:" + "2" * 64)

    # New day / new process: reload from disk.
    reg2 = sr.SourceRegistry(path)
    # Alias of the first work (same id, same bytes) -> duplicate-exact.
    assert reg2.resolve(source={"channel": "rss", "arxiv_id": "2401.01234"}, content_digest="sha256:" + "1" * 64)["status"] == "duplicate-exact"
    # A brand new work is admitted.
    assert reg2.resolve(source={"channel": "arxiv", "arxiv_id": "2402.02222"}, content_digest="sha256:" + "3" * 64)["status"] == "new"
    # A new revision of the first work -> version 2.
    v2 = reg2.resolve(source={"channel": "arxiv", "arxiv_id": "2401.01234"}, content_digest="sha256:" + "4" * 64)
    assert v2["status"] == "new-version" and v2["version"] == 2
    assert len(reg2.entries) == 3


def test_registry_rejects_tamper(tmp_path: Path):
    # Acceptance 7 (registry half): a tampered registry file is rejected, not
    # silently trusted.
    path = tmp_path / "registry.json"
    reg = sr.SourceRegistry(path)
    reg.resolve(source={"channel": "arxiv", "arxiv_id": "2401.01234"}, content_digest="sha256:" + "1" * 64)
    doc = json.loads(path.read_text())
    doc["entries"]["arxiv_id:2401.01234"]["versions"][0]["content_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(sr.SourceRegistryError, match="tamper|digest"):
        sr.SourceRegistry(path)
