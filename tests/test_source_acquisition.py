from __future__ import annotations

from pathlib import Path

import pytest

from orbenchlab import source_acquisition as sa


def _fetcher(payload: bytes):
    def fetch(url, max_bytes):
        return {"bytes": payload, "content_type": "application/pdf", "final_url": url}

    return fetch


def test_acquire_freezes_bytes_and_signs_provenance(tmp_path: Path):
    source = {"id": "s1", "url": "https://arxiv.org/pdf/2401.01234", "license": "CC-BY-4.0"}
    rec = sa.acquire_source(source=source, out=tmp_path, fetcher=_fetcher(b"PAPER BYTES"))
    assert rec["reused"] is False
    assert rec["byte_count"] == len(b"PAPER BYTES")
    assert rec["canonical_identity"] == "arxiv_id:2401.01234"
    assert rec["declared_license"] == "CC-BY-4.0"
    assert Path(rec["frozen_path"]).read_bytes() == b"PAPER BYTES"
    assert sa.verify_acquisition(receipt=rec, out=tmp_path) is True


def test_acquire_is_idempotent_by_content(tmp_path: Path):
    source = {"id": "s1", "url": "https://arxiv.org/pdf/2401.01234"}
    calls = []

    def counting_fetch(url, max_bytes):
        calls.append(url)
        return {"bytes": b"SAME", "content_type": "text/plain", "final_url": url}

    first = sa.acquire_source(source=source, out=tmp_path, fetcher=counting_fetch)
    second = sa.acquire_source(source=source, out=tmp_path, fetcher=counting_fetch)
    assert first["content_digest"] == second["content_digest"]
    assert second["reused"] is True


def test_verify_rejects_tampered_frozen_bytes(tmp_path: Path):
    # Acceptance 7 (provenance half): a tampered frozen artifact is rejected.
    source = {"id": "s1", "url": "https://arxiv.org/pdf/2401.01234"}
    rec = sa.acquire_source(source=source, out=tmp_path, fetcher=_fetcher(b"ORIGINAL"))
    Path(rec["frozen_path"]).write_bytes(b"TAMPERED")
    with pytest.raises(sa.SourceAcquisitionError, match="do not match"):
        sa.verify_acquisition(receipt=rec, out=tmp_path)


def test_acquire_rejects_oversize(tmp_path: Path):
    def big(url, max_bytes):
        return {"bytes": b"x" * (max_bytes + 5), "content_type": "text/plain", "final_url": url}

    with pytest.raises(sa.SourceAcquisitionError, match="byte cap"):
        sa.acquire_source(source={"url": "https://x/y"}, out=tmp_path, fetcher=big, max_bytes=10)
