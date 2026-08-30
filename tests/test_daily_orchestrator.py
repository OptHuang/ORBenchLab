from __future__ import annotations

from pathlib import Path

from orbenchlab import daily_orchestrator as do
from orbenchlab import source_triage as st


def _fetcher(bytes_by_url):
    def fetch(url, max_bytes):
        return {"bytes": bytes_by_url[url], "content_type": "application/pdf", "final_url": url}

    return fetch


def _good(anchor, admit=True):
    return {
        "anchor": anchor,
        "or_relevant": True,
        "novelty_within_bounded_corpus": True,
        "reproducible": True,
        "task_feasible": True,
        "verifier_feasible": True,
        "admit": admit,
        "source_kind": "paper",
        "task_nucleus": "scheduling under seeded durations",
        "difficulty_axes": ["instance-size", "seed-count"],
        "predicted_bottlenecks": ["feasible schedule construction"],
    }


class Reviews:
    """Counts how many reviewer sessions are actually run (paid)."""

    def __init__(self, admit=True):
        self.calls = 0
        self.admit = admit

    def make(self, acquisition):
        def run(reviewer_id, anchor, out_dir):
            self.calls += 1
            return {"decision": _good(anchor, self.admit), "session_status": "completed",
                    "session_receipt_digest": "sha256:" + "a" * 64}
        return run


def test_multichannel_same_work_admits_one_after_two_reviews(tmp_path: Path):
    # Acceptance 4: same licensed OR work from arXiv + RSS + repo -> ONE admitted
    # candidate after exactly two reviews (the duplicates are not re-triaged).
    same = b"IDENTICAL PAPER BYTES"
    sources = [
        {"url": "https://arxiv.org/abs/2401.01234", "channel": "arxiv", "arxiv_id": "2401.01234", "license": "MIT"},
        {"url": "https://rss.example/2401.01234", "channel": "rss", "arxiv_id": "2401.01234", "license": "MIT"},
        {"url": "https://github.com/org/repo", "channel": "repo", "arxiv_id": "2401.01234", "license": "MIT"},
    ]
    fetch = _fetcher({s["url"]: same for s in sources})
    reviews = Reviews(admit=True)
    receipt = do.run_daily(
        day="2026-08-28", sources=sources, out=tmp_path, fetcher=fetch,
        make_review_runner=reviews.make,
    )
    assert receipt["admitted_count"] == 1
    assert reviews.calls == 2  # exactly two reviews, once, for the single work
    assert receipt["status_counts"].get("duplicate") == 2
    from orbenchlab import source_manifest
    manifest = source_manifest.load_verified_manifest(receipt["candidate_manifest"])
    assert manifest["entry_count"] == 1


def test_unknown_license_defers_with_zero_downstream_calls(tmp_path: Path):
    # Acceptance 5: excellent semantics would admit, but an unknown license ->
    # deferred-license and ZERO reviewer (paid) calls.
    sources = [{"url": "https://arxiv.org/abs/2402.02222", "channel": "arxiv",
                "arxiv_id": "2402.02222", "license": "unknown"}]
    fetch = _fetcher({sources[0]["url"]: b"PAPER"})
    reviews = Reviews(admit=True)
    receipt = do.run_daily(
        day="d", sources=sources, out=tmp_path, fetcher=fetch, make_review_runner=reviews.make,
    )
    assert receipt["admitted_count"] == 0
    assert reviews.calls == 0
    assert receipt["status_counts"].get("deferred-license") == 1


def test_cross_day_dedup_and_new_admit(tmp_path: Path):
    # Acceptance 8: day-1 admits work A; day-2 dedupes A (no re-triage) and
    # admits a new work B, sharing a persistent registry.
    registry = tmp_path / "registry.json"
    a = {"url": "https://arxiv.org/abs/2401.01234", "channel": "arxiv", "arxiv_id": "2401.01234", "license": "MIT"}
    b = {"url": "https://arxiv.org/abs/2405.05555", "channel": "arxiv", "arxiv_id": "2405.05555", "license": "MIT"}
    fetch = _fetcher({a["url"]: b"PAPER-A", b["url"]: b"PAPER-B"})

    r1 = Reviews(admit=True)
    day1 = do.run_daily(day="2026-08-28", sources=[a], out=tmp_path / "d1", fetcher=fetch,
                        make_review_runner=r1.make, registry_path=registry)
    assert day1["admitted_count"] == 1 and r1.calls == 2

    r2 = Reviews(admit=True)
    day2 = do.run_daily(day="2026-08-29", sources=[a, b], out=tmp_path / "d2", fetcher=fetch,
                        make_review_runner=r2.make, registry_path=registry)
    # A is a cross-day duplicate (not re-triaged); only B is triaged and admitted.
    assert day2["status_counts"].get("duplicate") == 1
    assert day2["admitted_count"] == 1
    assert r2.calls == 2  # only B reviewed


def test_daily_resume_reuses_per_source_results(tmp_path: Path):
    sources = [{"url": "https://arxiv.org/abs/2401.01234", "channel": "arxiv",
                "arxiv_id": "2401.01234", "license": "MIT"}]
    fetch = _fetcher({sources[0]["url"]: b"PAPER"})
    r1 = Reviews(admit=True)
    first = do.run_daily(day="d", sources=sources, out=tmp_path, fetcher=fetch, make_review_runner=r1.make)
    r2 = Reviews(admit=True)
    second = do.run_daily(day="d", sources=sources, out=tmp_path, fetcher=fetch, make_review_runner=r2.make)
    assert first["admitted_count"] == second["admitted_count"] == 1
    assert r2.calls == 0  # resume reused the per-source result, no re-review


def test_sources_from_intake_maps_ids_and_skips_unfetchable():
    items = [
        {"canonical_url": "https://arxiv.org/abs/2401.01234", "source_kind": "arxiv",
         "external_id": "2401.01234v2", "content_digest": "sha256:aa"},
        {"canonical_url": "https://doi.org/10.1000/xyz", "source_kind": "rss",
         "external_id": "10.1000/xyz", "content_digest": "sha256:bb"},
        {"canonical_url": None, "source_kind": "arxiv", "external_id": "9999.99999"},  # unfetchable
    ]
    rows = do.sources_from_intake(items)
    assert len(rows) == 2
    assert rows[0]["arxiv_id"] == "2401.01234"  # version suffix stripped
    assert rows[0]["channel"] == "arxiv"
    assert rows[1]["doi"] == "10.1000/xyz"
    # No license from metadata-only intake -> the pipeline will defer it.
    assert "license" not in rows[0]
