"""Offline and safety tests for the metadata-only source intake prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import source_intake as intake
from orbenchlab.core import schema as schema_mod
from orbenchlab.cli import main


def _feeds() -> tuple[intake.FeedSpec, ...]:
    return intake.validate_config_mapping(
        {
            "version": 1,
            "feeds": [
                {
                    "id": "arxiv",
                    "kind": "arxiv",
                    "url": "https://export.arxiv.org/api/query?search_query=cat:math.OC",
                    "tags": ["paper", "or"],
                },
                {
                    "id": "digest",
                    "kind": "rss",
                    "url": "https://example.org/or.atom",
                    "tags": ["digest"],
                },
                {
                    "id": "github",
                    "kind": "github",
                    "url": "https://api.github.com/repos/example/or/releases",
                    "tags": ["benchmark"],
                },
            ],
        }
    )


def _fixture_fetcher(repo_root: Path):
    bodies = {
        "https://export.arxiv.org/api/query?search_query=cat:math.OC": (
            repo_root / "tests/fixtures/intake/arxiv.atom.xml"
        ).read_bytes(),
        "https://example.org/or.atom": (
            repo_root / "tests/fixtures/intake/rss.atom.xml"
        ).read_bytes(),
        "https://api.github.com/repos/example/or/releases": (
            repo_root / "tests/fixtures/intake/github.json"
        ).read_bytes(),
    }

    def fetch(url: str):
        return intake.FetchResponse(
            bodies[url],
            status=200,
            content_type="application/json" if url.startswith("https://api.github") else "application/atom+xml",
        )

    return fetch


def test_collect_parses_three_kinds_and_deduplicates_cross_feed(repo_root, tmp_path):
    result = intake.collect(
        _feeds(),
        fetcher=_fixture_fetcher(repo_root),
        created_at="2026-08-27T00:00:00Z",
    )

    # Two arXiv occurrences share a canonical URL; the merged item retains
    # both feed ids and an occurrence count rather than duplicating the queue.
    assert len(result.items) == 3
    arxiv_item = next(item for item in result.items if item.source_kind == "arxiv")
    assert arxiv_item.occurrence_count == 2
    assert arxiv_item.feed_ids == ("arxiv", "digest")
    assert arxiv_item.dedupe_status == "new"
    assert len(result.review_queue) == 3
    assert all(row["state"] == "pending" for row in result.review_queue)
    assert result.network_policy["model_calls"] == 0
    assert result.network_policy["credentials_read"] is False

    paths = intake.write_bundle(result, tmp_path / "bundle")
    assert set(paths) == {"intake", "review_queue", "manifest"}
    payload = json.loads(paths["intake"].read_text(encoding="utf-8"))
    schema = schema_mod.load_schema(schema_mod.schemas_dir() / "source_intake.schema.json")
    schema_mod.validate(payload, schema, name="intake")
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["model_calls"] == 0
    assert manifest["review_queue_digest"] == payload["review_queue_digest"]
    assert manifest["files"]["review_queue.jsonl"] == payload["review_queue_digest"]
    assert len(paths["review_queue"].read_text().splitlines()) == 3


def test_previous_snapshot_marks_duplicates_and_changed_metadata_updated(repo_root, tmp_path):
    fetcher = _fixture_fetcher(repo_root)
    first = intake.collect(_feeds(), fetcher=fetcher, created_at="2026-08-27T00:00:00Z")
    bundle = tmp_path / "first"
    intake.write_bundle(first, bundle)

    second = intake.collect(
        _feeds(),
        fetcher=fetcher,
        previous=bundle,
        created_at="2026-08-28T00:00:00Z",
    )
    assert all(item.dedupe_status == "duplicate" for item in second.items)
    assert second.review_queue == ()

    changed_body = (repo_root / "tests/fixtures/intake/rss.atom.xml").read_bytes().replace(
        b"Open scheduling benchmark release", b"Updated scheduling benchmark release"
    )

    def changed_fetch(url: str):
        if url == "https://example.org/or.atom":
            return intake.FetchResponse(changed_body, content_type="application/atom+xml")
        return fetcher(url)

    third = intake.collect(
        _feeds(),
        fetcher=changed_fetch,
        previous=bundle,
        created_at="2026-08-28T00:00:00Z",
    )
    assert any(item.dedupe_status == "updated" for item in third.items)
    assert len(third.review_queue) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/feed",
        "https://user:pass@example.org/feed",
        "https://example.org/feed#fragment",
        "https://example.org/feed?access_token=not-for-artifacts",
    ],
)
def test_feed_url_rejects_non_public_or_credential_bearing_urls(url):
    with pytest.raises(intake.SourceIntakeError):
        intake.validate_feed_url(url)


def test_feed_config_rejects_unknown_keys():
    with pytest.raises(intake.SourceIntakeError, match="unsupported key"):
        intake.validate_config_mapping(
            {
                "version": 1,
                "feeds": [
                    {
                        "id": "x",
                        "kind": "rss",
                        "url": "https://example.org/feed",
                        "headers": {"Authorization": "secret"},
                    }
                ],
            }
        )


def test_partial_feed_failure_is_recorded_without_erasing_success(repo_root):
    good = _fixture_fetcher(repo_root)

    def fetch(url: str):
        if url == "https://example.org/or.atom":
            raise intake.SourceIntakeError("network unavailable")
        return good(url)

    result = intake.collect(_feeds(), fetcher=fetch, created_at="2026-08-27T00:00:00Z")
    assert result.has_errors is True
    failed = next(row for row in result.feeds if row["id"] == "digest")
    assert failed["status"] == "error"
    assert failed["error"] == "network unavailable"
    assert len(result.items) == 2


def test_bundle_refuses_different_overwrite(repo_root, tmp_path):
    result = intake.collect(_feeds(), fetcher=_fixture_fetcher(repo_root), created_at="2026-08-27T00:00:00Z")
    out = tmp_path / "bundle"
    intake.write_bundle(result, out)
    altered = intake.collect(_feeds(), fetcher=_fixture_fetcher(repo_root), created_at="2026-08-28T00:00:00Z")
    with pytest.raises(intake.SourceIntakeError, match="refusing to overwrite"):
        intake.write_bundle(altered, out)


def test_cli_intake_validate_is_network_free(capsys, tmp_path):
    config = tmp_path / "feeds.yaml"
    config.write_text(
        "version: 1\nfeeds:\n  - id: x\n    kind: rss\n    url: https://example.org/feed\n",
        encoding="utf-8",
    )
    assert main(["intake", "validate", "--config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_requests"] == 0
    assert payload["model_calls"] == 0


def test_cli_collect_returns_nonzero_for_partial_feed_but_writes_bundle(
    monkeypatch, capsys, tmp_path
):
    config = tmp_path / "feeds.yaml"
    config.write_text(
        "version: 1\nfeeds:\n  - id: x\n    kind: rss\n    url: https://example.org/feed\n",
        encoding="utf-8",
    )

    def failed_fetch(_url: str):
        raise intake.SourceIntakeError("network unavailable")

    monkeypatch.setattr(intake, "fetch_url", failed_fetch)
    out = tmp_path / "bundle"
    assert main(["intake", "collect", "--config", str(config), "--out", str(out)]) == 8
    payload = json.loads(capsys.readouterr().out)
    assert payload["feed_errors"] == 1
    assert (out / "intake.json").is_file()
