from __future__ import annotations

import json
from pathlib import Path

from orbenchlab import cli, source_intake


class _Item:
    def to_dict(self):
        return {"canonical_url": "https://arxiv.org/abs/2401.01234", "source_kind": "arxiv",
                "external_id": "2401.01234", "content_digest": "sha256:aa"}


class _Result:
    items = (_Item(),)
    feed_errors = 0
    has_errors = False
    intake_id = "intake-1"


def test_source_daily_dry_run_collects_and_adapts(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(source_intake, "load_feed_config", lambda p: ())
    monkeypatch.setattr(source_intake, "collect", lambda feeds: _Result())
    rc = cli.main([
        "source-daily", "--feeds", str(tmp_path / "feeds.json"),
        "--out", str(tmp_path / "out"), "--day", "2026-08-30", "--dry-run",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["collected_items"] == 1
    assert out["fetchable_sources"] == 1
    assert out["intake_id"] == "intake-1"


def test_source_daily_requires_model_without_dry_run(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(source_intake, "load_feed_config", lambda p: ())
    monkeypatch.setattr(source_intake, "collect", lambda feeds: _Result())
    rc = cli.main([
        "source-daily", "--feeds", str(tmp_path / "feeds.json"),
        "--out", str(tmp_path / "out"), "--day", "2026-08-30",
    ])
    assert rc == 2  # missing --model/--claude-executable
