"""One-command daily source pipeline orchestrator (P0-E).

Ties the harness-owned gates and the semantic triage into a single idempotent
daily run:

    collected -> acquired -> exact-deduped -> license-allowed
              -> provenance-verified -> semantic-triaged -> admitted-for-authoring

The order is deliberate: a duplicate, an unknown/denied license, or unverifiable
provenance is settled BEFORE any triage or provider spend, so an ineligible
source triggers zero paid downstream calls.  The cross-day registry persists, so
a source already seen on a prior day is deduped without re-triage.  Per-source
results are content-addressed and reused on resume, and the worst-case triage
liability is capped before spending.  Admitted sources are written to a
digest-bound candidate manifest for the triaged-intake batch provider.

The fetcher and the per-source review-runner factory are injected, so the whole
orchestration is unit-tested with fakes and the identical driver runs real
sandboxed sessions on the host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import source_acquisition, source_manifest, source_triage
from .core.errors import ORBenchError
from .source_registry import LicenseAuthority, SourceRegistry

DAILY_SCHEMA = "orbenchlab.daily-source-run.v1"


class DailyOrchestratorError(ORBenchError):
    exit_code = 8


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sources_from_intake(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt deterministic G0 intake items into ``run_daily`` source rows.

    Only items with a fetchable canonical url are forwarded; the metadata-only
    intake carries no license, so the license stays unresolved and the harness
    defers it (never admits) until acquisition/authority settles it.
    """

    import re

    rows: list[dict[str, Any]] = []
    for item in items:
        url = item.get("canonical_url") or item.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        row: dict[str, Any] = {
            "url": url,
            "channel": item.get("source_kind") or "unknown",
            "intake_content_digest": item.get("content_digest"),
        }
        external = str(item.get("external_id") or "")
        if re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?", external):
            row["arxiv_id"] = re.sub(r"v[0-9]+$", "", external)
        elif re.fullmatch(r"10\.\d{4,9}/\S+", external):
            row["doi"] = external
        elif external:
            row["canonical_id"] = external
        if item.get("license"):
            row["license"] = item["license"]
        rows.append(row)
    return rows


def run_daily(
    *,
    day: str,
    sources: Sequence[Mapping[str, Any]],
    out: str | Path,
    fetcher: source_acquisition.Fetcher,
    make_review_runner: Callable[[Mapping[str, Any]], source_triage.ReviewRunner],
    make_adjudicator_runner: Callable[[Mapping[str, Any]], source_triage.ReviewRunner] | None = None,
    registry_path: str | Path | None = None,
    license_authority: LicenseAuthority | None = None,
    per_triage_session_usd: float = 0.25,
    max_daily_triage_liability_usd: float = 20.0,
) -> dict[str, Any]:
    """Run or resume one day of the source pipeline; return a signed receipt."""

    root = Path(out)
    registry = SourceRegistry(registry_path or (root / "registry.json"))
    authority = license_authority or LicenseAuthority()
    results: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    spent = 0.0

    for source in sources:
        source_key = _digest({"day": day, "source": dict(source)})[:24]
        result_path = root / "pipeline" / source_key / "result.json"
        # Content-addressed resume: a completed per-source result is reused.
        if result_path.is_file() and not result_path.is_symlink():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(cached)
            spent += float(cached.get("triage_liability_usd") or 0.0)
            if cached.get("status") == "admitted" and cached.get("_admit_row"):
                admitted.append(cached["_admit_row"])
            continue

        record: dict[str, Any] = {"source_key": source_key, "url": source.get("url") or source.get("source_url")}
        # 1. Acquire (freeze bytes + provenance).
        try:
            acquisition = source_acquisition.acquire_source(source=source, out=root, fetcher=fetcher)
        except source_acquisition.SourceAcquisitionError as exc:
            record.update(status="acquire-error", detail=str(exc))
            _finish(result_path, record, results)
            continue
        record["content_digest"] = acquisition["content_digest"]

        # 2. Exact / cross-day dedup.
        resolution = registry.resolve(source=source, content_digest=acquisition["content_digest"])
        record["registry"] = resolution
        if resolution["status"] == "duplicate-exact":
            record["status"] = "duplicate"
            _finish(result_path, record, results)
            continue

        # 3. License authority (before any spend).
        license_receipt = authority.decide(
            source_id=acquisition["source_id"],
            declared_license=acquisition.get("declared_license"),
            authority="spdx-allowlist",
        )
        record["license_decision"] = license_receipt["decision"]
        if license_receipt["decision"] != "allowed":
            record["status"] = "deferred-license" if license_receipt["decision"] == "deferred-license" else "license-denied"
            _finish(result_path, record, results)
            continue

        # 4. Provenance re-verification (before any spend).
        try:
            source_acquisition.verify_acquisition(receipt=acquisition, out=root)
        except source_acquisition.SourceAcquisitionError as exc:
            record.update(status="provenance-error", detail=str(exc))
            _finish(result_path, record, results)
            continue

        # 5. Triage — the first (and only) paid step. Cap worst-case liability.
        projected = spent + 3 * per_triage_session_usd
        if projected > max_daily_triage_liability_usd:
            record["status"] = "deferred-budget"
            _finish(result_path, record, results)
            continue
        triage = source_triage.run_triage(
            source_id=acquisition["source_id"],
            content_digest=acquisition["content_digest"],
            day=day,
            out=root / "triage" / source_key,
            review_runner=make_review_runner(acquisition),
            adjudicator_runner=(make_adjudicator_runner(acquisition) if make_adjudicator_runner else None),
        )
        # Liability = the sessions actually run (2 reviews + adjudicator if used).
        sessions = 2 + (1 if triage.get("adjudication") is not None else 0)
        triage_cost = sessions * per_triage_session_usd
        spent += triage_cost
        record["triage_liability_usd"] = round(triage_cost, 6)
        record["triage_verdict"] = triage["verdict"]
        if triage["verdict"] == "eligible_for_authoring":
            admit_row = {"acquisition": acquisition, "license": license_receipt, "triage": triage}
            record["status"] = "admitted"
            record["_admit_row"] = admit_row
            admitted.append(admit_row)
        else:
            record["status"] = "triaged-rejected"
        _finish(result_path, record, results)

    manifest = source_manifest.build_candidate_manifest(admitted=admitted, out=root / "manifest", day=day)
    receipt = {
        "schema_version": DAILY_SCHEMA,
        "day": day,
        "source_count": len(sources),
        "status_counts": _counts(results),
        "admitted_count": len(admitted),
        "total_triage_liability_usd": round(spent, 6),
        "candidate_manifest": str(root / "manifest" / "candidate-manifest.json"),
        "candidate_manifest_digest": manifest["manifest_digest"],
        "results": [{k: v for k, v in r.items() if k != "_admit_row"} for r in results],
    }
    receipt["receipt_digest"] = _digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    _atomic_json(root / f"daily-{day}.json", receipt)
    return receipt


def _finish(result_path: Path, record: Mapping[str, Any], results: list) -> None:
    _atomic_json(result_path, record)
    results.append(dict(record))


def _counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[str(r.get("status"))] = counts.get(str(r.get("status")), 0) + 1
    return counts


__all__ = ["DAILY_SCHEMA", "DailyOrchestratorError", "run_daily", "sources_from_intake"]
