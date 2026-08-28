"""Harness-owned eligibility backbone for the daily source pipeline (P0-E).

The harness alone decides bytes, license, identity, receipts, and eligibility;
the agent decides only semantics.  This module owns three deterministic pieces:

- :class:`LicenseAuthority` — an authority-backed license decision.  A source's
  declared SPDX/license id maps to ``allowed`` only from an explicit allowlist;
  a denylisted id is ``denied``; anything unknown or pending is
  ``deferred-license``.  Unknown is never allowed, so no downstream/paid call
  happens for an unresolved license.

- :class:`SourceRegistry` — a persistent, cross-day canonical/version/alias
  registry.  Exact content digests dedupe to one canonical entry; the same
  work arriving from arXiv + RSS + a repo (same canonical id) dedupes to one
  alias set; a new content digest under a known canonical id is a new version.
  Append-only and content-addressed, so a resume never forks or re-admits.

- ``eligibility`` — the combined harness verdict a source must earn (bytes
  acquired + provenance intact + license allowed + not a duplicate) before any
  semantic triage or paid factory work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.errors import ORBenchError

LICENSE_RECEIPT_SCHEMA = "orbenchlab.license-receipt.v1"
REGISTRY_SCHEMA = "orbenchlab.source-registry.v1"


class SourceRegistryError(ORBenchError):
    exit_code = 8


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# License authority


# Permissive licenses usable for OR-benchmark task derivation. Normalised to
# lowercase with '-'/' ' collapsed to '_'.
_ALLOWED_LICENSES = frozenset(
    {
        "mit",
        "bsd_2_clause",
        "bsd_3_clause",
        "apache_2.0",
        "apache_2_0",
        "cc0_1.0",
        "cc0_1_0",
        "cc_by_4.0",
        "cc_by_4_0",
        "cc_by_3.0",
        "cc_by_3_0",
        "isc",
        "unlicense",
        "public_domain",
    }
)
# Explicitly non-usable (share-alike/non-commercial/no-derivatives/proprietary).
_DENIED_LICENSES = frozenset(
    {
        "cc_by_nc_4.0",
        "cc_by_nc_4_0",
        "cc_by_nd_4.0",
        "cc_by_sa_4.0",
        "cc_by_nc_nd_4.0",
        "gpl_3.0",
        "gpl_2.0",
        "agpl_3.0",
        "proprietary",
        "all_rights_reserved",
        "arxiv_nonexclusive",  # arXiv's default license is not a redistribution grant
    }
)


def _normalise_license(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


@dataclass(frozen=True)
class LicenseAuthority:
    """Authority-backed license decision. Unknown is never allowed."""

    allowed: frozenset[str] = _ALLOWED_LICENSES
    denied: frozenset[str] = _DENIED_LICENSES

    def decide(self, *, source_id: str, declared_license: Any, authority: str) -> dict[str, Any]:
        norm = _normalise_license(declared_license)
        if not norm or norm in {"unknown", "none", "unspecified", "pending", "pending_human"}:
            decision = "deferred-license"
            reason = "license unknown or pending; no downstream spend permitted"
        elif norm in self.denied:
            decision = "denied"
            reason = "license is on the non-usable denylist"
        elif norm in self.allowed:
            decision = "allowed"
            reason = "license is on the permissive allowlist"
        else:
            decision = "deferred-license"
            reason = "license is not on the authority allowlist"
        receipt = {
            "schema_version": LICENSE_RECEIPT_SCHEMA,
            "source_id": source_id,
            "declared_license": str(declared_license or "") or None,
            "normalised_license": norm or None,
            "authority": authority,
            "decision": decision,
            "allowed_for_downstream": decision == "allowed",
            "reason": reason,
        }
        receipt["receipt_digest"] = _digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
        return receipt


# ---------------------------------------------------------------------------
# Cross-day canonical/version/alias registry


def canonical_identity(source: Mapping[str, Any]) -> str | None:
    """Derive a stable canonical identity for a source across channels.

    An arXiv id, DOI, or normalised repo URL identifies the same underlying
    work regardless of whether it arrived via arXiv, an RSS feed, or a repo.
    """

    for key in ("arxiv_id", "doi", "canonical_id"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip().lower()}"
    url = source.get("url") or source.get("source_url")
    if isinstance(url, str) and url.strip():
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
        if m:
            return f"arxiv_id:{m.group(1)}"
        m = re.search(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", url.lower())
        if m:
            return f"doi:{m.group(1)}"
        m = re.search(r"github\.com/([^/]+/[^/#?]+)", url.lower())
        if m:
            return f"repo:github.com/{m.group(1).removesuffix('.git')}"
    return None


class SourceRegistry:
    """Append-only, content-addressed cross-day source registry."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.is_file() and not self._path.is_symlink():
            doc = json.loads(self._path.read_text(encoding="utf-8"))
            if doc.get("schema_version") != REGISTRY_SCHEMA:
                raise SourceRegistryError("source registry schema mismatch")
            unsigned = {k: v for k, v in doc.items() if k != "registry_digest"}
            if doc.get("registry_digest") != _digest(unsigned):
                raise SourceRegistryError("source registry digest mismatch (possible tamper)")
            return doc
        return {"schema_version": REGISTRY_SCHEMA, "entries": {}, "by_content": {}, "next_ordinal": 1}

    def _save(self) -> None:
        unsigned = {k: v for k, v in self._state.items() if k != "registry_digest"}
        self._state["registry_digest"] = _digest(unsigned)
        _atomic_json(self._path, self._state)

    def resolve(self, *, source: Mapping[str, Any], content_digest: str) -> dict[str, Any]:
        """Classify a source as new / duplicate-alias / new-version / duplicate-exact.

        Idempotent and append-only: re-resolving an already-recorded
        (canonical, content) pair returns the same verdict without a new entry.
        """

        identity = canonical_identity(source) or f"content:{content_digest}"
        entries: dict[str, Any] = self._state["entries"]
        by_content: dict[str, str] = self._state["by_content"]

        # Exact content already seen anywhere -> duplicate-exact.
        if content_digest in by_content:
            canonical = by_content[content_digest]
            entry = entries[canonical]
            version = next(
                (v["version"] for v in entry["versions"] if v["content_digest"] == content_digest),
                entry["versions"][-1]["version"],
            )
            return {"status": "duplicate-exact", "canonical_id": canonical, "version": version, "reused": True}

        if identity in entries:
            entry = entries[identity]
            # Known canonical work, new bytes -> new version; also an alias if a
            # different channel/url than previously recorded.
            version = len(entry["versions"]) + 1
            entry["versions"].append({"version": version, "content_digest": content_digest})
            alias = self._channel(source)
            is_alias = alias not in entry["aliases"]
            if is_alias:
                entry["aliases"].append(alias)
            by_content[content_digest] = identity
            self._save()
            return {
                "status": "new-version",
                "canonical_id": identity,
                "version": version,
                "alias_added": is_alias,
                "reused": False,
            }

        ordinal = self._state["next_ordinal"]
        self._state["next_ordinal"] = ordinal + 1
        entries[identity] = {
            "canonical_id": identity,
            "ordinal": ordinal,
            "aliases": [self._channel(source)],
            "versions": [{"version": 1, "content_digest": content_digest}],
        }
        by_content[content_digest] = identity
        self._save()
        return {"status": "new", "canonical_id": identity, "version": 1, "reused": False}

    @staticmethod
    def _channel(source: Mapping[str, Any]) -> str:
        channel = source.get("channel") or source.get("source_kind") or "unknown"
        url = source.get("url") or source.get("source_url") or ""
        return f"{channel}:{url}"

    @property
    def entries(self) -> dict[str, Any]:
        return dict(self._state["entries"])


__all__ = [
    "LICENSE_RECEIPT_SCHEMA",
    "LicenseAuthority",
    "REGISTRY_SCHEMA",
    "SourceRegistry",
    "SourceRegistryError",
    "canonical_identity",
]
