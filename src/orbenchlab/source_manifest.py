"""Digest-bound candidate manifest for admitted sources (P0-E).

Turns sources that earned every harness gate — bytes acquired + provenance
intact + license ``allowed`` + not a duplicate + triage ``eligible_for_authoring``
— into a signed manifest the batch factory consumes.  Each entry binds the
acquisition, license and triage receipt digests and points at a per-candidate
``paper-provenance.json`` (license_status ``registry_resolved`` only because the
authority resolved a permissive SPDX id).  The manifest and every bound receipt
are re-verified before any paid downstream call, so a tampered receipt or entry
is rejected rather than authored.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.errors import ORBenchError

MANIFEST_SCHEMA = "orbenchlab.candidate-manifest.v1"
PROVENANCE_SCHEMA = "orbenchlab.source-paper-provenance.v1"


class CandidateManifestError(ORBenchError):
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


def build_candidate_manifest(
    *,
    admitted: Sequence[Mapping[str, Any]],
    out: str | Path,
    day: str,
) -> dict[str, Any]:
    """Write per-candidate provenance and a signed manifest for admitted sources.

    Each ``admitted`` row must carry ``acquisition``, ``license`` and ``triage``
    receipts; only rows whose license is ``allowed`` and whose triage verdict is
    ``eligible_for_authoring`` are admitted (others raise, since the caller must
    not pass ineligible rows).
    """

    root = Path(out)
    entries: list[dict[str, Any]] = []
    for row in admitted:
        acquisition = row["acquisition"]
        license_receipt = row["license"]
        triage = row["triage"]
        if license_receipt.get("decision") != "allowed":
            raise CandidateManifestError("only license-allowed sources may be admitted")
        if triage.get("verdict") != "eligible_for_authoring":
            raise CandidateManifestError("only eligible_for_authoring sources may be admitted")
        source_id = str(acquisition["source_id"])
        cand_dir = root / "candidates" / source_id
        provenance = {
            "schema_version": PROVENANCE_SCHEMA,
            "source_id": source_id,
            "canonical_id": acquisition.get("canonical_identity"),
            "content_digest": acquisition["content_digest"],
            "source_path": acquisition["frozen_path"],
            # The factory license gate reads license_status; the authority
            # resolved a permissive SPDX id, so it is registry_resolved.
            "license_status": "registry_resolved",
            "license_spdx": license_receipt.get("normalised_license"),
            "license_authority_decision": license_receipt.get("decision"),
            "license_receipt_digest": license_receipt["receipt_digest"],
            "acquisition_receipt_digest": acquisition["receipt_digest"],
            "triage_receipt_digest": triage["receipt_digest"],
            "day": day,
        }
        provenance["provenance_digest"] = _digest(
            {k: v for k, v in provenance.items() if k != "provenance_digest"}
        )
        provenance_path = cand_dir / "paper-provenance.json"
        _atomic_json(provenance_path, provenance)
        entries.append(
            {
                "source_id": source_id,
                "canonical_id": acquisition.get("canonical_identity"),
                "content_digest": acquisition["content_digest"],
                "paper_file": acquisition["frozen_path"],
                "paper_provenance": str(provenance_path),
                "provenance_digest": provenance["provenance_digest"],
                "acquisition_receipt_digest": acquisition["receipt_digest"],
                "license_receipt_digest": license_receipt["receipt_digest"],
                "triage_receipt_digest": triage["receipt_digest"],
            }
        )
    entries.sort(key=lambda e: e["source_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "day": day,
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest["manifest_digest"] = _digest(
        {k: v for k, v in manifest.items() if k != "manifest_digest"}
    )
    _atomic_json(root / "candidate-manifest.json", manifest)
    return manifest


def load_verified_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest, re-verifying its digest and every candidate provenance.

    Raises before returning if the manifest digest, a provenance digest, or a
    provenance file is tampered or missing — so no paid authoring runs on
    unverified evidence.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CandidateManifestError("candidate manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise CandidateManifestError("candidate manifest schema mismatch")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != _digest(unsigned):
        raise CandidateManifestError("candidate manifest digest mismatch (possible tamper)")
    for entry in manifest.get("entries", []):
        prov_path = Path(str(entry.get("paper_provenance") or ""))
        if not prov_path.is_file() or prov_path.is_symlink():
            raise CandidateManifestError(f"provenance missing for {entry.get('source_id')}")
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        prov_unsigned = {k: v for k, v in provenance.items() if k != "provenance_digest"}
        if (
            provenance.get("provenance_digest") != _digest(prov_unsigned)
            or provenance.get("provenance_digest") != entry.get("provenance_digest")
        ):
            raise CandidateManifestError(
                f"provenance digest mismatch for {entry.get('source_id')} (possible tamper)"
            )
    return manifest


__all__ = [
    "CandidateManifestError",
    "MANIFEST_SCHEMA",
    "PROVENANCE_SCHEMA",
    "build_candidate_manifest",
    "load_verified_manifest",
]
