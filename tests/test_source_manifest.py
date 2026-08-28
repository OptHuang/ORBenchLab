from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import factory_batch, source_manifest as sm


def _admitted_row(tmp_path: Path, source_id: str, *, license_decision="allowed", verdict="eligible_for_authoring"):
    frozen = tmp_path / f"{source_id}.bin"
    frozen.write_bytes(b"PAPER")
    return {
        "acquisition": {
            "source_id": source_id,
            "canonical_identity": f"arxiv_id:{source_id}",
            "content_digest": "sha256:" + "c" * 64,
            "frozen_path": str(frozen),
            "receipt_digest": "sha256:" + "1" * 64,
        },
        "license": {
            "decision": license_decision,
            "normalised_license": "mit",
            "receipt_digest": "sha256:" + "2" * 64,
        },
        "triage": {"verdict": verdict, "receipt_digest": "sha256:" + "3" * 64},
    }


def test_build_and_verify_manifest(tmp_path: Path):
    manifest = sm.build_candidate_manifest(
        admitted=[_admitted_row(tmp_path, "2401.01234")], out=tmp_path / "day", day="2026-08-28"
    )
    assert manifest["entry_count"] == 1
    loaded = sm.load_verified_manifest(tmp_path / "day" / "candidate-manifest.json")
    assert loaded["manifest_digest"] == manifest["manifest_digest"]
    prov = json.loads(Path(loaded["entries"][0]["paper_provenance"]).read_text())
    assert prov["license_status"] == "registry_resolved"
    assert prov["license_authority_decision"] == "allowed"


def test_manifest_refuses_ineligible_rows(tmp_path: Path):
    with pytest.raises(sm.CandidateManifestError, match="license-allowed"):
        sm.build_candidate_manifest(
            admitted=[_admitted_row(tmp_path, "s1", license_decision="deferred-license")],
            out=tmp_path / "d", day="d",
        )
    with pytest.raises(sm.CandidateManifestError, match="eligible_for_authoring"):
        sm.build_candidate_manifest(
            admitted=[_admitted_row(tmp_path, "s2", verdict="rejected")], out=tmp_path / "d2", day="d",
        )


def test_load_rejects_tampered_provenance(tmp_path: Path):
    # Acceptance 7: a tampered candidate provenance is rejected before yield.
    sm.build_candidate_manifest(
        admitted=[_admitted_row(tmp_path, "2401.01234")], out=tmp_path / "day", day="d"
    )
    prov_path = tmp_path / "day" / "candidates" / "2401.01234" / "paper-provenance.json"
    doc = json.loads(prov_path.read_text())
    doc["license_status"] = "proprietary-sneaked-in"
    prov_path.write_text(json.dumps(doc))
    with pytest.raises(sm.CandidateManifestError, match="provenance digest mismatch"):
        sm.load_verified_manifest(tmp_path / "day" / "candidate-manifest.json")


def test_triaged_intake_provider_yields_verified_candidates(tmp_path: Path):
    sm.build_candidate_manifest(
        admitted=[_admitted_row(tmp_path, "2401.01234")], out=tmp_path / "day", day="d"
    )
    manifest_path = str(tmp_path / "day" / "candidate-manifest.json")
    cands = factory_batch.discover_candidates(
        {"provider": {"kind": "triaged-intake", "manifest": manifest_path, "seed_task": "seed"}}
    )
    assert len(cands) == 1
    assert cands[0]["id"] == "2401-01234"  # arXiv id normalised to the id alphabet
    assert cands[0]["seed_task"] == "seed"

    # A tampered manifest is refused by the provider before any factory work.
    mpath = Path(manifest_path)
    doc = json.loads(mpath.read_text())
    doc["entries"][0]["paper_file"] = "/etc/passwd"
    mpath.write_text(json.dumps(doc))
    with pytest.raises(factory_batch.FactoryBatchError, match="verification"):
        factory_batch.discover_candidates(
            {"provider": {"kind": "triaged-intake", "manifest": manifest_path, "seed_task": "seed"}}
        )
