"""Structural checks for source-bound candidate genomes.

These files are design artifacts, not executable task packages.  The tests keep
their provenance and safety boundary explicit while allowing a later human
author to evolve the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_candidate_genomes_are_source_bound_and_not_packable():
    root = Path(__file__).resolve().parent.parent
    shortlist = json.loads(
        (root / "docs/task-genomes/source-shortlist-2026-08-27.json").read_text()
    )
    by_uid = {item["intake_item_uid"]: item for item in shortlist["items"]}
    paths = sorted((root / "docs/task-genomes").glob("*.yaml"))
    assert {path.stem for path in paths} == {
        "cir-constraint-audit",
        "multi-agent-vrp-recovery",
    }
    for path in paths:
        genome = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert genome["status"] == "candidate-not-packable"
        assert "hooks" not in genome
        source = genome["source"]
        assert source["intake_snapshot_id"] == shortlist["intake_snapshot_id"]
        assert source["intake_item_uid"] in by_uid
        assert source["source_content_digest"] == by_uid[source["intake_item_uid"]][
            "source_content_digest"
        ]
        assert source["intake_status"] == "metadata_only"
        assert source["human_checks_required"]
        assert source["url"].startswith("https://")

        contract = genome["task_contract"]
        assert contract["network_policy"] == "control-local"
        assert contract["verifier"]["independent"] is True
        assert contract["controls"]["oracle"]
        assert contract["controls"]["nop"]
        mutations = contract["controls"]["mutations"]
        assert len(mutations) >= 4
        assert len({mutation["id"] for mutation in mutations}) == len(mutations)
        acceptance = contract["acceptance"]
        assert acceptance["evidence"]["minimum_seeds_per_cell"] >= 3
        assert acceptance["evidence"]["minimum_difficulty_levels"] >= 3
        assert acceptance["evidence"]["minimum_agent_systems"] >= 2

        for axis in genome["difficulty_axes"].values():
            assert len(axis["levels"]) >= 3
            assert axis["expected_direction"] in {
                "harder_with_larger_value",
                "easier_with_larger_value",
                "measure_not_assume",
            }
        intervention = genome["interventions"]
        assert intervention["adapter_shape"] == "restart-with-hint"
        assert [row["level"] for row in intervention["hint_ladder"]] == [0, 1, 2, 3]
        assert "no_leak" in intervention


def test_shortlist_binding_contains_metadata_only_records():
    root = Path(__file__).resolve().parent.parent
    payload = json.loads(
        (root / "docs/task-genomes/source-shortlist-2026-08-27.json").read_text()
    )
    assert payload["policy"].startswith("metadata-only")
    for item in payload["items"]:
        assert set(item) <= {
            "intake_item_uid",
            "source_content_digest",
            "title",
            "url",
            "related_code_url",
            "kind",
        }
        assert item["url"].startswith("https://")
        assert item["source_content_digest"].startswith("sha256:")
