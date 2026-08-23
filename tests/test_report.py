"""Report generation, evidence labelling and the claim guard.

The golden files are the contract: if a change to the metric set or the renderer
alters what a report says, that shows up as a diff a reviewer has to look at.
Regenerate them with ``python tools/regenerate_fixtures.py``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from orbenchlab.core.errors import EvidenceError
from orbenchlab.core.evidence import (
    EvidenceGrade,
    EvidenceLabel,
    RepetitionClass,
    assert_no_comparative_claims,
    find_comparative_claims,
    repetition_class,
)
from orbenchlab.report import render as render_mod
from orbenchlab.report.model import NormalizedRollout, compute_metrics, validate_normalized

CONTROLS = "oragentbench-controls"
SMOKE = "oragentbench-smoke-r0"


@pytest.fixture
def controls(fixtures_dir: Path) -> NormalizedRollout:
    return NormalizedRollout.load(fixtures_dir / "normalized" / f"{CONTROLS}.json")


@pytest.fixture
def smoke(fixtures_dir: Path) -> NormalizedRollout:
    return NormalizedRollout.load(fixtures_dir / "normalized" / f"{SMOKE}.json")


def _fully_repeated(rollout: NormalizedRollout) -> NormalizedRollout:
    """The controls slice with its infrastructure exclusion healed.

    Every (task, agent) configuration then has three runs, which is exactly the
    threshold at which comparative claims become supportable.
    """
    restored = tuple(
        replace(
            trial,
            attribution="agent",
            counts_toward_capability=True,
            exclusion_basis=None,
            trace_status="complete",
            scores={"feasibility": 0.0, "quality": 0.0},
        )
        if not trial.counts_toward_capability
        else trial
        for trial in rollout.trials
    )
    return replace(rollout, trials=restored)


# --------------------------------------------------------------------------- #
# golden output
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem", [CONTROLS, SMOKE])
def test_report_matches_its_golden(stem, fixtures_dir, golden_dir):
    rollout = NormalizedRollout.load(fixtures_dir / "normalized" / f"{stem}.json")
    rendered = render_mod.build_report(rollout).markdown
    golden = (golden_dir / f"{stem}.summary.md").read_text(encoding="utf-8")
    assert rendered == golden, (
        f"{stem} report drifted from its golden; review the change and "
        "regenerate with 'python tools/regenerate_fixtures.py'"
    )


def test_evidence_index_matches_its_golden(controls, golden_dir):
    index = render_mod.build_report(controls).evidence_index
    golden = json.loads(
        (golden_dir / f"{CONTROLS}.evidence_index.json").read_text(encoding="utf-8")
    )
    assert index == golden


def test_evidence_index_does_not_leak_host_path(controls):
    index = render_mod.build_report(controls).evidence_index
    assert index["source"] == "oragentbench-controls.json"
    assert str(Path.cwd()) not in json.dumps(index)


def test_rendering_is_deterministic(controls):
    assert render_mod.build_report(controls).markdown == render_mod.build_report(controls).markdown


def test_report_carries_no_wall_clock_content(controls):
    """A timestamp would make the golden test meaningless."""
    markdown = render_mod.build_report(controls).markdown
    for token in ("2026-", "GMT", "UTC", "generated_at"):
        assert token not in markdown


def test_write_report_emits_all_three_artifacts(controls, tmp_path):
    paths = render_mod.write_report(render_mod.build_report(controls), tmp_path)
    assert set(paths) == {"summary_md", "summary_json", "evidence_index"}
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# evidence labelling
# --------------------------------------------------------------------------- #


def test_validated_downgrades_without_verified_replicas(controls):
    report = render_mod.build_report(controls)
    assert report.intended_label is EvidenceLabel.VALIDATED
    assert report.effective_label is EvidenceLabel.PARTIAL
    assert any("replica" in reason for reason in report.downgrade_reasons)
    assert "Downgraded from `validated`" in report.markdown


def test_validated_holds_when_replicas_are_verified(controls):
    verified = replace(
        controls, durability={"min_replica_count": 2, "verified": True}
    )
    assert render_mod.build_report(verified).effective_label is EvidenceLabel.VALIDATED


def test_partial_downgrades_when_nothing_was_repeated(smoke):
    report = render_mod.build_report(smoke)
    assert report.intended_label is EvidenceLabel.PARTIAL
    assert report.effective_label is EvidenceLabel.EXPLORATORY
    assert any("R0" in reason for reason in report.downgrade_reasons)


def test_downgrade_reason_is_always_shown_to_the_reader(controls, smoke):
    for rollout in (controls, smoke):
        report = render_mod.build_report(rollout)
        for reason in report.downgrade_reasons:
            assert reason in report.markdown


# --------------------------------------------------------------------------- #
# the comparative-claim guard
# --------------------------------------------------------------------------- #


def test_single_rollout_data_forbids_comparisons(smoke):
    report = render_mod.build_report(smoke)
    assert report.comparisons_allowed is False
    assert "case diagnosis only" in report.markdown


def test_repeated_data_permits_comparisons(controls):
    """Heal the infrastructure exclusion so every configuration reaches three runs."""
    repeated = _fully_repeated(controls)
    assert render_mod.build_report(repeated).comparisons_allowed is True


@pytest.mark.parametrize(
    "phrase",
    [
        "model A outperforms model B",
        "the oracle is better than the agent",
        "claude-code beats codex here",
        "this is the best-performing configuration",
        "agent X ranks above agent Y",
        "it is superior to the baseline",
    ],
)
def test_comparative_phrases_are_detected(phrase):
    assert find_comparative_claims(phrase)
    with pytest.raises(EvidenceError):
        assert_no_comparative_claims(phrase)


def test_case_diagnosis_language_is_permitted():
    assert_no_comparative_claims(
        "On task airport_gate_assignment the submission was feasible but did not reach the "
        "reference objective; the solver log shows a time limit."
    )


def test_a_comparative_claim_in_r0_output_is_rejected(smoke, monkeypatch):
    """Inject a comparison into the renderer and confirm the guard stops it."""
    original = render_mod._render_markdown

    def sabotaged(**kwargs):
        return original(**kwargs) + "\n\nThe oracle outperforms the nop agent.\n"

    monkeypatch.setattr(render_mod, "_render_markdown", sabotaged)
    with pytest.raises(EvidenceError) as excinfo:
        render_mod.build_report(smoke)
    assert "comparative claims" in str(excinfo.value)


def test_the_same_claim_is_allowed_once_repetition_supports_it(controls, monkeypatch):
    repeated = _fully_repeated(controls)
    original = render_mod._render_markdown
    monkeypatch.setattr(
        render_mod,
        "_render_markdown",
        lambda **kwargs: original(**kwargs) + "\n\nThe oracle outperforms the nop agent.\n",
    )
    assert render_mod.build_report(repeated).comparisons_allowed is True


# --------------------------------------------------------------------------- #
# the unbacked-claim guard
# --------------------------------------------------------------------------- #


def test_an_unbacked_claim_reference_is_rejected(controls, monkeypatch):
    original = render_mod._render_markdown
    monkeypatch.setattr(
        render_mod,
        "_render_markdown",
        lambda **kwargs: original(**kwargs) + "\n\nStrict pass improved `[C999]`.\n",
    )
    with pytest.raises(EvidenceError) as excinfo:
        render_mod.build_report(controls)
    assert "C999" in str(excinfo.value)


def test_every_claim_in_the_index_is_actually_rendered(controls):
    report = render_mod.build_report(controls)
    for claim in report.claims:
        assert f"`[{claim.claim_id}]`" in report.markdown or claim.claim_id in report.markdown


def test_every_claim_records_the_runs_behind_it(controls):
    report = render_mod.build_report(controls)
    by_id = {c["claim_id"]: c for c in report.evidence_index["claims"]}
    for claim in report.claims:
        entry = by_id[claim.claim_id]
        assert entry["definition"]
        assert entry["strength"] == claim.strength


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_controls_are_computed_and_meet_their_expectations(controls):
    by_name = {m.name: m for m in compute_metrics(controls) if m.scope.startswith("control:")}
    assert by_name["oracle_pass_rate"].value == 1.0
    assert by_name["nop_fail_rate"].value == 1.0
    oracle_ids = {
        trial.run_id
        for trial in controls.trials
        if trial.scaffold == "oracle" and trial.counts_toward_capability
    }
    nop_ids = {
        trial.run_id
        for trial in controls.trials
        if trial.scaffold == "nop" and trial.counts_toward_capability
    }
    assert set(by_name["oracle_pass_rate"].run_ids) == oracle_ids
    assert set(by_name["nop_fail_rate"].run_ids) == nop_ids


def test_repetition_class_follows_the_weakest_configuration():
    assert repetition_class(1) is RepetitionClass.R0
    assert repetition_class(2) is RepetitionClass.R0
    assert repetition_class(3) is RepetitionClass.R1
    assert repetition_class(3, panel_levels=3) is RepetitionClass.R2


def test_undefined_metrics_report_none_with_a_reason(controls):
    nop = [
        m
        for m in compute_metrics(controls)
        if m.scope == "agent:nop" and m.name == "cost_per_strict_pass_usd"
    ][0]
    assert nop.value is None
    assert "denominator is zero" in nop.unmet_requirement


def test_excluded_trials_leave_the_capability_denominator(controls):
    excluded = [t for t in controls.trials if not t.counts_toward_capability]
    assert excluded, "fixture should contain an infrastructure-excluded trial"
    assert all(t.exclusion_basis == "hard_infra_evidence" for t in excluded)
    assert len(controls.capability_trials) == len(controls.trials) - len(excluded)


def test_disclosures_are_always_present(controls):
    disclosure = [m for m in compute_metrics(controls) if m.scope == "disclosure"]
    names = {m.name for m in disclosure}
    assert names == {
        "infra_suspect_share",
        "excluded_trial_share",
        "orphan_trial_count",
        "no_load_sampling_share",
    }
    excluded_share = next(m for m in disclosure if m.name == "excluded_trial_share")
    assert set(excluded_share.run_ids) == {trial.run_id for trial in controls.trials}
    no_load_share = next(m for m in disclosure if m.name == "no_load_sampling_share")
    assert set(no_load_share.run_ids) == {trial.run_id for trial in controls.trials}


def test_only_the_verifier_grade_may_drive_a_gate():
    assert EvidenceGrade.DETERMINISTIC_VERIFIER.gate_eligible
    assert not EvidenceGrade.DETERMINISTIC_TRACE.gate_eligible
    assert not EvidenceGrade.LLM_ANNOTATION.gate_eligible
    assert not EvidenceGrade.HUMAN_REVIEW.gate_eligible


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def test_fixtures_match_the_published_schema(fixtures_dir):
    for path in sorted((fixtures_dir / "normalized").glob("*.json")):
        validate_normalized(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def test_a_malformed_slice_is_rejected(fixtures_dir):
    from orbenchlab.core.schema import SchemaError

    data = json.loads((fixtures_dir / "normalized" / f"{CONTROLS}.json").read_text())
    broken = copy.deepcopy(data)
    broken["trials"][0]["attribution"] = "gremlins"
    with pytest.raises(SchemaError) as excinfo:
        validate_normalized(broken)
    assert "attribution" in str(excinfo.value)


def test_a_slice_missing_its_strict_pass_rule_is_rejected(fixtures_dir):
    from orbenchlab.core.schema import SchemaError

    data = json.loads((fixtures_dir / "normalized" / f"{CONTROLS}.json").read_text())
    broken = copy.deepcopy(data)
    del broken["scoring"]["strict_pass_rule"]
    with pytest.raises(SchemaError):
        validate_normalized(broken)


def test_normalized_slices_carry_no_raw_bundle_content(fixtures_dir):
    """The slice is uploadable precisely because it holds none of this."""
    forbidden = ("trajectory", "test-stdout", "test-stderr", "reference_metrics", "instruction.md")
    for path in sorted((fixtures_dir / "normalized").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains raw bundle content: {token}"
