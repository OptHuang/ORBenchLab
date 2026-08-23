"""Report rendering with the evidence rules enforced in code.

Three rules the renderer applies to itself, so that a report cannot overstate
what its data supports:

1. **No unbacked claim.** Every number in the prose carries a claim id, and
   every claim id must resolve in the evidence index. The check runs against
   the rendered text, not against the author's intentions.
2. **Strength is capped.** A conclusion is only as strong as
   ``min(evidence grade, repetition class)``. Single-rollout (``R0``) content
   is scanned for comparative wording and rejected if any is found.
3. **Labels degrade, they do not aspire.** ``validated`` requires verified
   durable replicas; ``partial`` requires real repetition. When a precondition
   is unmet the report still renders — at the label its evidence supports, with
   the downgrade stated at the top.

Output is deterministic: no timestamps, no host details, no ordering that
depends on the machine. The same input renders byte-identically, which is what
makes the golden test meaningful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import __version__
from ..core.evidence import (
    REQUIRED_REPLICAS_FOR_VALIDATED,
    Claim,
    EvidenceError,
    EvidenceLabel,
    RepetitionClass,
    assert_no_comparative_claims,
)
from .model import MetricResult, NormalizedRollout, compute_metrics

REPORT_SCHEMA_VERSION = "1.0"

_CLAIM_REF_RE = re.compile(r"\[(C\d{3})\]")


@dataclass(frozen=True)
class Report:
    campaign_id: str
    effective_label: EvidenceLabel
    intended_label: EvidenceLabel
    downgrade_reasons: tuple[str, ...]
    comparisons_allowed: bool
    claims: tuple[Claim, ...]
    metrics: tuple[MetricResult, ...]
    markdown: str
    summary: dict[str, Any]
    evidence_index: dict[str, Any]


def build_report(rollout: NormalizedRollout) -> Report:
    metrics = tuple(compute_metrics(rollout))
    effective_label, reasons = _resolve_label(rollout, metrics)
    claims = tuple(
        metric.to_claim(f"C{index:03d}") for index, metric in enumerate(metrics, start=1)
    )
    comparisons_allowed = _comparisons_allowed(metrics)

    markdown = _render_markdown(
        rollout=rollout,
        metrics=metrics,
        claims=claims,
        effective_label=effective_label,
        reasons=reasons,
        comparisons_allowed=comparisons_allowed,
    )

    _assert_claims_resolve(markdown, claims)
    if not comparisons_allowed:
        assert_no_comparative_claims(markdown, context=f"report {rollout.campaign_id}")

    summary = _summary_dict(rollout, metrics, effective_label, reasons, comparisons_allowed)
    evidence_index = _evidence_index_dict(rollout, claims)
    return Report(
        campaign_id=rollout.campaign_id,
        effective_label=effective_label,
        intended_label=rollout.evidence_intent,
        downgrade_reasons=tuple(reasons),
        comparisons_allowed=comparisons_allowed,
        claims=claims,
        metrics=metrics,
        markdown=markdown,
        summary=summary,
        evidence_index=evidence_index,
    )


def write_report(report: Report, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_md": out / "summary.md",
        "summary_json": out / "summary.json",
        "evidence_index": out / "evidence_index.json",
    }
    paths["summary_md"].write_text(report.markdown, encoding="utf-8")
    paths["summary_json"].write_text(
        json.dumps(report.summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["evidence_index"].write_text(
        json.dumps(report.evidence_index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _resolve_label(
    rollout: NormalizedRollout, metrics: Iterable[MetricResult]
) -> tuple[EvidenceLabel, list[str]]:
    """Lower the intended label until its preconditions hold."""
    intended = rollout.evidence_intent
    reasons: list[str] = []
    effective = intended

    if intended is EvidenceLabel.VALIDATED:
        replicas = rollout.min_replica_count
        if replicas < REQUIRED_REPLICAS_FOR_VALIDATED or not rollout.durability.get("verified"):
            reasons.append(
                f"'validated' requires every referenced run to have at least "
                f"{REQUIRED_REPLICAS_FOR_VALIDATED} verified content-addressed replicas; "
                f"the slice reports min_replica_count={replicas} and "
                f"verified={bool(rollout.durability.get('verified'))}"
            )
            effective = EvidenceLabel.PARTIAL

    capability_metrics = [m for m in metrics if m.scope.startswith("agent:")]
    if effective is not EvidenceLabel.EXPLORATORY and capability_metrics:
        best = max((m.repetition.rank for m in capability_metrics), default=0)
        if best < RepetitionClass.R1.rank:
            reasons.append(
                "'partial' requires at least one capability measurement repeated 3 or more "
                "times for the same (task, agent) configuration; every configuration here "
                "was measured once (R0)"
            )
            effective = EvidenceLabel.EXPLORATORY

    return effective, reasons


def _comparisons_allowed(metrics: Iterable[MetricResult]) -> bool:
    agent_metrics = [m for m in metrics if m.scope.startswith("agent:")]
    if not agent_metrics:
        return False
    return all(m.repetition.allows_comparative_claims for m in agent_metrics)


def _render_markdown(
    *,
    rollout: NormalizedRollout,
    metrics: tuple[MetricResult, ...],
    claims: tuple[Claim, ...],
    effective_label: EvidenceLabel,
    reasons: list[str],
    comparisons_allowed: bool,
) -> str:
    claim_by_metric = {
        (claim.metric, metric.scope): claim
        for claim, metric in zip(claims, metrics)
    }
    lines: list[str] = []
    add = lines.append

    add(f"# ORBenchLab report — {rollout.campaign_id}")
    add("")
    add(f"**Evidence label: {effective_label.value.upper()}**")
    add("")
    if effective_label is not rollout.evidence_intent:
        add(f"Downgraded from `{rollout.evidence_intent.value}`:")
        add("")
        for reason in reasons:
            add(f"- {reason}")
        add("")
    add("## What this report may and may not say")
    add("")
    add(
        "Every figure below comes from an independent deterministic verifier (`E-V`) or "
        "from the trace (`E-T`). The strength of a statement is `min(evidence grade, "
        "repetition class)`."
    )
    add("")
    agent_metrics = [m for m in metrics if m.scope.startswith("agent:")]
    if comparisons_allowed:
        add(
            "Every capability measurement here reaches `R1` or better — each (task, agent) "
            "configuration was run at least three times — so rate comparisons between the "
            "listed agents are within scope."
        )
    else:
        weakest = min((m.repetitions for m in agent_metrics), default=0)
        add(
            f"Not every capability measurement reaches `R1`: the weakest (task, agent) "
            f"configuration was run {weakest} time(s), and `R1` requires at least 3. "
            "`R0` evidence supports **case diagnosis only**, so this report draws no "
            "cross-agent or cross-model ranking conclusions, and the renderer rejects "
            "comparative wording while any capability measurement is `R0`."
        )
    add("")

    add("## Campaign")
    add("")
    add("| field | value |")
    add("| --- | --- |")
    add(f"| campaign_id | `{rollout.campaign_id}` |")
    add(f"| integration | `{rollout.integration}` |")
    add(f"| site | `{rollout.site.get('name')}` |")
    add(f"| perf_isolated | `{str(rollout.site.get('perf_isolated')).lower()}` |")
    add(f"| load sampling | `{rollout.site.get('load_source')}` |")
    add(f"| trials | {len(rollout.trials)} |")
    add(f"| trials in capability denominator | {len(rollout.capability_trials)} |")
    add(f"| strict pass rule | {rollout.strict_pass_rule.description} |")
    add("")

    control_metrics = [m for m in metrics if m.scope.startswith("control:")]
    if control_metrics:
        add("## Task-health controls (zero model cost)")
        add("")
        add(
            "`oracle` and `nop` are Harbor built-in agents, so these controls cost nothing "
            "to run and are the cheapest evidence that a task set is well-formed."
        )
        add("")
        add(_table_header())
        for metric in control_metrics:
            add(_table_row(metric, claim_by_metric[(metric.name, metric.scope)]))
        add("")

    agent_scopes = sorted({m.scope for m in metrics if m.scope.startswith("agent:")})
    if agent_scopes:
        add("## Agent measurements")
        add("")
        for scope in agent_scopes:
            add(f"### `{scope.split(':', 1)[1]}`")
            add("")
            add(_table_header())
            for metric in [m for m in metrics if m.scope == scope]:
                add(_table_row(metric, claim_by_metric[(metric.name, metric.scope)]))
            add("")

    disclosure = [m for m in metrics if m.scope == "disclosure"]
    if disclosure:
        add("## Mandatory disclosures")
        add("")
        add(
            "These counts are always shown. `infra_suspect` and capability inclusion are "
            "separate fields: soft load evidence alone does not reassign blame, while hard "
            "or contradictory evidence may explicitly exclude a trial with an "
            "`exclusion_basis`."
        )
        add("")
        add(_table_header())
        for metric in disclosure:
            add(_table_row(metric, claim_by_metric[(metric.name, metric.scope)]))
        add("")

    add("## Evidence index")
    add("")
    add("| claim | metric | scope | strength | runs |")
    add("| --- | --- | --- | --- | --- |")
    for claim, metric in zip(claims, metrics):
        run_count = len(claim.run_ids)
        runs = f"{run_count} run id(s)" if run_count else "n/a"
        add(f"| `{claim.claim_id}` | `{claim.metric}` | `{metric.scope}` | `{claim.strength}` | {runs} |")
    add("")
    add(
        "Full run-id lists are in `evidence_index.json`. Each claim resolves to the run ids "
        "it was computed from, so any figure here can be traced back to the underlying runs."
    )
    add("")
    add(f"Generated by orbenchlab {__version__} — deterministic output, no wall-clock content.")
    add("")
    return "\n".join(lines)


def _table_header() -> str:
    return "| metric | value | n | strength | claim | definition |\n| --- | --- | --- | --- | --- | --- |"


def _table_row(metric: MetricResult, claim: Claim) -> str:
    value = _format_value(metric)
    return (
        f"| `{metric.name}` | {value} | {metric.n} | `{claim.strength}` | "
        f"`[{claim.claim_id}]` | {metric.formula} |"
    )


def _format_value(metric: MetricResult) -> str:
    if metric.value is None:
        reason = metric.unmet_requirement or "not derivable from this data"
        return f"n/a — {reason}"
    if metric.name.endswith("_count"):
        return str(int(metric.value))
    return f"{metric.value:.3f}"


def _assert_claims_resolve(markdown: str, claims: tuple[Claim, ...]) -> None:
    known = {claim.claim_id for claim in claims}
    referenced = {match.group(1) for match in _CLAIM_REF_RE.finditer(markdown)}
    unknown = sorted(referenced - known)
    if unknown:
        raise EvidenceError(
            f"report references claim id(s) {unknown} that are absent from the evidence "
            "index; a conclusion without an evidence entry is not rendered"
        )
    unused = sorted(known - referenced)
    if unused:
        raise EvidenceError(
            f"evidence index contains claim id(s) {unused} that the report never renders; "
            "the index and the report must describe the same set of claims"
        )


def _summary_dict(
    rollout: NormalizedRollout,
    metrics: tuple[MetricResult, ...],
    effective_label: EvidenceLabel,
    reasons: list[str],
    comparisons_allowed: bool,
) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generator": f"orbenchlab {__version__}",
        "campaign_id": rollout.campaign_id,
        "integration": rollout.integration,
        "site": rollout.site,
        "evidence": {
            "intended_label": rollout.evidence_intent.value,
            "effective_label": effective_label.value,
            "downgrade_reasons": reasons,
            "comparative_claims_allowed": comparisons_allowed,
            "rule": "conclusion strength = min(evidence grade, repetition class)",
        },
        "counts": {
            "trials": len(rollout.trials),
            "capability_trials": len(rollout.capability_trials),
            "orphan_trials": len(rollout.orphan_trials),
        },
        "strict_pass_rule": {
            "description": rollout.strict_pass_rule.description,
            "feasibility_key": rollout.strict_pass_rule.feasibility_key,
            "quality_key": rollout.strict_pass_rule.quality_key,
            "quality_threshold": rollout.strict_pass_rule.quality_threshold,
        },
        "metrics": [
            {
                "name": metric.name,
                "scope": metric.scope,
                "value": metric.value,
                "n": metric.n,
                "evidence_grade": metric.grade.code,
                "repetition_class": metric.repetition.value,
                "formula": metric.formula,
                "unmet_requirement": metric.unmet_requirement,
            }
            for metric in metrics
        ],
    }


def _evidence_index_dict(rollout: NormalizedRollout, claims: tuple[Claim, ...]) -> dict[str, Any]:
    # A report may be rebuilt on a developer laptop, a self-hosted runner, or a
    # GitHub-hosted runner.  Recording the absolute input path would make the
    # same evidence render differently on each host and would leak local
    # directory names into a public artifact.  The filename is enough to point
    # back to the normalized slice shipped alongside the report; content
    # identity belongs in the slice/ledger, not in a machine-specific path.
    source = Path(rollout.source_path).name if rollout.source_path else None
    return {
        "evidence_index_schema_version": REPORT_SCHEMA_VERSION,
        "campaign_id": rollout.campaign_id,
        "source": source,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "metric": claim.metric,
                "value": claim.value,
                "evidence_grade": claim.grade.code,
                "repetition_class": claim.repetition.value,
                "strength": claim.strength,
                "supports_comparison": claim.supports_comparison,
                "definition": claim.note,
                "run_ids": list(claim.run_ids),
            }
            for claim in claims
        ],
    }
