"""Normalized rollout data and the metrics computed from it.

The report is built from a small normalized slice, never from raw Harbor
bundles. That is both a size decision (bundles are gigabytes; this is kilobytes)
and a safety one: a normalized slice contains no trajectories, no verifier
stdout and no hidden reference data, so it is safe to upload as a CI artefact.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import schema as schema_mod
from ..core.evidence import (
    Claim,
    EvidenceGrade,
    EvidenceLabel,
    RepetitionClass,
    repetition_class,
)

NORMALIZED_SCHEMA = "normalized_rollout.schema.json"

#: Controls that cost nothing to run and say the most about task health.
ORACLE_SCAFFOLD = "oracle"
NOP_SCAFFOLD = "nop"


@dataclass(frozen=True)
class Trial:
    run_id: str
    task_name: str
    agent_id: str
    scaffold: str
    seed: int
    attempt: int
    attribution: str
    counts_toward_capability: bool
    infra_suspect: bool
    trace_status: str
    scores: dict[str, float]
    cost_usd: float = 0.0
    replica_count: int = 1
    exclusion_basis: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trial":
        return cls(
            run_id=data["run_id"],
            task_name=data["task_name"],
            agent_id=data["agent_id"],
            scaffold=data["scaffold"],
            seed=int(data["seed"]),
            attempt=int(data["attempt"]),
            attribution=data["attribution"],
            counts_toward_capability=bool(data["counts_toward_capability"]),
            infra_suspect=bool(data["infra_suspect"]),
            trace_status=data["trace_status"],
            scores={k: float(v) for k, v in (data.get("scores") or {}).items()},
            cost_usd=float(data.get("cost_usd", 0.0)),
            replica_count=int(data.get("replica_count", 1)),
            exclusion_basis=data.get("exclusion_basis"),
        )


@dataclass(frozen=True)
class StrictPassRule:
    """How strict pass is defined, stated in the output rather than implied."""

    description: str
    feasibility_key: str
    quality_key: str
    quality_threshold: float

    def holds_for(self, trial: Trial) -> bool:
        feasible = trial.scores.get(self.feasibility_key, 0.0) >= 1.0
        quality = trial.scores.get(self.quality_key)
        return bool(feasible and quality is not None and quality >= self.quality_threshold)


@dataclass(frozen=True)
class NormalizedRollout:
    campaign_id: str
    integration: str
    evidence_intent: EvidenceLabel
    site: dict[str, Any]
    scoring_reward_keys: tuple[str, ...]
    strict_pass_rule: StrictPassRule
    trials: tuple[Trial, ...]
    orphan_trials: tuple[dict[str, Any], ...] = ()
    durability: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "NormalizedRollout":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_normalized(data, name=str(path))
        rule = data["scoring"]["strict_pass_rule"]
        return cls(
            campaign_id=data["campaign_id"],
            integration=data["integration"],
            evidence_intent=EvidenceLabel(data["evidence_intent"]),
            site=data["site"],
            scoring_reward_keys=tuple(data["scoring"]["reward_keys"]),
            strict_pass_rule=StrictPassRule(
                description=rule["description"],
                feasibility_key=rule["feasibility_key"],
                quality_key=rule["quality_key"],
                quality_threshold=float(rule["quality_threshold"]),
            ),
            trials=tuple(Trial.from_dict(t) for t in data["trials"]),
            orphan_trials=tuple(data.get("orphan_trials") or []),
            durability=dict(data.get("durability") or {}),
            source_path=str(path),
        )

    @property
    def capability_trials(self) -> tuple[Trial, ...]:
        return tuple(t for t in self.trials if t.counts_toward_capability)

    @property
    def min_replica_count(self) -> int:
        if "min_replica_count" in self.durability:
            return int(self.durability["min_replica_count"])
        return min((t.replica_count for t in self.trials), default=0)


def validate_normalized(document: Any, *, name: str = "normalized rollout") -> None:
    schema = schema_mod.load_schema(schema_mod.schemas_dir() / NORMALIZED_SCHEMA)
    schema_mod.validate(document, schema, name=name)


@dataclass(frozen=True)
class MetricResult:
    """A metric with the evidence that limits what it may be used to say."""

    name: str
    value: float | None
    n: int
    grade: EvidenceGrade
    repetition: RepetitionClass
    formula: str
    run_ids: tuple[str, ...]
    scope: str = "campaign"
    unmet_requirement: str | None = None
    #: Repetitions of the *weakest* (task, agent) configuration behind this
    #: metric. The weakest configuration is what bounds the repetition class.
    repetitions: int = 0

    def to_claim(self, claim_id: str) -> Claim:
        return Claim(
            claim_id=claim_id,
            metric=self.name,
            value=self.value,
            grade=self.grade,
            repetition=self.repetition,
            run_ids=self.run_ids,
            note=self.formula,
        )


def compute_metrics(rollout: NormalizedRollout) -> list[MetricResult]:
    """Compute every metric the normalized slice can support.

    A metric whose repetition requirement is unmet returns ``value=None`` and
    records why, instead of returning a number that reads as if it meant
    something.
    """
    results: list[MetricResult] = []
    rule = rollout.strict_pass_rule
    capability = rollout.capability_trials

    results.extend(_agent_metrics(rollout, capability, rule))
    results.extend(_control_metrics(rollout, rule))
    results.extend(_disclosure_metrics(rollout))
    return results


def _agent_metrics(
    rollout: NormalizedRollout, capability: tuple[Trial, ...], rule: StrictPassRule
) -> list[MetricResult]:
    results: list[MetricResult] = []
    by_agent: dict[str, list[Trial]] = {}
    for trial in capability:
        by_agent.setdefault(trial.agent_id, []).append(trial)

    for agent_id in sorted(by_agent):
        trials = by_agent[agent_id]
        run_ids = tuple(sorted(t.run_id for t in trials))
        # Repetition is per (task, agent) configuration, not per agent: three
        # trials on three different tasks are three single measurements.
        per_config = _min_repetitions_per_config(trials)
        rep = repetition_class(per_config)

        feasible = [t for t in trials if t.scores.get(rule.feasibility_key, 0.0) >= 1.0]
        results.append(
            MetricResult(
                name="feasibility_rate",
                value=_ratio(len(feasible), len(trials)),
                n=len(trials),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula=f"count({rule.feasibility_key} >= 1) / count(capability trials)",
                run_ids=run_ids,
                scope=f"agent:{agent_id}",
                repetitions=per_config,
            )
        )
        strict = [t for t in trials if rule.holds_for(t)]
        results.append(
            MetricResult(
                name="strict_pass_rate",
                value=_ratio(len(strict), len(trials)),
                n=len(trials),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula=rule.description,
                run_ids=run_ids,
                scope=f"agent:{agent_id}",
                repetitions=per_config,
            )
        )
        qualities = [
            t.scores[rule.quality_key] for t in feasible if rule.quality_key in t.scores
        ]
        results.append(
            MetricResult(
                name="mean_quality_when_feasible",
                value=round(statistics.fmean(qualities), 6) if qualities else None,
                n=len(qualities),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula=f"mean({rule.quality_key}) over trials with {rule.feasibility_key} >= 1",
                run_ids=run_ids,
                scope=f"agent:{agent_id}",
                unmet_requirement=None if qualities else "no feasible trial produced a quality score",
                repetitions=per_config,
            )
        )
        total_cost = sum(t.cost_usd for t in trials)
        results.append(
            MetricResult(
                name="cost_per_strict_pass_usd",
                value=round(total_cost / len(strict), 6) if strict else None,
                n=len(trials),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula="sum(cost_usd) / count(strict pass)",
                run_ids=run_ids,
                scope=f"agent:{agent_id}",
                unmet_requirement=None if strict else "no strict pass; denominator is zero",
                repetitions=per_config,
            )
        )
    return results


def _control_metrics(rollout: NormalizedRollout, rule: StrictPassRule) -> list[MetricResult]:
    """Task-health controls. Both cost nothing: Harbor provides oracle and nop."""
    results: list[MetricResult] = []

    all_oracle = [t for t in rollout.trials if t.scaffold == ORACLE_SCAFFOLD]
    oracle = [t for t in all_oracle if t.counts_toward_capability]
    if all_oracle:
        passed = [t for t in oracle if rule.holds_for(t)]
        rep_count = _min_repetitions_per_config(oracle)
        rep = repetition_class(rep_count)
        results.append(
            MetricResult(
                name="oracle_pass_rate",
                value=_ratio(len(passed), len(oracle)),
                n=len(oracle),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula="count(oracle strict pass) / count(oracle trials); a healthy task set is 1.0",
                run_ids=tuple(sorted(t.run_id for t in oracle)),
                scope="control:oracle",
                unmet_requirement=(
                    None if oracle else "all oracle trials were excluded from capability metrics"
                ),
                repetitions=rep_count,
            )
        )

    all_nop = [t for t in rollout.trials if t.scaffold == NOP_SCAFFOLD]
    nop = [t for t in all_nop if t.counts_toward_capability]
    if all_nop:
        failed = [t for t in nop if not rule.holds_for(t)]
        rep_count = _min_repetitions_per_config(nop)
        rep = repetition_class(rep_count)
        results.append(
            MetricResult(
                name="nop_fail_rate",
                value=_ratio(len(failed), len(nop)),
                n=len(nop),
                grade=EvidenceGrade.DETERMINISTIC_VERIFIER,
                repetition=rep,
                formula="count(nop non-pass) / count(nop trials); a task passable by doing nothing is broken",
                run_ids=tuple(sorted(t.run_id for t in nop)),
                scope="control:nop",
                unmet_requirement=(
                    None if nop else "all nop trials were excluded from capability metrics"
                ),
                repetitions=rep_count,
            )
        )
    return results


def _disclosure_metrics(rollout: NormalizedRollout) -> list[MetricResult]:
    """Counts that must always be shown so uncertainty is not quietly dropped."""
    trials = rollout.trials
    run_ids = tuple(sorted(t.run_id for t in trials))
    suspect = [t for t in trials if t.infra_suspect]
    excluded = [t for t in trials if not t.counts_toward_capability]
    no_load = [t for t in trials if rollout.site.get("load_source") == "none"]
    return [
        MetricResult(
            name="infra_suspect_share",
            value=_ratio(len(suspect), len(trials)),
            n=len(trials),
            grade=EvidenceGrade.DETERMINISTIC_TRACE,
            repetition=RepetitionClass.R0,
            formula=(
                "count(infra_suspect) / count(trials); denominator inclusion is reported "
                "separately by counts_toward_capability and excluded_trial_share"
            ),
            run_ids=run_ids,
            scope="disclosure",
        ),
        MetricResult(
            name="excluded_trial_share",
            value=_ratio(len(excluded), len(trials)),
            n=len(trials),
            grade=EvidenceGrade.DETERMINISTIC_TRACE,
            repetition=RepetitionClass.R0,
            formula=(
                "count(trials excluded from capability metrics) / count(trials); "
                "exclusion is explicit and each excluded trial carries an exclusion_basis"
            ),
            # This is a ratio over the complete panel.  Keep both numerator and
            # denominator samples in the claim provenance; exclusion_basis on
            # each normalized trial identifies the numerator members.
            run_ids=run_ids,
            scope="disclosure",
        ),
        MetricResult(
            name="orphan_trial_count",
            value=float(len(rollout.orphan_trials)),
            n=len(trials),
            grade=EvidenceGrade.DETERMINISTIC_TRACE,
            repetition=RepetitionClass.R0,
            formula="count(trials with no plan-ledger match)",
            run_ids=(),
            scope="disclosure",
        ),
        MetricResult(
            name="no_load_sampling_share",
            value=_ratio(len(no_load), len(trials)),
            n=len(trials),
            grade=EvidenceGrade.DETERMINISTIC_TRACE,
            repetition=RepetitionClass.R0,
            formula=(
                "share of trials with load_source = 'none'; load-based suspicion is "
                "unavailable, but hard or contradictory evidence may still set infra_suspect"
            ),
            run_ids=run_ids,
            scope="disclosure",
        ),
    ]


def _min_repetitions_per_config(trials: list[Trial] | tuple[Trial, ...]) -> int:
    """Smallest number of repetitions of any (task, agent) configuration."""
    counts: dict[tuple[str, str], int] = {}
    for trial in trials:
        counts[(trial.task_name, trial.agent_id)] = counts.get((trial.task_name, trial.agent_id), 0) + 1
    return min(counts.values(), default=0)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)
