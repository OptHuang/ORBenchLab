"""Evidence labelling and the claim guard.

Two independent axes decide what a number is allowed to say:

* **evidence grade** — where the number came from (independent verifier, trace
  derivation, LLM annotation, human review);
* **repetition class** — how many times the configuration was repeated.

The strength of a conclusion is ``min(evidence grade, repetition class)``. A
single rollout (``R0``) can diagnose a case; it cannot rank models. That rule is
enforced here in code rather than left to whoever writes the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .errors import EvidenceError


class EvidenceLabel(str, Enum):
    """Campaign-level label printed at the top of every report."""

    EXPLORATORY = "exploratory"
    PARTIAL = "partial"
    VALIDATED = "validated"

    @property
    def rank(self) -> int:
        return _LABEL_RANK[self]


_LABEL_RANK = {
    EvidenceLabel.EXPLORATORY: 0,
    EvidenceLabel.PARTIAL: 1,
    EvidenceLabel.VALIDATED: 2,
}


class EvidenceGrade(str, Enum):
    """Provenance of a measurement."""

    DETERMINISTIC_VERIFIER = "deterministic_verifier"  # E-V
    DETERMINISTIC_TRACE = "deterministic_trace"  # E-T
    LLM_ANNOTATION = "llm_annotation"  # E-L
    HUMAN_REVIEW = "human_review"  # E-H

    @property
    def code(self) -> str:
        return _GRADE_CODE[self]

    @property
    def gate_eligible(self) -> bool:
        """Only an independent deterministic verifier may drive pass/fail."""
        return self is EvidenceGrade.DETERMINISTIC_VERIFIER


_GRADE_CODE = {
    EvidenceGrade.DETERMINISTIC_VERIFIER: "E-V",
    EvidenceGrade.DETERMINISTIC_TRACE: "E-T",
    EvidenceGrade.LLM_ANNOTATION: "E-L",
    EvidenceGrade.HUMAN_REVIEW: "E-H",
}


class RepetitionClass(str, Enum):
    R0 = "R0"  # single run — case diagnosis only
    R1 = "R1"  # n >= 3 same configuration — rate with CI
    R2 = "R2"  # capability panel x n >= 3 — difficulty, discrimination

    @property
    def rank(self) -> int:
        return _REPETITION_RANK[self]

    @property
    def allows_comparative_claims(self) -> bool:
        return self is not RepetitionClass.R0


_REPETITION_RANK = {
    RepetitionClass.R0: 0,
    RepetitionClass.R1: 1,
    RepetitionClass.R2: 2,
}

MIN_REPETITIONS_R1 = 3


def repetition_class(n_repetitions: int, *, panel_levels: int = 1) -> RepetitionClass:
    """Classify a measurement by how many times it was actually repeated."""
    if n_repetitions < MIN_REPETITIONS_R1:
        return RepetitionClass.R0
    if panel_levels >= 3:
        return RepetitionClass.R2
    return RepetitionClass.R1


#: Comparative wording that a single-rollout report may not use. Matched
#: case-insensitively on word boundaries.
COMPARATIVE_PATTERNS = (
    r"outperform\w*",
    r"underperform\w*",
    r"better than",
    r"worse than",
    r"stronger than",
    r"weaker than",
    r"superior to",
    r"inferior to",
    r"beats?\b",
    r"state[- ]of[- ]the[- ]art",
    r"best[- ]performing",
    r"worst[- ]performing",
    r"ranks? (?:above|below|higher|lower)",
    r"more capable than",
    r"less capable than",
)

_COMPARATIVE_RE = re.compile("|".join(COMPARATIVE_PATTERNS), re.IGNORECASE)


def find_comparative_claims(text: str) -> list[str]:
    """Return every comparative phrase found in ``text``."""
    return [match.group(0) for match in _COMPARATIVE_RE.finditer(text)]


def assert_no_comparative_claims(text: str, *, context: str = "report") -> None:
    """Reject comparative wording. Called by the renderer for R0 content."""
    hits = find_comparative_claims(text)
    if hits:
        raise EvidenceError(
            f"{context}: comparative claims are not derivable from single-rollout (R0) "
            f"evidence; found {sorted(set(h.lower() for h in hits))}. "
            "Raise the repetition class or reword as case diagnosis."
        )


@dataclass(frozen=True)
class Claim:
    """A rendered statement together with the evidence that backs it.

    The renderer refuses to emit a claim that has no entry here, which is more
    reliable than asking authors to remember citations.
    """

    claim_id: str
    metric: str
    value: float | int | None
    grade: EvidenceGrade
    repetition: RepetitionClass
    run_ids: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def strength(self) -> str:
        """``min(evidence grade, repetition class)`` in printable form."""
        return f"{self.grade.code}/{self.repetition.value}"

    @property
    def supports_comparison(self) -> bool:
        return self.grade.gate_eligible and self.repetition.allows_comparative_claims


def downgrade_label(
    intended: EvidenceLabel,
    *,
    reasons: Iterable[str],
    to: EvidenceLabel,
) -> tuple[EvidenceLabel, list[str]]:
    """Lower an intended label when its preconditions are unmet.

    Returns the effective label and the reasons. Never raises: a report that
    would have been ``validated`` still renders, but it renders as what its
    evidence actually supports, with the downgrade stated at the top.
    """
    reason_list = [r for r in reasons if r]
    if not reason_list:
        return intended, []
    effective = to if to.rank < intended.rank else intended
    return effective, reason_list


#: A campaign may only be rendered ``validated`` when each referenced run has at
#: least this many verified content-addressed replicas.
REQUIRED_REPLICAS_FOR_VALIDATED = 2
