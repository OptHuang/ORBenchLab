#!/usr/bin/env python3
"""Regenerate the report fixtures and their golden renderings.

Run from the repository root after changing the compiler, the metric set or the
renderer:

    python tools/regenerate_fixtures.py
    git diff -- fixtures tests/golden      # review before committing

Run ids come from the real compiler, so a fixture is always consistent with what
``orbench campaign plan`` emits for the same spec. The *scores* are synthetic and
say so in each fixture's ``_note`` field — no rollout was executed to produce
them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from orbenchlab.campaign import compile as compile_mod  # noqa: E402
from orbenchlab.campaign import spec as spec_mod  # noqa: E402
from orbenchlab.report import render as render_mod  # noqa: E402
from orbenchlab.report.model import NormalizedRollout  # noqa: E402

STRICT_RULE = {
    "description": (
        "strict pass = feasibility >= 1 and quality >= 1.0, where quality is the upstream "
        "ORAgentBench reward key (0-2, 1.0 means the reference objective was matched)"
    ),
    "feasibility_key": "feasibility",
    "quality_key": "quality",
    "quality_threshold": 1.0,
}

# Fixed synthetic quality per (task, seed) for the oracle control. Two tasks sit
# exactly on the strict-pass threshold and one clears it, so the fixture
# exercises the boundary rather than only the easy case.
ORACLE_QUALITY = {
    ("additive_microfactory_order_planning", 1): 1.0,
    ("additive_microfactory_order_planning", 2): 1.0,
    ("additive_microfactory_order_planning", 3): 1.0,
    ("airport_gate_assignment", 1): 1.42,
    ("airport_gate_assignment", 2): 1.42,
    ("airport_gate_assignment", 3): 1.41,
    ("bike_share_rebalancing", 1): 1.0,
    ("bike_share_rebalancing", 2): 1.0,
    ("bike_share_rebalancing", 3): 1.0,
}


def build_fixture(
    spec_path: str,
    out_path: str,
    *,
    seeds: list[int],
    evidence_intent: str,
    durability: dict,
    note: str,
    inject_infra: bool,
    slug: str | None = None,
) -> Path:
    raw = spec_mod.load_spec(REPO / spec_path)
    raw["seeds"] = seeds
    if slug:
        raw["slug"] = slug
    spec = spec_mod.validate(raw, sites_dir=REPO / "sites", source_path=spec_path)
    compiled = compile_mod.compile_campaign(spec)

    trials = []
    for run in sorted(compiled.runs, key=lambda r: (r.agent_id, r.seed, r.task_name)):
        oracle = run.agent_id == "oracle"
        trials.append(
            {
                "run_id": run.run_id,
                "task_name": run.task_name,
                "agent_id": run.agent_id,
                "scaffold": run.agent_id,
                "seed": run.seed,
                "attempt": run.attempt,
                "attribution": "agent",
                "counts_toward_capability": True,
                "infra_suspect": False,
                "exclusion_basis": None,
                "trace_status": "complete",
                "replica_count": durability["min_replica_count"],
                "cost_usd": 0.0,
                "scores": {
                    "feasibility": 1.0 if oracle else 0.0,
                    "quality": ORACLE_QUALITY[(run.task_name, run.seed)] if oracle else 0.0,
                },
            }
        )

    if inject_infra:
        # A container that was OOM-killed. Hard infrastructure evidence, so this
        # trial leaves the capability denominator — one of only three bases on
        # which a trial may be excluded.
        victim = next(t for t in trials if t["agent_id"] == "nop")
        victim.update(
            {
                "attribution": "infra",
                "counts_toward_capability": False,
                "exclusion_basis": "hard_infra_evidence",
                "trace_status": "missing",
                "scores": {},
            }
        )

    payload = {
        "normalized_schema_version": "1.0",
        "_note": note,
        "campaign_id": compiled.campaign_id,
        "integration": spec.integration,
        "evidence_intent": evidence_intent,
        "site": {"name": spec.site, "perf_isolated": False, "load_source": "none", "cpus": 4},
        "scoring": {"reward_keys": ["feasibility", "quality"], "strict_pass_rule": STRICT_RULE},
        "durability": durability,
        "orphan_trials": [
            {
                "trial_name": "airport_gate_assignment__k7Qm2Xz",
                "reason": (
                    "no plan-ledger match_key; recorded, excluded from the capability "
                    "denominator"
                ),
            }
        ],
        "trials": trials,
    }
    out = REPO / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fixture: {out.relative_to(REPO)} ({len(trials)} trials, {compiled.campaign_id})")
    return out


def render_golden(fixture: Path, golden_stem: str) -> None:
    report = render_mod.build_report(NormalizedRollout.load(fixture))
    golden_dir = REPO / "tests" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / f"{golden_stem}.summary.md").write_text(report.markdown, encoding="utf-8")
    print(f"golden:  tests/golden/{golden_stem}.summary.md ({report.effective_label.value})")
    if golden_stem == "oragentbench-controls":
        (golden_dir / f"{golden_stem}.evidence_index.json").write_text(
            json.dumps(report.evidence_index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"golden:  tests/golden/{golden_stem}.evidence_index.json")


def main() -> int:
    controls = build_fixture(
        "campaigns/oragentbench-controls.yaml",
        "fixtures/normalized/oragentbench-controls.json",
        seeds=[1, 2, 3],
        evidence_intent="validated",
        durability={"min_replica_count": 1, "verified": False},
        note=(
            "Synthetic scores over real planner output. Exercises the durability gate: the "
            "campaign asks for 'validated' but only one replica exists, so the report renders "
            "as 'partial'."
        ),
        inject_infra=True,
    )
    smoke = build_fixture(
        "campaigns/oragentbench-controls.yaml",
        "fixtures/normalized/oragentbench-smoke-r0.json",
        seeds=[1],
        evidence_intent="partial",
        durability={"min_replica_count": 2, "verified": True},
        note=(
            "Synthetic scores over real planner output. One rollout per configuration: asks "
            "for 'partial' but every measurement is R0, so the report renders as 'exploratory' "
            "and comparative wording is rejected."
        ),
        inject_infra=False,
        slug="oragentbench-smoke",
    )
    render_golden(controls, "oragentbench-controls")
    render_golden(smoke, "oragentbench-smoke-r0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
