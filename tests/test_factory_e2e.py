"""End-to-end unattended loop over deterministic fixtures.

This drives the *real* code path — pdftotext paper binding, the full
19-stage blueprint, Bubblewrap least-visibility sessions, deterministic
postcheck gates, both Harbor barriers, difficulty calibration and the
promotion chain — with scripted agent/Harbor executables.  The fixtures are
test doubles, clearly not model evidence; what this test proves is that the
harness itself can carry one paper to a promoted task card without a human
step, and that every receipt binds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from orbenchlab import (
    agentic_factory,
    factory_autopilot,
    factory_blueprints,
    factory_supervisor,
    task_authoring,
)
from orbenchlab.volc_review import REQUIRED_REVIEW_CRITERIA

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"
VOLC = "https://ark.cn-beijing.volces.com/api/coding"
PROVIDER = {"ANTHROPIC_BASE_URL": VOLC, "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}

_MINI_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 90 >> stream
BT /F1 12 Tf 72 720 Td (A tiny operations research paper about scheduling.) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
trailer << /Root 1 0 R >>
"""

_FAKE_CLAUDE = '''#!/usr/bin/env python3
import hashlib, json, shutil, sys
from pathlib import Path

envelope = json.loads(sys.stdin.readline())
prompt = envelope["message"]["content"]
if "Terminal-Bench Science task reviewer" in prompt:
    # Promotion semantic review runs as a least-visibility CLI session; emit a
    # strict passing verdict with all seven proposal criteria.
    criteria = ["difficult", "outcome_verified", "scientifically_grounded",
                "scope", "solvable", "verifiable", "well_specified"]
    verdict = {"decision": "promising", "shape_complete": True,
               "rubric_complete": True,
               "criteria": [{"name": n, "status": "pass", "evidence": "inspected " + n}
                            for n in criteria]}
    Path("review.json").write_text(json.dumps(verdict), encoding="utf-8")
    print(json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 0.01}))
    raise SystemExit(0)
contract = json.loads(prompt.split("ORBENCH_FACTORY_CONTRACT\\n", 1)[1])
stage = contract["stage_id"]
trusted = contract.get("trusted_json_digest_values", {})
SEED = Path("factory-input/seed-task")
PROV = Path("factory-input/paper-provenance.json")
SLUG = "alphaevolve-scheduling"

def generic(spec):
    path = Path(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "text":
        path.write_text(
            "# " + stage + "\\n\\nDeterministic fixture analysis for " + stage + ".\\n",
            encoding="utf-8",
        )
    elif spec["kind"] == "json":
        defaults = {
            "string": "fixture",
            "array": ["fixture"],
            "object": {"fixture": "value"},
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "null": None,
        }
        types = spec.get("json_key_types", {})
        document = {
            key: defaults[types.get(key, "string")]
            for key in spec.get("json_required_keys", [])
        }
        document.update(trusted.get(spec["path"], {}))
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")

def make_writable(root):
    # A real no-Bash agent recreates files with Write and never inherits the
    # 0444 bits of the immutable factory-input tree; this scripted double
    # copies trees wholesale, so it must normalise permissions itself.
    for child in Path(root).rglob("*"):
        child.chmod(0o755 if child.is_dir() else 0o644)
    Path(root).chmod(0o755)

def copy_task(dest_root):
    dest = Path(dest_root) / SLUG
    if Path(dest_root).exists():
        shutil.rmtree(dest_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SEED, dest)
    make_writable(dest)
    shutil.copy2(PROV, dest / "paper-provenance.json")
    (dest / "paper-provenance.json").chmod(0o644)
    return dest

if stage == "intervention-policy":
    Path("factory/analysis").mkdir(parents=True, exist_ok=True)
    Path("factory/analysis/intervention-policy.json").write_text(json.dumps({
        "bottleneck_id": "missing-seed-rule",
        "rationale": "Weak model forgets the seeded duration rule.",
        "trigger": {"kind": "assistant-event-index", "value": 1},
        "hint_level": 1,
        "hint_text": "Reminder: apply d + ((seed + operation_index) % 2) for every operation.",
    }, indent=2), encoding="utf-8")
elif stage == "task-author-v1":
    copy_task("factory/tasks/task-v1")
elif stage == "task-repair-v2":
    dest = copy_task("factory/tasks/task-v2")
    (dest / "data" / "repair-ledger.json").write_text(
        json.dumps({"resolved_findings": []}), encoding="utf-8"
    )
elif stage == "variant-author":
    base = Path("factory/tasks/task-v2") / SLUG
    root = Path("factory/tasks/variants")
    if root.exists():
        shutil.rmtree(root)
    rows = []
    for level in ("small", "medium", "large"):
        slug = SLUG + "-" + level
        dest = root / slug
        shutil.copytree(base, dest)
        toml = dest / "task.toml"
        toml.write_text(
            toml.read_text(encoding="utf-8").replace(
                "terminal-bench-science/" + SLUG,
                "terminal-bench-science/" + slug,
            ),
            encoding="utf-8",
        )
        (dest / "data" / "scale.json").write_text(
            json.dumps({"level": level}), encoding="utf-8"
        )
        rows.append(
            {
                "variant_id": slug,
                "relative_path": slug,
                "level": level,
                "axis_levels": {"instance_scale": level, "hint": 0},
            }
        )
    manifest = {
        "schema_version": "orbenchlab.variant-manifest.v1",
        "primary_axis": {
            "name": "instance_scale",
            "expected_direction": "solve rate nonincreasing with scale",
            "ordered_levels": ["small", "medium", "large"],
        },
        "secondary_axes": [
            {"name": "hint", "levels": [0], "meaning": "instruction hint richness"}
        ],
        "variants": rows,
        "evaluation_mode": "exploratory",
    }
    (root / "variant-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
elif stage == "final-synthesis":
    selected = "factory/tasks/task-v2/" + SLUG
    prov_digest = "sha256:" + hashlib.sha256(PROV.read_bytes()).hexdigest()
    genome = {
        "family": "alphaevolve_scheduling",
        "title": "AlphaEvolve scheduling benchmark",
        "design_goal": "Schedule jobs under paper-derived constraints.",
        "selected_task": selected,
        "source": {
            "title": "Fixture paper",
            "url": "https://example.org/paper",
            "paper_provenance_digest": prov_digest,
        },
        "difficulty_axes": {
            "instance_scale": {
                "levels": ["small", "medium", "large"],
                "meaning": "number of jobs in the instance",
                "expected_direction": "solve rate decreases with scale",
            }
        },
    }
    summary = {
        "selected_task": selected,
        "task_summary": "A scheduling task derived from the fixture paper.",
        "evidence_level": "E1-agent-session-process",
        "limitations": ["Semantic completion only; deterministic gates own promotion."],
    }
    Path("factory/final").mkdir(parents=True, exist_ok=True)
    Path("factory/final/task-genome.json").write_text(
        json.dumps(genome, indent=2), encoding="utf-8"
    )
    Path("factory/final/task-review-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
else:
    for spec in contract["required_outputs"]:
        generic(spec)
print("done")
'''

_FAKE_HARBOR = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

agent = value("--agent")
snapshot = Path(value("--path"))
name = snapshot.name
task_name = "terminal-bench-science/" + name
job = Path(value("--jobs-dir")) / value("--job-name")

if agent in ("oracle", "nop"):
    trial = job / (name + "__" + agent)
    (trial / "verifier").mkdir(parents=True)
    (trial / "artifacts").mkdir(parents=True)
    reward = 1.0 if agent == "oracle" else 0.0
    passed = 3 if agent == "oracle" else 0
    failed = 0 if agent == "oracle" else 3
    job.joinpath("result.json").write_text(json.dumps({
        "id": agent + "-id", "n_total_trials": 1,
        "stats": {"n_completed_trials": 1, "n_errored_trials": 0}}))
    trial.joinpath("result.json").write_text(json.dumps({
        "task_name": task_name,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None}))
    trial.joinpath("verifier/ctrf.json").write_text(json.dumps({
        "results": {"summary": {"tests": 3, "passed": passed, "failed": failed,
        "skipped": 0, "pending": 0, "other": 0}}}))
    trial.joinpath("verifier/reward.txt").write_text(str(reward) + "\\n")
    trial.joinpath("artifacts/manifest.json").write_text(json.dumps([
        {"source": "/root/submission/solver.py",
         "status": "ok" if agent == "oracle" else "failed"}]))
    raise SystemExit(0)

model = value("--model")
repetitions = int(value("--n-attempts"))
suffix = name.rsplit("-", 1)[-1]
level = suffix if suffix in ("small", "medium", "large") else "base"
passes = {
    "frontier": {"base": repetitions, "small": repetitions,
                 "medium": repetitions, "large": repetitions - 1},
    "weak": {"base": 0, "small": 3, "medium": 0, "large": 0},
}[model][level]
job.mkdir(parents=True, exist_ok=True)
for attempt in range(1, repetitions + 1):
    reward = 1.0 if attempt <= passes else 0.0
    trial = job / (name + "__" + str(attempt))
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)
    exception = None if reward else {"exception_type": "NonZeroAgentExitCodeError"}
    passed = 3 if reward else 0
    failed = 0 if reward else 3
    trial.joinpath("result.json").write_text(json.dumps({
        "task_name": task_name,
        "exception_info": exception,
        "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 5,
                         "n_output_tokens": 40, "cost_usd": 0.05},
        "verifier_result": {"rewards": {"reward": reward}}}))
    trial.joinpath("verifier/reward.txt").write_text(str(reward) + "\\n")
    trial.joinpath("verifier/ctrf.json").write_text(json.dumps({
        "results": {"summary": {"tests": 3, "passed": passed, "failed": failed,
        "skipped": 0, "pending": 0, "other": 0}}}))
    trial.joinpath("agent/trajectory.json").write_text(json.dumps({
        "schema_version": "ATIF-v1.0",
        "steps": [{"step_id": 1, "source": "agent"}, {"step_id": 2, "source": "verifier"}]}))
job.joinpath("result.json").write_text(json.dumps({
    "id": model + "-job", "n_total_trials": repetitions,
    "stats": {"n_completed_trials": repetitions,
              "n_errored_trials": repetitions - passes}}))
'''


@pytest.fixture()
def e2e_bin():
    # Bubblewrap sessions mount a private tmpfs over /tmp, so fixture
    # executables must live outside it to stay visible inside the sandbox.
    root = Path("/var/tmp") / f"orbench-e2e-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _paper_binding(root: Path) -> tuple[Path, Path]:
    paper = root / "paper.pdf"
    paper.write_bytes(_MINI_PDF)
    unsigned = {
        "paper_provenance_schema_version": "orbenchlab.paper-provenance.v1",
        "title": "Fixture paper",
        "url": "https://example.org/paper",
        "source_content_digest": "sha256:" + hashlib.sha256(paper.read_bytes()).hexdigest(),
        "license_status": "pending-human",
    }
    provenance = {
        **unsigned,
        "binding_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }
    provenance_path = root / "paper-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return paper, provenance_path


def _fake_semantic_review(monkeypatch) -> None:
    def fake_process(target, args, *, timeout_sec):
        task, _paper, static_json, output, _env, models, _timeout, _tokens = args
        static = json.loads(Path(static_json).read_text(encoding="utf-8"))
        paper = json.loads(
            (Path(task) / "paper-provenance.json").read_text(encoding="utf-8")
        )
        review = {
            "schema_version": "orbenchlab.volc-authoring-review.v1",
            "task_tree_digest": task_authoring._task_tree_digest(Path(task)),
            "static_receipt_digest": static["receipt_digest"],
            "paper_digest": paper["source_content_digest"],
            "aggregate_decision": "promising-needs-harbor",
            "models": list(models),
            "review_count": len(models),
            "reviewers": [
                {
                    "model": model,
                    "review": {
                        "decision": "promising",
                        "shape_complete": True,
                        "rubric_complete": True,
                        "criteria": [
                            {"name": name, "status": "pass", "evidence": "inspected"}
                            for name in sorted(REQUIRED_REVIEW_CRITERIA)
                        ],
                    },
                }
                for model in models
            ],
        }
        review["review_digest"] = factory_autopilot._value_digest(review)
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "volc-authoring-review.json").write_text(
            json.dumps(review, indent=2, sort_keys=True), encoding="utf-8"
        )
        return None

    monkeypatch.setattr(factory_supervisor, "_run_builtin_process", fake_process)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not shutil.which("bwrap"),
    reason="the paper factory requires the Bubblewrap sandbox",
)
@pytest.mark.skipif(not shutil.which("pdftotext"), reason="pdftotext is required")
def test_paper_to_promoted_task_card_without_human_steps(
    tmp_path: Path, e2e_bin: Path, monkeypatch
):
    claude = _write_executable(e2e_bin / "claude", _FAKE_CLAUDE)
    harbor = _write_executable(e2e_bin / "harbor", _FAKE_HARBOR)
    workdir = e2e_bin / "work"
    paper, provenance = _paper_binding(e2e_bin)
    manifest = factory_blueprints.prepare_workspace(
        paper_file=paper,
        paper_provenance=provenance,
        seed_task=GOOD_TASK,
        workdir=workdir,
    )
    plan = factory_blueprints.paper_to_benchmark_plan(
        source_binding_digest=manifest["workspace_binding_digest"],
        author_model="author-model",
        reviewer_models=["reviewer-a", "reviewer-b"],
        frontier_model="frontier",
        weak_model="weak",
    )
    agentic_factory.write_plan(plan, e2e_bin / "plan.json")
    state = factory_autopilot.run(
        plan,
        workdir=workdir,
        factory_out=e2e_bin / "factory-run",
        out=e2e_bin / "autopilot",
        harbor_executable=harbor,
        claude_executable=claude,
        frontier_model="frontier",
        weak_model="weak",
        provider_env=PROVIDER,
        repetitions=5,
        max_budget_usd=0.5,
        max_variants=3,
        max_harbor_liability_usd=100.0,
    )
    assert state["status"] == "promoted"
    assert state["factory_status"] == "semantic-complete-e1" or state.get(
        "factory_run_digest"
    )
    assert set(state["barriers"]) == {"baseline", "intervention", "difficulty"}
    # The intervention barrier ran the capability probe (study disabled by
    # default) and installed an honest machine-readable capability receipt.
    intervention = state["barriers"]["intervention"]
    assert intervention["same_session_hint_injection"] is True
    assert intervention["study_status"] == "not-run"
    assert intervention["study_reason"] == "disabled-by-configuration"
    icap = json.loads(
        (
            e2e_bin
            / "autopilot"
            / "intervention"
            / "trusted-source"
            / "runtime-capability.json"
        ).read_text()
    )
    assert icap["harbor_native"] is False
    assert icap["causal_intervention_claim_available"] is False
    assert state["barriers"]["difficulty"]["decision"] == "exploratory-promising"
    assert state["selected_task"] == "factory/tasks/task-v2/alphaevolve-scheduling"
    promotion = state["promotion"]
    assert promotion["promoted"] is True
    assert promotion["decision"] == "eligible-for-human-release-review"
    autopilot_out = e2e_bin / "autopilot"
    final = json.loads(
        (autopilot_out / "promotion" / "final" / "factory-finalization.json").read_text()
    )
    assert final["promoted"] is True and final["evidence_level"] == "E3"
    # The trusted baseline bundle carries controls, matrix, screening, the
    # passing static receipt and the honest runtime-capability receipt.
    trusted = workdir / "factory-input" / "trusted" / "baseline"
    for name in (
        "harbor-control-screening.json",
        "harbor-model-matrix.json",
        "screening-report.json",
        "static-authoring-receipt.json",
        "runtime-capability.json",
        "trusted-bundle-manifest.json",
    ):
        assert (trusted / name).is_file(), name
    capability = json.loads((trusted / "runtime-capability.json").read_text())
    assert capability["checkpoint_capability"] is False
    difficulty = json.loads(
        (autopilot_out / "difficulty" / "difficulty-matrix.json").read_text()
    )
    assert difficulty["decision"] == "exploratory-promising"
    assert [row["variant_id"] for row in difficulty["variants"]] == [
        "alphaevolve-scheduling-small",
        "alphaevolve-scheduling-medium",
        "alphaevolve-scheduling-large",
    ]
    assert difficulty["monotonicity"]["passed"] is True
    assert difficulty["difficulty_genome"]["secondary_axes"][0]["name"] == "hint"
    report = (autopilot_out / "promotion" / "final-report.md").read_text(encoding="utf-8")
    assert "難" not in report  # sanity: valid encoding
    for marker in ("任务是什么", "Frontier vs weak", "难度维度", "可复现命令"):
        assert marker in report, marker
    cards = json.loads(
        (autopilot_out / "promotion" / "cards" / "task-cards.json").read_text()
    )
    assert cards["cards"][0]["decision"] == "review-promising"
    assert cards["cards"][0]["task_id"] == "alphaevolve_scheduling"
    # Crash-safe resume: a second invocation is a no-op returning the same
    # terminal state without re-running Harbor jobs or sessions.
    jobs_before = sorted(
        p.name for p in (autopilot_out / "baseline" / "matrix" / "jobs").iterdir()
    )
    resumed = factory_autopilot.run(
        plan,
        workdir=workdir,
        factory_out=e2e_bin / "factory-run",
        out=autopilot_out,
        harbor_executable=harbor,
        claude_executable=claude,
        frontier_model="frontier",
        weak_model="weak",
        provider_env=PROVIDER,
        repetitions=5,
        max_budget_usd=0.5,
        max_variants=3,
        max_harbor_liability_usd=100.0,
    )
    assert resumed["status"] == "promoted"
    jobs_after = sorted(
        p.name for p in (autopilot_out / "baseline" / "matrix" / "jobs").iterdir()
    )
    assert jobs_before == jobs_after
