from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbenchlab import agentic_factory, factory_gates

ROOT = Path(__file__).resolve().parents[1]
GOOD_TASK = ROOT / "examples" / "tasks" / "alphaevolve-scheduling"
DIGEST = "sha256:" + "a" * 64
VOLC = "https://ark.cn-beijing.volces.com/api/coding"


def _environments() -> dict[str, dict[str, str]]:
    return {
        "claude-code": {
            "ANTHROPIC_BASE_URL": VOLC,
            "ANTHROPIC_AUTH_TOKEN": "fixture-secret",
        }
    }


def _gate_plan(*, max_attempts: int = 2) -> dict:
    return agentic_factory.compile_plan(
        name="postcheck gate factory",
        source_binding_digest=DIGEST,
        stages=[
            {
                "id": "task-repair-v2",
                "role": "task repair agent",
                "profile": "claude-code",
                "model": "fixture-model",
                "prompt": "Repair the task tree.",
                "depends_on": [],
                "timeout_sec": 30,
                "max_attempts": max_attempts,
                "max_budget_usd": 0.25,
                "max_output_bytes": 1024 * 1024,
                "required_outputs": [
                    {"path": "factory/tasks/task-v2", "kind": "directory"}
                ],
                "postchecks": ["tb-science-static-gate"],
            }
        ],
    )


def _workspace(path: Path) -> Path:
    workdir = path / "work"
    (workdir / "factory-input").mkdir(parents=True)
    shutil.copy2(
        GOOD_TASK / "paper-provenance.json",
        workdir / "factory-input" / "paper-provenance.json",
    )
    return workdir


def _repair_cli(path: Path, *, fix_on_retry: bool) -> Path:
    executable = path / "repair-agent"
    fix = (
        "rm -rf factory/tasks/task-v2 && mkdir -p factory/tasks/task-v2 && "
        f'cp -r "{GOOD_TASK}" factory/tasks/task-v2/alphaevolve-scheduling'
        if fix_on_retry
        else 'printf broken-differently > factory/tasks/task-v2/task.toml'
    )
    executable.write_text(
        f"""#!/bin/sh
cat >/dev/null
mkdir -p factory/tasks/task-v2
if [ -f factory/gate/task-repair-v2-postcheck.json ]; then
  {fix}
else
  printf broken > factory/tasks/task-v2/task.toml
fi
printf done
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_compile_rejects_unknown_postchecks_and_gate_directory_outputs():
    stage = {
        "id": "stage-a",
        "role": "agent",
        "profile": "claude-code",
        "model": "m",
        "prompt": "p",
        "depends_on": [],
        "timeout_sec": 5,
        "max_attempts": 1,
        "max_budget_usd": 0.1,
        "required_outputs": [{"path": "factory/out.json", "kind": "json"}],
        "postchecks": ["not-a-real-gate"],
    }
    with pytest.raises(agentic_factory.AgenticFactoryError):
        agentic_factory.compile_plan(
            name="x", source_binding_digest=DIGEST, stages=[stage]
        )
    stage["postchecks"] = []
    stage["required_outputs"] = [{"path": "factory/gate/out.json", "kind": "json"}]
    with pytest.raises(agentic_factory.AgenticFactoryError):
        agentic_factory.compile_plan(
            name="x", source_binding_digest=DIGEST, stages=[stage]
        )


def test_plans_without_postchecks_keep_stage_shape_and_digest_stability():
    stage = {
        "id": "stage-a",
        "role": "agent",
        "profile": "claude-code",
        "model": "m",
        "prompt": "p",
        "depends_on": [],
        "timeout_sec": 5,
        "max_attempts": 1,
        "max_budget_usd": 0.1,
        "required_outputs": [{"path": "factory/out.json", "kind": "json"}],
    }
    plan = agentic_factory.compile_plan(
        name="x", source_binding_digest=DIGEST, stages=[stage]
    )
    assert "postchecks" not in plan["stages"][0]
    # A persisted plan compiled before postchecks existed must revalidate to
    # the same digest under the new schema.
    assert agentic_factory.validate_plan(plan)["plan_digest"] == plan["plan_digest"]


def test_static_gate_postcheck_drives_repair_loop_to_completion(tmp_path: Path):
    plan = _gate_plan(max_attempts=2)
    workdir = _workspace(tmp_path)
    out = tmp_path / "run"
    executable = _repair_cli(tmp_path, fix_on_retry=True)
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"claude-code": executable},
    )
    assert result["status"] == "semantic-complete-e1"
    attempts = result["stages"]["task-repair-v2"]["attempts"]
    assert [row["status"] for row in attempts] == ["failed", "completed"]
    first = json.loads(
        (out / "stages" / "task-repair-v2" / "attempt-001.json").read_text()
    )
    assert first["failure_class"] == "deterministic_gate_failed"
    assert first["postcheck"]["passed"] is False
    assert first["failed_output_snapshots"] == []
    findings = json.loads(
        (out / "stages" / "task-repair-v2" / "postcheck-001.json").read_text()
    )
    assert findings["passed"] is False
    assert findings["postchecks"][0]["targets"][0]["decision"] == "blocked"
    second = json.loads(
        (out / "stages" / "task-repair-v2" / "attempt-002.json").read_text()
    )
    assert second["postcheck"]["passed"] is True
    # The advisory findings file is removed once the gate passes.
    assert not (workdir / "factory/gate/task-repair-v2-postcheck.json").exists()
    assert (
        workdir / "factory/tasks/task-v2/alphaevolve-scheduling/tests/test.sh"
    ).is_file()


def test_static_gate_postcheck_quarantines_after_attempt_cap(tmp_path: Path):
    plan = _gate_plan(max_attempts=2)
    workdir = _workspace(tmp_path)
    out = tmp_path / "run"
    executable = _repair_cli(tmp_path, fix_on_retry=False)
    result = agentic_factory.run_factory(
        plan,
        workdir=workdir,
        out=out,
        environments=_environments(),
        executables={"claude-code": executable},
    )
    assert result["status"] == "quarantined"
    assert result["quarantine"]["failure_class"] == "deterministic_gate_failed"
    # Gate-failed outputs stay in place for auditing and later repair rounds.
    assert (workdir / "factory/tasks/task-v2/task.toml").is_file()
    advisory = json.loads(
        (workdir / "factory/gate/task-repair-v2-postcheck.json").read_text()
    )
    assert advisory["passed"] is False


def _variant_stage_plan() -> dict:
    return agentic_factory.compile_plan(
        name="variant conformance factory",
        source_binding_digest=DIGEST,
        stages=[
            {
                "id": "task-repair-v2",
                "role": "base task owner",
                "profile": "claude-code",
                "model": "fixture-model",
                "prompt": "p",
                "depends_on": [],
                "timeout_sec": 5,
                "max_attempts": 1,
                "max_budget_usd": 0.1,
                "required_outputs": [
                    {"path": "factory/tasks/task-v2", "kind": "directory"}
                ],
            },
            {
                "id": "variant-author",
                "role": "variant author",
                "profile": "claude-code",
                "model": "fixture-model",
                "prompt": "p",
                "depends_on": ["task-repair-v2"],
                "timeout_sec": 5,
                "max_attempts": 1,
                "max_budget_usd": 0.1,
                "required_outputs": [
                    {"path": "factory/tasks/variants", "kind": "directory"}
                ],
                "postchecks": ["variant-conformance"],
            },
        ],
    )


def _write_variants(workdir: Path, *, distinct: bool) -> None:
    base = workdir / "factory/tasks/task-v2"
    shutil.copytree(GOOD_TASK, base)
    variants = workdir / "factory/tasks/variants"
    rows = []
    for level in ("small", "medium", "large"):
        if distinct:
            slug = f"alphaevolve-scheduling-{level}"
            target = variants / slug
            shutil.copytree(GOOD_TASK, target)
            toml_path = target / "task.toml"
            toml_path.write_text(
                toml_path.read_text(encoding="utf-8").replace(
                    "terminal-bench-science/alphaevolve-scheduling",
                    f"terminal-bench-science/{slug}",
                ),
                encoding="utf-8",
            )
            (target / "data" / "scale.json").write_text(
                json.dumps({"level": level}), encoding="utf-8"
            )
            relative = slug
        else:
            # A rename-only "variant": the task tree is an untouched copy of
            # the base task nested under a level directory.
            relative = f"{level}/alphaevolve-scheduling"
            target = variants / level / "alphaevolve-scheduling"
            shutil.copytree(GOOD_TASK, target)
        rows.append(
            {
                "variant_id": f"variant-{level}",
                "relative_path": relative,
                "level": level,
                "axis_levels": {"instance_scale": level},
            }
        )
    manifest = {
        "schema_version": "orbenchlab.variant-manifest.v1",
        "primary_axis": {
            "name": "instance_scale",
            "expected_direction": "solve rate decreases with scale",
            "ordered_levels": ["small", "medium", "large"],
        },
        "variants": rows,
        "evaluation_mode": "exploratory",
    }
    (variants / "variant-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def test_variant_conformance_rejects_byte_identical_variants(tmp_path: Path):
    plan = _variant_stage_plan()
    workdir = _workspace(tmp_path)
    stage = next(s for s in plan["stages"] if s["id"] == "variant-author")
    _write_variants(workdir, distinct=False)
    findings = factory_gates.run_postchecks(
        stage, workspace=workdir, plan=plan, attempt_number=1
    )
    assert findings["passed"] is False
    problems = findings["postchecks"][0]["problems"]
    assert any("byte-identical" in problem for problem in problems)


def test_variant_conformance_passes_distinct_static_clean_variants(tmp_path: Path):
    plan = _variant_stage_plan()
    workdir = _workspace(tmp_path)
    stage = next(s for s in plan["stages"] if s["id"] == "variant-author")
    _write_variants(workdir, distinct=True)
    findings = factory_gates.run_postchecks(
        stage, workspace=workdir, plan=plan, attempt_number=1
    )
    assert findings["passed"] is True
    targets = findings["postchecks"][0]["targets"]
    assert len(targets) == 3
    for target in targets:
        assert target["diff_vs_base"]["added"] == ["data/scale.json"]
        assert target["diff_vs_base"]["changed"] == ["task.toml"]
