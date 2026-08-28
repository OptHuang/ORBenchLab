from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import difficulty_matrix, volc_rollout
from orbenchlab.cli import main


def _digest(value):
    return difficulty_matrix._digest(value)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _controls(task: Path) -> dict:
    task_id = volc_rollout._task_id(task)
    tree = volc_rollout._task_tree_digest(task)
    digests = {key: "sha256:" + char * 64 for key, char in zip(
        ("job_result_digest", "trial_result_digest", "ctrf_digest", "reward_digest", "artifact_manifest_digest"), "bcdef"
    )}
    gates = {
        "oracle": {"gate": "pass", "control": "oracle", "task_name": task_id, "reward": 1.0,
                   "ctrf_summary": {"tests": 2, "passed": 2, "failed": 0, "skipped": 0, "pending": 0, "other": 0}, **digests},
        "nop": {"gate": "pass", "control": "nop", "task_name": task_id, "reward": 0.0,
                "ctrf_summary": {"tests": 2, "passed": 0, "failed": 2, "skipped": 0, "pending": 0, "other": 0}, **digests},
    }
    value = {"schema_version": "orbenchlab.screening-report.v1", "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
             "task_tree_digest": tree, "authoring_task_tree_digest": tree, "executed_task_tree_digest": tree,
             "tasks": [{"task": task_id, "family": task_id, "arms": {}, "control_gates": gates,
                        "decision": "collect-more-evidence", "evidence_level": "E3"}]}
    value["report_digest"] = _digest(value)
    return value


def _matrix(task: Path, frontier_pass: int, weak_pass: int, preregistration_digest=None) -> dict:
    task_id = volc_rollout._task_id(task)
    trials = []
    for model, passed in (("frontier", frontier_pass), ("weak", weak_pass)):
        for attempt in range(1, 6):
            status = "pass" if attempt <= passed else "fail"
            trials.append({"trial_id": _digest([model, attempt]), "model_id": model, "attempt": attempt,
                           "status": status, "reward": 1.0 if status == "pass" else 0.0,
                           "trial_result_digest": "sha256:" + "1" * 64, "reward_digest": "sha256:" + "2" * 64,
                           "ctrf_digest": "sha256:" + "3" * 64, "ctrf_summary": {}, "trajectories": []})
    value = {"schema_version": "orbenchlab.harbor-model-matrix.v1", "task": task_id,
             "task_tree_digest": volc_rollout._task_tree_digest(task), "models": ["frontier", "weak"],
             "repetitions": 5, "rectangular": True, "trials": trials, "jobs": [],
             "provider_route_digest": "sha256:" + "4" * 64,
             "preregistration_digest": preregistration_digest,
             "agent": {"name": "claude-code", "executable_digest": "sha256:" + "a" * 64,
                       "max_budget_usd_per_trial": 1.0, "max_turns_per_trial": 40,
                       "max_job_attempts_per_model": 2,
                       "maximum_model_liability_usd": 20.0,
                       "budget_enforcement": "claude-cli-max-budget-usd"},
             "evidence_level": "E3", "checkpoint_capability": False}
    value["receipt_digest"] = _digest(value)
    return value


def _fixture(tmp_path: Path, *, held_out: bool = False):
    root = tmp_path / "variants"
    evidence = {}
    rows = []
    for index, (name, counts) in enumerate((("easy", (5, 3)), ("medium", (5, 0)), ("hard", (4, 0)))):
        task = root / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'[task]\nname = "demo_{name}"\n', encoding="utf-8")
        matrix = _write(tmp_path / "evidence" / name / "matrix.json", _matrix(task, *counts))
        controls = _write(tmp_path / "evidence" / name / "controls.json", _controls(task))
        evidence[name] = {"model_matrix": str(matrix), "controls": str(controls)}
        rows.append({"variant_id": name, "relative_path": name, "level": index,
                     "axis_levels": {"size": index}})
    manifest = {"schema_version": "orbenchlab.variant-manifest.v1",
                "evaluation_mode": "held-out-confirmation" if held_out else "exploratory",
                "primary_axis": {"name": "size", "expected_direction": "harder", "ordered_levels": [0, 1, 2]},
                "variants": rows}
    manifest_path = _write(tmp_path / "manifest.json", manifest)
    preregistration_path = None
    if held_out:
        preregistration = difficulty_matrix.build_preregistration(
            manifest_path=manifest_path,
            variants_root=root,
            frontier_model="frontier",
            weak_model="weak",
            repetitions=5,
            max_budget_usd=1.0,
            max_turns=40,
            max_job_attempts=2,
            provider_route_digest="sha256:" + "4" * 64,
            claude_executable_digest="sha256:" + "a" * 64,
        )
        preregistration_path = difficulty_matrix.write_preregistration(
            preregistration, tmp_path / "preregistration"
        )
        for name, counts in (("easy", (5, 3)), ("medium", (5, 0)), ("hard", (4, 0))):
            _write(
                Path(evidence[name]["model_matrix"]),
                _matrix(root / name, *counts, preregistration["preregistration_digest"]),
            )
    return manifest_path, root, evidence, preregistration_path


def test_builds_exploratory_e3_difficulty_matrix(tmp_path: Path):
    manifest, root, evidence, _ = _fixture(tmp_path)
    receipt = difficulty_matrix.build_receipt(manifest_path=manifest, variants_root=root, evidence=evidence,
                                              frontier_model="frontier", weak_model="weak")
    assert receipt["decision"] == "exploratory-promising"
    assert receipt["evidence_level"] == "E3"
    assert receipt["checkpoint_capability"] is False
    assert receipt["monotonicity"]["passed"] is True
    assert receipt["monotonicity"]["weak_model_has_strict_drop"] is True
    assert receipt["discrimination"]["separated_levels"] == ["medium"]
    assert receipt["nondegenerate_cell_present"] is True
    assert len(receipt["raw_trials"]) == 30


def test_held_out_must_be_predeclared_and_is_confirmatory(tmp_path: Path):
    manifest, root, evidence, preregistration = _fixture(tmp_path, held_out=True)
    receipt = difficulty_matrix.build_receipt(manifest_path=manifest, variants_root=root, evidence=evidence,
        frontier_model="frontier", weak_model="weak", held_out=True,
        preregistration_path=preregistration)
    assert receipt["decision"] == "confirmed-promising"
    assert receipt["preregistration_digest"].startswith("sha256:")
    with pytest.raises(difficulty_matrix.DifficultyMatrixError, match="predeclared"):
        difficulty_matrix.build_receipt(manifest_path=manifest, variants_root=root, evidence=evidence,
            frontier_model="frontier", weak_model="weak", held_out=False)


def test_rejects_forged_controls_and_nonrectangular_matrix(tmp_path: Path):
    manifest, root, evidence, _ = _fixture(tmp_path)
    controls = Path(evidence["easy"]["controls"])
    value = json.loads(controls.read_text())
    value["tasks"][0]["control_gates"]["nop"]["reward"] = 1.0
    value["report_digest"] = _digest({k: v for k, v in value.items() if k != "report_digest"})
    _write(controls, value)
    with pytest.raises(difficulty_matrix.DifficultyMatrixError, match="strict validation"):
        difficulty_matrix.build_receipt(manifest_path=manifest, variants_root=root, evidence=evidence,
            frontier_model="frontier", weak_model="weak")


def test_quarantines_nonmonotonic_or_nondegenerate_evidence(tmp_path: Path):
    manifest, root, evidence, _ = _fixture(tmp_path)
    matrix_path = Path(evidence["hard"]["model_matrix"])
    hard_task = root / "hard"
    _write(matrix_path, _matrix(hard_task, 5, 5))
    receipt = difficulty_matrix.build_receipt(manifest_path=manifest, variants_root=root, evidence=evidence,
        frontier_model="frontier", weak_model="weak")
    assert receipt["decision"] == "quarantine"
    assert receipt["monotonicity"]["passed"] is False


def test_cli_writes_atomic_receipt(tmp_path: Path, capsys):
    manifest, root, evidence, _ = _fixture(tmp_path)
    evidence_path = _write(tmp_path / "evidence-map.json", evidence)
    out = tmp_path / "out"
    code = main(["difficulty-matrix", "--manifest", str(manifest), "--variants-root", str(root),
                 "--evidence-map", str(evidence_path), "--frontier-model", "frontier",
                 "--weak-model", "weak", "--out", str(out)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "exploratory-promising"
    assert json.loads((out / "difficulty-matrix.json").read_text())["receipt_digest"].startswith("sha256:")
    assert not list(out.glob(".*.tmp"))


def test_writer_is_idempotent_and_refuses_conflicting_receipt(tmp_path: Path):
    out = tmp_path / "out"
    first = {"receipt_digest": "sha256:" + "1" * 64, "decision": "quarantine"}
    assert difficulty_matrix.write_receipt(first, out) == difficulty_matrix.write_receipt(first, out)
    with pytest.raises(difficulty_matrix.DifficultyMatrixError, match="overwrite"):
        difficulty_matrix.write_receipt(
            {"receipt_digest": "sha256:" + "2" * 64, "decision": "confirmed-promising"}, out
        )


def test_secondary_axes_are_validated_and_recorded(tmp_path: Path):
    manifest_path, root, evidence, _ = _fixture(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["secondary_axes"] = [
        {"name": "hint", "meaning": "instruction hint richness", "levels": [0, 1]}
    ]
    for row in document["variants"]:
        row["axis_levels"]["hint"] = 0
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    receipt = difficulty_matrix.build_receipt(
        manifest_path=manifest_path,
        variants_root=root,
        evidence=evidence,
        frontier_model="frontier",
        weak_model="weak",
    )
    assert receipt["secondary_axes"][0]["name"] == "hint"
    genome = receipt["difficulty_genome"]
    assert {row["variant_id"] for row in genome["variants"]} == {"easy", "medium", "hard"}
    assert all(row["axis_levels"]["hint"] == 0 for row in genome["variants"])


def test_undeclared_axis_levels_are_rejected_when_axes_are_declared(tmp_path: Path):
    manifest_path, root, evidence, _ = _fixture(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["secondary_axes"] = [{"name": "hint", "levels": [0, 1]}]
    document["variants"][0]["axis_levels"]["undeclared_axis"] = 3
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(difficulty_matrix.DifficultyMatrixError):
        difficulty_matrix.load_variant_manifest(manifest_path)


def test_byte_identical_variants_cannot_carry_a_difficulty_claim(tmp_path: Path):
    manifest_path, root, evidence, _ = _fixture(tmp_path)
    # Make "medium" byte-identical to "easy": renaming alone is not difficulty.
    (root / "medium" / "task.toml").write_text(
        (root / "easy" / "task.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    from orbenchlab import volc_rollout

    for name in ("easy", "medium"):
        matrix_path = Path(evidence[name]["model_matrix"])
        controls_path = Path(evidence[name]["controls"])
        counts = (5, 3) if name == "easy" else (5, 0)
        _write(matrix_path, _matrix(root / name, *counts))
        _write(controls_path, _controls(root / name))
    with pytest.raises(difficulty_matrix.DifficultyMatrixError) as excinfo:
        difficulty_matrix.build_receipt(
            manifest_path=manifest_path,
            variants_root=root,
            evidence=evidence,
            frontier_model="frontier",
            weak_model="weak",
        )
    assert "pairwise distinct" in str(excinfo.value)


def test_variant_identical_to_base_task_is_rejected(tmp_path: Path):
    manifest_path, root, evidence, _ = _fixture(tmp_path)
    from orbenchlab import volc_rollout

    base_digest = volc_rollout._task_tree_digest(root / "easy")
    with pytest.raises(difficulty_matrix.DifficultyMatrixError) as excinfo:
        difficulty_matrix.build_receipt(
            manifest_path=manifest_path,
            variants_root=root,
            evidence=evidence,
            frontier_model="frontier",
            weak_model="weak",
            base_task_tree_digest=base_digest,
        )
    assert "byte-identical to the base task" in str(excinfo.value)
