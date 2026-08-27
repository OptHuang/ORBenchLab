from pathlib import Path

from orbenchlab.volc_rollout import _extract_solver, _hint, _task_id


ROOT = Path(__file__).parents[1]


def test_task_identity_matches_task_genome_family():
    assert _task_id(ROOT / "examples/tasks/alphaevolve-scheduling") == "alphaevolve_scheduling"
    assert _task_id(ROOT / "examples/tasks/vrp-recovery") == "vrp_recovery"


def test_solver_extractor_accepts_bounded_file_aliases():
    assert _extract_solver({"solver_py": "print(1)"}) == "print(1)"
    assert _extract_solver({"files": {"submission/solver.py": "print(2)"}}) == "print(2)"


def test_hint_ladder_is_task_specific_and_explicit():
    alpha = _hint(ROOT / "examples/tasks/alphaevolve-scheduling", 2)
    vrp = _hint(ROOT / "examples/tasks/vrp-recovery", 1)
    assert "alphaevolve-scheduling.solution.v1" in alpha
    assert "initial_routes.json" in vrp
