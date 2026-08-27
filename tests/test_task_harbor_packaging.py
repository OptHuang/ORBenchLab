from pathlib import Path


ROOT = Path(__file__).parents[1]
TASKS = ROOT / "examples/tasks"


def test_alpha_environment_materializes_frozen_inputs():
    task = TASKS / "alphaevolve-scheduling"
    assert (task / "environment/instance.json").read_bytes() == (task / "data/instance.json").read_bytes()
    assert (task / "environment/data/reference-bounds.json").read_bytes() == (task / "data/reference-bounds.json").read_bytes()
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert "COPY instance.json /root/instance.json" in dockerfile
    assert "COPY data/reference-bounds.json /root/data/reference-bounds.json" in dockerfile
    wrapper = (task / "solution/solve.sh").read_text()
    assert "cp /solution/solver.py /root/submission/solver.py" in wrapper
    assert "--instance /root/instance.json" in wrapper
    assert "--output /root/submission/solution.json" in wrapper
    assert (task / "tests/instance.json").read_bytes() == (task / "data/instance.json").read_bytes()
    assert (task / "tests/data/reference-bounds.json").read_bytes() == (task / "data/reference-bounds.json").read_bytes()
    verifier_dockerfile = (task / "tests/Dockerfile").read_text()
    assert "COPY . /tests/" in verifier_dockerfile
    assert "COPY instance.json /root/instance.json" in verifier_dockerfile
    assert "COPY data/reference-bounds.json /root/data/reference-bounds.json" in verifier_dockerfile
    test_sh = (task / "tests/test.sh").read_text()
    assert "/logs/verifier/reward.txt" in test_sh


def test_vrp_environment_materializes_frozen_inputs():
    task = TASKS / "vrp-recovery"
    assert (task / "environment/instance.json").read_bytes() == (task / "data/instance.json").read_bytes()
    assert (task / "environment/events.jsonl").read_bytes() == (task / "data/events.jsonl").read_bytes()
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert "COPY instance.json /root/instance.json" in dockerfile
    assert "COPY events.jsonl /root/events.jsonl" in dockerfile
    wrapper = (task / "solution/solve.sh").read_text()
    assert "cp /solution/solver.py /root/submission/solver.py" in wrapper
    assert "ORBENCH_TASK_ROOT=/root" in wrapper
    assert (task / "tests/instance.json").read_bytes() == (task / "data/instance.json").read_bytes()
    assert (task / "tests/events.jsonl").read_bytes() == (task / "data/events.jsonl").read_bytes()
    verifier_dockerfile = (task / "tests/Dockerfile").read_text()
    assert "COPY . /tests/" in verifier_dockerfile
    assert "COPY instance.json /root/instance.json" in verifier_dockerfile
    assert "COPY events.jsonl /root/events.jsonl" in verifier_dockerfile
    test_sh = (task / "tests/test.sh").read_text()
    assert "/logs/verifier/reward.txt" in test_sh
