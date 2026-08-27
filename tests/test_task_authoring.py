from __future__ import annotations

import json
from pathlib import Path

from orbenchlab import task_authoring


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_candidate(root: Path) -> Path:
    _write(
        root / "task.toml",
        '''schema_version = "1.4"
artifacts = ["/app/results.json"]

[task]
name = "terminal-bench-science/jobshop-replan"
description = "Replan a constrained job shop and emit a verifier-checkable schedule."
authors = [{ name = "ORBenchLab", email = "lab@example.org" }]

[metadata]
author_name = "ORBenchLab"
author_email = "lab@example.org"
author_organization = "Independent Researcher"
author_profile = "https://example.org/profile"
domain = "mathematical-sciences"
field = "operations-research"
subfield = "scheduling"
tags = ["optimization", "scheduling", "replanning"]
expert_time_estimate_hours = 12.0

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
network_mode = "public"
''',
    )
    _write(root / "instruction.md", """Produce the final schedule and machine-readable audit files.\n\nDo not modify the frozen input instance.\n""")
    _write(
        root / "README.md",
        """# Job-shop replan\n\n## Difficulty\n\nDifficulty comes from coupled machine capacity, precedence, and event-replanning constraints across multiple input files. The levels vary instance scale and disruption density while preserving the same scientific objective.\n\n## Reference solution\n\nThe reference solution constructs a feasible schedule, recomputes the objective, and writes the exact output files consumed by the verifier. It is included to demonstrate solvability and to document the important algorithmic choices.\n\n## Verification\n\nThe verifier executes the submitted program in a separate no-network environment, checks schema and feasibility, recomputes the objective, and emits a per-check CTRF report. Tests inspect final artifacts rather than the commands used to create them.\n""",
    )
    _write(root / "environment/Dockerfile", "FROM python:3.12-slim\n")
    _write(root / "tests/Dockerfile", "FROM python:3.12-slim\nRUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5\n")
    _write(root / "tests/test.sh", "#!/bin/sh\npytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA\n")
    _write(root / "tests/test_state.py", "def test_output():\n    assert True\n")
    _write(root / "solution/solve.py", "print('reference')\n")
    _write(root / "data/instance.json", "{}\n")
    return root


def _paper(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "title": "A reproducible job-shop re-planning study",
                "url": "https://arxiv.org/abs/2602.02029",
                "source_content_digest": "sha256:" + "a" * 64,
                "license_status": "verified",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_missing_task_is_blocked(tmp_path: Path):
    receipt = task_authoring.validate_task(tmp_path)
    assert receipt["decision"] == "blocked"
    assert receipt["counts"]["fail"] >= 1
    assert len(receipt["implementation_criteria"]) == 39
    assert any(item["name"] == "task_toml_schema" and item["status"] == "fail" for item in receipt["implementation_criteria"])


def test_complete_skeleton_is_ready_for_review_and_round_linked(tmp_path: Path):
    task = _make_candidate(tmp_path / "jobshop-replan")
    paper = _paper(tmp_path / "paper.json")
    first = task_authoring.validate_task(task, paper_provenance=paper, round_number=1)
    assert first["decision"] == "ready-for-human-review"
    assert first["counts"]["fail"] == 0
    assert first["counts"]["review"] >= 1
    paths = task_authoring.write_receipt(first, tmp_path / "round-1")
    second = task_authoring.validate_task(
        task,
        paper_provenance=paper,
        round_number=2,
        previous_receipt=paths["json"],
    )
    assert second["previous_receipt_digest"].startswith("sha256:")
    repeat = task_authoring.validate_task(
        task,
        paper_provenance=paper,
        round_number=2,
        previous_receipt=paths["json"],
    )
    assert second["receipt_digest"] == repeat["receipt_digest"]
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["decision"] == "ready-for-human-review"
    assert "verifiable" in paths["markdown"].read_text(encoding="utf-8")


def test_rubric_regression_catches_missing_ctrf_and_security_hazard(tmp_path: Path):
    task = _make_candidate(tmp_path / "jobshop-replan")
    paper = _paper(tmp_path / "paper.json")
    test_script = task / "tests/test.sh"
    test_script.write_text("#!/bin/sh\npytest /tests/test_state.py\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN cat ~/.ssh/id_rsa\n", encoding="utf-8"
    )
    receipt = task_authoring.validate_task(task, paper_provenance=paper)
    assert receipt["decision"] == "blocked"
    failed = {item["name"] for item in receipt["implementation_criteria"] if item["status"] == "fail"}
    assert {"ctrf_reporting", "task_security"} <= failed


def test_receipt_digest_is_independent_of_checkout_path(tmp_path: Path):
    first_root = _make_candidate(tmp_path / "checkout-a" / "jobshop-replan")
    second_root = _make_candidate(tmp_path / "checkout-b" / "jobshop-replan")
    first_paper = _paper(tmp_path / "checkout-a" / "paper.json")
    second_paper = _paper(tmp_path / "checkout-b" / "paper.json")
    first = task_authoring.validate_task(first_root, paper_provenance=first_paper)
    second = task_authoring.validate_task(second_root, paper_provenance=second_paper)
    assert first["receipt_digest"] == second["receipt_digest"]


def test_paper_source_digest_mismatch_is_blocked(tmp_path: Path):
    task = _make_candidate(tmp_path / "jobshop-replan")
    source = tmp_path / "paper.txt"
    source.write_text("paper bytes", encoding="utf-8")
    provenance = tmp_path / "paper.json"
    provenance.write_text(
        json.dumps(
            {
                "title": "Paper",
                "url": "https://arxiv.org/abs/2602.02029",
                "source_content_digest": "sha256:" + "b" * 64,
                "source_path": str(source),
                "license_status": "registry-resolved",
            }
        ),
        encoding="utf-8",
    )
    receipt = task_authoring.validate_task(task, paper_provenance=provenance)
    assert receipt["decision"] == "blocked"
    assert receipt["provenance_checks"][0]["status"] == "fail"
