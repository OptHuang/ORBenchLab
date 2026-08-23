"""Red-capable acceptance tests for the practical one-command workflow.

These tests intentionally exercise the product seam a user touches: one CLI
invocation must prepare a workspace, and a completed Harbor bundle must turn
into a normalized slice plus a report without hand-written glue.
"""

from __future__ import annotations

import json
import signal
import shutil
import subprocess
from pathlib import Path

import pytest

from orbenchlab.cli import main
from orbenchlab.core.errors import EvidenceError


def _copy_named_checkout(upstream_fixtures: Path, tmp_path: Path) -> Path:
    """ORAgentBench's upstream wrapper requires this exact checkout name."""
    source = tmp_path / "ORAgentBench"
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    return source


def test_one_command_prepares_an_auditable_workspace(
    capsys, upstream_fixtures: Path, tmp_path: Path
):
    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    workspace = tmp_path / "run workspace"

    code = main(
        [
            "run",
            "oragentbench",
            "--source",
            str(source),
            "--task",
            "single_task",
            "--agent",
            "oracle",
            "--date",
            "2026-08-24",
            "--workspace",
            str(workspace),
            "--prepare-only",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    run_root = Path(payload["run_root"])
    assert run_root.is_dir()
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "inspection.json").is_file()
    assert (run_root / "plan" / "plan.json").is_file()
    assert (run_root / "plan" / "plan_ledger.json").is_file()
    assert (run_root / "preflight.json").is_file()
    assert (run_root / "integrity.sha256").is_file()
    assert json.loads((run_root / "manifest.json").read_text())["state"] == "prepared"

    # The exact same command must resume/idempotently reuse the same identity.
    capsys.readouterr()
    second = main(
        [
            "run",
            "oragentbench",
            "--source",
            str(source),
            "--task",
            "single_task",
            "--agent",
            "oracle",
            "--date",
            "2026-08-24",
            "--workspace",
            str(workspace),
            "--prepare-only",
        ]
    )
    assert second == 0
    payload_2 = json.loads(capsys.readouterr().out)
    assert payload_2["run_root"] == payload["run_root"]
    assert payload_2["resumed"] is True


def test_harbor_bundle_ingest_builds_normalized_data_and_report(tmp_path: Path):
    from orbenchlab.ingest.harbor import ingest_harbor_bundle

    run_root = tmp_path / "run"
    plan = run_root / "plan"
    jobs = run_root / "jobs" / "job-oracle-seed-1"
    trial = jobs / "single_task__fixture"
    trajectory = trial / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)

    run_id = "0123456789abcdef"
    (plan / "jobs").mkdir(parents=True)
    (plan / "plan.json").write_text(
        json.dumps(
            {
                "campaign_id": "fixture-campaign",
                "integration": "oragentbench",
                "site": "local-docker",
                "evidence_intent": "exploratory",
                "runs": [
                    {
                        "run_id": run_id,
                        "task_name": "single_task",
                        "agent_id": "oracle",
                        "seed": 1,
                        "attempt": 1,
                        "job_name": "job-oracle-seed-1",
                        "match_key": "fixture-match-key",
                    }
                ],
            }
        )
        + "\n"
    )
    (plan / "plan_ledger.json").write_text(
        json.dumps(
            {
                "ledger_schema_version": "1.0",
                "campaign_id": "fixture-campaign",
                "entries": [
                    {
                        "run_id": run_id,
                        "task_name": "single_task",
                        "agent_id": "oracle",
                        "seed": 1,
                        "attempt": 1,
                        "job_name": "job-oracle-seed-1",
                        "match_key": "fixture-match-key",
                    }
                ],
            }
        )
        + "\n"
    )
    (plan / "jobs" / "job-oracle-seed-1.yaml").write_text(
        "job_name: job-oracle-seed-1\nagents:\n  - name: oracle\n"
    )
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.0",
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
            }
        )
        + "\n"
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "single_task__fixture",
                "task_name": "oragentbench/single_task",
                "task_checksum": "sha256:fixture",
                "agent_info": {"name": "oracle"},
                "agent_result": {"cost_usd": 0.0},
                "verifier_result": {
                    "rewards": {"feasibility": 1.0, "quality": 1.0}
                },
                "started_at": "2026-08-24T00:00:00Z",
                "finished_at": "2026-08-24T00:00:01Z",
            }
        )
        + "\n"
    )

    result = ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")

    assert result.trials == 1
    assert result.orphans == 0
    normalized = json.loads((run_root / "normalized" / "rollout.json").read_text())
    assert normalized["trials"][0]["run_id"] == run_id
    assert normalized["trials"][0]["scores"] == {
        "feasibility": 1.0,
        "quality": 1.0,
    }
    assert (run_root / "report" / "summary.md").is_file()
    assert (run_root / "evidence" / "integrity.sha256").is_file()


def test_one_command_prepares_a_codex_plan_without_serializing_a_secret(
    capsys, upstream_fixtures: Path, tmp_path: Path
):
    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    workspace = tmp_path / "runs"
    wrapper = source / "source" / "scripts" / "run_harbor_prebuild.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# upstream fixture wrapper\n", encoding="utf-8")

    code = main(
        [
            "run",
            "oragentbench",
            "--source",
            str(source),
            "--task",
            "single_task",
            "--agent",
            "codex",
            "--model",
            "gpt-5.5",
            "--auth-mode",
            "codex-auth-json",
            "--model-base-url",
            "https://router.example.test/v1",
            "--date",
            "2026-08-24",
            "--workspace",
            str(workspace),
            "--prepare-only",
        ]
    )

    assert code == 0
    run_root = Path(json.loads(capsys.readouterr().out)["run_root"])
    blob = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert "CODEX_FORCE_AUTH_JSON" in blob
    assert "OPENAI_API_KEY" not in blob
    assert "MODEL_API_KEY" not in blob


def test_doctor_checks_the_actual_codex_plan_runtime(
    capsys, monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    auth.chmod(0o600)
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth))
    monkeypatch.setenv("ORBENCH_MODEL_BASE_URL", "https://router.example.test/v1")
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    code = main(
        [
            "doctor",
            "oragentbench",
            "--source",
            str(source),
            "--task",
            "single_task",
            "--agent",
            "codex",
            "--model",
            "gpt-5.5",
            "--auth-mode",
            "codex-auth-json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["model_base_url_configured"] is True
    assert "router.example.test" not in json.dumps(payload)


def test_ingest_refuses_a_partial_job_that_silently_drops_a_planned_run(tmp_path: Path):
    from orbenchlab.ingest.harbor import ingest_harbor_bundle

    run_root = tmp_path / "run"
    plan_dir = run_root / "plan"
    trial = run_root / "jobs" / "job" / "task_a__fixture"
    (plan_dir / "jobs").mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)
    entries = [
        {"run_id": "a" * 16, "task_name": "task_a", "agent_id": "oracle", "seed": 1, "attempt": 1, "job_name": "job", "match_key": "a"},
        {"run_id": "b" * 16, "task_name": "task_b", "agent_id": "oracle", "seed": 1, "attempt": 1, "job_name": "job", "match_key": "b"},
    ]
    (plan_dir / "plan.json").write_text(
        json.dumps({"campaign_id": "c", "integration": "oragentbench", "site": "local-docker", "evidence_intent": "exploratory", "runs": entries}) + "\n"
    )
    (plan_dir / "plan_ledger.json").write_text(
        json.dumps({"campaign_id": "c", "entries": entries}) + "\n"
    )
    (plan_dir / "jobs" / "job.yaml").write_text("agents:\n  - name: oracle\n")
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.0", "steps": [{"step_id": 1}]}) + "\n"
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "task_a__fixture",
                "task_name": "task_a",
                "verifier_result": {"rewards": {"feasibility": 1, "quality": 1}},
            }
        )
        + "\n"
    )

    with pytest.raises(EvidenceError, match="missing planned run"):
        ingest_harbor_bundle(run_root=run_root, jobs_root=run_root / "jobs")


def test_wall_clock_timeout_terminates_the_upstream_process_group(monkeypatch, tmp_path: Path):
    from orbenchlab import workflow

    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4321

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -signal.SIGTERM

    monkeypatch.setattr(workflow.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(workflow.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    with (tmp_path / "stdout").open("wb") as stdout, (tmp_path / "stderr").open("wb") as stderr:
        code = workflow._run_process_group(
            ["fixture"], cwd=tmp_path, environ={}, stdout=stdout, stderr=stderr, timeout_sec=20
        )

    assert code == 124
    assert killed == [(4321, signal.SIGTERM)]
