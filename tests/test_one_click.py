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
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from orbenchlab.cli import main
from orbenchlab.core.errors import EvidenceError, PreconditionError


def _copy_named_checkout(upstream_fixtures: Path, tmp_path: Path) -> Path:
    """ORAgentBench's upstream wrapper requires this exact checkout name."""
    source = tmp_path / "ORAgentBench"
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    return source


def _install_passing_runner_probes(monkeypatch, execution) -> None:
    """Make execution-path tests independent of the developer's Docker host."""
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    def probe(argv, **kwargs):
        stdout = "harbor, version 0.16.2\n" if argv[-1] == "--version" else ""
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(execution.subprocess, "run", probe)


def test_different_campaigns_share_one_host_docker_alias_lock_until_fingerprinted(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    """A second workspace cannot race the fixed upstream Docker alias."""
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    first = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs-a",
        wall_clock_sec=20,
    )
    second = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-25",
        workspace=tmp_path / "runs-b",
        wall_clock_sec=20,
    )
    assert first.campaign_id != second.campaign_id
    _install_passing_runner_probes(monkeypatch, execution)
    monkeypatch.setattr(workflow, "_run_process_group", lambda *args, **kwargs: 0)
    monkeypatch.setattr(workflow, "ingest_harbor_bundle", lambda **kwargs: SimpleNamespace())
    environ = {"ORBENCH_HOST_LOCK_DIR": str(tmp_path / "host-locks")}
    attempted_while_fingerprinting = False

    def fingerprint(image):
        nonlocal attempted_while_fingerprinting
        if not attempted_while_fingerprinting:
            attempted_while_fingerprinting = True
            with pytest.raises(PreconditionError, match="shared Docker image alias"):
                workflow.execute_prepared_run(second, environ=environ)
            second_manifest = json.loads(
                (second.run_root / "manifest.json").read_text(encoding="utf-8")
            )
            assert second_manifest["state"] == "prepared"
        return {
            "requested_tag": image,
            "image_id": "sha256:" + "1" * 64,
            "repo_digests": [],
        }

    monkeypatch.setattr(execution, "docker_image_fingerprint", fingerprint)

    workflow.execute_prepared_run(first, environ=environ)
    assert attempted_while_fingerprinting

    # The same host lock is released after fingerprinting, so the rejected
    # campaign can subsequently run without repairing or recreating its plan.
    workflow.execute_prepared_run(second, environ=environ)
    second_manifest = json.loads(
        (second.run_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert second_manifest["state"] == "completed"
    assert second_manifest["runtime_image_alias_verification"][
        "matches_runtime_image"
    ] is True


def test_paid_runtime_image_must_match_the_fixed_upstream_alias(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="codex",
        scaffold_version="fixture-cli-1.2.3",
        model="gpt-5.5",
        auth_mode="api-key",
        model_base_url="https://router.example.test/v1",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    _install_passing_runner_probes(monkeypatch, execution)
    monkeypatch.setattr(workflow, "_run_process_group", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        workflow,
        "ingest_harbor_bundle",
        lambda **kwargs: pytest.fail("mismatched image reached result ingest"),
    )

    def fingerprint(image):
        image_id = "1" if image == prepared.runtime_image_tag else "2"
        return {
            "requested_tag": image,
            "image_id": "sha256:" + image_id * 64,
            "repo_digests": [],
        }

    monkeypatch.setattr(execution, "docker_image_fingerprint", fingerprint)
    environ = {
        "MODEL_API_KEY": "ephemeral-test-value",
        "MODEL_BASE_URL": "https://router.example.test/v1",
        "ORBENCH_HOST_LOCK_DIR": str(tmp_path / "host-locks"),
    }

    with pytest.raises(EvidenceError, match="fixed ORAgentBench base alias"):
        workflow.execute_prepared_run(
            prepared,
            acknowledge_cost="i-accept-model-costs",
            environ=environ,
        )

    manifest = json.loads(
        (prepared.run_root / "manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (prepared.run_root / "receipt.json").read_text(encoding="utf-8")
    )
    verification = manifest["runtime_image_alias_verification"]
    assert manifest["state"] == "failed"
    assert verification["fixed_alias"] == execution.ORAGENTBENCH_FIXED_BASE_IMAGE
    assert verification["matches_runtime_image"] is False
    assert receipt["runtime_image_alias_verification"] == verification
    assert set(manifest["runtime_image"]) == {
        "requested_tag",
        "image_id",
        "repo_digests",
    }

    # Failure does not strand the host-wide lock.
    with workflow._oragentbench_docker_alias_lock(environ=environ):
        pass


def test_host_docker_alias_lock_rejects_a_symlink_directory(
    monkeypatch, tmp_path: Path
):
    from orbenchlab import workflow

    target = tmp_path / "real-locks"
    target.mkdir(mode=0o700)
    link = tmp_path / "lock-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(PreconditionError, match="unavailable or unsafe"):
        with workflow._oragentbench_docker_alias_lock(
            environ={"ORBENCH_HOST_LOCK_DIR": str(link)}
        ):
            pytest.fail("unsafe host lock directory was accepted")


def test_host_docker_alias_lock_rejects_a_foreign_owner(
    monkeypatch, tmp_path: Path
):
    from orbenchlab import workflow

    lock_dir = tmp_path / "host-locks"
    lock_dir.mkdir(mode=0o700)
    actual_uid = workflow.os.getuid()
    monkeypatch.setattr(workflow.os, "getuid", lambda: actual_uid + 1)

    with pytest.raises(PreconditionError, match="not owned by the current user"):
        with workflow._oragentbench_docker_alias_lock(
            environ={"ORBENCH_HOST_LOCK_DIR": str(lock_dir)}
        ):
            pytest.fail("foreign-owned host lock directory was accepted")


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


def test_one_command_refuses_a_failed_upstream_inspection(
    capsys, upstream_fixtures: Path, tmp_path: Path
):
    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    (source / "metrics" / "per_dimension_reward.py").unlink()

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
            str(tmp_path / "runs"),
            "--prepare-only",
        ]
    )

    assert code == 5
    assert "official_metric_script" in capsys.readouterr().err


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
    # Harbor also writes a job-level aggregate named result.json beside the
    # trial directories. It is metadata, not an orphan trial.
    (jobs / "result.json").write_text(
        json.dumps({"n_total_trials": 1, "stats": {"n_completed": 1}}) + "\n"
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
            "--scaffold-version",
            "fixture-cli-1.2.3",
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
    _install_passing_runner_probes(monkeypatch, execution)

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
            "--scaffold-version",
            "fixture-cli-1.2.3",
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


def test_doctor_fails_when_docker_is_installed_but_the_daemon_is_unreachable(
    capsys, monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    def probe(argv, **kwargs):
        if argv[-1] == "--version":
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout="harbor, version 0.16.2\n", stderr=""
            )
        return subprocess.CompletedProcess(
            argv, returncode=1, stdout="", stderr="sensitive host detail"
        )

    monkeypatch.setattr(execution.subprocess, "run", probe)
    code = main(
        [
            "doctor",
            "oragentbench",
            "--source",
            str(source),
            "--task",
            "single_task",
            "--agent",
            "oracle",
        ]
    )

    assert code == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("docker info" in item for item in payload["checks"]["missing"])
    assert "sensitive host detail" not in json.dumps(payload)


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


def test_completed_resume_refreshes_integrity_after_reingest(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    run_root = prepared.run_root
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = "completed"
    workflow._atomic_json(manifest_path, manifest)
    workflow._write_integrity(run_root)
    _install_passing_runner_probes(monkeypatch, execution)

    def fake_ingest(**kwargs):
        (run_root / "normalized").mkdir()
        (run_root / "normalized" / "rollout.json").write_text("{}\n")
        return SimpleNamespace()

    monkeypatch.setattr(workflow, "ingest_harbor_bundle", fake_ingest)

    workflow.execute_prepared_run(prepared)

    integrity = (run_root / "integrity.sha256").read_text()
    assert "normalized/rollout.json" in integrity


def test_running_state_is_integrity_checked_before_upstream_starts(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    _install_passing_runner_probes(monkeypatch, execution)

    def inspect_integrity_before_launch(*args, **kwargs):
        workflow._verify_integrity(prepared.run_root)
        manifest = json.loads((prepared.run_root / "manifest.json").read_text())
        assert manifest["state"] == "running"
        assert manifest["runner_pid"]
        return 9

    monkeypatch.setattr(workflow, "_run_process_group", inspect_integrity_before_launch)
    with pytest.raises(Exception, match="exited with code 9"):
        workflow.execute_prepared_run(prepared)


def test_unexpected_ingest_failure_marks_campaign_failed_and_refreshes_integrity(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    _install_passing_runner_probes(monkeypatch, execution)
    monkeypatch.setattr(workflow, "_run_process_group", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        execution,
        "docker_image_fingerprint",
        lambda image: {
            "requested_tag": image,
            "image_id": "sha256:" + "1" * 64,
            "repo_digests": [],
        },
    )
    monkeypatch.setattr(
        workflow,
        "ingest_harbor_bundle",
        lambda **kwargs: (_ for _ in ()).throw(OSError("do not serialize this detail")),
    )

    with pytest.raises(OSError):
        workflow.execute_prepared_run(prepared)

    manifest = json.loads((prepared.run_root / "manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert manifest["failure_type"] == "OSError"
    assert "do not serialize" not in json.dumps(manifest)
    assert "runner_pid" not in manifest
    workflow._verify_integrity(prepared.run_root)


def test_successful_upstream_without_an_image_identity_fails_before_ingest(
    monkeypatch, upstream_fixtures: Path, tmp_path: Path
):
    """Exit zero is not completion unless Docker identifies what actually ran."""
    from orbenchlab import execution, workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    _install_passing_runner_probes(monkeypatch, execution)
    monkeypatch.setattr(workflow, "_run_process_group", lambda *args, **kwargs: 0)
    monkeypatch.setattr(execution, "docker_image_fingerprint", lambda image: None)
    monkeypatch.setattr(
        workflow,
        "ingest_harbor_bundle",
        lambda **kwargs: pytest.fail("unidentified image reached result ingest"),
    )

    with pytest.raises(EvidenceError, match="Docker image identity"):
        workflow.execute_prepared_run(prepared)

    manifest = json.loads((prepared.run_root / "manifest.json").read_text())
    receipt = json.loads((prepared.run_root / "receipt.json").read_text())
    assert manifest["state"] == "failed"
    assert manifest["failure"] == "upstream image identity could not be verified"
    assert manifest["runtime_image"] is None
    assert receipt["runtime_image"] is None
    assert "runner_pid" not in manifest
    workflow._verify_integrity(prepared.run_root)


def test_failed_agent_workspace_uses_upstream_resume_when_job_config_exists(
    upstream_fixtures: Path, tmp_path: Path
):
    from orbenchlab import workflow

    source = _copy_named_checkout(upstream_fixtures, tmp_path)
    wrapper = source / "source" / "scripts" / "run_harbor_prebuild.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# fixture wrapper\n")
    skill = source / "skills" / "fixture-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Fixture skill\n", encoding="utf-8")
    kwargs = dict(
        source=source,
        task="single_task",
        agent="codex",
        scaffold_version="fixture-cli-1.2.3",
        model="gpt-5.5",
        auth_mode="codex-auth-json",
        model_base_url="https://router.example.test/v1",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    first = workflow.prepare_oragentbench_run(**kwargs)
    plan = json.loads((first.run_root / "plan" / "plan.json").read_text())
    job_name = plan["jobs"][0]["job_name"]
    job_dir = first.run_root / "jobs" / job_name
    job_dir.mkdir(parents=True)
    import yaml

    config = yaml.safe_load(
        (first.run_root / "plan" / "jobs" / f"{job_name}.yaml").read_text()
    )
    config.pop("pre_build")
    config["agents"][0]["name"] = None
    config["agents"][0]["import_path"] = (
        "ORAgentBench.harbor_agents.prebuilt_agents:PrebuiltCodex"
    )
    dynamic_root = Path(tempfile.mkdtemp(prefix="oragentbench-skills-"))
    dynamic_dataset = dynamic_root / "harbor_tasks"
    dynamic_task = dynamic_dataset / "single_task"
    shutil.copytree(first.source / "harbor_tasks" / "single_task", dynamic_task)
    for directory in [dynamic_task, *dynamic_task.rglob("*")]:
        if directory.is_dir():
            directory.chmod(0o755)
        elif directory.is_file():
            directory.chmod(0o644)
    task_toml = dynamic_task / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            "[environment]\n", '[environment]\nskills_dir = "/skills"\n'
        ),
        encoding="utf-8",
    )
    shutil.copytree(
        first.source / "skills" / "fixture-skill",
        dynamic_task / "environment" / "skills" / "fixture-skill",
    )
    config["datasets"][0]["path"] = str(dynamic_dataset)
    (job_dir / "config.json").write_text(json.dumps(config) + "\n")
    manifest_path = first.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = "failed"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    workflow._write_integrity(first.run_root)

    resumed = workflow.prepare_oragentbench_run(**kwargs)

    assert resumed.resumed is True
    assert "--resume" in resumed.command.argv
    assert "--cleanup-before-resume" in resumed.command.argv
    binding = json.loads((resumed.run_root / "resume-binding.json").read_text())
    assert binding["job_name"] == job_name
    assert binding["scaffold_version"] == "fixture-cli-1.2.3"
