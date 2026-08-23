"""Regression tests for the remaining one-click lifecycle hardening gaps.

These tests deliberately live at product seams: runtime credential routing,
plan-directory reuse, interrupted-workspace integrity, and report semantics.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from orbenchlab import execution, workflow
from orbenchlab.campaign import compile as compile_mod
from orbenchlab.campaign import spec as spec_mod
from orbenchlab.core.errors import EvidenceError
from orbenchlab.core.evidence import EvidenceLabel
from orbenchlab.ingest import harbor as harbor_ingest
from orbenchlab.report import render as render_mod
from orbenchlab.report.model import NormalizedRollout, StrictPassRule, Trial


def _oab_source(upstream_fixtures: Path, tmp_path: Path) -> Path:
    source = tmp_path / execution.ORAGENTBENCH_CHECKOUT_DIRNAME
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    wrapper = source / execution.ORAGENTBENCH_PREBUILD_WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# fixture wrapper\n", encoding="utf-8")
    return source


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://router.example.test/v1",
        "https://user:LEAK_SENTINEL@router.example.test/v1",
        "https://router.example.test/v1?token=LEAK_SENTINEL",
        "https://router.example.test/v1#LEAK_SENTINEL",
    ],
)
def test_api_key_transport_rejects_unsafe_provider_url_without_echoing_it(
    upstream_fixtures: Path, tmp_path: Path, unsafe_url: str
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)

    report = execution.oragentbench_preconditions(
        source=source,
        task_name="single_task",
        scaffold="claude-code",
        model="deepseek-v4-pro",
        auth_mode="api-key",
        environ={"MODEL_API_KEY": "secret", "MODEL_BASE_URL": unsafe_url},
        require_docker=False,
        require_harbor=False,
    )

    rendered = json.dumps(report.to_dict())
    assert not report.ok
    assert "provider base URL" in rendered
    assert unsafe_url not in rendered
    assert "LEAK_SENTINEL" not in rendered


def test_reusing_plan_directory_removes_jobs_from_the_previous_campaign(
    campaigns_dir: Path, sites_dir: Path, tmp_path: Path
) -> None:
    raw = spec_mod.load_spec(campaigns_dir / "oragentbench-controls.yaml")
    first = compile_mod.compile_campaign(spec_mod.validate(raw, sites_dir=sites_dir))
    out = tmp_path / "plan"
    compile_mod.write_plan(first, out)

    smaller = copy.deepcopy(raw)
    smaller["agents"] = [{"id": "oracle", "scaffold": "oracle"}]
    smaller["seeds"] = [1]
    second = compile_mod.compile_campaign(
        spec_mod.validate(smaller, sites_dir=sites_dir)
    )
    compile_mod.write_plan(second, out)

    expected = {f"{job.job_name}.yaml" for job in second.jobs}
    actual = {path.name for path in (out / "jobs").glob("*.yaml")}
    assert actual == expected


def test_reusing_plan_directory_preserves_unowned_yaml(
    campaigns_dir: Path, sites_dir: Path, tmp_path: Path
) -> None:
    raw = spec_mod.load_spec(campaigns_dir / "oragentbench-controls.yaml")
    first = compile_mod.compile_campaign(spec_mod.validate(raw, sites_dir=sites_dir))
    out = tmp_path / "plan"
    compile_mod.write_plan(first, out)
    operator_file = out / "jobs" / "operator-owned.yaml"
    operator_file.write_text("job_name: do-not-touch\n", encoding="utf-8")

    smaller = copy.deepcopy(raw)
    smaller["agents"] = [{"id": "oracle", "scaffold": "oracle"}]
    smaller["seeds"] = [1]
    second = compile_mod.compile_campaign(
        spec_mod.validate(smaller, sites_dir=sites_dir)
    )
    compile_mod.write_plan(second, out)

    assert operator_file.read_text(encoding="utf-8") == "job_name: do-not-touch\n"


def test_prepared_workspace_rejects_an_unlisted_resume_control_file(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    kwargs = dict(
        source=source,
        task="single_task",
        agent="claude-code",
        scaffold_version="fixture-cli-1.2.3",
        model="deepseek-v4-pro",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
        model_base_url="https://router.example.test/v1",
    )
    first = workflow.prepare_oragentbench_run(**kwargs)
    plan = json.loads((first.run_root / "plan" / "plan.json").read_text())
    job_name = plan["jobs"][0]["job_name"]

    # This file changes the next upstream command to --resume.  Because it was
    # not covered by the prepared workspace's integrity manifest, it must not
    # be silently trusted as legitimate Harbor output.
    injected = first.run_root / "jobs" / job_name / "config.json"
    injected.parent.mkdir(parents=True)
    injected.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="integrity"):
        workflow.prepare_oragentbench_run(**kwargs)


def test_workspace_integrity_rejects_symlinks(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (prepared.run_root / "injected.json").symlink_to(outside)

    with pytest.raises(EvidenceError, match="symlink"):
        workflow._verify_integrity(prepared.run_root)


def test_running_crash_recovery_allows_only_upstream_output_roots(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    kwargs = dict(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    first = workflow.prepare_oragentbench_run(**kwargs)
    manifest_path = first.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = "running"
    manifest["runtime_image"] = {
        "requested_tag": "oragentbench-base:py311-scip",
        "image_id": "sha256:" + "1" * 64,
        "repo_digests": [],
    }
    manifest["runtime_image_alias_verification"] = {
        "fixed_alias": "oragentbench-base:py311-scip",
        "fixed_alias_image_id": "sha256:" + "1" * 64,
        "matches_runtime_image": True,
    }
    workflow._atomic_json(manifest_path, manifest)
    import yaml

    config = yaml.safe_load(
        (first.run_root / "plan" / "jobs" / f"{first.job_name}.yaml").read_text()
    )
    config_path = first.run_root / "jobs" / first.job_name / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    existing_log = first.run_root / "logs" / "upstream.stdout.log"
    existing_log.parent.mkdir()
    existing_log.write_text("before crash\n", encoding="utf-8")
    workflow._write_integrity(first.run_root)

    # Both a changed pre-existing log and a newly-created Harbor file are
    # expected after a SIGKILL.  Neither may relax checks on other roots.
    existing_log.write_text("after crash\n", encoding="utf-8")
    harbor_output = first.run_root / "jobs" / first.job_name / "result.json"
    harbor_output.parent.mkdir(parents=True, exist_ok=True)
    harbor_output.write_text("{}\n", encoding="utf-8")
    assert workflow.prepare_oragentbench_run(**kwargs).resumed is True

    (first.run_root / "untrusted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="absent from integrity manifest"):
        workflow.prepare_oragentbench_run(**kwargs)


def _write_interrupted_harbor_config(
    prepared: workflow.PreparedRun, config: dict[str, object], *, state: str = "failed"
) -> Path:
    config_path = prepared.run_root / "jobs" / prepared.job_name / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    manifest_path = prepared.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = state
    workflow._atomic_json(manifest_path, manifest)
    workflow._write_integrity(prepared.run_root)
    return config_path


def test_resume_rejects_matching_task_name_from_an_unrelated_dataset(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    """A task name alone cannot authorize a different benchmark tree."""
    source = _oab_source(upstream_fixtures, tmp_path)
    kwargs = dict(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    prepared = workflow.prepare_oragentbench_run(**kwargs)
    import yaml

    config = yaml.safe_load(
        (prepared.run_root / "plan" / "jobs" / f"{prepared.job_name}.yaml").read_text()
    )
    unrelated = Path(tempfile.mkdtemp(prefix="other-benchmark-")) / "harbor_tasks"
    shutil.copytree(prepared.source / "harbor_tasks", unrelated)
    config["datasets"][0]["path"] = str(unrelated)
    _write_interrupted_harbor_config(prepared, config)

    with pytest.raises(EvidenceError, match="dataset"):
        workflow.prepare_oragentbench_run(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda config: config.__setitem__("n_concurrent_trials", 8), "concurrency"),
        (
            lambda config: config["environment"].__setitem__(
                "extra_allowed_hosts", ["attacker.example"]
            ),
            "environment",
        ),
        (
            lambda config: config["agents"][0]["kwargs"].__setitem__(
                "tool_policy", {"allow": ["WebSearch"]}
            ),
            "agent",
        ),
    ],
)
def test_resume_rejects_runtime_config_that_changes_compiled_identity(
    upstream_fixtures: Path,
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    kwargs = dict(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    prepared = workflow.prepare_oragentbench_run(**kwargs)
    import yaml

    config = yaml.safe_load(
        (prepared.run_root / "plan" / "jobs" / f"{prepared.job_name}.yaml").read_text()
    )
    mutator(config)
    _write_interrupted_harbor_config(prepared, config)

    with pytest.raises(EvidenceError, match=message):
        workflow.prepare_oragentbench_run(**kwargs)


def _paid_interrupted_config(
    upstream_fixtures: Path,
    tmp_path: Path,
    *,
    agent: str = "codex",
    auth_mode: str = "codex-auth-json",
) -> tuple[dict[str, object], workflow.PreparedRun, dict[str, object], Path]:
    source = _oab_source(upstream_fixtures, tmp_path)
    skill = source / "skills" / "fixture-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Snapshot fixture skill\n", encoding="utf-8")
    kwargs: dict[str, object] = {
        "source": source,
        "task": "single_task",
        "agent": agent,
        "scaffold_version": "fixture-cli-1.2.3",
        "model": "gpt-5.5" if agent == "codex" else "deepseek-v4-pro",
        "auth_mode": auth_mode,
        "model_base_url": "https://router.example.test/v1",
        "date": "2026-08-24",
        "workspace": tmp_path / "runs",
        "wall_clock_sec": 20,
    }
    prepared = workflow.prepare_oragentbench_run(**kwargs)
    import yaml

    config = yaml.safe_load(
        (prepared.run_root / "plan" / "jobs" / f"{prepared.job_name}.yaml").read_text()
    )
    config.pop("pre_build")
    config["agents"][0]["name"] = None
    config["agents"][0]["import_path"] = (
        "ORAgentBench.harbor_agents.prebuilt_agents:"
        + ("PrebuiltCodex" if agent == "codex" else "PrebuiltClaudeCode")
    )
    if agent == "claude-code":
        config["agents"][0]["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"

    dynamic_root = Path(tempfile.mkdtemp(prefix="oragentbench-skills-"))
    dynamic_dataset = dynamic_root / "harbor_tasks"
    dynamic_task = dynamic_dataset / "single_task"
    shutil.copytree(prepared.source / "harbor_tasks" / "single_task", dynamic_task)
    for path in [dynamic_task, *dynamic_task.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    task_toml = dynamic_task / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            "[environment]\n", '[environment]\nskills_dir = "/skills"\n'
        ),
        encoding="utf-8",
    )
    shutil.copytree(
        prepared.source / "skills" / "fixture-skill",
        dynamic_task / "environment" / "skills" / "fixture-skill",
    )
    for path in (dynamic_task / "environment" / "skills").rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    config["datasets"][0]["path"] = str(dynamic_dataset)
    return kwargs, prepared, config, dynamic_dataset


def test_paid_resume_accepts_only_the_pinned_wrapper_task_copy(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    kwargs, prepared, config, _ = _paid_interrupted_config(
        upstream_fixtures, tmp_path
    )
    _write_interrupted_harbor_config(prepared, config)

    resumed = workflow.prepare_oragentbench_run(**kwargs)

    binding = json.loads((resumed.run_root / "resume-binding.json").read_text())
    assert binding["resume_binding_schema_version"] == "2.0"
    assert binding["dataset_content_digest"].startswith("sha256:")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("task-file", "task content"),
        ("task-toml", "task.toml"),
        ("task-toml-comment", "task.toml"),
        ("skill", "injected skills"),
        ("other-temp-root", "dataset path"),
    ],
)
def test_paid_resume_rejects_a_tampered_dynamic_dataset(
    upstream_fixtures: Path, tmp_path: Path, mutation: str, message: str
) -> None:
    kwargs, prepared, config, dataset = _paid_interrupted_config(
        upstream_fixtures, tmp_path
    )
    task = dataset / "single_task"
    if mutation == "task-file":
        (task / "tests" / "test.sh").write_text("tampered\n", encoding="utf-8")
    elif mutation == "task-toml":
        task_toml = task / "task.toml"
        task_toml.write_text(
            task_toml.read_text(encoding="utf-8").replace(
                "Single-step fixture task", "Different task"
            ),
            encoding="utf-8",
        )
    elif mutation == "task-toml-comment":
        task_toml = task / "task.toml"
        task_toml.write_text(
            "# unbound rewrite\n" + task_toml.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    elif mutation == "skill":
        (task / "environment" / "skills" / "fixture-skill" / "SKILL.md").write_text(
            "tampered\n", encoding="utf-8"
        )
    else:
        unrelated = Path(tempfile.mkdtemp(prefix="other-benchmark-")) / "harbor_tasks"
        shutil.copytree(dataset, unrelated)
        config["datasets"][0]["path"] = str(unrelated)
    _write_interrupted_harbor_config(prepared, config)

    with pytest.raises(EvidenceError, match=message):
        workflow.prepare_oragentbench_run(**kwargs)


@pytest.mark.parametrize(
    ("agent", "auth_mode", "env_key"),
    [
        ("codex", "codex-auth-json", "OPENAI_BASE_URL"),
        ("codex", "api-key", "OPENAI_API_KEY"),
        ("claude-code", "api-key", "CLAUDE_CODE_ATTRIBUTION_HEADER"),
    ],
)
def test_paid_resume_binds_agent_environment_values_without_logging_them(
    upstream_fixtures: Path,
    tmp_path: Path,
    agent: str,
    auth_mode: str,
    env_key: str,
) -> None:
    kwargs, prepared, config, _ = _paid_interrupted_config(
        upstream_fixtures,
        tmp_path,
        agent=agent,
        auth_mode=auth_mode,
    )
    config["agents"][0]["env"][env_key] = "LEAK_SENTINEL_CHANGED_VALUE"
    _write_interrupted_harbor_config(prepared, config)

    with pytest.raises(EvidenceError, match="agent") as exc_info:
        workflow.prepare_oragentbench_run(**kwargs)
    assert "LEAK_SENTINEL" not in str(exc_info.value)


def test_resume_binding_detects_dataset_change_after_initial_validation(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    kwargs, prepared, config, dataset = _paid_interrupted_config(
        upstream_fixtures, tmp_path
    )
    _write_interrupted_harbor_config(prepared, config)
    resumed = workflow.prepare_oragentbench_run(**kwargs)
    (dataset / "single_task" / "tests" / "test.sh").write_text(
        "changed after bind\n", encoding="utf-8"
    )

    with pytest.raises(EvidenceError, match="task content"):
        workflow._bind_resume_config(
            run_root=resumed.run_root,
            source=resumed.source,
            job_name=resumed.job_name,
            agent=resumed.agent,
            model=resumed.model,
            scaffold_version=resumed.scaffold_version,
            task=resumed.task,
        )


@pytest.mark.parametrize("phase", ["prepared", "completed", "failed"])
def test_nonrunning_phases_reject_every_unlisted_file(
    upstream_fixtures: Path, tmp_path: Path, phase: str
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    kwargs = dict(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    prepared = workflow.prepare_oragentbench_run(**kwargs)
    manifest_path = prepared.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = phase
    workflow._atomic_json(manifest_path, manifest)
    workflow._write_integrity(prepared.run_root)
    extra = prepared.run_root / "logs" / "unexpected.log"
    extra.parent.mkdir()
    extra.write_text("not covered\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="absent from integrity manifest"):
        workflow.prepare_oragentbench_run(**kwargs)


def test_execute_rechecks_integrity_after_prepare_before_launch(
    upstream_fixtures: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    plan = json.loads((prepared.run_root / "plan" / "plan.json").read_text())
    job_config = prepared.run_root / "plan" / plan["jobs"][0]["config_path"]
    job_config.write_text(job_config.read_text() + "# untrusted mutation\n")

    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        execution,
        "_probe_harbor_version",
        lambda *args, **kwargs: (True, "fixture Harbor is compatible"),
    )
    monkeypatch.setattr(
        execution,
        "_probe_uv_cli",
        lambda *args, **kwargs: (True, "fixture uv is compatible"),
    )
    monkeypatch.setattr(
        execution,
        "_probe_docker_daemon",
        lambda *args, **kwargs: (True, "fixture Docker is reachable"),
    )
    monkeypatch.setattr(
        workflow,
        "_run_process_group",
        lambda *args, **kwargs: pytest.fail("upstream launched before integrity verification"),
    )

    with pytest.raises(EvidenceError, match="integrity"):
        workflow.execute_prepared_run(prepared)


@pytest.mark.parametrize(
    "relative",
    [
        Path("harbor_tasks/single_task/task.toml"),
        Path("metrics/per_dimension_reward.py"),
    ],
    ids=["task-contract", "metric"],
)
def test_execute_uses_the_frozen_snapshot_when_the_operator_checkout_drifts(
    upstream_fixtures: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: Path,
) -> None:
    source = _oab_source(upstream_fixtures, tmp_path)
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    assert prepared.dataset_digest
    manifest_before = json.loads((prepared.run_root / "manifest.json").read_text())
    assert manifest_before["dataset_digest"] == prepared.dataset_digest
    assert manifest_before["source_commit"] == prepared.source_commit
    changed = source / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")

    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        execution,
        "_probe_harbor_version",
        lambda *args, **kwargs: (True, "fixture Harbor is compatible"),
    )
    monkeypatch.setattr(
        execution,
        "_probe_uv_cli",
        lambda *args, **kwargs: (True, "fixture uv is compatible"),
    )
    monkeypatch.setattr(
        execution,
        "_probe_docker_daemon",
        lambda *args, **kwargs: (True, "fixture Docker is reachable"),
    )
    launched: dict[str, object] = {}

    def capture_launch(argv, **kwargs):
        launched["argv"] = argv
        launched["cwd"] = kwargs["cwd"]
        return 9

    monkeypatch.setattr(workflow, "_run_process_group", capture_launch)

    with pytest.raises(Exception, match="exited with code 9"):
        workflow.execute_prepared_run(prepared)

    assert prepared.source != source
    assert str(prepared.source) in " ".join(launched["argv"])
    assert launched["cwd"] == str(prepared.source.parent)
    assert "# drift" not in (prepared.source / relative).read_text(encoding="utf-8")
    manifest = json.loads((prepared.run_root / "manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert "runner_pid" not in manifest


def test_malformed_harbor_json_error_does_not_echo_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"api_key":"LEAK_SENTINEL"', encoding="utf-8")

    with pytest.raises(EvidenceError) as exc_info:
        harbor_ingest._load_json(path, what="Harbor result")

    assert "LEAK_SENTINEL" not in str(exc_info.value)


def test_report_does_not_claim_every_infra_suspect_trial_stays_in_denominator() -> None:
    trial = Trial(
        run_id="0123456789abcdef",
        task_name="task",
        agent_id="oracle",
        scaffold="oracle",
        seed=1,
        attempt=1,
        attribution="unknown",
        counts_toward_capability=False,
        infra_suspect=True,
        exclusion_basis="ambiguous_reward_with_exception",
        trace_status="complete",
        scores={"feasibility": 1.0, "quality": 1.0},
    )
    rollout = NormalizedRollout(
        campaign_id="report-semantics",
        integration="oragentbench",
        evidence_intent=EvidenceLabel.EXPLORATORY,
        site={"name": "local-docker", "perf_isolated": False, "load_source": "none"},
        scoring_reward_keys=("feasibility", "quality"),
        strict_pass_rule=StrictPassRule(
            description="strict fixture pass",
            feasibility_key="feasibility",
            quality_key="quality",
            quality_threshold=1.0,
        ),
        trials=(trial,),
    )

    report = render_mod.build_report(rollout)

    assert "infra_suspect` stay in the capability denominator" not in report.markdown
    no_load = next(metric for metric in report.metrics if metric.name == "no_load_sampling_share")
    assert "infra_suspect is always false" not in no_load.formula
