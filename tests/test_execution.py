"""Upstream command construction, validation and receipt sanitization.

Every test here runs without a model key, a solver licence, Docker or a network
call. That is the point of keeping command construction a pure function: the
exact argv that will run on a self-hosted runner is asserted here, on a laptop,
for free.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from orbenchlab import execution
from orbenchlab.campaign import compile as compile_mod
from orbenchlab.campaign import spec as spec_mod
from orbenchlab.core.errors import PreconditionError, SpecError


# --------------------------------------------------------------------------- #
# fixtures: miniature checkouts with the shape the builders read
# --------------------------------------------------------------------------- #


@pytest.fixture
def oab_source(tmp_path: Path, upstream_fixtures: Path) -> Path:
    """A checkout named exactly as upstream's path resolver requires."""
    import shutil

    source = tmp_path / execution.ORAGENTBENCH_CHECKOUT_DIRNAME
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    wrapper = source / execution.ORAGENTBENCH_PREBUILD_WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# fixture stand-in for the upstream Harbor wrapper\n", encoding="utf-8")
    return source


@pytest.fixture
def fo_source(tmp_path: Path, upstream_fixtures: Path) -> Path:
    import shutil

    source = tmp_path / "frontieror"
    shutil.copytree(upstream_fixtures / "frontieror_min", source)
    (source / "paper_meta_info.json").write_text(
        json.dumps([{"paper_id": "bierwirth2017"}, {"paper_id": "liao2020"}]), encoding="utf-8"
    )
    return source


# --------------------------------------------------------------------------- #
# ORAgentBench: exact upstream commands
# --------------------------------------------------------------------------- #


def test_controls_command_delegates_to_upstream_runner_and_builds_on_fresh_host(oab_source, tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    command = execution.oragentbench_controls_command(source=oab_source, job_config=config)
    assert command.argv == (
        "bash",
        str(oab_source / execution.ORAGENTBENCH_ORACLE_RUNNER),
        "--config",
        str(config.resolve()),
    )
    assert "--skip-build" not in command.argv
    # Upstream runs Harbor from the checkout's *parent*, because the dataset
    # path in every job config is `ORAgentBench/harbor_tasks`.
    assert command.cwd == str(oab_source.parent)
    assert command.makes_model_calls is False
    assert "run_oracle_all.sh" in command.provenance


def test_controls_resume_restores_expected_base_then_delegates_to_harbor_job_resume(
    oab_source, tmp_path
):
    job_dir = tmp_path / "jobs" / "oracle-single-task"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text("{}\n", encoding="utf-8")
    expected_image_id = "sha256:" + "1" * 64

    command = execution.oragentbench_controls_resume_command(
        source=oab_source,
        job_dir=job_dir,
        expected_image_id=expected_image_id,
    )

    assert command.argv == (
        "bash",
        "-c",
        (
            'set -euo pipefail\n'
            'if [ -z "$3" ]; then\n'
            '  bash "$1" "$2"\n'
            'elif docker image inspect "$3" >/dev/null 2>&1; then\n'
            '  docker image tag "$3" "$2"\n'
            'else\n'
            '  bash "$1" "$2"\n'
            'fi\n'
            'actual_id="$(docker image inspect --format \'{{.Id}}\' "$2")"\n'
            'if ! printf \'%s\\n\' "$actual_id" | grep -Eq \'^sha256:[0-9a-f]{64}$\'; then\n'
            '  echo "fixed base image has no valid immutable ID" >&2\n'
            '  exit 1\n'
            'fi\n'
            'if [ -n "$3" ] && [ "$actual_id" != "$3" ]; then\n'
            '  echo "fixed base image identity mismatch" >&2\n'
            '  exit 1\n'
            'fi\n'
            'exec harbor job resume --job-path "$4"'
        ),
        "orbench-control-resume",
        str(oab_source / "scripts" / "build_base_image.sh"),
        "oragentbench-base:py311-scip",
        expected_image_id,
        str(job_dir.resolve()),
    )
    assert command.cwd == str(oab_source.parent)
    assert command.makes_model_calls is False
    assert "harbor job resume --job-path" in command.provenance


def test_controls_resume_without_prior_trials_builds_before_harbor(
    oab_source, tmp_path
):
    job_dir = tmp_path / "jobs" / "oracle-single-task"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text("{}\n", encoding="utf-8")

    command = execution.oragentbench_controls_resume_command(
        source=oab_source,
        job_dir=job_dir,
        expected_image_id=None,
    )

    assert command.argv[-2] == ""
    assert 'if [ -z "$3" ]; then\n  bash "$1" "$2"' in command.argv[2]
    assert "harbor job resume --job-path" in command.argv[2]


def test_controls_resume_image_mismatch_never_starts_harbor(
    oab_source, tmp_path
):
    job_dir = tmp_path / "jobs" / "oracle-single-task"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text("{}\n", encoding="utf-8")
    expected_image_id = "sha256:" + "1" * 64
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    harbor_marker = tmp_path / "harbor-started"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = \"image inspect\" ] && [ \"$3\" = \"--format\" ]; then\n"
        "  echo 'sha256:" + "2" * 64 + "'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    harbor = fake_bin / "harbor"
    harbor.write_text(
        f"#!/bin/sh\nprintf started > {harbor_marker}\nexit 0\n",
        encoding="utf-8",
    )
    harbor.chmod(0o755)
    command = execution.oragentbench_controls_resume_command(
        source=oab_source,
        job_dir=job_dir,
        expected_image_id=expected_image_id,
    )
    environ = dict(os.environ)
    environ["PATH"] = str(fake_bin) + os.pathsep + environ["PATH"]

    result = subprocess.run(
        command.argv,
        cwd=command.cwd,
        env=environ,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "fixed base image identity mismatch" in result.stderr
    assert not harbor_marker.exists()


def test_agent_command_is_the_upstream_prebuild_wrapper(oab_source, tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    command = execution.oragentbench_agent_command(
        source=oab_source, job_config=config, python="/usr/bin/python3"
    )
    assert command.argv == (
        "/usr/bin/python3",
        str(oab_source / execution.ORAGENTBENCH_PREBUILD_WRAPPER),
        "-c",
        str(config.resolve()),
    )
    assert command.cwd == str(oab_source.parent)
    assert command.env_overrides == {"PYTHONPATH": str(oab_source.parent)}
    assert command.makes_model_calls is True
    assert "run_claude_code.sh" in command.provenance


def test_agent_dry_run_costs_nothing(oab_source, tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    command = execution.oragentbench_agent_command(
        source=oab_source, job_config=config, dry_run=True
    )
    assert command.argv[-1] == "--dry-run"
    # Upstream's wrapper stops after printing the transformed config and the
    # harbor command, so nothing is spent.
    assert command.makes_model_calls is False


def test_agent_command_carries_secret_names_not_values(oab_source, tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    command = execution.oragentbench_agent_command(
        source=oab_source, job_config=config, required_env=("MODEL_API_KEY", "MODEL_BASE_URL")
    )
    assert command.required_env == ("MODEL_API_KEY", "MODEL_BASE_URL")
    assert not any("=" in token for token in command.argv)


def test_a_wrongly_named_checkout_is_refused(tmp_path, upstream_fixtures):
    """The name is load-bearing: upstream resolves the dataset path beside it."""
    import shutil

    wrong = tmp_path / "oragentbench-lowercase"
    shutil.copytree(upstream_fixtures / "oragentbench_min", wrong)
    with pytest.raises(PreconditionError) as excinfo:
        execution.validate_oragentbench_source(wrong)
    assert "ORAgentBench" in str(excinfo.value)
    assert "parent directory" in str(excinfo.value)


def test_a_checkout_without_the_wrapper_is_refused(oab_source, tmp_path):
    (oab_source / execution.ORAGENTBENCH_PREBUILD_WRAPPER).unlink()
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    with pytest.raises(PreconditionError):
        execution.oragentbench_agent_command(source=oab_source, job_config=config)


# --------------------------------------------------------------------------- #
# ORAgentBench: task validation
# --------------------------------------------------------------------------- #


def test_a_validated_task_resolves(oab_source):
    assert execution.validate_oragentbench_task(oab_source, "single_task") == "single_task"


def test_an_unknown_task_is_refused_with_a_hint(oab_source):
    with pytest.raises(SpecError) as excinfo:
        execution.validate_oragentbench_task(oab_source, "no_such_task")
    message = str(excinfo.value)
    assert "not a validated task" in message
    assert "single_task" in message


@pytest.mark.parametrize(
    "candidate",
    ["../etc", "a/b", "/absolute", "..", "with space", ""],
)
def test_path_traversal_and_junk_task_names_are_refused(oab_source, candidate):
    with pytest.raises(SpecError):
        execution.validate_oragentbench_task(oab_source, candidate)


def test_a_directory_without_a_task_toml_is_not_a_task(oab_source):
    (oab_source / "harbor_tasks" / "not_a_task").mkdir()
    with pytest.raises(SpecError):
        execution.validate_oragentbench_task(oab_source, "not_a_task")


# --------------------------------------------------------------------------- #
# ORAgentBench: campaign spec and the compiled job config
# --------------------------------------------------------------------------- #


def _agent_spec(oab_source, sites_dir, **overrides):
    raw = execution.oragentbench_agent_campaign_spec(
        slug="oab-agent-smoke",
        date="2026-08-22",
        dataset_digest="sha256:" + "a" * 64,
        task_name="single_task",
        scaffold=overrides.pop("scaffold", "claude-code"),
        scaffold_version=overrides.pop("scaffold_version", "fixture-cli-1.2.3"),
        model=overrides.pop("model", "deepseek-v4-pro"),
        model_base_url=overrides.pop(
            "model_base_url", "https://router.example.test/v1"
        ),
        **overrides,
    )
    return spec_mod.validate(raw, sites_dir=sites_dir)


def test_agent_campaign_spec_validates(oab_source, sites_dir):
    spec = _agent_spec(oab_source, sites_dir)
    assert spec.makes_model_calls is True
    assert spec.n_planned_runs == 1
    assert spec.agents[0].secret_names == ("MODEL_API_KEY", "MODEL_BASE_URL")


def test_compiled_job_config_matches_the_upstream_agent_shape(oab_source, sites_dir):
    """The fields upstream's own configs use, with no secret values anywhere."""
    compiled = compile_mod.compile_campaign(_agent_spec(oab_source, sites_dir))
    config = next(iter(compiled.job_configs.values()))
    agent = config["agents"][0]

    assert agent["name"] == "claude-code"
    assert agent["model_name"] == "deepseek-v4-pro"
    assert agent["override_setup_timeout_sec"] == 420
    assert agent["env"] == {
        "ANTHROPIC_AUTH_TOKEN": "${MODEL_API_KEY}",
        "ANTHROPIC_API_KEY": "${MODEL_API_KEY}",
        "ANTHROPIC_BASE_URL": "${MODEL_BASE_URL}",
        # A model name is a literal, not a credential, and upstream inlines it.
        "ANTHROPIC_MODEL": "deepseek-v4-pro",
    }
    assert agent["kwargs"]["disallowed_tools"] == "WebSearch,WebFetch"
    assert "provider_route_digest" not in json.dumps(config)
    assert config["datasets"][0]["path"] == execution.ORAGENTBENCH_DATASET_PATH
    assert config["n_attempts"] == 1
    # Consumed by upstream's wrapper to swap in its prebuilt agent class.
    assert config["pre_build"]["enabled"] is True
    assert config["pre_build"]["agent"] == "claude-code"
    assert config["pre_build"]["rebuild_base"] is True
    assert config["pre_build"]["image_tag"].startswith("orbench-oab-claude-code:")
    assert agent["kwargs"]["version"] == "fixture-cli-1.2.3"


def test_codex_profile_uses_the_openai_variables(oab_source, sites_dir):
    compiled = compile_mod.compile_campaign(
        _agent_spec(oab_source, sites_dir, scaffold="codex", model="gpt-5.3-codex")
    )
    agent = next(iter(compiled.job_configs.values()))["agents"][0]
    assert agent["env"] == {
        "OPENAI_API_KEY": "${MODEL_API_KEY}",
        "OPENAI_BASE_URL": "${MODEL_BASE_URL}",
    }
    assert agent["kwargs"]["web_search"] == "disabled"


def test_codex_auth_json_profile_contains_no_api_key_placeholder(oab_source, sites_dir):
    compiled = compile_mod.compile_campaign(
        _agent_spec(
            oab_source,
            sites_dir,
            scaffold="codex",
            model="gpt-5.5",
            auth_mode="codex-auth-json",
            model_base_url="https://router.example.test/v1",
        )
    )
    agent = next(iter(compiled.job_configs.values()))["agents"][0]

    assert agent["env"] == {
        "CODEX_FORCE_AUTH_JSON": "true",
        "OPENAI_BASE_URL": "https://router.example.test/v1",
    }
    assert "MODEL_API_KEY" not in json.dumps(agent)
    assert "OPENAI_API_KEY" not in agent["env"]


@pytest.mark.parametrize(
    "url",
    [
        "http://router.example.test/v1",
        "https://user:secret@router.example.test/v1",
        "https://router.example.test/v1?token=secret",
        "https://router.example.test/v1#fragment",
        "not-a-url",
    ],
)
def test_codex_auth_json_rejects_unsafe_provider_urls(oab_source, url):
    with pytest.raises(SpecError):
        execution.oragentbench_agent_campaign_spec(
            slug="unsafe-url",
            date="2026-08-24",
            dataset_digest="sha256:" + "a" * 64,
            task_name="single_task",
            scaffold="codex",
            scaffold_version="fixture-cli-1.2.3",
            model="gpt-5.5",
            auth_mode="codex-auth-json",
            model_base_url=url,
        )


def test_api_key_campaign_refuses_to_persist_an_unsafe_provider_url(oab_source):
    with pytest.raises(SpecError):
        execution.oragentbench_agent_campaign_spec(
            slug="unsafe-api-route",
            date="2026-08-24",
            dataset_digest="sha256:" + "a" * 64,
            task_name="single_task",
            scaffold="codex",
            scaffold_version="fixture-cli-1.2.3",
            model="gpt-5.5",
            auth_mode="api-key",
            model_base_url="https://user:credential@router.example.test/v1",
        )


def test_control_campaign_config_gains_no_pre_build_block(campaigns_dir, sites_dir):
    """Controls need no agent CLI, so their config stays as it was."""
    raw = spec_mod.load_spec(campaigns_dir / "oragentbench-controls.yaml")
    compiled = compile_mod.compile_campaign(spec_mod.validate(raw, sites_dir=sites_dir))
    for config in compiled.job_configs.values():
        assert "pre_build" not in config


def test_an_unknown_scaffold_has_no_profile(oab_source, sites_dir):
    with pytest.raises(SpecError) as excinfo:
        execution.oragentbench_agent_campaign_spec(
            slug="s",
            date="2026-08-22",
            dataset_digest="sha256:" + "a" * 64,
            task_name="single_task",
            scaffold="invented-agent",
            scaffold_version="fixture-cli-1.2.3",
            model="m",
        )
    assert "no recorded upstream agent profile" in str(excinfo.value)


def test_a_floating_model_is_refused_before_a_command_exists(oab_source):
    with pytest.raises(SpecError):
        execution.oragentbench_agent_campaign_spec(
            slug="s",
            date="2026-08-22",
            dataset_digest="sha256:" + "a" * 64,
            task_name="single_task",
            scaffold="claude-code",
            scaffold_version="fixture-cli-1.2.3",
            model="something-latest",
        )


def test_output_location_is_not_part_of_campaign_identity(oab_source, sites_dir):
    """What lets the output root be named after the campaign id."""
    base = execution.oragentbench_agent_campaign_spec(
        slug="relocation-check",
        date="2026-08-22",
        dataset_digest="sha256:" + "a" * 64,
        task_name="single_task",
        scaffold="claude-code",
        scaffold_version="fixture-cli-1.2.3",
        model="deepseek-v4-pro",
        model_base_url="https://router.example.test/v1",
    )
    first = compile_mod.compile_campaign(spec_mod.validate(base, sites_dir=sites_dir))
    moved = dict(base, harbor=dict(base["harbor"], jobs_dir="/some/other/place"))
    second = compile_mod.compile_campaign(spec_mod.validate(moved, sites_dir=sites_dir))
    assert first.campaign_id == second.campaign_id
    assert [r.run_id for r in first.runs] == [r.run_id for r in second.runs]


# --------------------------------------------------------------------------- #
# ORAgentBench: preconditions
# --------------------------------------------------------------------------- #


def _probe_result(
    argv: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        argv,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_preconditions_probe_the_docker_daemon_not_just_the_executable(
    oab_source, monkeypatch
):
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[-1] == "--version":
            return _probe_result(list(argv), stdout="harbor, version 0.16.2\n")
        return _probe_result(list(argv), returncode=1, stderr="daemon unavailable")

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="oracle",
        model="",
        environ={},
        command_runner=runner,
    )

    assert not report.ok
    assert calls == [
        ["/usr/bin/harbor", "--version"],
        ["/usr/bin/uv", "--version"],
        ["/usr/bin/docker", "info"],
    ]
    assert any("daemon" in item and "docker info" in item for item in report.missing)
    assert "daemon unavailable" not in json.dumps(report.to_dict())


@pytest.mark.parametrize(
    ("version_output", "expected_ok"),
    [
        ("0.16.1", True),
        ("harbor, version 0.16.0", True),
        ("harbor version v0.16.9", True),
        ("harbor, version 0.15.7", False),
        ("harbor development build", False),
        ("harbor development build with Python 3.12", False),
    ],
)
def test_preconditions_require_a_parseable_supported_harbor_version(
    oab_source, monkeypatch, version_output, expected_ok
):
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    def runner(argv, **kwargs):
        if argv[-1] == "--version":
            return _probe_result(list(argv), stdout=version_output)
        return _probe_result(list(argv))

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="oracle",
        model="",
        environ={},
        command_runner=runner,
    )

    assert report.ok is expected_ok
    relevant = report.satisfied if expected_ok else report.missing
    assert any("Harbor" in item and "0.16" in item for item in relevant)


def test_preconditions_require_uv_for_the_compiled_uv_script_metric(
    oab_source, monkeypatch
):
    monkeypatch.setattr(
        execution.shutil,
        "which",
        lambda name: None if name == "uv" else f"/usr/bin/{name}",
    )

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="oracle",
        model="",
        environ={},
        require_docker=False,
        command_runner=lambda argv, **kwargs: _probe_result(
            list(argv), stdout="harbor, version 0.16.1\n"
        ),
    )

    assert not report.ok
    assert any("uv" in item and "metric" in item for item in report.missing)


def test_preconditions_fail_closed_when_a_runner_probe_times_out(oab_source, monkeypatch):
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    def runner(argv, **kwargs):
        if argv[-1] == "--version":
            return _probe_result(list(argv), stdout="harbor, version 0.16.1")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="oracle",
        model="",
        environ={},
        command_runner=runner,
    )

    assert not report.ok
    assert any("timed out" in item and "docker info" in item for item in report.missing)


def test_preconditions_pass_with_everything_present(oab_source):
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="claude-code",
        model="deepseek-v4-pro",
        environ={
            "MODEL_API_KEY": "x",
            "MODEL_BASE_URL": "https://router.example.test/v1",
        },
        model_base_url="https://router.example.test/v1",
        require_docker=False,
        require_harbor=False,
    )
    assert report.ok, report.missing


@pytest.mark.parametrize(
    "url",
    [
        "http://router.example.test/v1",
        "https://user:secret@router.example.test/v1",
        "https://router.example.test/v1?token=secret",
        "https://router.example.test/v1#fragment",
        "not-a-url",
    ],
)
def test_api_key_preconditions_reject_an_unsafe_provider_url_without_echoing_it(
    oab_source, url
):
    api_key = "SENTINEL_API_VALUE_DO_NOT_LOG"
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="codex",
        model="gpt-5.5",
        environ={"MODEL_API_KEY": api_key, "MODEL_BASE_URL": url},
        model_base_url="https://pinned.example.test/v1",
        require_docker=False,
        require_harbor=False,
    )

    assert not report.ok
    rendered = json.dumps(report.to_dict())
    assert url not in rendered
    assert api_key not in rendered


def test_preconditions_fail_closed_without_the_secret(oab_source):
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="claude-code",
        model="deepseek-v4-pro",
        environ={},
        model_base_url="https://router.example.test/v1",
        require_docker=False,
        require_harbor=False,
    )
    assert not report.ok
    assert any("MODEL_API_KEY" in item for item in report.missing)
    with pytest.raises(PreconditionError):
        report.raise_if_unmet("test")


def test_an_upstream_dry_run_needs_no_credentials(oab_source):
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="claude-code",
        model="deepseek-v4-pro",
        environ={},
        require_docker=False,
        require_harbor=False,
        require_secrets=False,
        model_base_url="https://router.example.test/v1",
    )
    assert report.ok, report.missing


def test_controls_need_no_credentials_at_all(oab_source):
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="oracle",
        model="",
        environ={},
        require_docker=False,
        require_harbor=False,
    )
    assert report.ok, report.missing


def test_preconditions_report_a_missing_task(oab_source):
    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="nope",
        scaffold="oracle",
        model="",
        environ={},
        require_docker=False,
        require_harbor=False,
    )
    assert not report.ok


def test_codex_auth_json_preconditions_require_a_private_auth_file(oab_source, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode":"chatgpt"}\n')
    auth.chmod(0o600)
    environ = {"CODEX_AUTH_JSON_PATH": str(auth)}

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="codex",
        model="gpt-5.5",
        auth_mode="codex-auth-json",
        model_base_url="https://router.example.test/v1",
        environ=environ,
        require_docker=False,
        require_harbor=False,
    )
    assert report.ok, report.missing

    auth.chmod(0o644)
    exposed = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="codex",
        model="gpt-5.5",
        auth_mode="codex-auth-json",
        model_base_url="https://router.example.test/v1",
        environ=environ,
        require_docker=False,
        require_harbor=False,
    )
    assert not exposed.ok
    assert any("private" in item for item in exposed.missing)


@pytest.mark.parametrize("payload", ["not json", "[]", "{}"])
def test_codex_auth_json_preconditions_parse_a_nonempty_object(oab_source, tmp_path, payload):
    auth = tmp_path / "auth.json"
    auth.write_text(payload, encoding="utf-8")
    auth.chmod(0o600)

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="single_task",
        scaffold="codex",
        model="gpt-5.5",
        auth_mode="codex-auth-json",
        model_base_url="https://router.example.test/v1",
        environ={"CODEX_AUTH_JSON_PATH": str(auth)},
        require_docker=False,
        require_harbor=False,
    )

    assert not report.ok
    assert any("valid JSON" in item and "object" in item for item in report.missing)
    assert payload not in json.dumps(report.to_dict())


def test_codex_auth_json_is_blocked_for_an_internet_enabled_task(oab_source, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode":"chatgpt"}\n')
    auth.chmod(0o600)

    report = execution.oragentbench_preconditions(
        source=oab_source,
        task_name="multi_task",
        scaffold="codex",
        model="gpt-5.5",
        auth_mode="codex-auth-json",
        model_base_url="https://router.example.test/v1",
        environ={"CODEX_AUTH_JSON_PATH": str(auth)},
        require_docker=False,
        require_harbor=False,
    )

    assert not report.ok
    assert any("long-lived" in item and "internet" in item for item in report.missing)


def test_codex_base_url_can_be_discovered_without_reading_a_secret(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model_provider = "gateway"\n'
        '[model_providers.gateway]\n'
        'base_url = "https://router.example.test/v1"\n',
        encoding="utf-8",
    )

    assert execution.discover_codex_base_url(
        {"CODEX_CONFIG_PATH": str(config)}
    ) == "https://router.example.test/v1"


# --------------------------------------------------------------------------- #
# FrontierOR: the official command
# --------------------------------------------------------------------------- #


def _fo_command(fo_source, **overrides):
    kwargs = dict(
        source=fo_source,
        paper_id="bierwirth2017",
        primary_model="openai/gpt-5.4",
        stage1_instances=["tiny"],
        dev_instances=["large_1"],
        test_instances=["large_2"],
        run_id="orbench-agent-smoke",
        python="/usr/bin/python3",
    )
    kwargs.update(overrides)
    return execution.frontieror_agent_command(**kwargs)


def test_frontieror_command_matches_the_documented_invocation(fo_source):
    command = _fo_command(fo_source)
    assert command.argv == (
        "/usr/bin/python3",
        "-m",
        "frontieror.infra",
        "agent",
        "--paper-id",
        "bierwirth2017",
        "--primary-model",
        "openai/gpt-5.4",
        "--stage1-instances",
        "tiny",
        "--dev-set",
        "large_1",
        "--test-set",
        "large_2",
        "--coral-agent-count",
        "1",
        "--coral-attempts",
        "10",
        "--coral-max-steps",
        "10",
        "--coral-max-seconds",
        "auto",
        "--cpus",
        "1",
        "--memory",
        "128G",
        "--run-id",
        "orbench-agent-smoke",
    )
    assert command.cwd == str(fo_source)
    assert command.required_env == ("OPENROUTER_API_KEY", "GRB_LICENSE_FILE")


def test_frontieror_command_supplies_no_trusted_profile_flag(fo_source):
    """Upstream appends the profile itself; supplying it could only weaken it."""
    argv = set(_fo_command(fo_source).argv)
    assert not (argv & execution.FRONTIEROR_FORBIDDEN_AGENT_FLAGS)


@pytest.mark.parametrize(
    "flag",
    ["--framework", "--exec-mode", "--stage2-scorer", "--coral-model-access", "--anti-hack", "--coral-gateway"],
)
def test_trusted_profile_flags_are_refused(fo_source, flag):
    with pytest.raises(SpecError) as excinfo:
        _fo_command(fo_source, extra=[flag, "whatever"])
    assert "trusted-agent profile" in str(excinfo.value)


def test_undocumented_flags_are_refused(fo_source):
    with pytest.raises(SpecError) as excinfo:
        _fo_command(fo_source, extra=["--totally-made-up", "1"])
    assert "documented" in str(excinfo.value)


def test_documented_flags_pass_through(fo_source):
    command = _fo_command(fo_source, extra=["--wls-egress", "off"])
    assert command.argv[-2:] == ("--wls-egress", "off")


def test_zero_cost_contract_command(fo_source):
    command = execution.frontieror_contract_command(source=fo_source, python="/usr/bin/python3")
    assert command.argv == ("/usr/bin/python3", "-m", "frontieror.infra", "contract")
    assert command.makes_model_calls is False


def test_security_check_command(fo_source):
    command = execution.frontieror_security_check_command(
        source=fo_source, python="/usr/bin/python3"
    )
    assert command.argv[-2:] == ("--candidate-image", "frontieror-candidate:1")
    assert command.makes_model_calls is False


# --------------------------------------------------------------------------- #
# FrontierOR: input validation
# --------------------------------------------------------------------------- #


def test_an_unknown_paper_id_is_refused(fo_source):
    with pytest.raises(SpecError) as excinfo:
        _fo_command(fo_source, paper_id="notapaper2020")
    assert "paper_meta_info.json" in str(excinfo.value)


@pytest.mark.parametrize("name", ["large", "large_0", "small_1", "TINY", "../tiny", "large_x"])
def test_invalid_instance_names_are_refused(fo_source, name):
    with pytest.raises(SpecError):
        _fo_command(fo_source, dev_instances=[name])


@pytest.mark.parametrize("name", ["tiny", "large_1", "large_12"])
def test_valid_instance_names_are_accepted(fo_source, name):
    _fo_command(fo_source, dev_instances=[name], test_instances=["large_99"])


def test_a_leaky_dev_test_split_is_refused(fo_source):
    with pytest.raises(SpecError) as excinfo:
        _fo_command(fo_source, dev_instances=["large_2"], test_instances=["large_2"])
    assert "overlaps" in str(excinfo.value)


def test_stage1_leaking_into_the_test_set_is_refused(fo_source):
    with pytest.raises(SpecError):
        _fo_command(fo_source, stage1_instances=["tiny"], test_instances=["tiny"])


@pytest.mark.parametrize("route", ["gpt-5.4", "openai/gpt-latest", "openai/*", ""])
def test_bad_model_routes_are_refused(fo_source, route):
    with pytest.raises(SpecError):
        _fo_command(fo_source, primary_model=route)


def test_a_bad_run_id_is_refused(fo_source):
    with pytest.raises(SpecError):
        _fo_command(fo_source, run_id="../escape")


def test_a_checkout_without_the_official_cli_is_refused(tmp_path):
    with pytest.raises(PreconditionError) as excinfo:
        execution.validate_frontieror_source(tmp_path)
    assert "no official" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# FrontierOR: preconditions
# --------------------------------------------------------------------------- #


def test_frontieror_requires_isolation_licence_key_and_images(fo_source, tmp_path):
    licence = tmp_path / "gurobi.lic"
    licence.write_text("placeholder\n", encoding="utf-8")
    report = execution.frontieror_preconditions(
        source=fo_source,
        environ={
            "OPENROUTER_API_KEY": "x",
            "GRB_LICENSE_FILE": str(licence),
            "ORBENCH_PERF_ISOLATED": "true",
        },
        require_docker=False,
        available_images=execution.FRONTIEROR_REQUIRED_IMAGES,
    )
    assert report.ok, report.missing


def test_frontieror_preconditions_also_require_a_reachable_docker_daemon(
    fo_source, tmp_path, monkeypatch
):
    licence = tmp_path / "gurobi.lic"
    licence.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    report = execution.frontieror_preconditions(
        source=fo_source,
        environ={
            "OPENROUTER_API_KEY": "x",
            "GRB_LICENSE_FILE": str(licence),
            "ORBENCH_PERF_ISOLATED": "true",
        },
        available_images=execution.FRONTIEROR_REQUIRED_IMAGES,
        command_runner=lambda argv, **kwargs: _probe_result(list(argv), returncode=1),
    )

    assert not report.ok
    assert any("docker info" in item and "daemon" in item for item in report.missing)


def test_frontieror_refuses_a_shared_host(fo_source, tmp_path):
    licence = tmp_path / "gurobi.lic"
    licence.write_text("placeholder\n", encoding="utf-8")
    report = execution.frontieror_preconditions(
        source=fo_source,
        environ={"OPENROUTER_API_KEY": "x", "GRB_LICENSE_FILE": str(licence)},
        require_docker=False,
        available_images=execution.FRONTIEROR_REQUIRED_IMAGES,
    )
    assert not report.ok
    assert any("ORBENCH_PERF_ISOLATED" in item for item in report.missing)


def test_frontieror_refuses_a_missing_licence_file(fo_source):
    report = execution.frontieror_preconditions(
        source=fo_source,
        environ={
            "OPENROUTER_API_KEY": "x",
            "GRB_LICENSE_FILE": "/nonexistent/gurobi.lic",
            "ORBENCH_PERF_ISOLATED": "true",
        },
        require_docker=False,
        available_images=execution.FRONTIEROR_REQUIRED_IMAGES,
    )
    assert any("Gurobi licence" in item for item in report.missing)


def test_frontieror_refuses_unbuilt_images(fo_source, tmp_path):
    licence = tmp_path / "gurobi.lic"
    licence.write_text("placeholder\n", encoding="utf-8")
    report = execution.frontieror_preconditions(
        source=fo_source,
        environ={
            "OPENROUTER_API_KEY": "x",
            "GRB_LICENSE_FILE": str(licence),
            "ORBENCH_PERF_ISOLATED": "true",
        },
        require_docker=False,
        available_images=["frontieror-candidate:1"],
    )
    assert not report.ok
    assert any("frontieror-coral-agent:0.1" in item for item in report.missing)


# --------------------------------------------------------------------------- #
# receipts
# --------------------------------------------------------------------------- #


def test_secret_named_fields_are_redacted():
    receipt = execution.sanitize_receipt(
        {"MODEL_API_KEY": "sk-real-value", "nested": {"AUTH_TOKEN": "abc"}, "model": "gpt-5.4"}
    )
    assert receipt["MODEL_API_KEY"] == "<redacted>"
    assert receipt["nested"]["AUTH_TOKEN"] == "<redacted>"
    assert receipt["model"] == "gpt-5.4"


def test_presence_markers_survive_redaction():
    """A receipt is for debugging configuration; hiding set/unset defeats it."""
    receipt = execution.sanitize_receipt({"environment": {"MODEL_API_KEY": "<unset>"}})
    assert receipt["environment"]["MODEL_API_KEY"] == "<unset>"


def test_inline_credential_assignments_are_redacted():
    receipt = execution.sanitize_receipt({"note": "ran with OPENROUTER_API_KEY=sk-abc123 set"})
    assert "sk-abc123" not in receipt["note"]
    assert "OPENROUTER_API_KEY=<redacted>" in receipt["note"]


def test_env_presence_never_reveals_a_value():
    presence = execution.env_presence(["A", "B"], environ={"A": "super-secret"})
    assert presence == {"A": "<set>", "B": "<unset>"}


def test_receipt_is_shareable_and_claims_no_bundle(oab_source, tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text("job_name: x\n", encoding="utf-8")
    command = execution.oragentbench_agent_command(source=oab_source, job_config=config)
    receipt = execution.build_receipt(
        integration="oragentbench",
        mode="agent",
        command=command,
        campaign_id="c-20260822-abcdef01",
        preconditions=execution.PreconditionReport(satisfied=["ok"]),
        exit_code=0,
        evidence_label="exploratory",
        output_root=str(tmp_path),
    )
    execution.assert_receipt_is_shareable(receipt)
    assert receipt["raw_bundle_uploaded"] is False
    assert receipt["evidence_label"] == "exploratory"


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("Authorization: Bearer sk-live-secret", "sk-live-secret"),
        ("https://provider.test/v1?api_key=sk-query-secret", "sk-query-secret"),
        ("token=plain-token-value", "plain-token-value"),
    ],
)
def test_receipt_sanitizer_redacts_common_provider_secret_shapes(raw, secret):
    """Provider/CLI errors often contain headers or URL query parameters."""
    cleaned = execution.sanitize_receipt({"provider_error": raw})
    blob = json.dumps(cleaned)
    assert secret not in blob
    assert "<redacted>" in blob


@pytest.mark.parametrize("marker", ["trajectory.json", "test-stdout", "reference_objective"])
def test_a_receipt_naming_raw_evidence_is_refused(marker):
    with pytest.raises(PreconditionError) as excinfo:
        execution.assert_receipt_is_shareable({"notes": [f"see {marker}"]})
    assert "raw evidence" in str(excinfo.value)


def test_a_receipt_claiming_a_bundle_upload_is_refused():
    with pytest.raises(PreconditionError):
        execution.assert_receipt_is_shareable({"raw_bundle_uploaded": True})


# --------------------------------------------------------------------------- #
# the checked script
# --------------------------------------------------------------------------- #


def _script_main():
    """Import the tool by path; it lives outside the installed package."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "tools" / "run_benchmark_smoke.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_dry_run_builds_the_command_without_executing(oab_source, capsys, tmp_path):
    module = _script_main()
    code = module.main(
        [
            "oragentbench",
            "--source", str(oab_source),
            "--task", "single_task",
            "--scaffold", "claude-code",
            "--scaffold-version", "fixture-cli-1.2.3",
            "--model", "deepseek-v4-pro",
            "--date", "2026-08-22",
            "--campaign-slug", "script-check",
            "--dataset-digest", "sha256:" + "a" * 64,
            "--output-root", str(tmp_path / "out"),
            "--allow-missing-tooling",
            "--model-base-url", "https://router.example.test/v1",
        ]
    )
    assert code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["executed"] is False
    assert "--skip-build" not in receipt["upstream_command"]["argv"]
    assert receipt["evidence_label"] == "exploratory"
    assert any("dry run" in note for note in receipt["notes"])


def test_script_output_root_is_named_after_the_campaign(oab_source, capsys, tmp_path):
    module = _script_main()
    module.main(
        [
            "oragentbench",
            "--source", str(oab_source),
            "--task", "single_task",
            "--scaffold", "claude-code",
            "--scaffold-version", "fixture-cli-1.2.3",
            "--model", "deepseek-v4-pro",
            "--date", "2026-08-22",
            "--campaign-slug", "script-check",
            "--dataset-digest", "sha256:" + "a" * 64,
            "--output-root", str(tmp_path / "out"),
            "--allow-missing-tooling",
            "--model-base-url", "https://router.example.test/v1",
        ]
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["output_root"].endswith(receipt["campaign_id"])
    assert (Path(receipt["output_root"]) / "plan" / "plan_ledger.json").is_file()


def test_script_refuses_to_execute_without_the_tooling_it_skipped_checking(oab_source, tmp_path):
    module = _script_main()
    code = module.main(
        [
            "oragentbench",
            "--source", str(oab_source),
            "--task", "single_task",
            "--model", "m1",
            "--date", "2026-08-22",
            "--dataset-digest", "sha256:" + "a" * 64,
            "--output-root", str(tmp_path / "out"),
            "--allow-missing-tooling",
            "--execute",
        ]
    )
    assert code == 5


def test_script_refuses_a_paid_run_without_acknowledgement(oab_source, tmp_path, monkeypatch):
    module = _script_main()
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.setenv("MODEL_BASE_URL", "https://router.example.test/v1")
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")
    code = module.main(
        [
            "oragentbench",
            "--source", str(oab_source),
            "--task", "single_task",
            "--scaffold", "claude-code",
            "--scaffold-version", "fixture-cli-1.2.3",
            "--model", "deepseek-v4-pro",
            "--date", "2026-08-22",
            "--dataset-digest", "sha256:" + "a" * 64,
            "--output-root", str(tmp_path / "out"),
            "--execute",
        ]
    )
    # Exit 5 is PreconditionError: the acknowledgement is missing.
    assert code == 5


def test_script_requires_an_explicit_date(oab_source, tmp_path):
    module = _script_main()
    code = module.main(
        [
            "oragentbench",
            "--source", str(oab_source),
            "--task", "single_task",
            "--scaffold", "claude-code",
            "--model", "deepseek-v4-pro",
            "--dataset-digest", "sha256:" + "a" * 64,
            "--output-root", str(tmp_path / "out"),
            "--allow-missing-tooling",
        ]
    )
    assert code == 1


def test_script_refuses_a_trusted_profile_downgrade(fo_source, tmp_path):
    module = _script_main()
    code = module.main(
        [
            "frontieror",
            "--source", str(fo_source),
            "--paper-id", "bierwirth2017",
            "--primary-model", "openai/gpt-5.4",
            "--extra", "--exec-mode bare",
            "--allow-missing-tooling",
        ]
    )
    assert code == 2
