"""Campaign validation policy and compiler determinism."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from orbenchlab.campaign import compile as compile_mod
from orbenchlab.campaign import spec as spec_mod
from orbenchlab.core import ids as ids_mod
from orbenchlab.core import schema as schema_mod
from orbenchlab.core.errors import SpecError


@pytest.fixture
def controls_raw(campaigns_dir: Path) -> dict:
    return spec_mod.load_spec(campaigns_dir / "oragentbench-controls.yaml")


def _validate(raw: dict, sites_dir: Path) -> spec_mod.CampaignSpec:
    return spec_mod.validate(raw, sites_dir=sites_dir)


def _expect_rejection(raw: dict, sites_dir: Path, needle: str) -> str:
    with pytest.raises(SpecError) as excinfo:
        _validate(raw, sites_dir)
    message = str(excinfo.value)
    assert needle in message, message
    return message


# --------------------------------------------------------------------------- #
# shipped specs
# --------------------------------------------------------------------------- #


def test_shipped_controls_campaign_validates(controls_raw, sites_dir):
    spec = _validate(controls_raw, sites_dir)
    assert spec.integration == "oragentbench"
    assert spec.makes_model_calls is False
    assert spec.n_planned_runs == 18


def test_shipped_frontieror_campaign_is_rejected_on_a_shared_site(campaigns_dir, sites_dir):
    """This spec exists to be rejected; the rejection is the safety property."""
    raw = spec_mod.load_spec(campaigns_dir / "frontieror-contract-check.yaml")
    message = _expect_rejection(raw, sites_dir, "perf_isolated")
    assert "solver licence" in message


def test_every_shipped_campaign_parses_and_names_a_declared_site(campaigns_dir, sites_dir):
    for path in sorted(campaigns_dir.glob("*.yaml")):
        raw = spec_mod.load_spec(path)
        assert (sites_dir / f"{raw['site']}.yaml").is_file(), path


# --------------------------------------------------------------------------- #
# policy rejections
# --------------------------------------------------------------------------- #


def test_performance_scored_integration_needs_a_perf_isolated_site(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["integration"] = "frontieror"
    _expect_rejection(raw, sites_dir, "perf_isolated")


def test_capability_exceptions_may_not_be_retried(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["retry"] = {"max_retries": 2, "include_exceptions": ["AgentTimeoutError"]}
    _expect_rejection(raw, sites_dir, "capability-class exceptions")


def test_retries_require_an_explicit_allowlist(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["retry"] = {"max_retries": 3, "include_exceptions": []}
    _expect_rejection(raw, sites_dir, "explicit retry.include_exceptions allowlist")


def test_infra_exceptions_may_be_retried(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["retry"] = {"max_retries": 2, "include_exceptions": ["DockerBuildError"]}
    assert _validate(raw, sites_dir).retry["max_retries"] == 2


@pytest.mark.parametrize("field", ["include_exceptions", "exclude_exceptions"])
@pytest.mark.parametrize("value", [None, ("AgentTimeoutError",)])
def test_retry_exception_sets_must_be_explicit_lists(
    controls_raw, sites_dir, field, value
):
    raw = copy.deepcopy(controls_raw)
    raw["retry"][field] = value
    _expect_rejection(raw, sites_dir, f"retry.{field} must be a list")


@pytest.mark.parametrize("field", ["include_exceptions", "exclude_exceptions"])
def test_retry_exception_sets_reject_duplicates_and_non_strings(
    controls_raw, sites_dir, field
):
    duplicate = copy.deepcopy(controls_raw)
    duplicate["retry"][field] = ["AgentTimeoutError", "AgentTimeoutError"]
    _expect_rejection(duplicate, sites_dir, f"retry.{field} must not contain duplicates")
    non_string = copy.deepcopy(controls_raw)
    non_string["retry"][field] = ["AgentTimeoutError", 1]
    _expect_rejection(non_string, sites_dir, f"retry.{field} must contain only strings")


def test_attempts_must_be_one(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["attempts"] = 3
    _expect_rejection(raw, sites_dir, "attempts must be 1")


def test_validated_intent_fails_closed(controls_raw, sites_dir):
    """Durability verification is unimplemented, so 'validated' cannot be planned."""
    raw = copy.deepcopy(controls_raw)
    raw["evidence_intent"] = "validated"
    message = _expect_rejection(raw, sites_dir, "durability.replicas >= 2")
    assert "not implemented" in message


def test_validated_intent_still_fails_when_replicas_are_declared(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["evidence_intent"] = "validated"
    raw["durability"] = {"replicas": 2}
    _expect_rejection(raw, sites_dir, "orbench durability verify")


def test_model_calling_agent_needs_a_pinned_model(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [{"id": "a", "scaffold": "claude-code", "env_keys": ["KEY"]}]
    raw["budget"]["max_cost_usd"] = 10
    _expect_rejection(raw, sites_dir, "model must be pinned")


def test_floating_model_alias_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {"id": "a", "scaffold": "claude-code", "model": "some-model-latest", "env_keys": ["KEY"]}
    ]
    raw["budget"]["max_cost_usd"] = 10
    _expect_rejection(raw, sites_dir, "floating alias")


def test_model_calling_agent_must_declare_env_keys(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [{"id": "a", "scaffold": "claude-code", "model": "pinned-1"}]
    raw["budget"]["max_cost_usd"] = 10
    _expect_rejection(raw, sites_dir, "must declare env_keys")


def test_codex_auth_json_is_a_narrow_secretless_exception(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {
            "id": "codex-plan",
                "scaffold": "codex",
                "scaffold_version": "fixture-cli-1.2.3",
                "model": "gpt-5.5",
            "auth_mode": "codex-auth-json",
            "env_literals": {
                "CODEX_FORCE_AUTH_JSON": "true",
                "OPENAI_BASE_URL": "https://router.example.test/v1",
            },
        }
    ]
    raw["budget"]["max_cost_usd"] = 1

    spec = _validate(raw, sites_dir)

    assert spec.agents[0].auth_mode == "codex-auth-json"
    assert spec.agents[0].secret_names == ()
    assert spec.agents[0].provider_route_digest is not None


@pytest.mark.parametrize(
    ("scaffold", "literals", "needle"),
    [
        (
            "claude-code",
            {"CODEX_FORCE_AUTH_JSON": "true", "OPENAI_BASE_URL": "https://x.test/v1"},
            "only valid for the codex scaffold",
        ),
        ("codex", {"OPENAI_BASE_URL": "https://x.test/v1"}, "CODEX_FORCE_AUTH_JSON"),
        ("codex", {"CODEX_FORCE_AUTH_JSON": "true"}, "OPENAI_BASE_URL"),
    ],
)
def test_codex_auth_json_fails_closed_when_misconfigured(
    controls_raw, sites_dir, scaffold, literals, needle
):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {
            "id": "plan",
            "scaffold": scaffold,
            "model": "pinned-1",
            "auth_mode": "codex-auth-json",
            "env_literals": literals,
        }
    ]
    raw["budget"]["max_cost_usd"] = 1
    _expect_rejection(raw, sites_dir, needle)


def test_env_keys_may_not_carry_values(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {"id": "a", "scaffold": "claude-code", "model": "pinned-1", "env_keys": ["KEY=secret"]}
    ]
    raw["budget"]["max_cost_usd"] = 10
    _expect_rejection(raw, sites_dir, "variable names only")


def test_model_campaign_with_zero_budget_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {"id": "a", "scaffold": "claude-code", "model": "pinned-1", "env_keys": ["KEY"]}
    ]
    raw["budget"]["max_cost_usd"] = 0
    _expect_rejection(raw, sites_dir, "includes model-calling agents")


def test_control_campaign_with_nonzero_budget_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["budget"]["max_cost_usd"] = 5
    _expect_rejection(raw, sites_dir, "zero-cost control")


def test_undeclared_site_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["site"] = "some-laptop"
    _expect_rejection(raw, sites_dir, "not declared")


def test_relative_date_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["date"] = "today"
    _expect_rejection(raw, sites_dir, "YYYY-MM-DD")


def test_dataset_digest_must_be_content_addressed(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["dataset"]["digest"] = "whatever"
    _expect_rejection(raw, sites_dir, "dataset.digest")


def test_unknown_integration_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["integration"] = "not-a-benchmark"
    _expect_rejection(raw, sites_dir, "unknown integration")


def test_unsupported_schema_version_is_rejected(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["schema_version"] = "9.9"
    _expect_rejection(raw, sites_dir, "unsupported schema_version")


def test_all_problems_are_reported_together(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["date"] = "today"
    raw["tasks"] = []
    message = _expect_rejection(raw, sites_dir, "YYYY-MM-DD")
    assert "tasks must list at least one task" in message


# --------------------------------------------------------------------------- #
# compiler
# --------------------------------------------------------------------------- #


def test_compilation_is_byte_identical_across_runs(controls_raw, sites_dir, tmp_path):
    spec = _validate(controls_raw, sites_dir)
    first, second = tmp_path / "a", tmp_path / "b"
    compile_mod.write_plan(compile_mod.compile_campaign(spec), first)
    compile_mod.write_plan(compile_mod.compile_campaign(spec), second)

    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files, "compiler wrote nothing"
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    for relative in files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_run_ids_are_unique_and_reproducible(controls_raw, sites_dir):
    spec = _validate(controls_raw, sites_dir)
    first = compile_mod.compile_campaign(spec)
    second = compile_mod.compile_campaign(spec)
    ids_first = [run.run_id for run in first.runs]
    assert len(set(ids_first)) == len(ids_first)
    assert ids_first == [run.run_id for run in second.runs]


def test_one_job_carries_one_agent_seed_attempt(controls_raw, sites_dir):
    """The constraint that makes (job_name, task_name) an exact ledger lookup."""
    spec = _validate(controls_raw, sites_dir)
    compiled = compile_mod.compile_campaign(spec)
    for job in compiled.jobs:
        members = [run for run in compiled.runs if run.job_name == job.job_name]
        assert len({(run.agent_id, run.seed, run.attempt) for run in members}) == 1
        assert compiled.job_configs[job.job_name]["n_attempts"] == 1


def test_compiled_retry_policy_survives_harbor_lock_round_trip(
    controls_raw, sites_dir
):
    """Harbor 0.16 omits nulls from lock.json, then rehydrates its defaults."""
    compiled = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    for config in compiled.job_configs.values():
        retry = config["retry"]
        assert retry["exclude_exceptions"] == []
        assert compile_mod.harbor_016_retry_lock_roundtrip(retry) == retry


def test_compiled_retry_policy_preserves_explicit_exclusions(
    controls_raw, sites_dir
):
    raw = copy.deepcopy(controls_raw)
    raw["retry"]["exclude_exceptions"] = [
        "VerifierTimeoutError",
        "AgentTimeoutError",
    ]
    compiled = compile_mod.compile_campaign(_validate(raw, sites_dir))
    for config in compiled.job_configs.values():
        assert config["retry"]["exclude_exceptions"] == [
            "AgentTimeoutError",
            "VerifierTimeoutError",
        ]
        assert compile_mod.harbor_016_retry_lock_roundtrip(config["retry"]) == config[
            "retry"
        ]


def test_job_config_contract_version_changes_campaign_and_run_identity(
    monkeypatch, controls_raw, sites_dir
):
    first = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    legacy_digest_without_contract = ids_mod.campaign_cfg_digest(first.spec.raw)
    monkeypatch.setattr(compile_mod, "JOB_CONFIG_CONTRACT_VERSION", "next-test-version")
    second = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))

    assert first.job_config_contract_version == "2.0"
    assert first.cfg_digest != legacy_digest_without_contract
    assert second.job_config_contract_version == "next-test-version"
    assert first.cfg_digest != second.cfg_digest
    assert first.campaign_id != second.campaign_id
    assert [run.run_id for run in first.runs] != [run.run_id for run in second.runs]
    assert first.plan_dict()["job_config_contract_version"] == "2.0"
    assert first.ledger_dict()["job_config_contract_version"] == "2.0"


def test_match_keys_are_unique_within_a_job(controls_raw, sites_dir):
    spec = _validate(controls_raw, sites_dir)
    compiled = compile_mod.compile_campaign(spec)
    for job in compiled.jobs:
        keys = [run.match_key for run in compiled.runs if run.job_name == job.job_name]
        assert len(set(keys)) == len(keys)


def test_sharding_partitions_every_run_exactly_once(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["shards"] = 4
    compiled = compile_mod.compile_campaign(_validate(raw, sites_dir))
    assert all(0 <= run.shard < 4 for run in compiled.runs)
    per_shard = sum(len(job.task_names) for job in compiled.jobs)
    assert per_shard == len(compiled.runs)


def test_job_configs_contain_no_secret_values(controls_raw, sites_dir):
    raw = copy.deepcopy(controls_raw)
    raw["agents"] = [
        {
            "id": "pinned",
                "scaffold": "claude-code",
                "scaffold_version": "fixture-cli-1.2.3",
                "model": "pinned-model-1",
            "env_keys": ["ANTHROPIC_AUTH_TOKEN"],
            "provider_route_digest": "sha256:" + "a" * 64,
        }
    ]
    raw["budget"]["max_cost_usd"] = 10
    compiled = compile_mod.compile_campaign(_validate(raw, sites_dir))
    for config in compiled.job_configs.values():
        env = config["agents"][0]["env"]
        assert env == {"ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}"}


def test_job_configs_are_campaign_labelled_for_scoped_cleanup(controls_raw, sites_dir):
    compiled = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    for config in compiled.job_configs.values():
        labels = config["environment"]["kwargs"]["labels"]
        assert labels[compile_mod.CAMPAIGN_LABEL_KEY] == compiled.campaign_id


def test_written_job_configs_are_parsable_yaml(controls_raw, sites_dir, tmp_path):
    compiled = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    compile_mod.write_plan(compiled, tmp_path)
    for path in sorted((tmp_path / "jobs").glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["n_attempts"] == 1
        assert loaded["datasets"][0]["path"] == compiled.spec.dataset_path


def test_ledger_matches_its_schema_and_precedes_execution(controls_raw, sites_dir, tmp_path):
    compiled = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    compile_mod.write_plan(compiled, tmp_path)
    ledger = json.loads((tmp_path / "plan_ledger.json").read_text(encoding="utf-8"))
    schema = schema_mod.load_schema(schema_mod.schemas_dir() / "plan_ledger.schema.json")
    schema_mod.validate(ledger, schema, name="plan_ledger.json")
    assert ledger["job_config_contract_version"] == "2.0"
    legacy_v1_ledger = copy.deepcopy(ledger)
    legacy_v1_ledger.pop("job_config_contract_version")
    schema_mod.validate(legacy_v1_ledger, schema, name="legacy-plan_ledger.json")
    assert ledger["written_before_execution"] is True
    assert len(ledger["entries"]) == len(compiled.runs)


def test_plan_records_that_the_budget_is_not_a_circuit_breaker(controls_raw, sites_dir):
    compiled = compile_mod.compile_campaign(_validate(controls_raw, sites_dir))
    note = compiled.plan_dict()["cost"]["note"]
    assert "circuit breaker" in note
    assert "provider-side key cap" in note
