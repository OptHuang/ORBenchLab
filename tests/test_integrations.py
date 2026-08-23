"""Integration registry and inspectors.

These run against miniature upstream stand-ins so the suite needs no network.
The same inspectors run against the pinned upstream commits in
``.github/workflows/integration-contract.yml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orbenchlab.core import schema as schema_mod
from orbenchlab.core.errors import IntegrationError
from orbenchlab.integrations import base, frontieror, oragentbench, registry

INSPECTION_SCHEMA = "inspection_report.schema.json"


def _validate_report(report) -> dict:
    payload = report.to_dict()
    schema = schema_mod.load_schema(schema_mod.schemas_dir() / INSPECTION_SCHEMA)
    schema_mod.validate(payload, schema, name="inspection report")
    return payload


def _checks(payload: dict) -> dict[str, dict]:
    return {check["id"]: check for check in payload["checks"]}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_both_first_class_integrations_are_registered():
    assert registry.names() == ["oragentbench", "frontieror"]


def test_unknown_integration_names_the_known_ones():
    with pytest.raises(IntegrationError) as excinfo:
        registry.get("nope")
    assert "oragentbench" in str(excinfo.value)


def test_every_integration_declares_its_requirements():
    for row in registry.summary_rows():
        assert row["upstream_repo"].startswith("https://")
        assert len(row["pinned_commit"]) == 40
        assert isinstance(row["performance_scored"], bool)


def test_the_two_integrations_take_different_forms():
    """The point of shipping both: they exercise opposite integration shapes."""
    kinds = {row["name"]: row["kind"] for row in registry.summary_rows()}
    assert kinds["oragentbench"] == "harbor-native"
    assert kinds["frontieror"] == "official-external-harness"


def test_out_of_tree_integrations_can_register(tmp_path):
    import types

    module = types.ModuleType("fake")
    module.NAME = "fake-bench"
    module.describe = lambda: {"name": "fake-bench"}
    module.inspect = lambda source: None
    try:
        registry.register(module)
        assert "fake-bench" in registry.names()
    finally:
        registry.unregister("fake-bench")
    assert "fake-bench" not in registry.names()


def test_registering_an_incomplete_module_is_rejected():
    import types

    module = types.ModuleType("broken")
    module.NAME = "broken"
    with pytest.raises(IntegrationError):
        registry.register(module)


# --------------------------------------------------------------------------- #
# ORAgentBench
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def oab_payload(request) -> dict:
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    return _validate_report(oragentbench.inspect(source))


def test_oragentbench_finds_harbor_native_task_packages(oab_payload):
    check = _checks(oab_payload)["harbor_native_task_packages"]
    assert check["status"] == "pass"
    assert check["evidence"]["task_count"] == 2


def test_oragentbench_writes_no_adapter(oab_payload):
    """The core decision, stated machine-readably rather than only in prose."""
    assert oab_payload["decisions"]["adapter_required"] is False
    assert "in place" in oab_payload["decisions"]["task_materialization"]


def test_oragentbench_discovers_the_reward_keys_rather_than_assuming_them(oab_payload):
    assert oab_payload["facts"]["reward_keys"] == ["feasibility", "quality"]
    assert _checks(oab_payload)["reward_channel"]["status"] == "pass"


def test_oragentbench_dataset_digest_is_content_addressed(oab_payload):
    digest = oab_payload["facts"]["dataset_digest"]
    assert digest.startswith("sha256:") and len(digest) == 71


def test_oragentbench_dataset_digest_changes_with_task_content(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    before = oragentbench.inspect(copy).facts["dataset_digest"]
    toml = copy / "harbor_tasks" / "single_task" / "task.toml"
    toml.write_text(toml.read_text() + '\nkeywords = ["changed"]\n', encoding="utf-8")
    assert oragentbench.inspect(copy).facts["dataset_digest"] != before


def test_oragentbench_dataset_digest_changes_with_verifier_content(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    before = oragentbench.inspect(copy).facts["dataset_digest"]
    verifier = copy / "harbor_tasks" / "single_task" / "tests" / "test.sh"
    verifier.write_text(verifier.read_text() + "\n# verifier changed\n", encoding="utf-8")
    assert oragentbench.inspect(copy).facts["dataset_digest"] != before


def test_oragentbench_dataset_digest_changes_with_metric_content(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    before = oragentbench.inspect(copy).facts["dataset_digest"]
    metric = copy / "metrics" / "per_dimension_reward.py"
    metric.write_text(metric.read_text() + "\n# metric changed\n", encoding="utf-8")
    assert oragentbench.inspect(copy).facts["dataset_digest"] != before


def test_oragentbench_dataset_digest_ignores_generated_python_bytecode(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    before = oragentbench.inspect(copy).facts["dataset_digest"]
    # Use a declared identity root that already exists.  Importing one of its
    # modules may create this cache, but must not mutate benchmark identity.
    cache = copy / "source" / "scripts" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "prebuilt_agents.cpython-312.pyc").write_bytes(b"generated-bytecode")
    assert oragentbench.inspect(copy).facts["dataset_digest"] == before


def test_oragentbench_identity_ignores_generated_names_only_inside_checkout(
    tmp_path, request
):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "__pycache__" / "ORAgentBench"
    shutil.copytree(source, copy)
    before_report = oragentbench.inspect(copy)
    before = before_report.facts["dataset_digest"]
    assert before_report.facts["identity_file_count"] > 0
    toml = copy / "harbor_tasks" / "single_task" / "task.toml"
    toml.write_text(toml.read_text() + '\nkeywords = ["ancestor-regression"]\n')
    after = oragentbench.inspect(copy)
    assert after.facts["dataset_digest"] != before
    assert after.status != "failed"


def test_oragentbench_fails_when_an_identity_root_is_missing(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "ORAgentBench"
    shutil.copytree(source, copy)
    (copy / "metrics" / "per_dimension_reward.py").unlink()
    report = oragentbench.inspect(copy)
    check = _checks(report.to_dict())["dataset_identity_inputs"]
    assert report.status == "failed"
    assert check["status"] == "fail"
    assert "metrics/per_dimension_reward.py" in check["evidence"]["paths"]


def test_oragentbench_dataset_digest_tracks_executable_mode(tmp_path, request):
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/oragentbench_min"
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    verifier = copy / "harbor_tasks" / "single_task" / "tests" / "test.sh"
    verifier.chmod(verifier.stat().st_mode ^ 0o100)
    before = oragentbench.inspect(source).facts["dataset_digest"]
    assert oragentbench.inspect(copy).facts["dataset_digest"] != before


def test_oragentbench_flags_multi_step_regrade_limitation(oab_payload):
    check = _checks(oab_payload)["multi_step_tasks"]
    assert check["status"] == "warn"
    assert "regrade" in check["evidence"]["impact"]


def test_oragentbench_flags_agent_internet_access(oab_payload):
    assert _checks(oab_payload)["agent_network_isolation"]["status"] == "warn"


def test_oragentbench_infers_solver_grant_from_task_declarations(oab_payload):
    # One fixture task declares requires_gurobi = true.
    assert oab_payload["facts"]["solver_license_grant"] == "verifier"


def test_oragentbench_does_not_claim_to_verify_the_harbor_builtin_nop(oab_payload):
    """Absent upstream is not the same as broken upstream."""
    check = _checks(oab_payload)["nop_control_available"]
    assert check["status"] == "skip"
    assert "Harbor built-in" in check["detail"]


def test_oragentbench_fails_closed_on_a_tree_with_no_tasks(tmp_path):
    payload = _validate_report(oragentbench.inspect(tmp_path))
    assert payload["status"] == "failed"
    assert _checks(payload)["harbor_native_task_packages"]["status"] == "fail"


def test_inspecting_a_missing_directory_raises(tmp_path):
    with pytest.raises(IntegrationError):
        oragentbench.inspect(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# FrontierOR
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fo_payload(request) -> dict:
    source = Path(request.config.rootpath) / "tests/fixtures/upstream/frontieror_min"
    return _validate_report(frontieror.inspect(source))


def test_frontieror_finds_the_official_entrypoint(fo_payload):
    check = _checks(fo_payload)["official_harness_entrypoint"]
    assert check["status"] == "pass"
    assert fo_payload["facts"]["official_commands"] == [
        "agent",
        "submission",
        "contract",
        "security-check",
    ]


def test_frontieror_falls_back_without_claiming_success(fo_payload):
    """The stand-in is not runnable, so the contract read degrades — visibly."""
    check = _checks(fo_payload)["public_scoring_contract"]
    assert check["status"] == "warn"
    assert check["evidence"]["contract_version"] == "staged-qte-v1"
    assert "do not treat a run from this environment as official" in check["evidence"]["impact"]


def test_frontieror_assumes_performance_scoring_when_it_cannot_confirm(fo_payload):
    """Fail-closed: an unreadable contract must not relax the site requirement."""
    assert fo_payload["facts"]["requires_perf_isolated_site"] is True
    check = _checks(fo_payload)["performance_isolation_required"]
    assert check["evidence"]["assumed"] is True


def test_frontieror_does_not_materialize_harbor_tasks(fo_payload):
    decisions = fo_payload["decisions"]
    assert decisions["integration_form"] == "official-external-harness"
    assert decisions["harbor_tasks_materialized"] is False
    assert decisions["harbor_conversion_parity_safe"] is False


def test_frontieror_records_why_conversion_is_not_parity_safe(fo_payload):
    blockers = {b["id"] for b in fo_payload["facts"]["harbor_conversion_blockers"]}
    assert {
        "trusted_host_timing",
        "undistributed_reference_data",
        "upstream_security_boundary",
        "multi_candidate_inner_loop",
    } <= blockers


def test_frontieror_preconditions_all_fail_closed(fo_payload):
    preconditions = fo_payload["facts"]["preconditions"]
    assert {p["id"] for p in preconditions} == {
        "dataset",
        "solver_license",
        "model_api_key",
        "container_images",
        "perf_isolated_runner",
    }
    assert all(p["fail_closed"] for p in preconditions)


def test_frontieror_references_sample_checkers_without_copying_them(fo_payload):
    assert _checks(fo_payload)["trusted_checkers_present"]["status"] == "pass"
    assert _checks(fo_payload)["no_vendored_upstream_copy"]["status"] == "pass"


def test_frontieror_fails_closed_without_an_entrypoint(tmp_path):
    payload = _validate_report(frontieror.inspect(tmp_path))
    assert payload["status"] == "failed"
    assert _checks(payload)["official_harness_entrypoint"]["status"] == "fail"


# --------------------------------------------------------------------------- #
# repository-level invariants
# --------------------------------------------------------------------------- #


def test_this_repository_vendors_no_upstream_benchmark_code(repo_root):
    """Vendoring is how an integration silently becomes a fork."""
    names = ("harbor_tasks",) + frontieror.FORBIDDEN_VENDORED_NAMES
    assert base.find_vendored(repo_root, names) == []


def test_inspection_never_reports_model_calls(oab_payload, fo_payload):
    for payload in (oab_payload, fo_payload):
        assert payload["execution"] == {
            "model_calls": 0,
            "benchmark_executed": False,
            "network_access": False,
            "reads_credentials": False,
        }


# --------------------------------------------------------------------------- #
# real upstream checkouts (opt-in)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("ORBENCH_ORAGENTBENCH_SOURCE"),
    reason="set ORBENCH_ORAGENTBENCH_SOURCE to a real checkout to run this",
)
def test_real_oragentbench_checkout_inspects_cleanly():
    payload = _validate_report(
        oragentbench.inspect(Path(os.environ["ORBENCH_ORAGENTBENCH_SOURCE"]))
    )
    assert payload["status"] in {"ok", "degraded"}
    assert payload["facts"]["task_count"] > 0


@pytest.mark.skipif(
    not os.environ.get("ORBENCH_FRONTIEROR_SOURCE"),
    reason="set ORBENCH_FRONTIEROR_SOURCE to a real checkout to run this",
)
def test_real_frontieror_checkout_exposes_its_scoring_contract():
    payload = _validate_report(frontieror.inspect(Path(os.environ["ORBENCH_FRONTIEROR_SOURCE"])))
    assert payload["status"] in {"ok", "degraded"}
    contract = payload["facts"]["scoring_contract"]
    assert contract["contract_version"] == "staged-qte-v1"
    assert contract["scorer"] == "staged_qte"
