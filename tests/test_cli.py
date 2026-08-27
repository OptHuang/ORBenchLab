"""End-to-end CLI behaviour, including exit codes.

Exit codes are part of the interface: CI branches on them, so a change here is a
change to how every workflow behaves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab.cli import main


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "orbench" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------- #
# integrations
# --------------------------------------------------------------------------- #


def test_integrations_list_table(capsys):
    assert main(["integrations", "list"]) == 0
    out = capsys.readouterr().out
    assert "oragentbench" in out and "frontieror" in out
    assert "perf-isolated-site" in out


def test_integrations_list_json(capsys):
    assert main(["integrations", "list", "--json"]) == 0
    payload = _json_out(capsys)
    names = [row["name"] for row in payload["integrations"]]
    assert names == ["oragentbench", "frontieror"]


def test_integration_describe(capsys):
    assert main(["integration", "describe", "frontieror"]) == 0
    payload = _json_out(capsys)
    assert payload["kind"] == "official-external-harness"
    assert payload["performance_scored"] is True


def test_integration_inspect_emits_machine_readable_json(capsys, upstream_fixtures):
    code = main(
        ["integration", "inspect", "oragentbench", "--source", str(upstream_fixtures / "oragentbench_min")]
    )
    assert code == 0
    payload = _json_out(capsys)
    assert payload["integration"] == "oragentbench"
    assert payload["status"] in {"ok", "degraded"}
    assert payload["execution"]["model_calls"] == 0


def test_integration_inspect_writes_the_json_file(capsys, upstream_fixtures, tmp_path):
    out = tmp_path / "nested" / "report.json"
    code = main(
        [
            "integration",
            "inspect",
            "oragentbench",
            "--source",
            str(upstream_fixtures / "oragentbench_min"),
            "--json",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text())["integration"] == "oragentbench"


def test_integration_inspect_fails_on_a_broken_checkout(capsys, tmp_path):
    assert main(["integration", "inspect", "oragentbench", "--source", str(tmp_path)]) == 3


def test_fail_on_warn_turns_a_degraded_report_into_a_failure(capsys, upstream_fixtures):
    source = str(upstream_fixtures / "oragentbench_min")
    assert main(["integration", "inspect", "oragentbench", "--source", source]) == 0
    capsys.readouterr()
    assert main(["integration", "inspect", "oragentbench", "--source", source, "--fail-on-warn"]) == 3


def test_unknown_integration_exits_with_the_integration_code(capsys):
    assert main(["integration", "describe", "nope"]) == 3
    assert "unknown integration" in capsys.readouterr().err


def test_inspect_of_a_missing_source_exits_nonzero(capsys, tmp_path):
    assert main(["integration", "inspect", "frontieror", "--source", str(tmp_path / "missing")]) == 3


# --------------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------------- #


def test_campaign_validate_accepts_the_controls_spec(capsys, campaigns_dir):
    code = main(["campaign", "validate", str(campaigns_dir / "oragentbench-controls.yaml")])
    assert code == 0
    payload = _json_out(capsys)
    assert payload["valid"] is True
    assert payload["zero_cost"] is True


def test_campaign_validate_rejects_a_performance_scored_spec_on_a_shared_site(
    capsys, campaigns_dir
):
    code = main(["campaign", "validate", str(campaigns_dir / "frontieror-contract-check.yaml")])
    assert code == 2
    assert "perf_isolated" in capsys.readouterr().err


def test_campaign_validate_reports_a_missing_file_clearly(capsys, tmp_path):
    assert main(["campaign", "validate", str(tmp_path / "nope.yaml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_campaign_plan_writes_plan_ledger_and_jobs(capsys, campaigns_dir, tmp_path):
    code = main(
        [
            "campaign",
            "plan",
            str(campaigns_dir / "oragentbench-controls.yaml"),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = _json_out(capsys)
    assert payload["runs"] == 18
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "plan_ledger.json").is_file()
    assert len(list((tmp_path / "jobs").glob("*.yaml"))) == payload["jobs"]


def test_campaign_plan_json_flag_prints_the_whole_plan(capsys, campaigns_dir, tmp_path):
    main(
        [
            "campaign",
            "plan",
            str(campaigns_dir / "oragentbench-controls.yaml"),
            "--out",
            str(tmp_path),
            "--json",
        ]
    )
    payload = _json_out(capsys)
    assert payload["plan_schema_version"] == "1.0"
    assert len(payload["runs"]) == 18


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_report_build_writes_all_three_files(capsys, fixtures_dir, tmp_path):
    code = main(
        [
            "report",
            "build",
            "--input",
            str(fixtures_dir / "normalized" / "oragentbench-controls.json"),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = _json_out(capsys)
    assert payload["effective_label"] == "partial"
    assert payload["intended_label"] == "validated"
    for name in ("summary.md", "summary.json", "evidence_index.json"):
        assert (tmp_path / name).is_file()


def test_report_build_refuses_to_over_label(capsys, fixtures_dir, tmp_path):
    code = main(
        [
            "report",
            "build",
            "--input",
            str(fixtures_dir / "normalized" / "oragentbench-smoke-r0.json"),
            "--out",
            str(tmp_path),
            "--require-label",
            "validated",
        ]
    )
    assert code == 4
    assert "weaker than the required" in capsys.readouterr().err


def test_report_build_accepts_a_label_the_evidence_supports(capsys, fixtures_dir, tmp_path):
    code = main(
        [
            "report",
            "build",
            "--input",
            str(fixtures_dir / "normalized" / "oragentbench-controls.json"),
            "--out",
            str(tmp_path),
            "--require-label",
            "partial",
        ]
    )
    assert code == 0


def test_pipeline_run_writes_final_task_cards(capsys, tmp_path):
    task = tmp_path / "tasks" / "demo.yaml"
    task.parent.mkdir()
    task.write_text(
        "family: demo\n"
        "title: Demo task\n"
        "design_goal: Test a coupled rule\n"
        "difficulty_axes:\n"
        "  scale:\n"
        "    levels: [tiny, small]\n",
        encoding="utf-8",
    )
    screening = tmp_path / "screening.json"
    screening.write_text(
        json.dumps(
            {
                "schema_version": "orbenchlab.screening-report.v1",
                "tasks": [
                    {
                        "task": "demo",
                        "arms": {
                            "route-a": {
                                "n": 1,
                                "complete": 1,
                                "metric_n": 1,
                                "solve_rate": 1.0,
                                "quality_pass_rate": 1.0,
                                "mean_feasibility": 1.0,
                                "infra_exceptions": [],
                            }
                        },
                        "decision": "collect-more-evidence",
                        "evidence_level": "E3",
                        "limitations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["pipeline", "run", "--tasks", str(task), "--screenings", str(screening), "--out", str(out)]) == 0
    capsys.readouterr()
    cards = json.loads((out / "task-cards.json").read_text())
    assert cards["cards"][0]["task_id"] == "demo"
    assert cards["cards"][0]["performance"]["models"][0]["solve_rate"] == 1.0
    assert "难度如何调控" in (out / "task-cards.md").read_text()


def test_task_author_validate_writes_blocked_receipt(capsys, tmp_path):
    out = tmp_path / "authoring"
    task_dir = tmp_path / "missing-task"
    task_dir.mkdir()
    code = main(
        [
            "task-author",
            "validate",
            "--task-dir",
            str(task_dir),
            "--out",
            str(out),
        ]
    )
    assert code == 8
    payload = _json_out(capsys)
    assert payload["decision"] == "blocked"
    assert (out / "authoring-receipt.json").is_file()
    assert (out / "authoring-receipt.md").is_file()


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_schema_list(capsys):
    assert main(["schema", "list"]) == 0
    files = {entry["file"] for entry in _json_out(capsys)["schemas"]}
    assert "normalized_rollout.schema.json" in files
    assert "plan_ledger.schema.json" in files


def test_schema_validate_accepts_a_shipped_fixture(capsys, fixtures_dir):
    code = main(
        [
            "schema",
            "validate",
            str(fixtures_dir / "normalized" / "oragentbench-controls.json"),
            "--schema",
            "normalized_rollout.schema.json",
        ]
    )
    assert code == 0
    assert _json_out(capsys)["valid"] is True


def test_schema_validate_rejects_a_mismatched_document(capsys, fixtures_dir, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"normalized_schema_version": "1.0"}), encoding="utf-8")
    assert main(["schema", "validate", str(broken), "--schema", "normalized_rollout.schema.json"]) == 6
    assert "missing required property" in capsys.readouterr().err


def test_schema_validate_names_available_schemas_on_a_typo(capsys, fixtures_dir):
    code = main(
        [
            "schema",
            "validate",
            str(fixtures_dir / "normalized" / "oragentbench-controls.json"),
            "--schema",
            "nope.json",
        ]
    )
    assert code == 6
    assert "normalized_rollout.schema.json" in capsys.readouterr().err
