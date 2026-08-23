from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orbenchlab import cli
import orbenchlab.export as export_mod
from orbenchlab.core.errors import EvidenceError, PreconditionError
from orbenchlab.export import _assert_shareable_text, export_shareable_run
from orbenchlab.report import render as render_mod
from orbenchlab.report.model import NormalizedRollout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_integrity(run_root: Path) -> None:
    rows = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        if path.name == "integrity.sha256":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(run_root).as_posix()}")
    (run_root / "integrity.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def completed_run(tmp_path: Path) -> Path:
    run_root = tmp_path / "runs" / "oab-fixture-20260824-deadbeef"
    host_source = tmp_path / "private" / "ORAgentBench"
    host_output = run_root.resolve()
    secret = "sk-private-do-not-share"
    manifest = {
        "manifest_schema_version": "1.0",
        "state": "completed",
        "campaign_id": "oab-fixture-20260824-deadbeef",
        "integration": "oragentbench",
        "task": "additive_microfactory_order_planning",
        "agent": "codex",
        "agent_id": "codex-gpt-fixture",
        "model": "gpt-fixture",
        "date": "2026-08-24",
        "source": str(host_source),
        "source_commit": "a" * 40,
        "auth_mode": "api-key",
        "provider_route_digest": "sha256:" + "c" * 64,
        "dataset_digest": "sha256:" + "b" * 64,
        "source_snapshot_digest": "sha256:" + "d" * 64,
        "scaffold_version": "1.2.3",
        "runtime_image": {
            "requested_tag": "orbench-oab-codex:fixture",
            "image_id": "sha256:" + "e" * 64,
            "repo_digests": [],
        },
        "runtime_image_evidence": "docker-image-inspect",
        "runtime_image_alias_verification": {
            "fixed_alias": "oragentbench-base:py311-scip",
            "fixed_alias_image_id": "sha256:" + "e" * 64,
            "matches_runtime_image": True,
        },
        "raw_evidence_local_only": True,
        "exit_code": 0,
    }
    receipt = {
        "receipt_schema_version": "1.0",
        "integration": "oragentbench",
        "mode": "agent",
        "campaign_id": manifest["campaign_id"],
        "evidence_label": "exploratory",
        "upstream_command": {
            "argv": ["harbor", "run", "-c", str(run_root / "plan" / "job.yaml")],
            "cwd": str(host_source.parent),
            "provenance": "native Harbor CLI",
            "makes_model_calls": True,
        },
        "environment": {"OPENAI_API_KEY": "<set>"},
        "preconditions": {"satisfied": [f"provider configured at {host_source}"]},
        "exit_code": 0,
        "executed": True,
        "output_root": str(host_output),
        "raw_bundle_uploaded": False,
        "agent_id": manifest["agent_id"],
        "scaffold_version": manifest["scaffold_version"],
        "source_snapshot_digest": manifest["source_snapshot_digest"],
        "runtime_image": manifest["runtime_image"],
        "runtime_image_evidence": manifest["runtime_image_evidence"],
        "runtime_image_alias_verification": manifest[
            "runtime_image_alias_verification"
        ],
        "notes": [f"Authorization: Bearer {secret}"],
    }
    normalized = {
        "normalized_schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "integration": "oragentbench",
        "evidence_intent": "exploratory",
        "site": {"name": "local-docker", "perf_isolated": False, "load_source": "none"},
        "scoring": {
            "reward_keys": ["feasibility", "quality"],
            "strict_pass_rule": {
                "description": "fixture",
                "feasibility_key": "feasibility",
                "quality_key": "quality",
                "quality_threshold": 1,
            },
        },
        "trials": [
            {
                "run_id": "1" * 16,
                "task_name": manifest["task"],
                "agent_id": manifest["agent_id"],
                "scaffold": manifest["agent"],
                "seed": 1,
                "attempt": 1,
                "attribution": "agent",
                "counts_toward_capability": True,
                "infra_suspect": False,
                "exclusion_basis": None,
                "trace_status": "complete",
                "scores": {"feasibility": 1, "quality": 1},
            }
        ],
    }
    _write_json(run_root / "manifest.json", manifest)
    _write_json(run_root / "receipt.json", receipt)
    normalized_path = run_root / "normalized" / "rollout.json"
    _write_json(normalized_path, normalized)
    report = render_mod.build_report(NormalizedRollout.load(normalized_path))
    render_mod.write_report(report, run_root / "report")
    (run_root / "jobs" / "raw" / "trajectory.json").parent.mkdir(parents=True)
    (run_root / "jobs" / "raw" / "trajectory.json").write_text(secret, encoding="utf-8")
    (run_root / "logs").mkdir()
    (run_root / "logs" / "upstream.stderr.log").write_text(secret, encoding="utf-8")
    _refresh_integrity(run_root)
    return run_root


def test_export_builds_a_path_free_whitelisted_share_bundle(
    completed_run: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "shared" / "run"

    result = export_shareable_run(completed_run, destination)

    assert result.destination == destination.resolve()
    exported = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert exported == {
        "normalized/rollout.json",
        "report/summary.md",
        "report/summary.json",
        "report/evidence_index.json",
        "public-manifest.json",
        "public-receipt.json",
        "share-integrity.sha256",
    }
    manifest = json.loads((destination / "public-manifest.json").read_text())
    receipt = json.loads((destination / "public-receipt.json").read_text())
    assert manifest == {
        "agent": "codex",
        "agent_id": "codex-gpt-fixture",
        "campaign_id": "oab-fixture-20260824-deadbeef",
        "dataset_digest": "sha256:" + "b" * 64,
        "exit_code": 0,
        "integration": "oragentbench",
        "model": "gpt-fixture",
        "provider_route_digest": "sha256:" + "c" * 64,
        "runtime_image": {
            "image_id": "sha256:" + "e" * 64,
            "repo_digests": [],
            "requested_tag": "orbench-oab-codex:fixture",
        },
        "runtime_image_alias_verification": {
            "fixed_alias": "oragentbench-base:py311-scip",
            "fixed_alias_image_id": "sha256:" + "e" * 64,
            "matches_runtime_image": True,
        },
        "scaffold_version": "1.2.3",
        "public_manifest_schema_version": "1.0",
        "source_commit": "a" * 40,
        "source_snapshot_digest": "sha256:" + "d" * 64,
        "state": "completed",
        "task": "additive_microfactory_order_planning",
    }
    assert receipt == {
        "agent": "codex",
        "agent_id": "codex-gpt-fixture",
        "campaign_id": "oab-fixture-20260824-deadbeef",
        "evidence_label": "exploratory",
        "executed": True,
        "exit_code": 0,
        "integration": "oragentbench",
        "mode": "agent",
        "model": "gpt-fixture",
        "public_receipt_schema_version": "1.0",
        "runtime_image": {
            "image_id": "sha256:" + "e" * 64,
            "repo_digests": [],
            "requested_tag": "orbench-oab-codex:fixture",
        },
        "runtime_image_alias_verification": {
            "fixed_alias": "oragentbench-base:py311-scip",
            "fixed_alias_image_id": "sha256:" + "e" * 64,
            "matches_runtime_image": True,
        },
        "runtime_image_evidence": "docker-image-inspect",
        "scaffold_version": "1.2.3",
        "source_snapshot_digest": "sha256:" + "d" * 64,
        "task": "additive_microfactory_order_planning",
        "upstream_command": {
            "makes_model_calls": True,
            "provenance": "native Harbor CLI",
        },
    }
    blob = b"\n".join(path.read_bytes() for path in destination.rglob("*") if path.is_file())
    assert str(completed_run).encode() not in blob
    assert str(tmp_path / "private").encode() not in blob
    assert b"provider.private.example" not in blob
    assert b"sk-private-do-not-share" not in blob
    assert b"trajectory.json" not in blob

    lines = (destination / "share-integrity.sha256").read_text().splitlines()
    assert len(lines) == 6
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert _sha256(destination / relative) == expected


def test_export_is_idempotent_but_refuses_an_existing_conflict(
    completed_run: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "shared"
    first = export_shareable_run(completed_run, destination)
    second = export_shareable_run(completed_run, destination)
    assert first.destination == second.destination
    assert second.reused is True

    (destination / "report" / "summary.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="conflicts"):
        export_shareable_run(completed_run, destination)


@pytest.mark.parametrize("state", ["prepared", "running", "failed"])
def test_export_requires_a_completed_run(
    completed_run: Path, tmp_path: Path, state: str
) -> None:
    manifest_path = completed_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = state
    _write_json(manifest_path, manifest)
    _refresh_integrity(completed_run)

    with pytest.raises(PreconditionError, match="completed"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_tampering_even_of_a_non_exported_raw_file(
    completed_run: Path, tmp_path: Path
) -> None:
    (completed_run / "jobs" / "raw" / "trajectory.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(EvidenceError, match="integrity"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_an_allowlisted_file_changed_after_ledger_verification(
    completed_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = export_mod._verify_completed_workspace

    def verify_then_mutate(run_root: Path) -> dict[str, str]:
        expected = original(run_root)
        normalized_path = run_root / "normalized" / "rollout.json"
        normalized_path.write_text("{}\n", encoding="utf-8")
        return expected

    monkeypatch.setattr(export_mod, "_verify_completed_workspace", verify_then_mutate)
    destination = tmp_path / "shared"

    with pytest.raises(EvidenceError, match="changed after workspace integrity"):
        export_shareable_run(completed_run, destination)
    assert not destination.exists()


def test_export_publishes_only_the_validated_in_memory_snapshot(
    completed_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = export_mod._assert_shareable_text
    injected = "sk-concurrent-source-mutation-123456"
    mutated = False

    def scan_then_mutate(
        text: str, *, label: str, forbidden_paths: tuple[str, ...]
    ) -> None:
        nonlocal mutated
        original(text, label=label, forbidden_paths=forbidden_paths)
        if label == "report markdown" and not mutated:
            mutated = True
            (completed_run / "report" / "summary.md").write_text(
                text + f"\nOPENAI_API_KEY={injected}\n", encoding="utf-8"
            )

    monkeypatch.setattr(export_mod, "_assert_shareable_text", scan_then_mutate)
    destination = tmp_path / "shared"

    export_shareable_run(completed_run, destination)

    assert mutated is True
    exported = (destination / "report" / "summary.md").read_text(encoding="utf-8")
    assert injected not in exported


def test_export_rejects_symlinks(completed_run: Path, tmp_path: Path) -> None:
    target = completed_run / "report" / "summary.md"
    target.unlink()
    target.symlink_to(tmp_path / "outside.md")
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="symlink"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_an_unsafe_integrity_path(
    completed_run: Path, tmp_path: Path
) -> None:
    ledger = completed_run / "integrity.sha256"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + f"{'0' * 64}  ../outside\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="unsafe"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_a_missing_allowlisted_file(
    completed_run: Path, tmp_path: Path
) -> None:
    (completed_run / "report" / "summary.md").unlink()
    with pytest.raises(EvidenceError, match="integrity"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_requires_inspected_runtime_image_identity(
    completed_run: Path, tmp_path: Path
) -> None:
    for filename in ("manifest.json", "receipt.json"):
        path = completed_run / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("runtime_image", None)
        payload.pop("runtime_image_evidence", None)
        _write_json(path, payload)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="Docker image"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_conflicting_runtime_image_evidence(
    completed_run: Path, tmp_path: Path
) -> None:
    receipt_path = completed_run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runtime_image"]["image_id"] = "sha256:" + "f" * 64
    _write_json(receipt_path, receipt)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="conflicting runtime image"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_an_alias_that_does_not_match_the_runtime_image(
    completed_run: Path, tmp_path: Path
) -> None:
    for filename in ("manifest.json", "receipt.json"):
        path = completed_run / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["runtime_image_alias_verification"]["fixed_alias_image_id"] = (
            "sha256:" + "f" * 64
        )
        _write_json(path, payload)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="conflicting Docker alias"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_a_synchronized_substitute_for_the_fixed_alias(
    completed_run: Path, tmp_path: Path
) -> None:
    """A self-consistent forged alias and refreshed ledger remain untrusted."""
    for filename in ("manifest.json", "receipt.json"):
        path = completed_run / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["runtime_image_alias_verification"]["fixed_alias"] = "evil:tag"
        _write_json(path, payload)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="fixed ORAgentBench Docker alias"):
        export_shareable_run(completed_run, tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("task_name", "different_task", "conflicting task"),
        ("scaffold", "different-scaffold", "conflicting scaffold"),
        ("agent_id", "different-agent-id", "compiled agent identity"),
    ],
)
def test_export_binds_normalized_trial_identity_to_the_local_manifest(
    completed_run: Path,
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    normalized_path = completed_run / "normalized" / "rollout.json"
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized["trials"][0][field] = replacement
    _write_json(normalized_path, normalized)
    report = render_mod.build_report(NormalizedRollout.load(normalized_path))
    render_mod.write_report(report, completed_run / "report")
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match=message):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_binds_model_to_the_compiled_agent_identity(
    completed_run: Path, tmp_path: Path
) -> None:
    for filename in ("manifest.json", "receipt.json"):
        path = completed_run / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model"] = "different-model"
        _write_json(path, payload)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="model conflicts"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_refuses_non_shareable_provenance(
    completed_run: Path, tmp_path: Path
) -> None:
    receipt_path = completed_run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["upstream_command"]["provenance"] = "/home/private/runner.py"
    _write_json(receipt_path, receipt)
    _refresh_integrity(completed_run)
    with pytest.raises(EvidenceError, match="provenance"):
        export_shareable_run(completed_run, tmp_path / "shared")


def test_export_rejects_an_unknown_normalized_debug_path_without_echoing_it(
    completed_run: Path, tmp_path: Path
) -> None:
    normalized_path = completed_run / "normalized" / "rollout.json"
    normalized = json.loads(normalized_path.read_text())
    normalized["debug_path"] = "/home/alice/private/trace.json"
    _write_json(normalized_path, normalized)
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError) as excinfo:
        export_shareable_run(completed_run, tmp_path / "shared")
    assert "strict export validation" in str(excinfo.value)
    assert "/home/alice" not in str(excinfo.value)
    assert not (tmp_path / "shared").exists()


def test_export_rejects_a_secret_in_report_markdown_without_echoing_it(
    completed_run: Path, tmp_path: Path
) -> None:
    summary_path = completed_run / "report" / "summary.md"
    secret = "sk-report-secret-123456"
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + f"\nOPENAI_API_KEY={secret}\n",
        encoding="utf-8",
    )
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError) as excinfo:
        export_shareable_run(completed_run, tmp_path / "shared")
    assert "credential-like" in str(excinfo.value)
    assert secret not in str(excinfo.value)
    assert not (tmp_path / "shared").exists()


def test_export_rejects_a_private_key_marker_in_report_markdown(
    completed_run: Path, tmp_path: Path
) -> None:
    summary_path = completed_run / "report" / "summary.md"
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8")
        + "\n-----BEGIN "
        + "OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    _refresh_integrity(completed_run)

    with pytest.raises(EvidenceError, match="credential-like"):
        export_shareable_run(completed_run, tmp_path / "shared")


@pytest.mark.parametrize(
    "text",
    [
        "trace at /scratch/alice/run.json",
        "cache=/data/private/result.json",
        "mounted from /nfs/benchmarks/oab",
        "mac artifact /Volumes/External/trace.json",
        r"windows artifact D:\\Users\\alice\\trace.json",
    ],
)
def test_share_guard_rejects_generic_embedded_absolute_paths(text: str) -> None:
    with pytest.raises(EvidenceError, match="host path"):
        _assert_shareable_text(text, label="fixture", forbidden_paths=())


@pytest.mark.parametrize(
    "text",
    [
        "quality = feasible / attempted",
        "ratio 1 / n remains bounded",
        "documentation: https://example.test/report/v1",
        "artifact name normalized/rollout.json",
    ],
)
def test_share_guard_allows_math_urls_and_relative_artifact_names(text: str) -> None:
    _assert_shareable_text(text, label="fixture", forbidden_paths=())


def test_export_refuses_a_destination_inside_the_run(completed_run: Path) -> None:
    with pytest.raises(PreconditionError, match="inside"):
        export_shareable_run(completed_run, completed_run / "share")


def test_export_cli_emits_machine_readable_result(
    completed_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "shared"
    code = cli.main(
        [
            "export",
            "--run-root",
            str(completed_run),
            "--destination",
            str(destination),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["campaign_id"] == "oab-fixture-20260824-deadbeef"
    assert payload["destination"] == str(destination.resolve())
    assert payload["files"] == 7
