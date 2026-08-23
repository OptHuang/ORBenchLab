"""Provider-route binding is part of a paid agent's reproducible identity.

These tests exercise the product seam rather than only the URL helper: a paid
run must bind one credential-safe provider route before a workspace exists,
and execution must refuse to send the key anywhere else.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from orbenchlab import execution, workflow
from orbenchlab.cli import main
from orbenchlab.core.errors import PreconditionError, SpecError
from orbenchlab.core.urls import provider_route_digest


def _source(upstream_fixtures: Path, tmp_path: Path) -> Path:
    source = tmp_path / execution.ORAGENTBENCH_CHECKOUT_DIRNAME
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    wrapper = source / execution.ORAGENTBENCH_PREBUILD_WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# fixture wrapper\n", encoding="utf-8")
    return source


@pytest.mark.parametrize("scaffold", ["claude-code", "codex"])
def test_api_key_prepare_requires_a_pinned_provider_route(
    upstream_fixtures: Path, tmp_path: Path, scaffold: str
) -> None:
    source = _source(upstream_fixtures, tmp_path)

    with pytest.raises(SpecError, match="provider route"):
        workflow.prepare_oragentbench_run(
            source=source,
            task="single_task",
            agent=scaffold,
            model="pinned-model-2026-08-24",
            date="2026-08-24",
            workspace=tmp_path / "runs",
        )

    assert not (tmp_path / "runs").exists()


def test_cli_safely_resolves_api_key_route_from_model_base_url(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    upstream_fixtures: Path,
    tmp_path: Path,
) -> None:
    source = _source(upstream_fixtures, tmp_path)
    route = "https://Router.Example.Test:443/v1"
    monkeypatch.setenv("MODEL_BASE_URL", route)

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
            "gpt-5.5-2026-08-24",
            "--date",
            "2026-08-24",
            "--workspace",
            str(tmp_path / "runs"),
            "--prepare-only",
        ]
    )

    assert code == 0
    run_root = Path(json.loads(capsys.readouterr().out)["run_root"])
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider_route_digest"] == provider_route_digest(route)
    assert "model_base_url" not in manifest
    blob = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert route not in blob
    assert "router.example.test" not in blob.lower()


def test_runtime_route_must_match_the_pinned_route_without_echoing_either(
    monkeypatch: pytest.MonkeyPatch,
    upstream_fixtures: Path,
    tmp_path: Path,
) -> None:
    source = _source(upstream_fixtures, tmp_path)
    pinned = "https://pinned.example.test/v1"
    runtime = "https://other.example.test/v1"
    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="codex",
        scaffold_version="fixture-cli-1.2.3",
        model="gpt-5.5-2026-08-24",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        model_base_url=pinned,
    )
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/usr/bin/{name}")

    def probe(argv, **kwargs):
        stdout = "harbor, version 0.16.2\n" if argv[-1] == "--version" else ""
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(execution.subprocess, "run", probe)

    with pytest.raises(PreconditionError) as excinfo:
        workflow.execute_prepared_run(
            prepared,
            acknowledge_cost="i-accept-model-costs",
            environ={"MODEL_API_KEY": "secret", "MODEL_BASE_URL": runtime},
        )

    message = str(excinfo.value)
    assert "does not match" in message
    assert pinned not in message
    assert runtime not in message


def test_different_provider_routes_produce_different_campaign_identities(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    source = _source(upstream_fixtures, tmp_path)
    common = dict(
        source=source,
        task="single_task",
        agent="claude-code",
        scaffold_version="fixture-cli-1.2.3",
        model="claude-opus-5-2026-08-24",
        date="2026-08-24",
        workspace=tmp_path / "runs",
    )

    first = workflow.prepare_oragentbench_run(
        **common, model_base_url="https://first.example.test/v1"
    )
    second = workflow.prepare_oragentbench_run(
        **common, model_base_url="https://second.example.test/v1"
    )

    assert first.campaign_id != second.campaign_id
    first_manifest = json.loads((first.run_root / "manifest.json").read_text())
    second_manifest = json.loads((second.run_root / "manifest.json").read_text())
    assert first_manifest["provider_route_digest"] != second_manifest["provider_route_digest"]


def test_equivalent_route_spellings_have_one_identity() -> None:
    assert provider_route_digest(
        "https://Router.Example.Test:443/v1"
    ) == provider_route_digest("https://router.example.test/v1")


def test_runtime_route_is_compared_after_canonicalization(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    source = _source(upstream_fixtures, tmp_path)
    report = execution.oragentbench_preconditions(
        source=source,
        task_name="single_task",
        scaffold="codex",
        model="gpt-5.5-2026-08-24",
        model_base_url="https://router.example.test/v1",
        environ={
            "MODEL_API_KEY": "secret",
            "MODEL_BASE_URL": "https://Router.Example.Test:443/v1",
        },
        require_docker=False,
        require_harbor=False,
    )

    assert report.ok, report.missing


def test_mini_swe_agent_does_not_require_a_provider_route() -> None:
    raw = execution.oragentbench_agent_campaign_spec(
        slug="mini-swe-route-free",
        date="2026-08-24",
        dataset_digest="sha256:" + "a" * 64,
        task_name="single_task",
        scaffold="mini-swe-agent",
        scaffold_version="fixture-cli-1.2.3",
        model="qwen3-coder-2026-08-24",
    )

    assert "provider_route_digest" not in raw["agents"][0]
