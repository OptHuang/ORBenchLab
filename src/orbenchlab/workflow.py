"""Product-level orchestration for a complete ORBenchLab run workspace.

The lower layers remain pure and benchmark-native.  This module is the small
piece that makes them usable together: inspect, plan, preflight, execute,
ingest, and report under one deterministic run directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import execution
from .campaign import compile as compile_mod
from .campaign import spec as spec_mod
from .core.errors import EvidenceError, PreconditionError, SpecError
from .ingest.harbor import HarborIngestResult, ingest_harbor_bundle
from .integrations import registry


@dataclass(frozen=True)
class PreparedRun:
    run_root: Path
    campaign_id: str
    command: execution.UpstreamCommand
    preconditions: execution.PreconditionReport
    resumed: bool
    agent: str
    model: str
    task: str
    source: Path
    wall_clock_sec: int
    auth_mode: str
    model_base_url: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity(run_root: Path) -> Path:
    target = run_root / "integrity.sha256"
    rows = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        if path == target or ".tmp-" in path.name or path.name == ".run.lock":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(run_root).as_posix()}")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(temporary, target)
    return target


def _verify_integrity(run_root: Path) -> None:
    manifest = run_root / "integrity.sha256"
    if not manifest.is_file():
        raise EvidenceError(f"existing run {run_root} has no integrity.sha256")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            raise EvidenceError(f"invalid integrity line in {manifest}: {line!r}") from None
        path = run_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise EvidenceError(f"existing run workspace failed integrity check: {relative}")


@contextmanager
def _campaign_lock(workspace: Path, campaign_id: str) -> Iterator[None]:
    lock_dir = workspace / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{campaign_id}.lock"
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreconditionError(
                f"campaign {campaign_id} is already being prepared or executed in {workspace}"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _slug(task: str, agent: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", f"oab-{task}-{agent}".lower()).strip("-")
    if len(value) <= 63:
        return value
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[:54].rstrip('-')}-{suffix}"


def _control_spec(
    *, task: str, agent: str, date: str, digest: str, jobs_dir: str, wall_clock_sec: int
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slug": _slug(task, agent),
        "date": date,
        "integration": "oragentbench",
        "site": "local-docker",
        "evidence_intent": "exploratory",
        "dataset": {"path": execution.ORAGENTBENCH_DATASET_PATH, "digest": digest},
        "tasks": [task],
        "agents": [{"id": agent, "scaffold": agent}],
        "budget": {"wall_clock_sec": wall_clock_sec, "max_cost_usd": 0},
        "seeds": [1],
        "attempts": 1,
        "shards": 1,
        "harbor": {
            "jobs_dir": jobs_dir,
            "n_concurrent_trials": 1,
            "environment_type": "docker",
        },
        "retry": {"max_retries": 0, "include_exceptions": []},
        "metrics": [
            {
                "type": "uv-script",
                "script_path": "ORAgentBench/metrics/per_dimension_reward.py",
            }
        ],
    }


def _git_head(source: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_process_group(
    argv: list[str],
    *,
    cwd: str | Path,
    environ: Mapping[str, str],
    stdout: Any,
    stderr: Any,
    timeout_sec: int,
) -> int:
    """Run upstream under one killable process group and enforce the budget.

    Harbor and Docker both create children.  Killing only the wrapper on a
    timeout leaves those children consuming a shared runner and can corrupt the
    next resume attempt, so the timeout owns a fresh process group.
    """
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(environ),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        return int(process.wait(timeout=timeout_sec))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 124


def prepare_oragentbench_run(
    *,
    source: str | Path,
    task: str,
    agent: str,
    model: str,
    date: str,
    workspace: str | Path,
    wall_clock_sec: int = 2700,
    max_cost_usd: float = 25.0,
    auth_mode: str = "api-key",
    model_base_url: str = "",
) -> PreparedRun:
    source = execution.validate_oragentbench_source(Path(source))
    execution.validate_oragentbench_task(source, task)
    inspection = registry.inspect("oragentbench", source)
    digest = str(inspection.facts["dataset_digest"])
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if agent in execution.CONTROL_SCAFFOLDS:
        raw = _control_spec(
            task=task,
            agent=agent,
            date=date,
            digest=digest,
            jobs_dir="jobs",
            wall_clock_sec=wall_clock_sec,
        )
    else:
        raw = execution.oragentbench_agent_campaign_spec(
            slug=_slug(task, agent),
            date=date,
            dataset_digest=digest,
            task_name=task,
            scaffold=agent,
            model=model,
            wall_clock_sec=wall_clock_sec,
            max_cost_usd=max_cost_usd,
            auth_mode=auth_mode,
            model_base_url=model_base_url,
        )

    sites_dir = Path(__file__).resolve().parents[2] / "sites"
    if not sites_dir.is_dir():
        # Installed wheels do not carry repository site declarations.  The
        # built-in local site is materialized inside the workspace instead of
        # making behavior depend on the caller's cwd.
        sites_dir = workspace / ".sites"
        sites_dir.mkdir(parents=True, exist_ok=True)
        site_file = sites_dir / "local-docker.yaml"
        if not site_file.exists():
            site_file.write_text(
                "name: local-docker\nperf_isolated: false\nsolver_license_slots: 0\n",
                encoding="utf-8",
            )
    validated = spec_mod.validate(raw, sites_dir=sites_dir)
    campaign_id = compile_mod.compile_campaign(validated).campaign_id
    run_root = workspace / campaign_id
    raw["harbor"]["jobs_dir"] = str(run_root / "jobs")
    validated = spec_mod.validate(raw, sites_dir=sites_dir)
    compiled = compile_mod.compile_campaign(validated)
    if compiled.campaign_id != campaign_id:
        raise SpecError("output path unexpectedly changed campaign identity")

    with _campaign_lock(workspace, campaign_id):
        resumed = run_root.exists()
        if resumed:
            _verify_integrity(run_root)
            manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
            expected = {
                "campaign_id": campaign_id,
                "integration": "oragentbench",
                "task": task,
                "agent": agent,
                "model": model or None,
                "date": date,
                "source_commit": _git_head(source),
                "auth_mode": auth_mode if agent not in execution.CONTROL_SCAFFOLDS else None,
                "model_base_url": model_base_url or None,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise EvidenceError(
                        f"existing run {run_root} has conflicting {key}: "
                        f"{manifest.get(key)!r} != {value!r}"
                    )
        else:
            stage = workspace / f".{campaign_id}.tmp-{os.getpid()}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            compile_mod.write_plan(compiled, stage / "plan")
            _atomic_json(stage / "inspection.json", inspection.to_dict())
            _atomic_json(
                stage / "manifest.json",
                {
                    "manifest_schema_version": "1.0",
                    "state": "prepared",
                    "campaign_id": campaign_id,
                    "integration": "oragentbench",
                    "task": task,
                    "agent": agent,
                    "model": model or None,
                    "date": date,
                    "source": str(source),
                    "source_commit": _git_head(source),
                    "auth_mode": auth_mode if agent not in execution.CONTROL_SCAFFOLDS else None,
                    "model_base_url": model_base_url or None,
                    "dataset_digest": digest,
                    "raw_evidence_local_only": True,
                },
            )

            final_job = run_root / "plan" / "jobs" / f"{compiled.jobs[0].job_name}.yaml"
            if agent in execution.CONTROL_SCAFFOLDS:
                command = execution.oragentbench_controls_command(
                    source=source, job_config=final_job
                )
            else:
                command = execution.oragentbench_agent_command(
                    source=source,
                    job_config=final_job,
                    required_env=validated.agents[0].secret_names,
                )
            preconditions = execution.oragentbench_preconditions(
                source=source,
                task_name=task,
                scaffold=agent,
                model=model,
                require_docker=False,
                require_harbor=False,
                require_secrets=False,
                auth_mode=auth_mode,
                model_base_url=model_base_url,
            )
            _atomic_json(
                stage / "preflight.json",
                {"command": command.to_dict(), "preconditions": preconditions.to_dict()},
            )
            _write_integrity(stage)
            try:
                os.replace(stage, run_root)
            except FileExistsError:
                shutil.rmtree(stage)
                _verify_integrity(run_root)
                resumed = True

    final_job = run_root / "plan" / "jobs" / f"{compiled.jobs[0].job_name}.yaml"
    if agent in execution.CONTROL_SCAFFOLDS:
        command = execution.oragentbench_controls_command(source=source, job_config=final_job)
    else:
        command = execution.oragentbench_agent_command(
            source=source,
            job_config=final_job,
            required_env=validated.agents[0].secret_names,
        )
    preconditions = execution.oragentbench_preconditions(
        source=source,
        task_name=task,
        scaffold=agent,
        model=model,
        require_docker=False,
        require_harbor=False,
        require_secrets=False,
        auth_mode=auth_mode,
        model_base_url=model_base_url,
    )
    return PreparedRun(
        run_root=run_root,
        campaign_id=campaign_id,
        command=command,
        preconditions=preconditions,
        resumed=resumed,
        agent=agent,
        model=model,
        task=task,
        source=source,
        wall_clock_sec=wall_clock_sec,
        auth_mode=auth_mode,
        model_base_url=model_base_url,
    )


def execute_prepared_run(
    prepared: PreparedRun,
    *,
    acknowledge_cost: str = "",
    environ: Mapping[str, str] | None = None,
) -> HarborIngestResult:
    environ = dict(os.environ if environ is None else environ)
    paid = prepared.agent not in execution.CONTROL_SCAFFOLDS
    if paid and acknowledge_cost != "i-accept-model-costs":
        raise PreconditionError(
            "this path makes model calls; pass --acknowledge-cost i-accept-model-costs"
        )
    preconditions = execution.oragentbench_preconditions(
        source=prepared.source,
        task_name=prepared.task,
        scaffold=prepared.agent,
        model=prepared.model,
        environ=environ,
        require_docker=True,
        require_harbor=True,
        require_secrets=paid,
        auth_mode=prepared.auth_mode,
        model_base_url=prepared.model_base_url,
    )
    preconditions.raise_if_unmet("oragentbench run")

    with _campaign_lock(prepared.run_root.parent, prepared.campaign_id):
        manifest_path = prepared.run_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") == "completed":
            return ingest_harbor_bundle(
                run_root=prepared.run_root, jobs_root=prepared.run_root / "jobs"
            )
        manifest["state"] = "running"
        manifest["runner_pid"] = os.getpid()
        _atomic_json(manifest_path, manifest)
        logs = prepared.run_root / "logs"
        logs.mkdir(exist_ok=True)
        try:
            with (logs / "upstream.stdout.log").open("wb") as stdout, (
                logs / "upstream.stderr.log"
            ).open("wb") as stderr:
                exit_code = _run_process_group(
                    list(prepared.command.argv),
                    cwd=prepared.command.cwd,
                    environ=environ,
                    stdout=stdout,
                    stderr=stderr,
                    timeout_sec=prepared.wall_clock_sec,
                )
        except OSError as exc:
            manifest["state"] = "failed"
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(manifest_path, execution.sanitize_receipt(manifest))
            _write_integrity(prepared.run_root)
            raise PreconditionError(f"could not start upstream command: {exc}") from None

        receipt = execution.build_receipt(
            integration="oragentbench",
            mode="controls" if not paid else "agent",
            command=prepared.command,
            campaign_id=prepared.campaign_id,
            preconditions=preconditions,
            exit_code=exit_code,
            evidence_label="exploratory",
            output_root=str(prepared.run_root),
            notes=["raw Harbor bundle remains local; normalized report is derived after ingest"],
        )
        _atomic_json(prepared.run_root / "receipt.json", receipt)
        if exit_code != 0:
            manifest["state"] = "failed"
            manifest["exit_code"] = exit_code
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise PreconditionError(f"upstream benchmark exited with code {exit_code}")

        try:
            ingested = ingest_harbor_bundle(
                run_root=prepared.run_root, jobs_root=prepared.run_root / "jobs"
            )
        except EvidenceError:
            manifest["state"] = "failed"
            manifest["failure"] = "upstream exited successfully but result ingest failed"
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise
        manifest["state"] = "completed"
        manifest["exit_code"] = 0
        manifest.pop("runner_pid", None)
        _atomic_json(manifest_path, manifest)
        _write_integrity(prepared.run_root)
        return ingested
