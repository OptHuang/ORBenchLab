"""Product-level orchestration for a complete ORBenchLab run workspace.

The lower layers remain pure and benchmark-native.  This module is the small
piece that makes them usable together: inspect, plan, preflight, execute,
ingest, and report under one deterministic run directory.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from . import execution
from .campaign import compile as compile_mod
from .campaign import spec as spec_mod
from .core.errors import EvidenceError, PreconditionError, SpecError
from .core.urls import provider_route_digest, validate_https_base_url
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
    agent_id: str
    model: str
    task: str
    source: Path
    wall_clock_sec: int
    auth_mode: str
    model_base_url: str
    provider_route_digest: str | None = None
    dataset_digest: str = ""
    source_commit: str | None = None
    source_snapshot_digest: str = ""
    scaffold_version: str | None = None
    runtime_image_tag: str = ""
    job_name: str = ""


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


_SNAPSHOT_IGNORED_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", "ORAgentBench-trajectories"}
)
_SNAPSHOT_IGNORED_SUFFIXES = (".pyc", ".pyo")


def _source_snapshot_rows(source: Path) -> list[tuple[str, str, int]]:
    """Return the regular-file tree that may reach the upstream process."""
    rows: list[tuple[str, str, int]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in _SNAPSHOT_IGNORED_DIRS for part in relative.parts):
            continue
        if path.name == ".DS_Store" or path.name.endswith(_SNAPSHOT_IGNORED_SUFFIXES):
            continue
        if path.is_symlink():
            raise EvidenceError("ORAgentBench source snapshot input contains a symlink")
        if path.is_file():
            executable = 1 if path.stat().st_mode & 0o111 else 0
            rows.append((relative.as_posix(), _sha256(path), executable))
    if not rows:
        raise EvidenceError("ORAgentBench source snapshot would be empty")
    return rows


def _source_snapshot_digest(source: Path) -> str:
    digest = hashlib.sha256(b"orbench-source-snapshot.v1\0")
    for relative, content, executable in _source_snapshot_rows(source):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(executable).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _materialize_source_snapshot(source: Path, workspace: Path) -> tuple[Path, str]:
    """Create/reuse a content-addressed execution checkout."""
    rows = _source_snapshot_rows(source)
    digest_builder = hashlib.sha256(b"orbench-source-snapshot.v1\0")
    for relative, content, executable in rows:
        digest_builder.update(relative.encode("utf-8"))
        digest_builder.update(b"\0")
        digest_builder.update(content.encode("ascii"))
        digest_builder.update(b"\0")
        digest_builder.update(str(executable).encode("ascii"))
        digest_builder.update(b"\0")
    digest = f"sha256:{digest_builder.hexdigest()}"
    snapshots = workspace / ".source-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    destination_root = snapshots / digest.removeprefix("sha256:")
    destination = destination_root / execution.ORAGENTBENCH_CHECKOUT_DIRNAME
    if destination.exists():
        if _source_snapshot_digest(destination) != digest:
            raise EvidenceError("existing ORAgentBench source snapshot failed its content digest")
        _freeze_snapshot(destination)
        return destination, digest

    stage_root = snapshots / f".{digest.removeprefix('sha256:')}.tmp-{os.getpid()}"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage = stage_root / execution.ORAGENTBENCH_CHECKOUT_DIRNAME
    stage.mkdir(parents=True)
    try:
        for relative, _, executable in rows:
            source_file = source / relative
            target_file = stage / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target_file)
            target_file.chmod(0o755 if executable else 0o644)
        if _source_snapshot_digest(stage) != digest:
            raise EvidenceError("ORAgentBench source changed while its snapshot was copied")
        try:
            os.replace(stage_root, destination_root)
        except OSError:
            # A concurrent campaign may have won the same content-addressed
            # destination.  Accept only an exact snapshot and discard ours.
            if not destination.is_dir() or _source_snapshot_digest(destination) != digest:
                raise
            shutil.rmtree(stage_root)
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        raise
    _freeze_snapshot(destination)
    return destination, digest


def _freeze_snapshot(source: Path) -> None:
    harbor_tasks = source / "harbor_tasks"
    for path in source.rglob("*"):
        if path.is_file():
            if path.name == "task.toml" and path.is_relative_to(harbor_tasks):
                # The pinned paid wrapper copies a task with shutil.copytree
                # and then injects skills_dir into the copied task.toml.  The
                # copied file inherits this mode, so it must remain writable.
                path.chmod(0o644)
            else:
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    for directory in sorted(
        (path for path in source.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        # The same wrapper replaces environment/skills after copytree, whose
        # directory modes are inherited from this snapshot.  Keep only the
        # task package tree owner-writable; all content remains bound by the
        # source digest checked before and after execution.
        directory.chmod(0o755 if directory.is_relative_to(harbor_tasks) else 0o555)
    source.chmod(0o555)


def _write_integrity(run_root: Path) -> Path:
    target = run_root / "integrity.sha256"
    rows = []
    candidates = sorted(run_root.rglob("*"))
    symlinks = [path.relative_to(run_root).as_posix() for path in candidates if path.is_symlink()]
    if symlinks:
        raise EvidenceError(
            "run workspace contains symlink(s), which cannot be bound into evidence: "
            + ", ".join(symlinks)
        )
    for path in (p for p in candidates if p.is_file()):
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
    symlinks = [
        path.relative_to(run_root).as_posix()
        for path in sorted(run_root.rglob("*"))
        if path.is_symlink()
    ]
    if symlinks:
        raise EvidenceError(
            "run workspace contains symlink(s), which are not valid evidence: "
            + ", ".join(symlinks)
        )
    expected_files: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            raise EvidenceError(f"invalid integrity line in {manifest}: {line!r}") from None
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvidenceError("integrity manifest contains an unsafe relative path")
        if relative in expected_files:
            raise EvidenceError(f"integrity manifest contains duplicate path: {relative}")
        expected_files[relative] = expected

    # Trust the phase only after verifying the manifest that records it.  A
    # running process may legitimately create or update Harbor outputs before
    # a SIGKILL, but immutable plan/configuration files must still match.
    manifest_relative = "manifest.json"
    manifest_path = run_root / manifest_relative
    manifest_digest = expected_files.get(manifest_relative)
    if (
        manifest_digest is None
        or not manifest_path.is_file()
        or _sha256(manifest_path) != manifest_digest
    ):
        raise EvidenceError("existing run workspace failed integrity check: manifest.json")
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_data, dict):
            raise AttributeError
        phase = manifest_data.get("state")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        raise EvidenceError("existing run manifest is not a readable JSON object") from None
    if phase not in {"prepared", "running", "completed", "failed"}:
        raise EvidenceError(f"existing run manifest has unsupported state {phase!r}")

    job_name = manifest_data.get("job_name") if isinstance(manifest_data, dict) else None
    job_prefix = f"jobs/{job_name}/" if isinstance(job_name, str) and job_name else ""
    if phase == "running" and job_prefix:
        # config.json is byte-mutable because Harbor serializes set-valued retry
        # fields in arbitrary order. Once a resume binding exists, keep the
        # mutable byte layer tied to its canonical execution semantics here.
        binding_path = run_root / "resume-binding.json"
        config_path = run_root / job_prefix / "config.json"
        if binding_path.is_file():
            if binding_path.is_symlink() or config_path.is_symlink() or not config_path.is_file():
                raise EvidenceError("running Harbor resume binding has no safe config")
            try:
                binding = json.loads(binding_path.read_bytes())
                config_bytes = config_path.read_bytes()
                config = json.loads(config_bytes)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise EvidenceError("running Harbor resume evidence is unreadable") from None
            if not isinstance(binding, dict) or not isinstance(config, dict):
                raise EvidenceError("running Harbor resume evidence is invalid")
            version = binding.get("resume_binding_schema_version")
            if version == "3.0":
                bound = binding.get("config_semantic_sha256")
                current = _resume_config_semantic_sha256(config)
            elif version == "2.0":
                bound = binding.get("config_sha256")
                current = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
            else:
                raise EvidenceError("running Harbor resume binding schema is unsupported")
            if bound != current:
                raise EvidenceError("running Harbor config changed after resume binding")
    upstream_attempt = manifest_data.get("upstream_attempt")
    mutable_log_files: set[str] | None = None
    if upstream_attempt is not None:
        if not isinstance(upstream_attempt, dict):
            raise EvidenceError("existing run manifest has invalid upstream attempt evidence")
        attempt_id = upstream_attempt.get("id")
        stdout_log = upstream_attempt.get("stdout_log")
        stderr_log = upstream_attempt.get("stderr_log")
        receipt_path = upstream_attempt.get("receipt")
        if (
            not isinstance(attempt_id, str)
            or re.fullmatch(r"attempt-[0-9]{4,}", attempt_id) is None
            or stdout_log != f"logs/{attempt_id}/upstream.stdout.log"
            or stderr_log != f"logs/{attempt_id}/upstream.stderr.log"
            or receipt_path != f"logs/{attempt_id}/receipt.json"
        ):
            raise EvidenceError("existing run manifest has invalid upstream attempt evidence")
        mutable_log_files = {stdout_log, stderr_log, receipt_path}

    def running_mutable(relative: str) -> bool:
        if relative == "receipt.json":
            # This is only the canonical alias for the latest attempt-local
            # receipt.  It is refreshed atomically while the attempt is still
            # running; historical receipts remain immutable under logs/.
            return True
        if relative.startswith("logs/"):
            # Legacy workspaces used two mutable top-level files.  New runs
            # identify the current attempt explicitly, keeping every prior
            # attempt hash-bound even while a resumed process is running.
            return mutable_log_files is None or relative in mutable_log_files
        if not job_prefix or not relative.startswith(job_prefix):
            return False
        # Harbor may rewrite config.json after resolving set-valued fields. Its
        # bytes are therefore outside the ledger while running; the binding
        # check above (when present) and prepare/execute _bind calls enforce its
        # semantic contract. Trial/result files remain upstream-owned output.
        return True

    for relative, expected in expected_files.items():
        if relative == manifest_relative:
            continue
        # On crash recovery only upstream-owned output roots are mutable.  The
        # plan, preflight and identity evidence stay hash-checked in all phases.
        if phase == "running" and running_mutable(relative):
            continue
        path = run_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise EvidenceError(f"existing run workspace failed integrity check: {relative}")

    actual_files = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
        and path != manifest
        and ".tmp-" not in path.name
        and path.name != ".run.lock"
    }
    unexpected = sorted(actual_files - set(expected_files))
    if phase == "running":
        # config.json is first created by Harbor after the process starts, so
        # it may initially be absent from our ledger. prepare() validates and
        # binds it before selecting --resume. No other job-name subtree is
        # accepted.
        unexpected = [
            relative
            for relative in unexpected
            if not (
                running_mutable(relative)
                or (job_prefix and relative == f"{job_prefix}config.json")
            )
        ]
    if unexpected:
        raise EvidenceError(
            "existing run workspace has file(s) absent from integrity manifest: "
            + ", ".join(unexpected)
        )


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


_HOST_LOCK_ENV = "ORBENCH_HOST_LOCK_DIR"
_ORAGENTBENCH_DOCKER_LOCK = "oragentbench-docker-alias.lock"


def _open_host_lock_directory(environ: Mapping[str, str]) -> int:
    """Open a host-stable, user-owned directory for cross-workspace locks."""
    configured = str(environ.get(_HOST_LOCK_ENV, "")).strip()
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "orbenchlab" / "host-locks"
    )
    if not path.is_absolute():
        raise PreconditionError(f"{_HOST_LOCK_ENV} must be an absolute path")
    directory_fd: int | None = None
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise PreconditionError(
            "ORAgentBench host lock directory is unavailable or unsafe"
        ) from exc
    try:
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PreconditionError("ORAgentBench host lock path is not a directory")
        if metadata.st_uid != os.getuid():
            raise PreconditionError(
                "ORAgentBench host lock directory is not owned by the current user"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PreconditionError(
                "ORAgentBench host lock directory must not be accessible by group or others"
            )
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


@contextmanager
def _oragentbench_docker_alias_lock(
    *, environ: Mapping[str, str]
) -> Iterator[None]:
    """Fail closed within one runner account while upstream retags its alias.

    The upstream control and paid wrappers both build through a fixed base-image
    alias.  A campaign-local lock cannot protect that daemon-global Docker state,
    so every runner account uses one non-blocking, user-owned lock file.  A
    Docker daemon shared by different Unix accounts requires external
    serialization or, preferably, separate isolated daemons.
    """
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = _open_host_lock_directory(environ)
    try:
        file_fd = os.open(
            _ORAGENTBENCH_DOCKER_LOCK,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise PreconditionError(
            "ORAgentBench Docker alias lock is unavailable or unsafe"
        ) from exc
    finally:
        os.close(directory_fd)

    with os.fdopen(file_fd, "r+", encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise PreconditionError(
                "ORAgentBench Docker alias lock is not a current-user regular file"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PreconditionError(
                "ORAgentBench Docker alias lock must not be accessible by group or others"
            )
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreconditionError(
                "another ORAgentBench campaign is using the shared Docker image alias "
                "on this host; retry after that execution completes"
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
    value = result.stdout.strip() if result.returncode == 0 else ""
    return value or None


_PREBUILT_IMPORTS = {
    "claude-code": "ORAgentBench.harbor_agents.prebuilt_agents:PrebuiltClaudeCode",
    "codex": "ORAgentBench.harbor_agents.prebuilt_agents:PrebuiltCodex",
    "mini-swe-agent": "ORAgentBench.harbor_agents.prebuilt_agents:PrebuiltMiniSweAgent",
}

_HARBOR_TOP_LEVEL_DEFAULTS: dict[str, Any] = {
    "agent_setup_timeout_multiplier": None,
    "agent_timeout_multiplier": None,
    "artifacts": [],
    "debug": False,
    "environment_build_timeout_multiplier": None,
    "extra_instruction_paths": [],
    "install_only": False,
    "quiet": False,
    "verifier": {
        "override_timeout_sec": None,
        "max_timeout_sec": None,
        "env": {},
        "disable": False,
    },
    "verifier_timeout_multiplier": None,
}

_HARBOR_ENVIRONMENT_DEFAULTS: dict[str, Any] = {
    "import_path": None,
    "cpu_enforcement_policy": "auto",
    "memory_enforcement_policy": "auto",
    "override_cpus": None,
    "override_memory_mb": None,
    "override_storage_mb": None,
    "override_gpus": None,
    "override_tpu": None,
    "mounts": None,
    "extra_docker_compose": [],
    "env": {},
    "extra_allowed_hosts": [],
}

_HARBOR_RETRY_DEFAULTS: dict[str, Any] = {
    "wait_multiplier": 1.0,
    "min_wait_sec": 1.0,
    "max_wait_sec": 60.0,
}

_HARBOR_AGENT_DEFAULTS: dict[str, Any] = {
    "n_concurrent": None,
    "concurrency_group": None,
    "skills": [],
    "override_timeout_sec": None,
    "max_timeout_sec": None,
    "extra_allowed_hosts": [],
    "mcp_servers": [],
}

_HARBOR_DATASET_DEFAULTS: dict[str, Any] = {
    "name": None,
    "version": None,
    "ref": None,
    "registry_url": None,
    "registry_path": None,
    "repo": None,
    "overwrite": False,
    "download_dir": None,
    "n_tasks": None,
}


def _same_resume_value(actual: Any, expected: Any) -> bool:
    """Compare config scalars without accepting bool as an integer."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == float(expected)
        )
    return actual == expected


def _canonical_resume_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only Harbor fields whose JSON order is not semantic."""
    canonical = copy.deepcopy(dict(config))
    retry = canonical.get("retry")
    if not isinstance(retry, dict):
        raise EvidenceError("interrupted Harbor config has a conflicting retry policy")
    for field in ("include_exceptions", "exclude_exceptions"):
        value = retry.get(field)
        if value is None:
            continue
        if (
            not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
            or len(set(value)) != len(value)
        ):
            raise EvidenceError("interrupted Harbor config has a conflicting retry policy")
        retry[field] = sorted(value)
    return canonical


def _resume_config_semantic_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_resume_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"orbench-resume-config.v1\0" + payload).hexdigest()
    return f"sha256:{digest}"


def _resume_bindings_match(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Compare v3 bindings semantically and upgrade v2 only by raw proof."""
    identity_keys = {
        "dataset_content_digest",
        "job_name",
        "agent",
        "model",
        "scaffold_version",
        "task",
    }
    if any(existing.get(key) != current.get(key) for key in identity_keys):
        return False
    raw = existing.get("config_sha256")
    if not isinstance(raw, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", raw) is None:
        return False
    version = existing.get("resume_binding_schema_version")
    if version == "2.0":
        allowed = {"resume_binding_schema_version", "config_sha256", *identity_keys}
        return set(existing) == allowed and raw == current.get("config_sha256")
    if version != "3.0":
        return False
    allowed = {
        "resume_binding_schema_version",
        "config_sha256",
        "config_semantic_sha256",
        *identity_keys,
    }
    semantic = existing.get("config_semantic_sha256")
    return (
        set(existing) == allowed
        and isinstance(semantic, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", semantic) is not None
        and semantic == current.get("config_semantic_sha256")
    )


def _validate_mapping_with_harbor_defaults(
    actual: Any,
    expected: Any,
    defaults: Mapping[str, Any],
    *,
    area: str,
    unordered_string_lists: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise EvidenceError(f"interrupted Harbor config has a conflicting {area}")
    allowed = set(expected) | set(defaults)
    if set(actual) - allowed:
        raise EvidenceError(f"interrupted Harbor config has a conflicting {area}")

    def same(key: str, actual_value: Any, expected_value: Any) -> bool:
        if key not in unordered_string_lists:
            return _same_resume_value(actual_value, expected_value)
        if actual_value is None or expected_value is None:
            return actual_value is None and expected_value is None
        if not isinstance(actual_value, list) or not isinstance(expected_value, list):
            return False
        if not all(isinstance(value, str) for value in [*actual_value, *expected_value]):
            return False
        if len(set(actual_value)) != len(actual_value) or len(set(expected_value)) != len(
            expected_value
        ):
            return False
        return set(actual_value) == set(expected_value)

    for key, expected_value in expected.items():
        if key not in actual or not same(key, actual[key], expected_value):
            raise EvidenceError(f"interrupted Harbor config has a conflicting {area}")
    for key, default_value in defaults.items():
        if key in actual and not same(key, actual[key], default_value):
            raise EvidenceError(f"interrupted Harbor config has a conflicting {area}")
    return actual


def _tree_hashes(root: Path, *, excluded_prefix: str = "") -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("interrupted Harbor dataset is not a safe directory")
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if excluded_prefix and (
            relative == excluded_prefix or relative.startswith(f"{excluded_prefix}/")
        ):
            continue
        if path.is_symlink():
            raise EvidenceError("interrupted Harbor dataset contains a symlink")
        if path.is_file():
            rows[relative] = _sha256(path)
        elif not path.is_dir():
            raise EvidenceError("interrupted Harbor dataset contains a special file")
    return rows


def _toml_without_skills_dir(path: Path) -> tuple[dict[str, Any], Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("interrupted Harbor dataset has no safe task.toml")
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise EvidenceError("interrupted Harbor dataset has an invalid task.toml") from None
    environment = parsed.get("environment")
    if environment is None:
        environment_copy: dict[str, Any] = {}
    elif isinstance(environment, dict):
        environment_copy = dict(environment)
    else:
        raise EvidenceError("interrupted Harbor dataset has an invalid environment table")
    skills_dir = environment_copy.pop("skills_dir", None)
    parsed["environment"] = environment_copy
    return parsed, skills_dir


def _task_toml_with_default_skills(source_text: str) -> str:
    """Mirror the pinned wrapper's sole permitted task.toml rewrite."""
    lines = source_text.splitlines()
    target_line = 'skills_dir = "/skills"'
    environment_start: int | None = None
    environment_end = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[environment]":
            environment_start = index
            continue
        if (
            environment_start is not None
            and index > environment_start
            and stripped.startswith("[")
            and stripped.endswith("]")
        ):
            environment_end = index
            break
    if environment_start is None:
        lines.extend(["", "[environment]", target_line])
    else:
        for index in range(environment_start + 1, environment_end):
            if lines[index].strip().startswith("skills_dir"):
                lines[index] = target_line
                break
        else:
            lines.insert(environment_start + 1, target_line)
    return "\n".join(lines) + "\n"


def _default_snapshot_skill_dirs(source: Path) -> list[Path]:
    grouped = source / "skills" / "base-Skills"
    root = grouped if grouped.is_dir() else source / "skills"
    if root.is_dir() and (root / "SKILL.md").is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "SKILL.md").is_file()
        ]
    else:
        candidates = []
    if not candidates:
        raise EvidenceError("interrupted Harbor dataset has no snapshot skill source")
    return candidates


def _validate_dynamic_task_copy(
    *, source: Path, dataset_path: Path, task: str
) -> str:
    source_task = source / "harbor_tasks" / task
    runtime_task = dataset_path / task
    if not source_task.is_dir() or not runtime_task.is_dir() or runtime_task.is_symlink():
        raise EvidenceError("interrupted Harbor dataset has a conflicting task copy")

    source_toml_path = source_task / "task.toml"
    runtime_toml_path = runtime_task / "task.toml"
    source_toml, _ = _toml_without_skills_dir(source_toml_path)
    runtime_toml, runtime_skills_dir = _toml_without_skills_dir(runtime_toml_path)
    try:
        expected_runtime_text = _task_toml_with_default_skills(
            source_toml_path.read_text(encoding="utf-8")
        )
        runtime_text = runtime_toml_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise EvidenceError("interrupted Harbor dataset has an unreadable task.toml") from None
    if (
        runtime_toml != source_toml
        or runtime_skills_dir != "/skills"
        or runtime_text != expected_runtime_text
    ):
        raise EvidenceError("interrupted Harbor dataset changed task.toml beyond skills_dir")

    excluded = "environment/skills"
    source_files = _tree_hashes(source_task, excluded_prefix=excluded)
    runtime_files = _tree_hashes(runtime_task, excluded_prefix=excluded)
    source_files.pop("task.toml", None)
    runtime_files.pop("task.toml", None)
    if runtime_files != source_files:
        raise EvidenceError("interrupted Harbor dataset changed the selected task content")

    expected_skills: dict[str, str] = {}
    for skill_dir in _default_snapshot_skill_dirs(source):
        for relative, digest in _tree_hashes(skill_dir).items():
            expected_skills[f"{skill_dir.name}/{relative}"] = digest
    actual_skills = _tree_hashes(runtime_task / excluded)
    if actual_skills != expected_skills:
        raise EvidenceError("interrupted Harbor dataset has conflicting injected skills")

    digest = hashlib.sha256(b"orbench-resume-dataset.v1\0")
    for relative, content in sorted(
        {
            **runtime_files,
            "task.toml": _sha256(runtime_task / "task.toml"),
            **{f"{excluded}/{key}": value for key, value in actual_skills.items()},
        }.items()
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _validate_resume_dataset_path(
    *, source: Path, agent: str, task: str, raw_path: Any
) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError("interrupted Harbor config has a conflicting dataset path")
    source_dataset = source / "harbor_tasks"
    candidate = Path(raw_path).expanduser()
    if agent in execution.CONTROL_SCAFFOLDS:
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (source.parent / candidate).resolve()
        )
        if resolved != source_dataset.resolve():
            raise EvidenceError("interrupted Harbor config has a conflicting dataset path")
        task_root = resolved / task
        files = _tree_hashes(task_root)
        digest = hashlib.sha256(b"orbench-resume-dataset.v1\0")
        for relative, content in sorted(files.items()):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content.encode("ascii"))
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"

    if not candidate.is_absolute():
        raise EvidenceError("interrupted Harbor config has a conflicting dataset path")
    try:
        resolved = candidate.resolve(strict=True)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    except OSError:
        raise EvidenceError("interrupted Harbor config has an unavailable dataset path") from None
    dynamic_root = resolved.parent
    if (
        resolved.name != source_dataset.name
        or dynamic_root.parent != temp_root
        or re.fullmatch(r"oragentbench-skills-[A-Za-z0-9_.-]+", dynamic_root.name)
        is None
        or dynamic_root.is_symlink()
        or candidate.is_symlink()
    ):
        raise EvidenceError("interrupted Harbor config has a conflicting dataset path")
    metadata = dynamic_root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise EvidenceError("interrupted Harbor config has an unsafe dataset path")
    entries = sorted(path.name for path in resolved.iterdir())
    if entries != [task]:
        raise EvidenceError("interrupted Harbor dataset has a conflicting task selection")
    return _validate_dynamic_task_copy(
        source=source, dataset_path=resolved, task=task
    )


def _validated_resume_binding(
    *,
    run_root: Path,
    source: Path,
    job_name: str,
    agent: str,
    model: str,
    scaffold_version: str | None,
    task: str,
) -> dict[str, Any]:
    """Bind every execution-relevant field that controls Harbor ``--resume``."""
    config_path = run_root / "jobs" / job_name / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise EvidenceError("interrupted Harbor job has no safe resumable config")
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError("interrupted Harbor config is not readable JSON") from None
    if not isinstance(config, dict):
        raise EvidenceError("interrupted Harbor config is not a JSON object")

    plan_path = run_root / "plan" / "jobs" / f"{job_name}.yaml"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise EvidenceError("interrupted Harbor job has no safe compiled config")
    try:
        expected = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise EvidenceError("compiled Harbor config is not readable YAML") from None
    if not isinstance(expected, dict):
        raise EvidenceError("compiled Harbor config is not a YAML object")

    allowed_top_level = (
        set(expected) - {"pre_build", "skills"}
    ) | set(_HARBOR_TOP_LEVEL_DEFAULTS)
    if set(config) - allowed_top_level:
        raise EvidenceError("interrupted Harbor config has unsupported execution fields")
    for key, default_value in _HARBOR_TOP_LEVEL_DEFAULTS.items():
        if key in config and not _same_resume_value(config[key], default_value):
            raise EvidenceError("interrupted Harbor config conflicts with Harbor defaults")
    if config.get("job_name") != job_name:
        raise EvidenceError("interrupted Harbor config has a conflicting job name")
    for key, area in (
        ("n_attempts", "attempt count"),
        ("n_concurrent_trials", "concurrency"),
        ("timeout_multiplier", "timeout"),
    ):
        if key not in config or not _same_resume_value(config[key], expected.get(key)):
            raise EvidenceError(f"interrupted Harbor config has a conflicting {area}")
    jobs_dir = config.get("jobs_dir")
    expected_jobs_dir = expected.get("jobs_dir")
    if (
        not isinstance(jobs_dir, str)
        or not isinstance(expected_jobs_dir, str)
        or Path(jobs_dir).resolve() != (run_root / "jobs").resolve()
        or Path(expected_jobs_dir).resolve() != (run_root / "jobs").resolve()
    ):
        raise EvidenceError("interrupted Harbor config has a conflicting jobs directory")

    _validate_mapping_with_harbor_defaults(
        config.get("environment"),
        expected.get("environment"),
        _HARBOR_ENVIRONMENT_DEFAULTS,
        area="environment",
    )
    _validate_mapping_with_harbor_defaults(
        config.get("retry"),
        expected.get("retry"),
        _HARBOR_RETRY_DEFAULTS,
        area="retry policy",
        unordered_string_lists=frozenset(
            {"include_exceptions", "exclude_exceptions"}
        ),
    )
    if config.get("metrics") != expected.get("metrics"):
        raise EvidenceError("interrupted Harbor config has a conflicting metric set")
    if config.get("tasks") != expected.get("tasks"):
        raise EvidenceError("interrupted Harbor config has a conflicting task set")

    agents = config.get("agents")
    expected_agents = expected.get("agents")
    if not isinstance(agents, list) or len(agents) != 1 or not isinstance(agents[0], dict):
        raise EvidenceError("interrupted Harbor config has a conflicting agent set")
    if (
        not isinstance(expected_agents, list)
        or len(expected_agents) != 1
        or not isinstance(expected_agents[0], dict)
    ):
        raise EvidenceError("compiled Harbor config has a conflicting agent set")
    actual_agent = agents[0]
    expected_agent = expected_agents[0]
    allowed_agent_keys = set(expected_agent) | set(_HARBOR_AGENT_DEFAULTS)
    if set(actual_agent) - allowed_agent_keys:
        raise EvidenceError("interrupted Harbor config has a conflicting agent identity")
    for key, default_value in _HARBOR_AGENT_DEFAULTS.items():
        if key in actual_agent and not _same_resume_value(
            actual_agent[key], default_value
        ):
            raise EvidenceError("interrupted Harbor config has a conflicting agent identity")

    if agent in execution.CONTROL_SCAFFOLDS:
        expected_name = agent
        expected_import = None
    else:
        expected_name = None
        expected_import = _PREBUILT_IMPORTS.get(agent)
    expected_env = expected_agent.get("env") or {}
    actual_env = actual_agent.get("env") or {}
    expected_kwargs = expected_agent.get("kwargs") or {}
    actual_kwargs = actual_agent.get("kwargs") or {}
    expected_env_keys = set(expected_env)
    if agent == "claude-code":
        expected_env_keys.add("CLAUDE_CODE_ATTRIBUTION_HEADER")
    expected_runtime_env = dict(expected_env)
    if agent == "claude-code":
        expected_runtime_env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    agent_ok = (
        (expected_import is not None or agent in execution.CONTROL_SCAFFOLDS)
        and actual_agent.get("name") == expected_name
        and actual_agent.get("import_path") == expected_import
        and actual_agent.get("model_name") == (model or None)
        and _same_resume_value(
            actual_agent.get("override_setup_timeout_sec"),
            expected_agent.get("override_setup_timeout_sec"),
        )
        and isinstance(actual_env, dict)
        and set(actual_env) == expected_env_keys
        and actual_env == expected_runtime_env
        and isinstance(actual_kwargs, dict)
        and actual_kwargs == expected_kwargs
    )
    if agent not in execution.CONTROL_SCAFFOLDS:
        agent_ok = agent_ok and actual_kwargs.get("version") == scaffold_version
    if not agent_ok:
        raise EvidenceError("interrupted Harbor config has a conflicting agent identity")

    datasets = config.get("datasets")
    expected_datasets = expected.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise EvidenceError("interrupted Harbor config has a conflicting dataset set")
    if (
        not isinstance(expected_datasets, list)
        or len(expected_datasets) != 1
        or not isinstance(expected_datasets[0], dict)
    ):
        raise EvidenceError("compiled Harbor config has a conflicting dataset set")
    actual_dataset = datasets[0]
    expected_dataset = expected_datasets[0]
    allowed_dataset_keys = set(expected_dataset) | set(_HARBOR_DATASET_DEFAULTS)
    if set(actual_dataset) - allowed_dataset_keys:
        raise EvidenceError("interrupted Harbor config has a conflicting dataset set")
    for key, default_value in _HARBOR_DATASET_DEFAULTS.items():
        if key in actual_dataset and not _same_resume_value(
            actual_dataset[key], default_value
        ):
            raise EvidenceError("interrupted Harbor config has a conflicting dataset set")
    if (
        actual_dataset.get("task_names") != [task]
        or actual_dataset.get("exclude_task_names") is not None
        or expected_dataset.get("task_names") != [task]
        or expected_dataset.get("exclude_task_names") is not None
    ):
        raise EvidenceError("interrupted Harbor config has a conflicting task selection")
    dataset_content_digest = _validate_resume_dataset_path(
        source=source,
        agent=agent,
        task=task,
        raw_path=actual_dataset.get("path"),
    )
    return {
        "resume_binding_schema_version": "3.0",
        "config_sha256": f"sha256:{hashlib.sha256(config_bytes).hexdigest()}",
        "config_semantic_sha256": _resume_config_semantic_sha256(config),
        "dataset_content_digest": dataset_content_digest,
        "job_name": job_name,
        "agent": agent,
        "model": model or None,
        "scaffold_version": scaffold_version,
        "task": task,
    }


def _bind_resume_config(**kwargs: Any) -> dict[str, Any]:
    run_root = Path(kwargs["run_root"])
    binding = _validated_resume_binding(**kwargs)
    path = run_root / "resume-binding.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise EvidenceError("resume binding is unreadable") from None
        if not isinstance(existing, dict) or not _resume_bindings_match(existing, binding):
            raise EvidenceError("interrupted Harbor config changed after resume binding")
        if existing.get("resume_binding_schema_version") == "2.0":
            _atomic_json(path, binding)
    else:
        _atomic_json(path, binding)
    return binding


def _allocate_upstream_attempt_logs(
    run_root: Path, *, starting_number: int = 1
) -> tuple[dict[str, str], Path, Path]:
    """Reserve a new append-only log namespace under the campaign lock."""
    logs = run_root / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    number = starting_number
    while True:
        attempt_id = f"attempt-{number:04d}"
        attempt_dir = logs / attempt_id
        try:
            attempt_dir.mkdir(mode=0o700)
        except FileExistsError:
            number += 1
            continue
        stdout_path = attempt_dir / "upstream.stdout.log"
        stderr_path = attempt_dir / "upstream.stderr.log"
        record = {
            "id": attempt_id,
            "stdout_log": stdout_path.relative_to(run_root).as_posix(),
            "stderr_log": stderr_path.relative_to(run_root).as_posix(),
            "receipt": (attempt_dir / "receipt.json").relative_to(run_root).as_posix(),
        }
        return record, stdout_path, stderr_path


def _atomic_copy_file(source: Path, target: Path) -> None:
    """Publish an exact byte copy without exposing a partial evidence file."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _sha256(target) != _sha256(source):
            raise EvidenceError("existing legacy attempt receipt conflicts with canonical receipt")
        return
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legacy_upstream_attempt(
    run_root: Path, *, archive_canonical: bool
) -> dict[str, str | None] | None:
    """Describe pre-attempt-layout logs without moving or rewriting them."""
    logs = run_root / "logs"
    stdout_path = logs / "upstream.stdout.log"
    stderr_path = logs / "upstream.stderr.log"
    if not stdout_path.is_file() and not stderr_path.is_file():
        return None
    canonical_receipt = run_root / "receipt.json"
    archived_receipt = logs / "legacy-attempt-0001" / "receipt.json"
    if archive_canonical and canonical_receipt.is_file():
        _atomic_copy_file(canonical_receipt, archived_receipt)
    receipt = (
        archived_receipt.relative_to(run_root).as_posix()
        if archived_receipt.is_file()
        else None
    )
    return {
        "id": "legacy-attempt-0001",
        "stdout_log": "logs/upstream.stdout.log" if stdout_path.is_file() else None,
        "stderr_log": "logs/upstream.stderr.log" if stderr_path.is_file() else None,
        "receipt": receipt,
    }


def _write_upstream_attempt_receipt(
    *,
    prepared: PreparedRun,
    preconditions: execution.PreconditionReport,
    upstream_attempt: Mapping[str, str],
    paid: bool,
    exit_code: int | None,
    runtime_image: Mapping[str, Any] | None = None,
    runtime_image_evidence: str | None = None,
    alias_verification: Mapping[str, Any] | None = None,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Atomically persist the attempt receipt, then refresh its canonical alias."""
    receipt = execution.build_receipt(
        integration="oragentbench",
        mode="agent" if paid else "controls",
        command=prepared.command,
        campaign_id=prepared.campaign_id,
        preconditions=preconditions,
        exit_code=exit_code,
        evidence_label="exploratory",
        output_root=str(prepared.run_root),
        notes=notes,
    )
    receipt["scaffold_version"] = prepared.scaffold_version
    receipt["agent_id"] = prepared.agent_id
    receipt["source_snapshot_digest"] = prepared.source_snapshot_digest
    receipt["runtime_image"] = runtime_image
    receipt["runtime_image_evidence"] = runtime_image_evidence
    receipt["runtime_image_alias_verification"] = alias_verification
    receipt["upstream_attempt"] = dict(upstream_attempt)
    receipt = execution.sanitize_receipt(receipt)
    attempt_receipt = prepared.run_root / upstream_attempt["receipt"]
    _atomic_json(attempt_receipt, receipt)
    _atomic_json(prepared.run_root / "receipt.json", receipt)
    return receipt


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


def _command_environment(
    command: execution.UpstreamCommand, environ: Mapping[str, str]
) -> dict[str, str]:
    merged = dict(environ)
    for name, value in command.env_overrides.items():
        if name == "PYTHONPATH" and merged.get(name):
            merged[name] = value + os.pathsep + merged[name]
        else:
            merged[name] = value
    return merged


def _is_oragentbench_resume_command(command: execution.UpstreamCommand) -> bool:
    """Recognize both upstream agent and control recovery entry points."""
    argv = command.argv
    if "--resume" in argv:
        return True
    return (
        len(argv) >= 4
        and argv[0:2] == ("bash", "-c")
        and argv[3] == "orbench-control-resume"
        and "exec harbor job resume --job-path" in argv[2]
    )


def _resume_runtime_image_id(run_root: Path, job_dir: Path) -> str | None:
    """Return the image behind existing trials, or None before any trial began."""
    try:
        manifest = json.loads(
            (run_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError("interrupted job manifest is not readable JSON") from None
    runtime_image = manifest.get("runtime_image") if isinstance(manifest, dict) else None
    image_id = runtime_image.get("image_id") if isinstance(runtime_image, dict) else None
    if image_id is None:
        try:
            has_trial_directory = any(child.is_dir() for child in job_dir.iterdir())
        except OSError:
            raise EvidenceError(
                "interrupted Harbor job directory cannot be inspected safely"
            ) from None
        if not has_trial_directory:
            return None
        raise EvidenceError(
            "interrupted Harbor job has trial directories but no valid immutable "
            "Docker image ID; refusing to mix trials with an unknown runtime"
        )
    if not isinstance(image_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ) is None:
        raise EvidenceError("interrupted Harbor runtime image evidence is inconsistent")
    alias = manifest.get("runtime_image_alias_verification")
    evidence_is_consistent = (
        isinstance(runtime_image, dict)
        and runtime_image.get("requested_tag")
        == execution.ORAGENTBENCH_FIXED_BASE_IMAGE
        and isinstance(alias, dict)
        and alias.get("fixed_alias") == execution.ORAGENTBENCH_FIXED_BASE_IMAGE
        and alias.get("fixed_alias_image_id") == image_id
        and alias.get("matches_runtime_image") is True
    )
    if not evidence_is_consistent:
        raise EvidenceError("interrupted Harbor runtime image evidence is inconsistent")
    return image_id


def prepare_oragentbench_run(
    *,
    source: str | Path,
    task: str,
    agent: str,
    model: str,
    scaffold_version: str = "",
    date: str,
    workspace: str | Path,
    wall_clock_sec: int = 2700,
    max_cost_usd: float = 25.0,
    auth_mode: str = "api-key",
    model_base_url: str = "",
) -> PreparedRun:
    route_digest: str | None = None
    if agent in execution.ROUTE_PINNED_SCAFFOLDS and auth_mode != "codex-login":
        if not model_base_url:
            raise SpecError(f"scaffold {agent!r} requires a pinned HTTPS provider route")
        model_base_url = validate_https_base_url(model_base_url)
        route_digest = provider_route_digest(model_base_url)
    elif model_base_url:
        model_base_url = validate_https_base_url(model_base_url)
        route_digest = provider_route_digest(model_base_url)

    original_source = execution.validate_oragentbench_source(Path(source))
    execution.validate_oragentbench_task(original_source, task)
    inspection = registry.inspect("oragentbench", original_source)
    if inspection.status == "failed":
        failures = "; ".join(
            f"{check.id}: {check.detail}" for check in inspection.failures()
        )
        raise PreconditionError(f"ORAgentBench inspection failed: {failures}")
    digest = str(inspection.facts["dataset_digest"])
    source_commit = _git_head(original_source)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source, snapshot_digest = _materialize_source_snapshot(original_source, workspace)
    snapshot_inspection = registry.inspect("oragentbench", source)
    if snapshot_inspection.status == "failed" or str(
        snapshot_inspection.facts.get("dataset_digest") or ""
    ) != digest:
        raise EvidenceError(
            "content-addressed ORAgentBench source snapshot failed identity verification"
        )

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
            scaffold_version=scaffold_version,
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

    job_name = compiled.jobs[0].job_name
    runtime_image_tag = (
        execution.ORAGENTBENCH_FIXED_BASE_IMAGE
        if agent in execution.CONTROL_SCAFFOLDS
        else str(validated.harbor["pre_build"]["image_tag"])
    )

    with _campaign_lock(workspace, campaign_id):
        resumed = run_root.exists()
        if resumed:
            _verify_integrity(run_root)
            manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
            expected = {
                "campaign_id": campaign_id,
                "job_config_contract_version": compiled.job_config_contract_version,
                "integration": "oragentbench",
                "task": task,
                "agent": agent,
                "agent_id": validated.agents[0].id,
                "model": model or None,
                "date": date,
                "source_commit": source_commit,
                "source_snapshot_digest": snapshot_digest,
                "scaffold_version": (
                    scaffold_version if agent not in execution.CONTROL_SCAFFOLDS else None
                ),
                "auth_mode": auth_mode if agent not in execution.CONTROL_SCAFFOLDS else None,
                "provider_route_digest": route_digest,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise EvidenceError(
                        f"existing run {run_root} has conflicting {key}: "
                        f"{manifest.get(key)!r} != {value!r}"
                    )
            resume_config = run_root / "jobs" / job_name / "config.json"
            if manifest.get("state") == "running" and not resume_config.is_file():
                raise EvidenceError(
                    "interrupted Harbor job has no config.json; refusing to start a duplicate job"
                )
            if manifest.get("state") in {"running", "failed"} and resume_config.is_file():
                _bind_resume_config(
                    run_root=run_root,
                    source=source,
                    job_name=job_name,
                    agent=agent,
                    model=model,
                    scaffold_version=(
                        scaffold_version
                        if agent not in execution.CONTROL_SCAFFOLDS
                        else None
                    ),
                    task=task,
                )
                _write_integrity(run_root)
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
                    "job_config_contract_version": compiled.job_config_contract_version,
                    "integration": "oragentbench",
                    "task": task,
                    "agent": agent,
                    "agent_id": validated.agents[0].id,
                    "model": model or None,
                    "date": date,
                    "source": str(source),
                    "source_commit": source_commit,
                    "source_snapshot_digest": snapshot_digest,
                    "scaffold_version": (
                        scaffold_version if agent not in execution.CONTROL_SCAFFOLDS else None
                    ),
                    "runtime_image_tag": runtime_image_tag,
                    "job_name": job_name,
                    "auth_mode": auth_mode if agent not in execution.CONTROL_SCAFFOLDS else None,
                    "provider_route_digest": route_digest,
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
    upstream_job_dir = run_root / "jobs" / compiled.jobs[0].job_name
    resume_upstream = (upstream_job_dir / "config.json").is_file()
    if agent in execution.CONTROL_SCAFFOLDS:
        expected_image_id: str | None = None
        if resume_upstream:
            with _campaign_lock(workspace, campaign_id):
                _verify_integrity(run_root)
                expected_image_id = _resume_runtime_image_id(
                    run_root, upstream_job_dir
                )
        command = (
            execution.oragentbench_controls_resume_command(
                source=source,
                job_dir=upstream_job_dir,
                expected_image_id=expected_image_id,
            )
            if resume_upstream
            else execution.oragentbench_controls_command(
                source=source, job_config=final_job
            )
        )
    else:
        command = execution.oragentbench_agent_command(
            source=source,
            job_config=final_job,
            required_env=validated.agents[0].secret_names,
            resume=resume_upstream,
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
        agent_id=validated.agents[0].id,
        model=model,
        task=task,
        source=source,
        wall_clock_sec=wall_clock_sec,
        auth_mode=auth_mode,
        model_base_url=model_base_url,
        provider_route_digest=route_digest,
        dataset_digest=digest,
        source_commit=source_commit,
        source_snapshot_digest=snapshot_digest,
        scaffold_version=(
            scaffold_version if agent not in execution.CONTROL_SCAFFOLDS else None
        ),
        runtime_image_tag=runtime_image_tag,
        job_name=job_name,
    )


def execute_prepared_run(
    prepared: PreparedRun,
    *,
    acknowledge_cost: str = "",
    environ: Mapping[str, str] | None = None,
) -> HarborIngestResult:
    environ = dict(os.environ if environ is None else environ)
    if prepared.auth_mode in execution.CODEX_AUTH_FILE_MODES:
        raise PreconditionError(
            "direct auth-file benchmark execution is disabled until a host-side "
            "credential broker keeps the personal login outside the task boundary; "
            "codex-login and legacy codex-auth-json are prepare/doctor only, so use "
            "the api-key route for rollout"
        )
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
        # Preparation and execution are separate calls.  Re-check under the
        # campaign lock so no mutation in that gap can reach the upstream
        # process or alter resume behavior without being detected.
        _verify_integrity(prepared.run_root)
        manifest_path = prepared.run_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prepared.dataset_digest:
            if manifest.get("dataset_digest") != prepared.dataset_digest:
                raise EvidenceError(
                    "prepared run identity conflicts with its integrity-protected manifest"
                )
            if manifest.get("source_commit") != prepared.source_commit:
                raise EvidenceError(
                    "prepared checkout commit conflicts with its integrity-protected manifest"
                )
        if manifest.get("source_snapshot_digest") != prepared.source_snapshot_digest:
            raise EvidenceError(
                "prepared source snapshot conflicts with its integrity-protected manifest"
            )
        if manifest.get("provider_route_digest") != prepared.provider_route_digest:
            raise EvidenceError(
                "prepared provider route conflicts with its integrity-protected manifest"
            )
        if manifest.get("state") == "completed":
            ingested = ingest_harbor_bundle(
                run_root=prepared.run_root, jobs_root=prepared.run_root / "jobs"
            )
            _write_integrity(prepared.run_root)
            return ingested
        if _is_oragentbench_resume_command(prepared.command):
            _bind_resume_config(
                run_root=prepared.run_root,
                source=prepared.source,
                job_name=prepared.job_name,
                agent=prepared.agent,
                model=prepared.model,
                scaffold_version=prepared.scaffold_version,
                task=prepared.task,
            )

        # The command never reads the operator checkout. It runs from a
        # content-addressed workspace snapshot and binds that exact
        # tree again immediately before launch.
        if _source_snapshot_digest(prepared.source) != manifest.get(
            "source_snapshot_digest"
        ):
            raise EvidenceError("ORAgentBench execution snapshot changed after preparation")
        current_inspection = registry.inspect("oragentbench", prepared.source)
        if current_inspection.status == "failed":
            failed_checks = ", ".join(
                check.id for check in current_inspection.failures()
            )
            raise EvidenceError(
                "ORAgentBench checkout no longer passes inspection before execution"
                + (f" ({failed_checks})" if failed_checks else "")
            )
        current_digest = str(current_inspection.facts.get("dataset_digest") or "")
        if not current_digest or current_digest != manifest.get("dataset_digest"):
            raise EvidenceError(
                "ORAgentBench checkout content changed after campaign preparation; "
                "prepare a new campaign identity"
            )
        # Both upstream wrappers build through a fixed Docker alias.  Hold one
        # host-wide lock from before any upstream process can retag it until we
        # have captured the post-run immutable image identity.  Campaign locks
        # alone are insufficient because campaigns may use different roots.
        with _oragentbench_docker_alias_lock(environ=environ):
            prior_attempts = manifest.get("upstream_attempts", [])
            if not isinstance(prior_attempts, list) or not all(
                isinstance(item, dict) for item in prior_attempts
            ):
                raise EvidenceError("run manifest has invalid upstream attempt history")
            legacy_attempt = _legacy_upstream_attempt(
                prepared.run_root, archive_canonical=not prior_attempts
            )
            if legacy_attempt is not None and not prior_attempts:
                prior_attempts = [legacy_attempt]
            upstream_attempt, stdout_path, stderr_path = _allocate_upstream_attempt_logs(
                prepared.run_root,
                starting_number=2 if legacy_attempt is not None else 1,
            )
            manifest["upstream_attempts"] = [*prior_attempts, upstream_attempt]
            manifest["upstream_attempt"] = upstream_attempt
            manifest["state"] = "running"
            manifest["runner_pid"] = os.getpid()
            _atomic_json(manifest_path, manifest)
            # State is mutable evidence too.  Refresh before the upstream process
            # starts so a power loss/SIGKILL leaves a resumable, integrity-valid
            # workspace instead of permanently bricking the campaign.
            _write_integrity(prepared.run_root)
            try:
                with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                    exit_code = _run_process_group(
                        list(prepared.command.argv),
                        cwd=prepared.command.cwd,
                        environ=_command_environment(prepared.command, environ),
                        stdout=stdout,
                        stderr=stderr,
                        timeout_sec=prepared.wall_clock_sec,
                    )
            except OSError as exc:
                _write_upstream_attempt_receipt(
                    prepared=prepared,
                    preconditions=preconditions,
                    upstream_attempt=upstream_attempt,
                    paid=paid,
                    exit_code=None,
                    notes=("upstream command could not be started",),
                )
                manifest["state"] = "failed"
                manifest["failure"] = f"{type(exc).__name__}: {exc}"
                manifest.pop("runner_pid", None)
                _atomic_json(manifest_path, execution.sanitize_receipt(manifest))
                _write_integrity(prepared.run_root)
                raise PreconditionError(f"could not start upstream command: {exc}") from None

            # A resume config has a stricter domain contract than the general
            # workspace ledger. Validate it first so a changed Harbor field is
            # diagnosed precisely rather than flattened into a hash mismatch.
            if _is_oragentbench_resume_command(prepared.command):
                try:
                    _bind_resume_config(
                        run_root=prepared.run_root,
                        source=prepared.source,
                        job_name=prepared.job_name,
                        agent=prepared.agent,
                        model=prepared.model,
                        scaffold_version=prepared.scaffold_version,
                        task=prepared.task,
                    )
                except EvidenceError:
                    _write_upstream_attempt_receipt(
                        prepared=prepared,
                        preconditions=preconditions,
                        upstream_attempt=upstream_attempt,
                        paid=paid,
                        exit_code=exit_code,
                        notes=("interrupted Harbor resume identity changed",),
                    )
                    manifest["state"] = "failed"
                    manifest["failure"] = (
                        "interrupted Harbor resume identity changed during execution"
                    )
                    manifest.pop("runner_pid", None)
                    _atomic_json(manifest_path, manifest)
                    _write_integrity(prepared.run_root)
                    raise

            # The ledger was refreshed after selecting the current attempt but
            # before its files were opened.  In the running phase only that
            # attempt's stdout, stderr and receipt are mutable; this check
            # prevents a wrapper from rewriting older evidence and having that
            # rewrite blessed below.
            try:
                _verify_integrity(prepared.run_root)
            except EvidenceError:
                _write_upstream_attempt_receipt(
                    prepared=prepared,
                    preconditions=preconditions,
                    upstream_attempt=upstream_attempt,
                    paid=paid,
                    exit_code=exit_code,
                    notes=("upstream changed integrity-bound run inputs",),
                )
                manifest["state"] = "failed"
                manifest["failure"] = (
                    "upstream changed integrity-bound run inputs during execution"
                )
                manifest.pop("runner_pid", None)
                _atomic_json(manifest_path, manifest)
                _write_integrity(prepared.run_root)
                raise

            if _source_snapshot_digest(prepared.source) != manifest.get(
                "source_snapshot_digest"
            ):
                _write_upstream_attempt_receipt(
                    prepared=prepared,
                    preconditions=preconditions,
                    upstream_attempt=upstream_attempt,
                    paid=paid,
                    exit_code=exit_code,
                    notes=("content-addressed source snapshot changed",),
                )
                manifest["state"] = "failed"
                manifest["failure"] = (
                    "content-addressed source snapshot changed during execution"
                )
                manifest.pop("runner_pid", None)
                _atomic_json(manifest_path, manifest)
                _write_integrity(prepared.run_root)
                raise EvidenceError(
                    "ORAgentBench execution snapshot changed during execution"
                )
            runtime_image = execution.docker_image_fingerprint(
                prepared.runtime_image_tag
            )
            alias_image = (
                runtime_image
                if prepared.runtime_image_tag
                == execution.ORAGENTBENCH_FIXED_BASE_IMAGE
                else execution.docker_image_fingerprint(
                    execution.ORAGENTBENCH_FIXED_BASE_IMAGE
                )
            )
            runtime_image_id = (
                runtime_image.get("image_id")
                if isinstance(runtime_image, Mapping)
                else None
            )
            alias_image_id = (
                alias_image.get("image_id")
                if isinstance(alias_image, Mapping)
                else None
            )
            alias_verification = {
                "fixed_alias": execution.ORAGENTBENCH_FIXED_BASE_IMAGE,
                "fixed_alias_image_id": alias_image_id,
                "matches_runtime_image": bool(
                    runtime_image_id
                    and alias_image_id
                    and runtime_image_id == alias_image_id
                ),
            }
            manifest["runtime_image"] = runtime_image
            manifest["runtime_image_evidence"] = (
                "docker-image-inspect"
                if runtime_image is not None
                else "unavailable; run is not independently bound to a Docker image ID"
            )
            manifest["runtime_image_alias_verification"] = alias_verification

        _write_upstream_attempt_receipt(
            prepared=prepared,
            preconditions=preconditions,
            upstream_attempt=upstream_attempt,
            paid=paid,
            exit_code=exit_code,
            runtime_image=runtime_image,
            runtime_image_evidence=manifest["runtime_image_evidence"],
            alias_verification=alias_verification,
            notes=(
                "raw Harbor bundle remains local; normalized report is derived after ingest",
            ),
        )
        if exit_code != 0:
            manifest["state"] = "failed"
            manifest["exit_code"] = exit_code
            manifest.pop("runner_pid", None)
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise PreconditionError(f"upstream benchmark exited with code {exit_code}")
        if runtime_image is None:
            # An exit-zero run without Docker's immutable image identity cannot
            # be audited later.  Keep its logs/receipt locally, but never ingest
            # or promote it to a completed campaign.
            manifest["state"] = "failed"
            manifest["failure"] = "upstream image identity could not be verified"
            manifest.pop("runner_pid", None)
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise EvidenceError(
                "upstream exited successfully, but its Docker image identity "
                "could not be verified"
            )
        if not alias_verification["matches_runtime_image"]:
            manifest["state"] = "failed"
            manifest["failure"] = (
                "runtime Docker image does not match the fixed ORAgentBench base alias"
            )
            manifest.pop("runner_pid", None)
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise EvidenceError(
                "runtime Docker image identity does not match the fixed "
                "ORAgentBench base alias"
            )

        try:
            ingested = ingest_harbor_bundle(
                run_root=prepared.run_root, jobs_root=prepared.run_root / "jobs"
            )
        except Exception as exc:
            manifest["state"] = "failed"
            manifest["failure"] = "upstream exited successfully but result ingest failed"
            manifest["failure_type"] = type(exc).__name__
            manifest.pop("runner_pid", None)
            _atomic_json(manifest_path, manifest)
            _write_integrity(prepared.run_root)
            raise
        manifest["state"] = "completed"
        manifest["exit_code"] = 0
        manifest.pop("runner_pid", None)
        _atomic_json(manifest_path, manifest)
        _write_integrity(prepared.run_root)
        return ingested
