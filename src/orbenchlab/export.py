"""Create a small, shareable evidence bundle from a completed local run.

The run workspace is intentionally host-private: it contains raw Harbor jobs,
logs, command arguments and absolute paths.  This module is the only supported
boundary for moving run evidence off that host.  It verifies the complete local
workspace first, then copies a fixed normalized/report allowlist and derives new
public metadata instead of copying the local manifest or receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from . import execution
from .core import schema as schema_mod
from .core.errors import EvidenceError, ORBenchError, PreconditionError
from .report import render as render_mod
from .report.model import NORMALIZED_SCHEMA, NormalizedRollout


_ALLOWLIST = (
    "normalized/rollout.json",
    "report/summary.md",
    "report/summary.json",
    "report/evidence_index.json",
)
_PUBLIC_METADATA = ("public-manifest.json", "public-receipt.json")
_SHARE_INTEGRITY = "share-integrity.sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROVIDER_ROUTE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,254}$")
_DOCKER_REPO_DIGEST_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,190}@sha256:[0-9a-f]{64}$"
)
_UNIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s=(\[{'\"]))/+(?!/)[^\s,;)\]}'\"]+"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:(?<=^)|(?<=[\s=(\[{'\"]))[a-z]:\\[^\s,;)\]}'\"]+"
)
_SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|credential)\s*[=:]\s*)([^&\s,;]+)"
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)(\bauthorization\s*:\s*(?:bearer|basic)\s+)([^\s,;]+)"
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_COMMON_HOST_PATH_RE = re.compile(
    r"(?i)(?:file://)?(?:/(?:Users|home|private|tmp|var|opt|root|mnt|srv|etc|workspace)/|[a-z]:\\(?:Users|Windows|ProgramData)\\)"
)
_ENV_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|LICENSE)[A-Z0-9_]*\s*[=:]\s*(?![\"']?<(?:set|unset|redacted)>)[\"']?[^\s,;\"'}]+"
)
_JSON_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)[\"'](?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|credential)[\"']\s*:\s*(?![\"']?<(?:set|unset|redacted)>)[\"']?[^\s,;\"'}]+"
)
_CREDENTIAL_PREFIX_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{8,}|\bghp_[A-Za-z0-9]{12,}|\bgithub_pat_[A-Za-z0-9_]{12,}|\bAKIA[A-Z0-9]{16}\b)"
)


@dataclass(frozen=True)
class ExportResult:
    """Machine-readable result returned by :func:`export_shareable_run`."""

    campaign_id: str
    destination: Path
    files: int
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "destination": str(self.destination),
            "files": self.files,
            "reused": self.reused,
        }


def export_shareable_run(run_root: str | Path, destination: str | Path) -> ExportResult:
    """Export a completed ORAgentBench run into a sanitized evidence directory.

    The source workspace must be integrity-complete and contain no symlinks.
    ``destination`` is created atomically from a sibling staging directory.  An
    existing byte-identical bundle is reused; any other existing destination is
    treated as a conflict and left untouched.
    """

    source = Path(run_root).expanduser()
    if source.is_symlink():
        raise EvidenceError("run workspace may not be a symlink")
    try:
        source = source.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise PreconditionError("run workspace does not exist") from None
    if not source.is_dir():
        raise PreconditionError("run workspace is not a directory")

    target_input = Path(destination).expanduser()
    if target_input.is_symlink():
        raise EvidenceError("export destination may not be a symlink")
    _reject_symlink_path(target_input.absolute().parent)
    target = target_input.resolve(strict=False)
    if _is_relative_to(target, source):
        raise PreconditionError("export destination may not be inside the run workspace")

    expected = _verify_completed_workspace(source)
    manifest = _load_verified_json_object(
        source,
        "manifest.json",
        label="run manifest",
        expected_digest=expected["manifest.json"],
    )
    receipt = _load_verified_json_object(
        source,
        "receipt.json",
        label="run receipt",
        expected_digest=expected["receipt.json"],
    )
    normalized = _load_verified_json_object(
        source,
        "normalized/rollout.json",
        label="normalized rollout",
        expected_digest=expected["normalized/rollout.json"],
    )
    summary = _load_verified_json_object(
        source,
        "report/summary.json",
        label="report summary",
        expected_digest=expected["report/summary.json"],
    )
    evidence_index = _load_verified_json_object(
        source,
        "report/evidence_index.json",
        label="evidence index",
        expected_digest=expected["report/evidence_index.json"],
    )
    summary_markdown = _read_verified_text(
        source,
        "report/summary.md",
        label="report markdown",
        expected_digest=expected["report/summary.md"],
    )
    if manifest.get("state") != "completed":
        raise PreconditionError("only a completed run can be exported")
    if manifest.get("integration") != "oragentbench":
        raise PreconditionError("only an ORAgentBench run workspace can be exported")
    _strict_validate_normalized(normalized)
    forbidden_paths = _host_paths_from_local_metadata(manifest, receipt)
    for label, document in (
        ("normalized rollout", normalized),
        ("report summary", summary),
        ("evidence index", evidence_index),
    ):
        _assert_shareable_document(
            document, label=label, forbidden_paths=forbidden_paths
        )
    _assert_shareable_text(
        summary_markdown, label="report markdown", forbidden_paths=forbidden_paths
    )
    _validate_cross_file_identity(manifest, receipt, normalized, summary, evidence_index)
    public_manifest, public_receipt = _public_metadata(manifest, receipt)
    _assert_shareable_document(
        public_manifest,
        label="public manifest",
        forbidden_paths=forbidden_paths,
    )
    _assert_shareable_document(
        public_receipt,
        label="public receipt",
        forbidden_paths=forbidden_paths,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(target.parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        # Write only the already hash-bound, validated in-memory values.  A
        # second copy from the live workspace would reopen a TOCTOU window
        # between leak validation and publication.
        _write_json(stage / "normalized" / "rollout.json", normalized)
        _write_json(stage / "report" / "summary.json", summary)
        _write_json(stage / "report" / "evidence_index.json", evidence_index)
        (stage / "report" / "summary.md").write_text(
            summary_markdown, encoding="utf-8"
        )
        _validate_report_outputs(
            stage / "normalized" / "rollout.json",
            summary=summary,
            evidence_index=evidence_index,
            summary_markdown=summary_markdown,
        )
        _write_json(stage / "public-manifest.json", public_manifest)
        _write_json(stage / "public-receipt.json", public_receipt)
        _write_share_integrity(stage)

        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise EvidenceError("export destination conflicts with an existing path")
            if not _directories_equal(stage, target):
                raise EvidenceError("export destination conflicts with different content")
            return ExportResult(
                campaign_id=str(manifest["campaign_id"]),
                destination=target,
                files=len(_ALLOWLIST) + len(_PUBLIC_METADATA) + 1,
                reused=True,
            )

        os.replace(stage, target)
        stage = Path()  # ownership moved to the final destination
        return ExportResult(
            campaign_id=str(manifest["campaign_id"]),
            destination=target,
            files=len(_ALLOWLIST) + len(_PUBLIC_METADATA) + 1,
            reused=False,
        )
    finally:
        if stage != Path() and stage.exists():
            shutil.rmtree(stage)


def _verify_completed_workspace(run_root: Path) -> dict[str, str]:
    """Strictly verify the workflow integrity ledger without trusting paths."""

    for path in run_root.rglob("*"):
        if path.is_symlink():
            raise EvidenceError("run workspace contains a symlink")

    ledger = run_root / "integrity.sha256"
    if not ledger.is_file():
        raise EvidenceError("run workspace has no integrity manifest")
    expected: dict[str, str] = {}
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise EvidenceError("run integrity manifest is unreadable") from None
    for line in lines:
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            raise EvidenceError("run integrity manifest contains an invalid line") from None
        pure = PurePosixPath(relative)
        if (
            not _SHA256_RE.fullmatch(digest)
            or not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in expected
            or "\\" in relative
        ):
            raise EvidenceError("run integrity manifest contains an unsafe entry")
        expected[relative] = digest

    actual: dict[str, Path] = {}
    for path in run_root.rglob("*"):
        if not path.is_file() or path == ledger:
            continue
        if ".tmp-" in path.name or path.name == ".run.lock":
            continue
        relative = path.relative_to(run_root).as_posix()
        actual[relative] = path

    if set(actual) != set(expected):
        raise EvidenceError("run workspace failed integrity check: file set mismatch")
    for relative, path in actual.items():
        if _sha256(path) != expected[relative]:
            raise EvidenceError(f"run workspace failed integrity check: {relative}")

    manifest = _load_json_object(run_root / "manifest.json", label="run manifest")
    if manifest.get("state") != "completed":
        raise PreconditionError("only a completed run can be exported")
    if manifest.get("integration") != "oragentbench":
        raise PreconditionError("only an ORAgentBench run workspace can be exported")
    return expected


def _validate_cross_file_identity(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    normalized: Mapping[str, Any],
    summary: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
) -> None:
    campaign_id = _public_scalar(
        _required_string(manifest, "campaign_id", "run manifest"),
        field="campaign_id",
        allow_none=False,
    )
    integration = _public_scalar(
        _required_string(manifest, "integration", "run manifest"),
        field="integration",
        allow_none=False,
    )
    assert campaign_id is not None
    assert integration is not None
    if integration != "oragentbench":
        raise PreconditionError("only an ORAgentBench run workspace can be exported")
    for label, document in (
        ("run receipt", receipt),
        ("normalized rollout", normalized),
        ("report summary", summary),
        ("evidence index", evidence_index),
    ):
        if document.get("campaign_id") != campaign_id:
            raise EvidenceError(f"{label} has a conflicting campaign id")
    for label, document in (
        ("run receipt", receipt),
        ("normalized rollout", normalized),
    ):
        if document.get("integration") != integration:
            raise EvidenceError(f"{label} has a conflicting integration")
    trials = normalized.get("trials")
    if not isinstance(trials, list) or len(trials) != 1 or not isinstance(trials[0], Mapping):
        raise EvidenceError(
            "normalized rollout does not match the one-command single-trial campaign"
        )
    trial = trials[0]
    for trial_key, manifest_key, label in (
        ("task_name", "task", "task"),
        ("scaffold", "agent", "scaffold"),
        ("agent_id", "agent_id", "compiled agent identity"),
    ):
        if trial.get(trial_key) != manifest.get(manifest_key):
            raise EvidenceError(f"normalized rollout has a conflicting {label}")
    manifest_exit = manifest.get("exit_code")
    receipt_exit = receipt.get("exit_code")
    if (
        isinstance(manifest_exit, bool)
        or not isinstance(manifest_exit, int)
        or isinstance(receipt_exit, bool)
        or not isinstance(receipt_exit, int)
        or manifest_exit != 0
        or receipt_exit != manifest_exit
    ):
        raise EvidenceError("completed run has inconsistent exit-code evidence")
    for key, label in (
        ("agent_id", "compiled agent identity"),
        ("source_snapshot_digest", "source snapshot"),
        ("scaffold_version", "scaffold version"),
        ("runtime_image", "runtime image"),
        ("runtime_image_evidence", "runtime image evidence"),
        ("runtime_image_alias_verification", "runtime image alias"),
    ):
        if receipt.get(key) != manifest.get(key):
            raise EvidenceError(f"completed run has conflicting {label} evidence")


def _public_metadata(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_id = _public_scalar(
        _required_string(manifest, "campaign_id", "run manifest"),
        field="campaign_id",
        allow_none=False,
    )
    integration = _public_scalar(
        _required_string(manifest, "integration", "run manifest"),
        field="integration",
        allow_none=False,
    )
    assert campaign_id is not None
    assert integration is not None
    task = _public_scalar(manifest.get("task"), field="task", allow_none=False)
    agent = _public_scalar(manifest.get("agent"), field="agent", allow_none=False)
    model = _public_scalar(manifest.get("model"), field="model", allow_none=True)
    source_commit = _public_scalar(
        manifest.get("source_commit"), field="source_commit", allow_none=False
    )
    dataset_digest = _public_scalar(
        manifest.get("dataset_digest"), field="dataset_digest", allow_none=False
    )
    provider_route_digest = _public_scalar(
        manifest.get("provider_route_digest"),
        field="provider_route_digest",
        allow_none=True,
    )
    if provider_route_digest is not None and not _PROVIDER_ROUTE_DIGEST_RE.match(
        provider_route_digest
    ):
        raise EvidenceError("run manifest has an invalid provider-route digest")
    source_snapshot_digest = _public_scalar(
        manifest.get("source_snapshot_digest"),
        field="source_snapshot_digest",
        allow_none=False,
    )
    if source_snapshot_digest is None or not _PROVIDER_ROUTE_DIGEST_RE.fullmatch(
        source_snapshot_digest
    ):
        raise EvidenceError("run manifest has an invalid source-snapshot digest")
    scaffold_version = _public_scalar(
        manifest.get("scaffold_version"),
        field="scaffold_version",
        allow_none=True,
    )
    agent_id = _public_scalar(
        manifest.get("agent_id"), field="agent_id", allow_none=False
    )
    assert agent_id is not None
    expected_agent_id = (
        agent
        if model is None
        else f"{agent}-{model}".lower().replace("/", "-").replace(".", "-")
    )
    if agent_id != expected_agent_id:
        raise EvidenceError("run manifest model conflicts with its compiled agent identity")
    runtime_image = _public_runtime_image(manifest.get("runtime_image"))
    alias_verification = _public_runtime_image_alias(
        manifest.get("runtime_image_alias_verification"), runtime_image=runtime_image
    )
    if manifest.get("runtime_image_evidence") != "docker-image-inspect":
        raise EvidenceError("completed run is not bound to inspected Docker image evidence")
    command = receipt.get("upstream_command")
    if not isinstance(command, Mapping):
        raise EvidenceError("run receipt has no upstream-command provenance")
    provenance = _public_scalar(
        command.get("provenance"), field="provenance", allow_none=False
    )
    makes_model_calls = command.get("makes_model_calls")
    if not isinstance(makes_model_calls, bool):
        raise EvidenceError("run receipt has invalid model-call provenance")
    executed = receipt.get("executed")
    if executed is not True:
        raise EvidenceError("completed run receipt does not record execution")

    public_manifest = {
        "public_manifest_schema_version": "1.0",
        "state": "completed",
        "campaign_id": campaign_id,
        "integration": integration,
        "source_commit": source_commit,
        "dataset_digest": dataset_digest,
        "source_snapshot_digest": source_snapshot_digest,
        "agent": agent,
        "agent_id": agent_id,
        "model": model,
        "scaffold_version": scaffold_version,
        "runtime_image": runtime_image,
        "runtime_image_alias_verification": alias_verification,
        "task": task,
        "exit_code": 0,
    }
    if provider_route_digest is not None:
        public_manifest["provider_route_digest"] = provider_route_digest
    public_receipt = {
        "public_receipt_schema_version": "1.0",
        "campaign_id": campaign_id,
        "integration": integration,
        "mode": _public_scalar(receipt.get("mode"), field="mode", allow_none=False),
        "evidence_label": _public_scalar(
            receipt.get("evidence_label"), field="evidence_label", allow_none=False
        ),
        "agent": agent,
        "agent_id": agent_id,
        "model": model,
        "scaffold_version": scaffold_version,
        "source_snapshot_digest": source_snapshot_digest,
        "runtime_image": runtime_image,
        "runtime_image_alias_verification": alias_verification,
        "runtime_image_evidence": "docker-image-inspect",
        "task": task,
        "upstream_command": {
            "provenance": provenance,
            "makes_model_calls": makes_model_calls,
        },
        "exit_code": 0,
        "executed": True,
    }
    return public_manifest, public_receipt


def _public_runtime_image(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError("completed run has no inspected Docker image identity")
    if set(value) != {"requested_tag", "image_id", "repo_digests"}:
        raise EvidenceError("completed run has malformed Docker image identity")
    requested_tag = value.get("requested_tag")
    image_id = value.get("image_id")
    repo_digests = value.get("repo_digests")
    if not isinstance(requested_tag, str) or not _DOCKER_TAG_RE.fullmatch(requested_tag):
        raise EvidenceError("completed run has malformed Docker image tag evidence")
    if not isinstance(image_id, str) or not _DOCKER_IMAGE_ID_RE.fullmatch(image_id):
        raise EvidenceError("completed run has malformed Docker image-id evidence")
    if (
        not isinstance(repo_digests, list)
        or any(
            not isinstance(item, str) or not _DOCKER_REPO_DIGEST_RE.fullmatch(item)
            for item in repo_digests
        )
        or repo_digests != sorted(set(repo_digests))
    ):
        raise EvidenceError("completed run has malformed Docker repo-digest evidence")
    return {
        "requested_tag": requested_tag,
        "image_id": image_id,
        "repo_digests": list(repo_digests),
    }


def _public_runtime_image_alias(
    value: Any, *, runtime_image: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "fixed_alias",
        "fixed_alias_image_id",
        "matches_runtime_image",
    }:
        raise EvidenceError("completed run has malformed Docker alias evidence")
    fixed_alias = value.get("fixed_alias")
    fixed_alias_image_id = value.get("fixed_alias_image_id")
    if not isinstance(fixed_alias, str) or not _DOCKER_TAG_RE.fullmatch(fixed_alias):
        raise EvidenceError("completed run has malformed Docker alias tag evidence")
    if fixed_alias != execution.ORAGENTBENCH_FIXED_BASE_IMAGE:
        raise EvidenceError(
            "completed run is not bound to the fixed ORAgentBench Docker alias"
        )
    if not isinstance(fixed_alias_image_id, str) or not _DOCKER_IMAGE_ID_RE.fullmatch(
        fixed_alias_image_id
    ):
        raise EvidenceError("completed run has malformed Docker alias image-id evidence")
    if value.get("matches_runtime_image") is not True:
        raise EvidenceError("completed run Docker alias did not match its runtime image")
    if fixed_alias_image_id != runtime_image.get("image_id"):
        raise EvidenceError("completed run has conflicting Docker alias image identity")
    return {
        "fixed_alias": fixed_alias,
        "fixed_alias_image_id": fixed_alias_image_id,
        "matches_runtime_image": True,
    }


def _strict_validate_normalized(document: Mapping[str, Any]) -> None:
    """Validate the normalized slice and close every declared object shape.

    The repository schema intentionally remains forward-compatible.  Export is
    a narrower trust boundary: an unrecognised field must not hitch a ride in a
    public artifact, so every object that declares properties is treated as
    closed for this check.
    """

    try:
        schema = schema_mod.load_schema(schema_mod.schemas_dir() / NORMALIZED_SCHEMA)
        schema_mod.validate(document, schema, name="normalized rollout")
        _reject_unknown_properties(document, schema)
    except (schema_mod.SchemaError, schema_mod.SchemaFeatureError, ValueError, TypeError):
        raise EvidenceError("normalized rollout failed strict export validation") from None


def _reject_unknown_properties(value: Any, schema: Mapping[str, Any]) -> None:
    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            if set(value) - set(properties):
                raise ValueError("unknown property")
            for key, item in value.items():
                child = properties.get(key)
                if isinstance(child, Mapping):
                    _reject_unknown_properties(item, child)
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for item in value:
                _reject_unknown_properties(item, item_schema)


def _validate_report_outputs(
    normalized_path: Path,
    *,
    summary: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
    summary_markdown: str,
) -> None:
    """Require reports to be exactly reproducible from the normalized slice."""

    try:
        expected = render_mod.build_report(NormalizedRollout.load(normalized_path))
    except (ORBenchError, OSError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        raise EvidenceError("normalized rollout could not reproduce its report") from None
    if (
        dict(summary) != expected.summary
        or dict(evidence_index) != expected.evidence_index
        or summary_markdown != expected.markdown
    ):
        raise EvidenceError("report artifacts do not match the normalized rollout")


def _host_paths_from_local_metadata(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[str, ...]:
    candidates: list[Any] = [manifest.get("source"), receipt.get("output_root")]
    command = receipt.get("upstream_command")
    if isinstance(command, Mapping):
        candidates.append(command.get("cwd"))
        argv = command.get("argv")
        if isinstance(argv, list):
            candidates.extend(argv)
    paths = {
        value
        for value in candidates
        if isinstance(value, str) and _is_absolute_host_path(value)
    }
    return tuple(sorted(paths, key=len, reverse=True))


def _assert_shareable_text(
    text: str, *, label: str, forbidden_paths: tuple[str, ...]
) -> None:
    """Fail closed on credential or host-path evidence without echoing it."""

    if (
        any(path in text for path in forbidden_paths)
        or _COMMON_HOST_PATH_RE.search(text)
        or _UNIX_ABSOLUTE_PATH_RE.search(text)
        or _WINDOWS_ABSOLUTE_PATH_RE.search(text)
    ):
        raise EvidenceError(f"{label} contains a non-shareable host path")
    if (
        _AUTHORIZATION_RE.search(text)
        or _SENSITIVE_TEXT_RE.search(text)
        or _ENV_SECRET_ASSIGNMENT_RE.search(text)
        or _JSON_SECRET_ASSIGNMENT_RE.search(text)
        or _PRIVATE_KEY_RE.search(text)
        or _CREDENTIAL_PREFIX_RE.search(text)
    ):
        raise EvidenceError(f"{label} contains non-shareable credential-like text")


def _assert_shareable_document(
    document: Mapping[str, Any], *, label: str, forbidden_paths: tuple[str, ...]
) -> None:
    # Scan the serialized form to catch secret-shaped JSON keys, and raw string
    # values to catch Windows paths before JSON escaping changes their spelling.
    _assert_shareable_text(
        _canonical_json_text(document), label=label, forbidden_paths=forbidden_paths
    )
    stack: list[Any] = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            if _is_absolute_host_path(value.strip()):
                raise EvidenceError(f"{label} contains a non-shareable host path")
            _assert_shareable_text(
                value, label=label, forbidden_paths=forbidden_paths
            )


def _is_absolute_host_path(value: str) -> bool:
    return value.startswith(("/", "file:///")) or bool(
        re.match(r"(?i)^[a-z]:\\", value)
    )


def _canonical_json_text(document: Mapping[str, Any]) -> str:
    try:
        return json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise EvidenceError("shareable JSON contains an unsupported value") from None


def _read_verified_text(
    root: Path,
    relative: str,
    *,
    label: str,
    expected_digest: str,
) -> str:
    path = _regular_file_beneath(root, relative)
    try:
        payload = path.read_bytes()
    except (OSError, UnicodeDecodeError):
        raise EvidenceError(f"{label} is not readable UTF-8 text") from None
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise EvidenceError(f"{label} changed after workspace integrity verification")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceError(f"{label} is not readable UTF-8 text") from None


def _public_scalar(value: Any, *, field: str, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"run metadata has invalid {field}")
    cleaned = _sanitize_public_text(value)
    if cleaned != value:
        # Identity and provenance must not silently change in a public package.
        # Refuse the export instead of producing ambiguous metadata.
        raise EvidenceError(f"run metadata {field} contains non-shareable content")
    return cleaned


def _required_string(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} has invalid {key}")
    return value


def _sanitize_public_text(value: str) -> str:
    value = _AUTHORIZATION_RE.sub(r"\1<redacted>", value)
    value = _SENSITIVE_TEXT_RE.sub(r"\1<redacted>", value)
    value = _UNIX_ABSOLUTE_PATH_RE.sub("<host-path-omitted>", value)
    return _WINDOWS_ABSOLUTE_PATH_RE.sub("<host-path-omitted>", value)


def _regular_file_beneath(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"shareable run artifact is missing or unsafe: {relative}")
    resolved = path.resolve(strict=True)
    if not _is_relative_to(resolved, root):
        raise EvidenceError(f"shareable run artifact escapes the workspace: {relative}")
    return resolved


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvidenceError(f"{label} is not a readable JSON object") from None
    if not isinstance(data, dict):
        raise EvidenceError(f"{label} is not a readable JSON object")
    return data


def _load_verified_json_object(
    root: Path,
    relative: str,
    *,
    label: str,
    expected_digest: str,
) -> dict[str, Any]:
    text = _read_verified_text(
        root,
        relative,
        label=label,
        expected_digest=expected_digest,
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, ValueError):
        raise EvidenceError(f"{label} is not a readable JSON object") from None
    if not isinstance(data, dict):
        raise EvidenceError(f"{label} is not a readable JSON object")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_share_integrity(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == _SHARE_INTEGRITY:
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / _SHARE_INTEGRITY).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _directories_equal(first: Path, second: Path) -> bool:
    first_files = _safe_file_map(first)
    second_files = _safe_file_map(second)
    if set(first_files) != set(second_files):
        return False
    return all(_sha256(first_files[key]) == _sha256(second_files[key]) for key in first_files)


def _safe_file_map(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise EvidenceError("export destination contains a symlink")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def _reject_symlink_path(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise EvidenceError("export destination path contains a symlink")
        if current == current.parent:
            return
        current = current.parent


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
