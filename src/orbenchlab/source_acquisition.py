"""Frozen source acquisition + provenance receipts (P0-E: collected -> acquired).

Given a discovered source, the harness fetches the full source bytes once,
freezes them at a content-addressed path, and writes a signed provenance
receipt that binds the exact bytes to the source URL, canonical identity, and
declared license.  Acquisition is idempotent by content digest, and the frozen
bytes are re-verified against the receipt on read, so a tampered or truncated
artifact is rejected before any triage or paid factory work.

The fetcher is injected: the default performs a bounded HTTPS GET; tests pass a
fake.  Fetched bytes are untrusted (hostile input) and are only ever stored and
digested here — they are parsed/interpreted later inside the disposable
minimal-root sandbox, never on the host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .core.errors import ORBenchError
from .source_registry import canonical_identity

ACQUISITION_SCHEMA = "orbenchlab.source-acquisition.v1"
DEFAULT_MAX_BYTES = 32 * 1024 * 1024


class SourceAcquisitionError(ORBenchError):
    exit_code = 8


# fetcher(url, max_bytes) -> {"bytes": b"...", "content_type": str, "final_url": str}
Fetcher = Callable[[str, int], Mapping[str, Any]]


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _frozen_dir(out: Path, content_digest: str) -> Path:
    return out / "frozen" / content_digest.removeprefix("sha256:")[:16]


def acquire_source(
    *,
    source: Mapping[str, Any],
    out: str | Path,
    fetcher: Fetcher,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch, freeze and receipt one source. Idempotent by content digest."""

    url = source.get("url") or source.get("source_url")
    if not isinstance(url, str) or not url.strip():
        raise SourceAcquisitionError("source has no fetchable url")
    result = fetcher(url, max_bytes)
    data = result.get("bytes")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise SourceAcquisitionError("fetcher returned no bytes")
    data = bytes(data)
    if len(data) > max_bytes:
        raise SourceAcquisitionError("fetched source exceeds the byte cap")
    content_digest = _digest_bytes(data)
    frozen_dir = _frozen_dir(Path(out), content_digest)
    frozen_path = frozen_dir / "source.bin"
    receipt_path = frozen_dir / "acquisition-receipt.json"

    # Idempotent reuse: an existing receipt with intact bytes is returned as-is.
    if receipt_path.is_file() and frozen_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            existing.get("content_digest") == content_digest
            and _digest_bytes(frozen_path.read_bytes()) == content_digest
        ):
            unsigned = {k: v for k, v in existing.items() if k != "receipt_digest"}
            if existing.get("receipt_digest") == _digest(unsigned):
                return {**existing, "reused": True, "frozen_path": str(frozen_path)}

    _atomic_bytes(frozen_path, data)
    receipt = {
        "schema_version": ACQUISITION_SCHEMA,
        "source_id": str(source.get("id") or source.get("source_id") or content_digest[:24]),
        "url": url,
        "final_url": str(result.get("final_url") or url),
        "canonical_identity": canonical_identity(source),
        "content_digest": content_digest,
        "byte_count": len(data),
        "content_type": str(result.get("content_type") or "application/octet-stream"),
        "declared_license": source.get("license") or source.get("declared_license"),
        "channel": source.get("channel") or source.get("source_kind"),
        "frozen_relpath": frozen_path.name,
    }
    receipt["receipt_digest"] = _digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    _atomic_json(receipt_path, receipt)
    return {**receipt, "reused": False, "frozen_path": str(frozen_path)}


def verify_acquisition(*, receipt: Mapping[str, Any], out: str | Path) -> bool:
    """Re-verify frozen bytes against a provenance receipt; raise on tamper."""

    content_digest = str(receipt.get("content_digest") or "")
    frozen_path = _frozen_dir(Path(out), content_digest) / "source.bin"
    if not frozen_path.is_file():
        raise SourceAcquisitionError("frozen source bytes are missing")
    if _digest_bytes(frozen_path.read_bytes()) != content_digest:
        raise SourceAcquisitionError("frozen source bytes do not match the provenance receipt")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    if receipt.get("receipt_digest") != _digest(unsigned):
        raise SourceAcquisitionError("acquisition receipt digest mismatch (possible tamper)")
    return True


def bounded_https_fetcher(url: str, max_bytes: int) -> dict[str, Any]:
    """Default fetcher: a bounded, redirect-limited HTTPS GET.

    Rejects non-HTTPS and reads at most ``max_bytes`` + 1 to detect overflow.
    """

    import urllib.request
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise SourceAcquisitionError("source url must be https")
    request = urllib.request.Request(url, method="GET", headers={"user-agent": "orbench-source/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
        return {
            "bytes": data[:max_bytes] if len(data) > max_bytes else data,
            "content_type": response.headers.get("content-type", "application/octet-stream"),
            "final_url": response.geturl(),
        }


__all__ = [
    "ACQUISITION_SCHEMA",
    "DEFAULT_MAX_BYTES",
    "Fetcher",
    "SourceAcquisitionError",
    "acquire_source",
    "bounded_https_fetcher",
    "verify_acquisition",
]
