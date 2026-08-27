"""Auditable, model-free intake of operations-research sources.

The intake layer is deliberately narrower than a task authoring system.  It
fetches public feed *metadata* (RSS/Atom, arXiv Atom, or GitHub's public JSON
and Atom endpoints), normalizes and de-duplicates entries, and writes a queue
for a human to review.  It never calls a model, executes benchmark code,
rewrites ``raw/``, or turns a source into a task automatically.

Only digests and bounded metadata are persisted.  The response body itself is
not written to the bundle; this keeps the bundle small and avoids silently
redistributing source text while retaining enough evidence to identify exactly
which response was observed.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import yaml

from .core.errors import ORBenchError
from .core import schema as schema_mod


INTAKE_SCHEMA_VERSION = "1.0"
SUPPORTED_KINDS = frozenset({"rss", "arxiv", "github"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SECRET_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "credential",
        "key",
        "secret",
        "token",
    }
)
_MAX_TITLE = 500
_MAX_SUMMARY = 2_000
_MAX_AUTHORS = 20
_MAX_TAGS = 40
_MAX_ENTRIES_PER_FEED = 10_000


class SourceIntakeError(ORBenchError):
    """A source-intake configuration, response, or bundle is invalid."""

    exit_code = 8


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _normalise_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_text(str(value), limit=100)
    return text or None


def _safe_error_message(exc: BaseException) -> str:
    """Return a short, non-sensitive diagnostic for an artifact row."""

    message = str(exc)
    # A custom fetcher is outside our control.  Never persist a message that
    # looks like a bearer/token/query credential; the feed row still records
    # the exception class for audit and retry decisions.
    if re.search(r"(?:token|secret|password|api[_-]?key|authorization)\s*[:=]", message, re.I):
        return "feed operation failed (sensitive diagnostic elided)"
    if len(message) > 240 or "http://" in message or "https://" in message:
        return "feed operation failed (diagnostic elided)"
    return message or "feed operation failed"


def _clean_text(value: str, *, limit: int) -> str:
    """Collapse markup/whitespace and bound source-provided text."""

    # Feed summaries are often XHTML fragments.  We retain a short readable
    # excerpt, never arbitrary markup or scripts.
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = " ".join(value.split())
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


def validate_feed_url(value: str) -> str:
    """Validate and canonicalize a public feed URL.

    Query strings are allowed because arXiv search feeds use them.  Query
    parameter names that conventionally carry credentials are rejected before
    any request is made, so a secret cannot accidentally enter an artifact.
    """

    candidate = str(value).strip()
    if not candidate:
        raise SourceIntakeError("feed URL is required")
    if any(ord(c) < 0x20 or c.isspace() for c in candidate):
        raise SourceIntakeError("feed URL may not contain whitespace or controls")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        raise SourceIntakeError("feed URL is malformed") from None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise SourceIntakeError("feed URL must use HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise SourceIntakeError("feed URL may not contain userinfo or credentials")
    if parsed.fragment:
        raise SourceIntakeError("feed URL may not contain a fragment")
    for name, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_name = name.lower().replace("-", "_")
        if normalized_name in _SECRET_QUERY_NAMES or normalized_name.endswith(
            ("_token", "_key", "_secret")
        ):
            raise SourceIntakeError("feed URL may not contain credential-like query parameters")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return urlunsplit(("https", netloc, parsed.path, parsed.query, ""))


@dataclass(frozen=True)
class FeedSpec:
    """One public source endpoint in an intake configuration."""

    id: str
    kind: str
    url: str
    tags: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int = 0) -> "FeedSpec":
        if not isinstance(value, Mapping):
            raise SourceIntakeError(f"feeds[{index}] must be an object")
        unknown = sorted(set(value) - {"id", "kind", "url", "tags", "enabled"})
        if unknown:
            raise SourceIntakeError(f"feeds[{index}] has unsupported key(s): {unknown}")
        missing = [key for key in ("id", "kind", "url") if key not in value]
        if missing:
            raise SourceIntakeError(f"feeds[{index}] missing required key(s): {missing}")
        feed_id = str(value["id"]).strip()
        if not _ID_RE.fullmatch(feed_id):
            raise SourceIntakeError(f"feeds[{index}].id is not a safe identifier")
        kind = str(value["kind"]).strip().lower()
        if kind not in SUPPORTED_KINDS:
            raise SourceIntakeError(
                f"feeds[{index}].kind must be one of {sorted(SUPPORTED_KINDS)}"
            )
        url = validate_feed_url(str(value["url"]))
        raw_tags = value.get("tags", [])
        if isinstance(raw_tags, str) or not isinstance(raw_tags, Sequence):
            raise SourceIntakeError(f"feeds[{index}].tags must be a list")
        tags: list[str] = []
        for tag in raw_tags:
            cleaned = _clean_text(str(tag), limit=80)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        if len(tags) > _MAX_TAGS:
            raise SourceIntakeError(f"feeds[{index}].tags has too many entries")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise SourceIntakeError(f"feeds[{index}].enabled must be boolean")
        return cls(feed_id, kind, url, tuple(sorted(tags)), enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "url": self.url,
            "tags": list(self.tags),
            "enabled": self.enabled,
        }


def load_feed_config(path: str | Path) -> tuple[FeedSpec, ...]:
    """Load and validate a YAML/JSON feed config without touching ``raw/``."""

    config_path = Path(path)
    if not config_path.is_file():
        raise SourceIntakeError(f"feed config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceIntakeError(f"could not read feed config: {type(exc).__name__}") from None
    if not isinstance(raw, Mapping):
        raise SourceIntakeError("feed config root must be an object")
    unknown = sorted(set(raw) - {"version", "feeds"})
    if unknown:
        raise SourceIntakeError(f"feed config has unsupported key(s): {unknown}")
    if raw.get("version", 1) != 1:
        raise SourceIntakeError("feed config version must be 1")
    feeds = raw.get("feeds")
    if not isinstance(feeds, list) or not feeds:
        raise SourceIntakeError("feed config must contain a non-empty feeds list")
    parsed = tuple(FeedSpec.from_mapping(item, index=i) for i, item in enumerate(feeds))
    ids = [feed.id for feed in parsed]
    if len(set(ids)) != len(ids):
        raise SourceIntakeError("feed ids must be unique")
    return parsed


@dataclass(frozen=True)
class FetchResponse:
    """Bounded response metadata returned by a fetcher.

    A custom fetcher is useful for offline tests and mirrors the network
    fetcher's contract.  It may return this type or raw bytes.
    """

    body: bytes
    status: int = 200
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None


def fetch_url(url: str, *, timeout_sec: int = 20, max_bytes: int = 2_000_000) -> FetchResponse:
    """Fetch one public feed with a bounded body and no credential lookup."""

    canonical = validate_feed_url(url)
    if timeout_sec < 1 or max_bytes < 1:
        raise SourceIntakeError("timeout_sec and max_bytes must be >= 1")
    request = urllib.request.Request(
        canonical,
        headers={
            "Accept": "application/atom+xml, application/rss+xml, application/json;q=0.9, text/xml;q=0.8",
            "User-Agent": "orbenchlab-source-intake/1.0 (+metadata-only)",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", 200))
            content_type = str(response.headers.get("Content-Type", ""))
            body = response.read(max_bytes + 1)
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Do not echo exception text: proxies sometimes include the full URL.
        raise SourceIntakeError(f"feed request failed ({type(exc).__name__})") from None
    if len(body) > max_bytes:
        raise SourceIntakeError(f"feed response exceeds {max_bytes} bytes")
    if status < 200 or status >= 300:
        raise SourceIntakeError(f"feed returned HTTP status {status}")
    return FetchResponse(body, status, content_type, etag, last_modified)


@dataclass(frozen=True)
class _ParsedEntry:
    kind: str
    feed_id: str
    external_id: str | None
    title: str
    canonical_url: str | None
    summary: str
    authors: tuple[str, ...]
    published_at: str | None
    updated_at: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class IntakeItem:
    """A normalized, de-duplicated source record suitable for review."""

    item_uid: str
    identity_key: str
    source_kind: str
    title: str
    canonical_url: str | None
    external_id: str | None
    summary: str
    authors: tuple[str, ...]
    published_at: str | None
    updated_at: str | None
    tags: tuple[str, ...]
    feed_ids: tuple[str, ...]
    occurrence_count: int
    content_digest: str
    dedupe_status: str = "new"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_uid": self.item_uid,
            "identity_key": self.identity_key,
            "source_kind": self.source_kind,
            "title": self.title,
            "canonical_url": self.canonical_url,
            "external_id": self.external_id,
            "summary": self.summary,
            "authors": list(self.authors),
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "feed_ids": list(self.feed_ids),
            "occurrence_count": self.occurrence_count,
            "content_digest": self.content_digest,
            "dedupe_status": self.dedupe_status,
        }


@dataclass(frozen=True)
class IntakeResult:
    """Complete metadata-only intake snapshot and its human queue."""

    schema_version: str
    intake_id: str
    created_at: str
    config_digest: str
    snapshot_digest: str
    feeds: tuple[dict[str, Any], ...]
    items: tuple[IntakeItem, ...]
    review_queue: tuple[dict[str, Any], ...]
    network_policy: dict[str, Any]

    @property
    def feed_errors(self) -> int:
        return sum(1 for feed in self.feeds if feed.get("status") == "error")

    @property
    def has_errors(self) -> bool:
        return self.feed_errors > 0

    @property
    def review_queue_digest(self) -> str:
        """Digest of the exact JSONL bytes written beside ``intake.json``."""

        return _digest(_queue_bytes(self.review_queue))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intake_id": self.intake_id,
            "created_at": self.created_at,
            "config_digest": self.config_digest,
            "snapshot_digest": self.snapshot_digest,
            "feeds": list(self.feeds),
            "items": [item.to_dict() for item in self.items],
            "review_queue_count": len(self.review_queue),
            "review_queue_digest": self.review_queue_digest,
            "network_policy": dict(self.network_policy),
        }


Fetcher = Callable[[str], FetchResponse | bytes]


def _canonical_url(value: Any, *, kind: str) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return None
    for name, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_name = name.lower().replace("-", "_")
        if normalized_name in _SECRET_QUERY_NAMES or normalized_name.endswith(
            ("_token", "_key", "_secret")
        ):
            return None
    host = parsed.hostname.lower()
    if kind == "arxiv" and host in {"arxiv.org", "export.arxiv.org"}:
        # arXiv commonly emits both http://export.arxiv.org and
        # https://arxiv.org forms for the same work.
        path = parsed.path.rstrip("/")
        match = re.match(r"^/abs/(.+)$", path, flags=re.I)
        if match:
            return "https://arxiv.org/abs/" + match.group(1)
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = host if port in (None, 80 if parsed.scheme.lower() == "http" else 443) else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/") or "/", parsed.query, ""))


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, names: Iterable[str]) -> str:
    wanted = set(names)
    for child in list(element):
        if _xml_local_name(child.tag) in wanted:
            return _clean_text("".join(child.itertext()), limit=_MAX_SUMMARY)
    return ""


def _xml_links(element: ET.Element) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for child in list(element):
        if _xml_local_name(child.tag) != "link":
            continue
        href = str(child.attrib.get("href", "")).strip()
        text = _clean_text("".join(child.itertext()), limit=1000)
        value = href or text
        if value:
            links.append((str(child.attrib.get("rel", "alternate")), value))
    return links


def _parse_xml(body: bytes, feed: FeedSpec) -> list[_ParsedEntry]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise SourceIntakeError("feed XML could not be parsed") from None
    nodes = [node for node in root.iter() if _xml_local_name(node.tag) in {"item", "entry"}]
    entries: list[_ParsedEntry] = []
    for node in nodes[:_MAX_ENTRIES_PER_FEED]:
        title = _child_text(node, {"title"})
        links = _xml_links(node)
        link = next((value for rel, value in links if rel.lower() == "alternate"), None)
        link = link or (links[0][1] if links else None)
        external_id = _child_text(node, {"id", "guid", "identifier"}) or None
        canonical = _canonical_url(link or external_id, kind=feed.kind)
        # arXiv's <id> is the canonical abs URL; RSS feeds may only provide a
        # GUID, so preserve it as an external id when no URL is available.
        if feed.kind == "arxiv" and canonical is None:
            canonical = _canonical_url(external_id, kind="arxiv")
        summary = _child_text(node, {"summary", "description", "content", "abstract"})
        published = _child_text(node, {"published", "pubdate", "date", "issued"}) or None
        updated = _child_text(node, {"updated", "modified", "lastmod"}) or None
        authors: list[str] = []
        for child in node.iter():
            if _xml_local_name(child.tag) in {"author", "creator"}:
                text = _clean_text("".join(child.itertext()), limit=200)
                if text and text not in authors:
                    authors.append(text)
        if not title and not canonical and not external_id:
            continue
        entries.append(
            _ParsedEntry(
                kind=feed.kind,
                feed_id=feed.id,
                external_id=_clean_text(external_id, limit=500) if external_id else None,
                title=_clean_text(title or (canonical or external_id or "untitled"), limit=_MAX_TITLE),
                canonical_url=canonical,
                summary=summary,
                authors=tuple(authors[:_MAX_AUTHORS]),
                published_at=_normalise_timestamp(published),
                updated_at=_normalise_timestamp(updated),
                tags=feed.tags,
            )
        )
    return entries


def _json_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("items", "entries", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    # A single GitHub release/commit object is also a useful response.
    if any(key in payload for key in ("html_url", "node_id", "tag_name", "sha")):
        return [payload]
    return []


def _nested_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "login", "title", "id"):
            if value.get(key) is not None:
                return str(value[key])
    return str(value) if value is not None else ""


def _parse_github_json(body: bytes, feed: FeedSpec) -> list[_ParsedEntry]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SourceIntakeError("GitHub response is not valid JSON") from None
    entries: list[_ParsedEntry] = []
    for item in _json_entries(payload)[:_MAX_ENTRIES_PER_FEED]:
        title = _nested_text(
            item.get("title")
            or item.get("name")
            or item.get("tag_name")
            or item.get("sha")
            or "untitled"
        )
        canonical = _canonical_url(item.get("html_url") or item.get("url"), kind="github")
        external = item.get("node_id") or item.get("id") or item.get("sha") or item.get("tag_name")
        summary = _nested_text(item.get("body") or item.get("description") or item.get("summary"))
        author = item.get("author") or item.get("user") or item.get("owner")
        authors = (_nested_text(author),) if author is not None and _nested_text(author) else ()
        published = item.get("published_at") or item.get("created_at") or item.get("date")
        updated = item.get("updated_at") or item.get("committed_at")
        entries.append(
            _ParsedEntry(
                kind=feed.kind,
                feed_id=feed.id,
                external_id=_clean_text(str(external), limit=500) if external is not None else None,
                title=_clean_text(title, limit=_MAX_TITLE),
                canonical_url=canonical,
                summary=_clean_text(summary, limit=_MAX_SUMMARY),
                authors=authors,
                published_at=_normalise_timestamp(published),
                updated_at=_normalise_timestamp(updated),
                tags=feed.tags,
            )
        )
    return entries


def _parse_response(response: FetchResponse, feed: FeedSpec) -> list[_ParsedEntry]:
    body = response.body
    content_type = response.content_type.lower()
    stripped = body.lstrip()
    if feed.kind == "github" and ("json" in content_type or stripped.startswith((b"[", b"{"))):
        return _parse_github_json(body, feed)
    if stripped.startswith((b"<", b"<?xml")) or "xml" in content_type or "atom" in content_type or "rss" in content_type:
        return _parse_xml(body, feed)
    if feed.kind == "github":
        return _parse_github_json(body, feed)
    raise SourceIntakeError("feed response format is unsupported")


def _identity(entry: _ParsedEntry) -> str:
    if entry.canonical_url:
        return "url:" + entry.canonical_url
    if entry.external_id:
        return f"{entry.kind}:id:{entry.external_id}"
    return "text:" + _digest(
        _canonical_json(
            {
                "kind": entry.kind,
                "title": entry.title,
                "published_at": entry.published_at,
            }
        )
    )


def _item_from_group(identity_key: str, entries: Sequence[_ParsedEntry]) -> IntakeItem:
    ordered = sorted(
        entries,
        key=lambda item: (
            item.feed_id,
            item.canonical_url or "",
            item.external_id or "",
            item.title,
        ),
    )
    # Deterministically choose the richest representation while still merging
    # provenance tags and feed ids from every occurrence.
    chosen = max(
        ordered,
        key=lambda item: (
            bool(item.summary),
            len(item.summary),
            bool(item.canonical_url),
            len(item.title),
            item.feed_id,
        ),
    )
    feed_ids = tuple(sorted({item.feed_id for item in ordered}))
    authors = tuple(sorted({author for item in ordered for author in item.authors if author}))[:_MAX_AUTHORS]
    tags = tuple(sorted({tag for item in ordered for tag in item.tags if tag}))[:_MAX_TAGS]
    payload = {
        "identity_key": identity_key,
        "source_kind": chosen.kind,
        "title": chosen.title,
        "canonical_url": chosen.canonical_url,
        "external_id": chosen.external_id,
        "summary": chosen.summary,
        "authors": list(authors),
        "published_at": chosen.published_at,
        "updated_at": chosen.updated_at,
        "tags": list(tags),
        "feed_ids": list(feed_ids),
    }
    content_digest = _digest(_canonical_json(payload))
    item_uid = _digest("orbench-intake-item.v1\0" + identity_key)
    return IntakeItem(
        item_uid=item_uid,
        identity_key=identity_key,
        source_kind=chosen.kind,
        title=chosen.title,
        canonical_url=chosen.canonical_url,
        external_id=chosen.external_id,
        summary=chosen.summary,
        authors=authors,
        published_at=chosen.published_at,
        updated_at=chosen.updated_at,
        tags=tags,
        feed_ids=feed_ids,
        occurrence_count=len(entries),
        content_digest=content_digest,
    )


def _load_previous(previous: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if previous is None:
        return {}
    if isinstance(previous, Mapping):
        payload = dict(previous)
    else:
        path = Path(previous)
        if path.is_dir():
            path = path / "intake.json"
        if not path.is_file():
            raise SourceIntakeError(f"previous intake not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise SourceIntakeError("previous intake is not valid JSON") from None
    if not isinstance(payload, Mapping):
        raise SourceIntakeError("previous intake root must be an object")
    previous_items = payload.get("items", [])
    if not isinstance(previous_items, list):
        raise SourceIntakeError("previous intake items must be a list")
    by_identity: dict[str, Any] = {}
    for item in previous_items:
        if not isinstance(item, Mapping):
            continue
        identity = str(item.get("identity_key", ""))
        if identity:
            by_identity[identity] = item
    return by_identity


def collect(
    feeds: Sequence[FeedSpec],
    *,
    fetcher: Fetcher | None = None,
    previous: str | Path | Mapping[str, Any] | None = None,
    created_at: str | None = None,
    timeout_sec: int = 20,
    max_bytes: int = 2_000_000,
) -> IntakeResult:
    """Collect and normalize feeds into an auditable human-review snapshot.

    ``fetcher`` is injectable so the same parser can be tested entirely offline.
    A failing feed is recorded in the bundle and does not erase successful
    feeds; :attr:`IntakeResult.has_errors` lets a caller fail CI or alert an
    operator without pretending the partial snapshot is complete.
    """

    if fetcher is None:
        fetcher = fetch_url
    if not feeds:
        raise SourceIntakeError("at least one feed is required")
    if timeout_sec < 1:
        raise SourceIntakeError("timeout_sec must be >= 1")
    if max_bytes < 1:
        raise SourceIntakeError("max_bytes must be >= 1")
    # Revalidate programmatic callers too; config validation is not the only API
    # boundary.
    validated: list[FeedSpec] = []
    ids: set[str] = set()
    for feed in feeds:
        checked = FeedSpec.from_mapping(feed.to_dict()) if isinstance(feed, FeedSpec) else FeedSpec.from_mapping(feed)
        if checked.id in ids:
            raise SourceIntakeError("feed ids must be unique")
        ids.add(checked.id)
        validated.append(checked)
    created = created_at or _utc_now()
    # Ensure output identity does not depend on local timezone or random UUID.
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        raise SourceIntakeError("created_at must be an ISO-8601 timestamp") from None
    if parsed_created.tzinfo is None:
        raise SourceIntakeError("created_at must include a timezone")
    config_payload = {"version": 1, "feeds": [feed.to_dict() for feed in validated]}
    config_digest = _digest(_canonical_json(config_payload))
    previous_by_identity = _load_previous(previous)

    feed_records: list[dict[str, Any]] = []
    occurrences: list[_ParsedEntry] = []
    for feed in sorted(validated, key=lambda item: item.id):
        if not feed.enabled:
            feed_records.append(
                {
                    "id": feed.id,
                    "kind": feed.kind,
                    "url": feed.url,
                    "enabled": False,
                    "status": "disabled",
                    "fetched_at": created,
                    "response_digest": None,
                    "item_count": 0,
                    "warnings": [],
                }
            )
            continue
        record: dict[str, Any] = {
            "id": feed.id,
            "kind": feed.kind,
            "url": feed.url,
            "enabled": True,
            "fetched_at": created,
            "response_digest": None,
            "item_count": 0,
            "warnings": [],
        }
        try:
            if fetcher is fetch_url:
                response = fetch_url(
                    feed.url, timeout_sec=timeout_sec, max_bytes=max_bytes
                )
            else:
                # Injected fetchers intentionally receive only the URL; this
                # keeps offline tests deterministic and avoids masking a
                # genuine TypeError raised by a custom parser.
                response = fetcher(feed.url)
            if isinstance(response, bytes):
                response = FetchResponse(response)
            if not isinstance(response, FetchResponse):
                raise SourceIntakeError("custom fetcher returned an unsupported response")
            if len(response.body) > max_bytes:
                raise SourceIntakeError(f"feed response exceeds {max_bytes} bytes")
            record["http_status"] = int(response.status)
            if response.status < 200 or response.status >= 300:
                raise SourceIntakeError(f"feed returned HTTP status {response.status}")
            record["status"] = "ok"
            record["content_type"] = response.content_type
            record["response_digest"] = _digest(response.body)
            if response.etag:
                record["etag"] = _clean_text(response.etag, limit=200)
            if response.last_modified:
                record["last_modified"] = _clean_text(response.last_modified, limit=100)
            parsed = _parse_response(response, feed)
            record["item_count"] = len(parsed)
            occurrences.extend(parsed)
        except SourceIntakeError as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            # Error messages are generated locally and intentionally do not
            # contain response bodies or credential values.
            record["error"] = _safe_error_message(exc)
        except Exception as exc:  # defensive parser boundary; retain audit row
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = "unexpected feed parser failure"
        feed_records.append(record)

    grouped: dict[str, list[_ParsedEntry]] = {}
    for entry in occurrences:
        grouped.setdefault(_identity(entry), []).append(entry)
    items: list[IntakeItem] = []
    for identity_key, group in grouped.items():
        item = _item_from_group(identity_key, group)
        old = previous_by_identity.get(identity_key)
        if old is None:
            status = "new"
        elif str(old.get("content_digest", "")) == item.content_digest:
            status = "duplicate"
        else:
            status = "updated"
        items.append(
            IntakeItem(
                **{**item.__dict__, "dedupe_status": status}
            )
        )
    items.sort(key=lambda item: (item.source_kind, item.published_at or "", item.title.lower(), item.item_uid))

    queue: list[dict[str, Any]] = []
    for item in items:
        if item.dedupe_status not in {"new", "updated"}:
            continue
        queue.append(
            {
                "queue_id": item.item_uid,
                "item_uid": item.item_uid,
                "state": "pending",
                "priority": "normal",
                "dedupe_status": item.dedupe_status,
                "source_kind": item.source_kind,
                "title": item.title,
                "canonical_url": item.canonical_url,
                "content_digest": item.content_digest,
                "review_dimensions": [
                    "or_relevance",
                    "novelty",
                    "task_potential",
                    "reproducibility",
                ],
                "task_authoring": "human_review_required",
            }
        )
    queue.sort(key=lambda row: (row["source_kind"], row["title"].lower(), row["item_uid"]))

    # The source snapshot digest describes the fetched material, not the
    # caller's history.  In particular, ``dedupe_status`` changes from
    # ``new`` to ``duplicate`` on the next day but must not make identical
    # feed bytes look like different source content.  Fetch timestamps are
    # likewise run metadata, while response/status/item fields are evidence.
    snapshot_feeds = []
    for record in feed_records:
        snapshot_record = dict(record)
        snapshot_record.pop("fetched_at", None)
        snapshot_feeds.append(snapshot_record)
    snapshot_items = []
    for item in items:
        snapshot_item = item.to_dict()
        snapshot_item.pop("dedupe_status", None)
        snapshot_items.append(snapshot_item)
    snapshot_payload = {"feeds": snapshot_feeds, "items": snapshot_items}
    snapshot_digest = _digest(_canonical_json(snapshot_payload))
    intake_id = _digest(
        _canonical_json(
            {
                "schema_version": INTAKE_SCHEMA_VERSION,
                "collection_date": created[:10],
                "config_digest": config_digest,
                "snapshot_digest": snapshot_digest,
            }
        )
    )
    return IntakeResult(
        schema_version=INTAKE_SCHEMA_VERSION,
        intake_id=intake_id,
        created_at=created,
        config_digest=config_digest,
        snapshot_digest=snapshot_digest,
        feeds=tuple(feed_records),
        items=tuple(items),
        review_queue=tuple(queue),
        network_policy={
            "model_calls": 0,
            "credentials_read": False,
            "raw_sources_written": False,
            "task_generation": "disabled",
            "task_publishing": "disabled",
            "metadata_only": True,
        },
    )


def _write_atomic_if_same(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_bytes() == data:
                return
        except OSError:
            pass
        raise SourceIntakeError(f"refusing to overwrite existing intake artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _queue_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _check_existing(path: Path, data: bytes) -> None:
    if not path.exists():
        return
    try:
        same = path.read_bytes() == data
    except OSError:
        same = False
    if not same:
        raise SourceIntakeError(f"refusing to overwrite existing intake artifact: {path.name}")


def write_bundle(result: IntakeResult, destination: str | Path) -> dict[str, Path]:
    """Write an idempotent intake bundle and a hash manifest.

    The bundle contains no feed response body.  Re-running with the same
    result is safe; attempting to replace an existing artifact with a
    different result fails closed.
    """

    # Validate exactly what will be persisted before creating any artifact.
    # This catches accidental contract drift in future parser extensions.
    schema = schema_mod.load_schema(schema_mod.schemas_dir() / "source_intake.schema.json")
    schema_mod.validate(result.to_dict(), schema, name="source intake")
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    intake_bytes = (_canonical_json(result.to_dict()) + b"\n")
    queue_bytes = _queue_bytes(result.review_queue)
    manifest_payload = {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "intake_id": result.intake_id,
        "files": {
            "intake.json": _digest(intake_bytes),
            "review_queue.jsonl": _digest(queue_bytes),
        },
        "review_queue_count": len(result.review_queue),
        "review_queue_digest": result.review_queue_digest,
        "raw_response_bodies": "not_written",
        "credentials": "not_read",
        "model_calls": 0,
        "task_publication": "disabled",
    }
    manifest_bytes = (_canonical_json(manifest_payload) + b"\n")
    planned = (
        (out / "intake.json", intake_bytes),
        (out / "review_queue.jsonl", queue_bytes),
        (out / "intake-manifest.json", manifest_bytes),
    )
    # Preflight all paths so a conflicting third file cannot leave a partially
    # updated bundle behind.
    for path, data in planned:
        _check_existing(path, data)
    for path, data in planned:
        _write_atomic_if_same(path, data)
    return {
        "intake": out / "intake.json",
        "review_queue": out / "review_queue.jsonl",
        "manifest": out / "intake-manifest.json",
    }


def load_intake(path: str | Path) -> dict[str, Any]:
    """Load a previously written intake JSON for inspection/replay."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate /= "intake.json"
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SourceIntakeError(f"intake artifact is not valid JSON: {candidate}") from None
    if not isinstance(payload, dict) or payload.get("schema_version") != INTAKE_SCHEMA_VERSION:
        raise SourceIntakeError("intake artifact has an unsupported schema version")
    return payload


def validate_config_mapping(value: Mapping[str, Any]) -> tuple[FeedSpec, ...]:
    """Validate an in-memory config (useful for callers and tests)."""

    if not isinstance(value, Mapping):
        raise SourceIntakeError("feed config root must be an object")
    unknown = sorted(set(value) - {"version", "feeds"})
    if unknown:
        raise SourceIntakeError(f"feed config has unsupported key(s): {unknown}")
    if value.get("version", 1) != 1:
        raise SourceIntakeError("feed config version must be 1")
    feeds = value.get("feeds")
    if not isinstance(feeds, list) or not feeds:
        raise SourceIntakeError("feed config must contain a non-empty feeds list")
    parsed = tuple(FeedSpec.from_mapping(item, index=i) for i, item in enumerate(feeds))
    if len({feed.id for feed in parsed}) != len(parsed):
        raise SourceIntakeError("feed ids must be unique")
    return parsed
