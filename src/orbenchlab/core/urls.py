"""Validation for provider endpoints that are persisted in run artifacts.

Provider URLs are configuration, but they still decide where credentials are
sent.  Treat them as a security boundary: only a plain HTTPS origin/path may
be serialized into a plan.  Credentials, query strings and fragments never
belong in an auditable benchmark artifact.
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit

from .errors import SpecError


def validate_https_base_url(value: str) -> str:
    """Return a canonical HTTPS provider URL or fail without echoing it.

    Canonicalization is deliberately conservative: the scheme and host are
    lower-cased and an explicit default port is removed, while the path is
    retained byte-for-byte because provider gateways may distinguish routes
    below the origin.  A trailing slash therefore remains an identity change.
    """
    candidate = str(value).strip()
    if not candidate:
        raise SpecError("provider base URL is required")
    if any(character.isspace() or ord(character) < 0x20 for character in candidate):
        raise SpecError("provider base URL may not contain whitespace or control characters")
    try:
        parsed = urlsplit(candidate)
        # Accessing ``port`` also validates malformed bracket/port syntax.
        _ = parsed.port
    except ValueError:
        raise SpecError("provider base URL is malformed") from None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise SpecError("provider base URL must use HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise SpecError("provider base URL may not contain userinfo or credentials")
    if parsed.query or parsed.fragment:
        raise SpecError("provider base URL may not contain a query string or fragment")
    if "\\" in candidate:
        raise SpecError("provider base URL may not contain backslashes")

    host = parsed.hostname.lower()
    # urlsplit strips brackets from IPv6 hostnames; put them back in a netloc.
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return urlunsplit(("https", netloc, parsed.path, "", ""))


def provider_route_digest(value: str) -> str:
    """Content identity for a credential destination, never the URL itself."""
    normalized = validate_https_base_url(value)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
