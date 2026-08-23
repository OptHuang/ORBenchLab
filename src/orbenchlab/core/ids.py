"""Deterministic identifier algebra.

Every function here is a pure function of its inputs: no I/O, no randomness, no
clock. That is the whole point. Harbor generates trial names with
``ShortUUID().random(7)``, so a run cannot be identified by anything Harbor
writes; ORBenchLab therefore derives its own content-addressed ``run_id`` at
compile time and reconciles it back to Harbor trials through a plan ledger.

See ``docs/decisions/0002-plan-ledger-run-ids.md``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

CANONICAL_JSON_KWARGS = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}

RUN_ID_LENGTH = 16
CAMPAIGN_DIGEST_LENGTH = 8

#: Keys excluded from ``campaign_cfg_digest``. Harbor's ``JobConfig.__eq__``
#: excludes ``job_name`` and ``debug`` for replay identity; we stay consistent,
#: and additionally drop free-text annotation fields and output locations.
#:
#: ``jobs_dir`` is excluded for the same reason as ``job_name``: where results
#: are written is not part of what was measured. Excluding it is also what lets
#: a campaign be compiled once, have its output root derived from its own id,
#: and be recompiled into that root without the id shifting underneath.
CFG_DIGEST_EXCLUDED_KEYS = ("job_name", "jobs_dir", "debug", "comments", "description")


def canon(obj: Any) -> bytes:
    """Canonical JSON encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, **CANONICAL_JSON_KWARGS).encode("utf-8")  # type: ignore[arg-type]


def sha(*parts: bytes | str) -> str:
    """sha256 hexdigest over the concatenation of ``parts``."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8") if isinstance(part, str) else part)
    return digest.hexdigest()


def short(hexdigest: str, n: int) -> str:
    return hexdigest[:n]


def task_uid(bench: str, dataset_digest: str, task_name: str, task_checksum: str) -> str:
    """``{bench}@{dataset_digest}/{task_name}#{task_checksum}``.

    ``task_checksum`` comes from Harbor's ``TrialResult.task_checksum`` once a
    trial exists. At compile time it is not yet known, so callers pass the
    planned placeholder ``"pending"``; the ledger records which form was used.
    """
    return f"{bench}@{dataset_digest}/{task_name}#{task_checksum}"


def agent_uid(
    *,
    scaffold: str,
    scaffold_version: str | None,
    model_id: str | None,
    prompt_digest: str | None,
    tool_policy: Mapping[str, Any] | None,
    agent_kwargs: Mapping[str, Any] | None,
    env_keys: Iterable[str] | None,
    env_from_secret: Mapping[str, str] | None = None,
    env_literals: Mapping[str, str] | None = None,
    auth_mode: str | None = None,
    provider_route_digest: str | None = None,
    import_path: str | None = None,
    setup_timeout_sec: int | None = None,
) -> str:
    """Identity of an agent configuration.

    Only environment variable *names* participate — both the variable the
    scaffold reads and the name of the secret supplying it. Values are never
    hashed and never written to disk, so rotating a credential does not change
    any id while renaming the variable (a real configuration change) does.

    **Canonicalization rule:** the seven original fields always participate;
    the later fields participate only when set to something other than their
    default. Explicitly setting a field to its default is therefore not a
    configuration change, which is what keeps ids stable across releases that
    add expressiveness rather than changing meaning.
    """
    payload: dict[str, Any] = {
        "scaffold": scaffold,
        "scaffold_version": scaffold_version,
        "model_id": model_id,
        "prompt_digest": prompt_digest,
        "tool_policy": dict(tool_policy or {}),
        "agent_kwargs": dict(agent_kwargs or {}),
        "env_keys": sorted(env_keys or []),
    }
    if env_from_secret:
        # Both sides are names. A value never reaches this function.
        payload["env_from_secret"] = {str(k): str(v) for k, v in sorted(env_from_secret.items())}
    if env_literals:
        payload["env_literals"] = {str(k): str(v) for k, v in sorted(env_literals.items())}
    if auth_mode is not None and auth_mode != "api-key":
        # API-key is the historical default; only a non-default transport
        # changes identity so existing campaigns keep stable ids.
        payload["auth_mode"] = auth_mode
    if provider_route_digest:
        # This is a digest of the normalized credential destination, never the
        # URL or a credential. Provider routing changes what was measured and
        # must therefore change agent and run identity.
        payload["provider_route_digest"] = provider_route_digest
    if import_path:
        payload["import_path"] = import_path
    if setup_timeout_sec is not None:
        payload["setup_timeout_sec"] = int(setup_timeout_sec)
    return sha(canon(payload))


def budget_uid(budget: Mapping[str, Any]) -> str:
    return sha(canon(dict(budget)))


def campaign_cfg_digest(spec: Mapping[str, Any]) -> str:
    """Digest of a campaign spec with non-semantic fields removed."""
    return sha(canon(_strip_excluded(spec)))


def _strip_excluded(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            key: _strip_excluded(value)
            for key, value in obj.items()
            if key not in CFG_DIGEST_EXCLUDED_KEYS
        }
    if isinstance(obj, list):
        return [_strip_excluded(item) for item in obj]
    return obj


def run_id(
    *,
    cfg_digest: str,
    task_uid_value: str,
    agent_uid_value: str,
    budget_uid_value: str,
    seed: int,
    attempt: int,
) -> str:
    """Stable external run id: a pure function of frozen inputs.

    Seed is a separate component rather than part of ``agent_uid`` so that agent
    identity stays stable across seeds.
    """
    return short(
        sha(
            canon(
                [
                    cfg_digest,
                    task_uid_value,
                    agent_uid_value,
                    budget_uid_value,
                    int(seed),
                    int(attempt),
                ]
            )
        ),
        RUN_ID_LENGTH,
    )


def campaign_id(slug: str, date_yyyymmdd: str, cfg_digest: str) -> str:
    return f"{slug}-{date_yyyymmdd}-{short(cfg_digest, CAMPAIGN_DIGEST_LENGTH)}"


def score_run_id(run_id_value: str, verifier_digest: str, rescore_reason: str) -> str:
    """Re-scoring produces a new id, so one rollout can carry several verdicts."""
    return short(sha(canon([run_id_value, verifier_digest, rescore_reason])), RUN_ID_LENGTH)


def shard_of(run_id_value: str, n_shards: int) -> int:
    """Pure-function sharding: every machine computes the same assignment."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    return int(run_id_value[:8], 16) % n_shards


def match_key(*, task_name: str, agent_uid_value: str, seed: int, attempt: int) -> str:
    """Key used to reconcile a Harbor trial back to a ledger entry.

    Harbor never round-trips our ``run_id``, so ingest matches on this instead.
    The compiler guarantees one ``(agent_uid, seed, attempt)`` per job with
    ``n_attempts = 1``, which makes ``(job_name, task_name)`` uniquely
    identifying and reduces matching to an exact lookup.
    """
    return sha(canon([task_name, agent_uid_value, int(seed), int(attempt)]))
