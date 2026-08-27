"""Volcengine-backed authoring review for paper-derived TB-Science tasks.

The client uses only Volcengine routes: the Anthropic-compatible Coding Plan
endpoint for its configured alias and the same host's OpenAI-compatible Ark
endpoint for explicit model ids.  It is not a generic provider router:
refusing non-Volc hosts keeps the user's requirement that all agent work in
this pipeline uses the Volc route auditable.  Prompts contain
only public paper metadata, a bounded task-tree snapshot, and the static
authoring receipt.  Raw prompts/responses are never written to artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .core.errors import ORBenchError


class VolcReviewError(ORBenchError):
    """A Volcengine review could not be completed safely."""

    exit_code = 8


DEFAULT_MAX_TOKENS = 1600
DEFAULT_TIMEOUT_SEC = 120
MAX_FILE_BYTES = 64_000
MAX_SNAPSHOT_BYTES = 256_000
VOLC_HOST_SUFFIXES = ("volces.com", "volcengine.com")
REQUIRED_REVIEW_CRITERIA = frozenset(
    {
        "verifiable",
        "well_specified",
        "solvable",
        "difficult",
        "scientifically_grounded",
        "scope",
        "outcome_verified",
    }
)
_SECRET_FILENAME = re.compile(
    r"(?:api[_-]?key|apikey|password|passwd|bearer|token|secret|credential|auth|"
    r"private[_-]?key|id_rsa|id_dsa|id_ed25519)",
    re.I,
)
_SECRET_CONTENT = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(
        r"(?:api[_-]?key|apikey|password|passwd|token|secret|authorization|bearer|"
        r"aws_access_key_id|aws_secret_access_key|github_token)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9_./+=:-]{8,}",
        re.I,
    ),
)


@dataclass(frozen=True)
class VolcConfig:
    base_url: str
    token: str
    default_model: str
    timeout_sec: int = DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls, *, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> "VolcConfig":
        base_url = str(os.environ.get("ANTHROPIC_BASE_URL", "")).strip().rstrip("/")
        token = str(os.environ.get("ANTHROPIC_AUTH_TOKEN", "")).strip()
        model = str(os.environ.get("ANTHROPIC_MODEL", "ark-code-latest")).strip()
        if not base_url:
            raise VolcReviewError("ANTHROPIC_BASE_URL is missing")
        if not token:
            raise VolcReviewError("ANTHROPIC_AUTH_TOKEN is missing")
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise VolcReviewError("ANTHROPIC_BASE_URL must be an HTTPS URL")
        host = parsed.hostname.lower().rstrip(".")
        if not host.endswith(VOLC_HOST_SUFFIXES):
            raise VolcReviewError("agent review refuses a non-Volcengine endpoint")
        if not model:
            raise VolcReviewError("ANTHROPIC_MODEL is missing")
        if timeout_sec <= 0:
            raise VolcReviewError("Volc review timeout must be positive")
        return cls(base_url=base_url, token=token, default_model=model, timeout_sec=timeout_sec)

    @property
    def endpoint(self) -> str:
        return self.base_url + "/v1/messages"

    @property
    def openai_endpoint(self) -> str:
        parsed = urlsplit(self.base_url)
        return urlunsplit((parsed.scheme, parsed.netloc, "/api/v3/chat/completions", "", ""))

    def public_dict(self) -> dict[str, Any]:
        parsed = urlsplit(self.base_url)
        return {
            "provider": "volcengine",
            "base_host": parsed.hostname,
            "base_path": parsed.path,
            "token_present": bool(self.token),
            "model_default": self.default_model,
            "route_digest": _digest({"host": parsed.hostname, "path": parsed.path}),
        }


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_public_text(path: str, text: str | None = None) -> None:
    name = PurePath(path).name
    lower = name.lower()
    if (
        name.startswith(".")
        or _SECRET_FILENAME.search(name)
        or lower.endswith((".pem", ".key", ".p12", ".pfx", ".keystore"))
    ):
        raise VolcReviewError(f"task snapshot refuses hidden or credential-like file: {path}")
    if text is not None and any(pattern.search(text) for pattern in _SECRET_CONTENT):
        raise VolcReviewError(f"task snapshot refuses credential-like content: {path}")


def _bounded_task_snapshot(task_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    preview_bytes = 0
    for path in sorted(
        (p for p in task_dir.rglob("*") if p.is_file() and not p.is_symlink()),
        key=lambda p: p.relative_to(task_dir).as_posix(),
    ):
        relative = path.relative_to(task_dir).as_posix()
        parts = path.relative_to(task_dir).parts
        if any(part.startswith(".") for part in parts):
            raise VolcReviewError(f"task snapshot refuses hidden or credential-like file: {relative}")
        _assert_public_text(relative)
        entry: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "digest": _file_digest(path)}
        if path.stat().st_size <= MAX_FILE_BYTES and path.suffix.lower() in {".md", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".py", ".sh"}:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                text = ""
            # The reviewer needs structure, not an unbounded data dump.  Do not
            # include files in common secret/data locations even if present.
            if not any(part.lower() in {"secret", "secrets", ".ssh", ".aws", ".config"} for part in path.parts):
                preview = text[:MAX_FILE_BYTES]
                _assert_public_text(relative, preview)
                preview_bytes += len(preview.encode("utf-8"))
                if preview_bytes > MAX_SNAPSHOT_BYTES:
                    raise VolcReviewError("task snapshot previews exceed 256000 UTF-8 bytes")
                entry["preview"] = preview
        files.append(entry)
    return files


def _task_tree_digest(task_dir: Path) -> str:
    entries = []
    for path in sorted(task_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            entries.append(
                {
                    "path": path.relative_to(task_dir).as_posix(),
                    "content_digest": _file_digest(path),
                }
            )
    return _digest(entries)


def _validate_review_inputs(
    task_dir: Path,
    paper: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    round_number: int,
) -> None:
    """Reject stale or forged authoring inputs before any paid model call."""

    if receipt.get("authoring_schema_version") != "orbenchlab.tbscience-authoring.v1":
        raise VolcReviewError("authoring receipt schema is unsupported")
    supplied = receipt.get("receipt_digest")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if not isinstance(supplied, str) or supplied != _digest(unsigned):
        raise VolcReviewError("authoring receipt digest does not match its contents")
    if receipt.get("task_dir") != task_dir.name:
        raise VolcReviewError("authoring receipt task directory does not match current task")
    current_tree = _task_tree_digest(task_dir)
    if receipt.get("task_tree_digest") != current_tree:
        raise VolcReviewError("authoring receipt task-tree digest is stale")
    if (
        not isinstance(round_number, int)
        or isinstance(round_number, bool)
        or round_number <= 0
        or receipt.get("round") != round_number
    ):
        raise VolcReviewError("authoring receipt round does not match review round")
    receipt_paper = receipt.get("paper")
    if not isinstance(receipt_paper, Mapping) or (
        receipt_paper.get("source_content_digest") != paper.get("source_content_digest")
    ):
        raise VolcReviewError("paper digest does not match authoring receipt")
    # Materialize the bounded snapshot now so secret-like filenames and the
    # aggregate preview cap fail before the first provider request.
    _bounded_task_snapshot(task_dir)


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        # Models sometimes add a short preamble or trailing prose.  Decode the
        # first complete object instead of greedily matching through unrelated
        # braces in that prose; a truncated object still fails closed.
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise VolcReviewError("Volc reviewer returned malformed JSON") from None
    if not isinstance(value, dict):
        raise VolcReviewError("Volc reviewer response must be a JSON object")
    return value


def _safe_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, list):
        raise VolcReviewError("Volc response has no message content")
    text = "".join(item.get("text", "") for item in content if isinstance(item, Mapping))
    if not text.strip():
        raise VolcReviewError("Volc response content is empty")
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    return {
        "parsed": _json_object(text),
        "response_digest": _digest({"text": text}),
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        },
    }


def _safe_openai_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    text = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(text, str) or not text.strip():
        raise VolcReviewError("Volc OpenAI-compatible response content is empty")
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    return {
        "parsed": _json_object(text),
        "response_digest": _digest({"text": text}),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    }


def _normalize_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize heterogeneous model JSON into the review contract.

    Some Volc models answer a constrained prompt with one difficulty-axis
    object rather than the requested envelope.  Preserve that evidence as a
    partial, human-review result instead of treating a parser success as a
    complete review.
    """

    allowed_decisions = {"revise", "promising", "needs-human"}
    decision = str(value.get("decision") or value.get("verdict") or "needs-human").lower()
    if decision not in allowed_decisions:
        decision = "needs-human"
    axes = value.get("difficulty_axes")
    if not isinstance(axes, list):
        axes = []
    if value.get("name") is not None and value.get("levels") is not None:
        axes = [
            {
                "name": value.get("name"),
                "levels": value.get("levels"),
                "evidence": value.get("evidence", ""),
                "risk": value.get("risk", ""),
            }
        ]
    def _strings(*keys: str) -> list[str]:
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [str(item) for item in candidate]
            if isinstance(candidate, str) and candidate.strip():
                return [candidate]
        return []
    raw_criteria = value.get("criteria") if isinstance(value.get("criteria"), list) else []
    criteria: list[dict[str, Any]] = []
    names: list[str] = []
    for item in raw_criteria:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        status = str(item.get("status") or "review")
        evidence = item.get("evidence")
        if name not in REQUIRED_REVIEW_CRITERIA or status not in {"pass", "fail", "review"}:
            continue
        criteria.append(
            {
                "name": name,
                "status": status,
                "evidence": str(evidence or ""),
                "next_action": str(item.get("next_action") or ""),
            }
        )
        names.append(name)
    rubric_complete = (
        len(names) == len(set(names))
        and set(names) == REQUIRED_REVIEW_CRITERIA
        and all(row["evidence"].strip() for row in criteria)
    )
    shape_complete = all(
        key in value
        for key in (
            "decision",
            "task_summary",
            "blocking_findings",
            "difficulty_axes",
            "criteria",
            "suggested_edits",
        )
    )
    if decision == "promising" and (
        not shape_complete
        or not rubric_complete
        or any(row["status"] != "pass" for row in criteria)
    ):
        decision = "needs-human"
    return {
        "decision": decision,
        "task_summary": str(value.get("task_summary") or value.get("summary") or ""),
        "blocking_findings": _strings("blocking_findings", "blocking_issues", "findings"),
        "difficulty_axes": axes,
        "criteria": criteria,
        "suggested_edits": _strings("suggested_edits", "recommendations", "next_actions"),
        "shape_complete": shape_complete,
        "rubric_complete": rubric_complete,
        "source_keys": sorted(str(key) for key in value.keys()),
    }


def call_reviewer(
    config: VolcConfig,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Call one Volc model and return only structured, digestable evidence."""

    if not model or max_tokens <= 0:
        raise VolcReviewError("model and max_tokens must be positive")
    protocol = "anthropic" if model in {config.default_model, "ark-code-latest"} else "openai"
    if protocol == "anthropic":
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        endpoint = config.endpoint
        headers = {
            "x-api-key": config.token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        endpoint = config.openai_endpoint
        headers = {
            "authorization": f"Bearer {config.token}",
            "content-type": "application/json",
        }
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VolcReviewError(f"Volc reviewer request failed: {type(exc).__name__}") from None
    if not isinstance(response_payload, Mapping):
        raise VolcReviewError("Volc reviewer returned a non-object response")
    result = (
        _safe_response(response_payload)
        if protocol == "anthropic"
        else _safe_openai_response(response_payload)
    )
    result.update(
        {
            "model": model,
            "protocol": protocol,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "request_digest": _digest(
                {"model": model, "protocol": protocol, "system": system, "user": user}
            ),
        }
    )
    return result


def _review_prompt(task_dir: Path, paper: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    safe_paper = {
        key: paper.get(key)
        for key in ("title", "url", "source_content_digest", "license_status")
        if key in paper
    }
    safe_receipt = {
        "decision": receipt.get("decision"),
        "round": receipt.get("round"),
        "counts": receipt.get("counts"),
        "task_tree_digest": receipt.get("task_tree_digest"),
        "implementation_criteria": receipt.get("implementation_criteria", []),
        "provenance_checks": receipt.get("provenance_checks", []),
    }
    payload = {"paper": safe_paper, "receipt": safe_receipt, "task_files": _bounded_task_snapshot(task_dir)}
    return (
        "You are reviewing a paper-derived Terminal-Bench Science task. "
        "Use only the supplied evidence; do not invent paper claims or hidden tests. "
        "Return JSON only with keys: decision (revise|promising|needs-human), "
        "task_summary (string), blocking_findings (array of strings), "
        "difficulty_axes (array of objects with name, levels, evidence, risk), "
        "criteria (array of objects with name, status pass|fail|review, evidence, next_action), "
        "suggested_edits (array of strings). Criteria must contain each of these names exactly once: "
        + ", ".join(sorted(REQUIRED_REVIEW_CRITERIA))
        + ". A promising decision requires non-empty evidence and pass status for every criterion. "
        "A static blocked receipt cannot become accepted.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def review_task(
    task_dir: str | Path,
    *,
    paper_provenance: Mapping[str, Any],
    receipt: Mapping[str, Any],
    config: VolcConfig,
    models: Sequence[str],
    round_number: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Run independent Volc reviews and aggregate a conservative round report."""

    root = Path(task_dir)
    if not root.is_dir() or root.is_symlink():
        raise VolcReviewError("task directory must be a real directory")
    selected = [str(model).strip() for model in models if str(model).strip()]
    if not selected:
        selected = [config.default_model]
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise VolcReviewError("authoring review requires at least two distinct reviewer model ids")
    _validate_review_inputs(
        root,
        paper_provenance,
        receipt,
        round_number=round_number,
    )
    system = (
        "You are an evidence-calibrated TB-Science task reviewer. "
        "Do not expose secrets. Do not claim Harbor acceptance. Keep semantic judgments as review when evidence is insufficient."
    )
    reports: list[dict[str, Any]] = []
    prompt = _review_prompt(root, paper_provenance, receipt)
    for model in selected:
        result = call_reviewer(config, model=model, system=system, user=prompt, max_tokens=max_tokens)
        parsed = result.pop("parsed")
        result["review"] = _normalize_review(parsed)
        reports.append(result)
    decisions = [str((item.get("review") or {}).get("decision", "needs-human")) for item in reports]
    if receipt.get("decision") == "blocked":
        aggregate = "blocked-static-gate"
    elif any(decision == "revise" for decision in decisions):
        aggregate = "revise"
    elif all(decision == "promising" for decision in decisions):
        aggregate = "promising-needs-harbor"
    else:
        aggregate = "needs-human"
    return {
        "schema_version": "orbenchlab.volc-authoring-review.v1",
        "round": int(round_number),
        "provider": config.public_dict(),
        "task_dir": root.name,
        "task_tree_digest": receipt.get("task_tree_digest"),
        "paper_digest": paper_provenance.get("source_content_digest"),
        "static_receipt_digest": receipt.get("receipt_digest"),
        "models": selected,
        "max_tokens": max_tokens,
        "review_count": len(reports),
        "aggregate_decision": aggregate,
        "evidence_level": "E1-model-review",
        "reviewers": reports,
        "limitations": [
            "Model review is a proposal for the next authoring round, not TB-Science acceptance.",
            "No hidden verifier, model trajectory, or Harbor runtime result was supplied to this review.",
            "All reviewer text is bounded to structured output; raw prompt/response bodies are not persisted.",
        ],
    }


def write_review(review: Mapping[str, Any], out: str | Path) -> dict[str, Path]:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(review)
    payload["review_digest"] = _digest(payload)
    json_path = output / "volc-authoring-review.json"
    markdown_path = output / "volc-authoring-review.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Volcengine authoring review",
        "",
        f"- Decision: **{payload.get('aggregate_decision')}**",
        f"- Round: `{payload.get('round')}`",
        f"- Models: `{', '.join(payload.get('models', []))}`",
        f"- Evidence: `{payload.get('evidence_level')}`",
        f"- Review digest: `{payload['review_digest']}`",
        "",
    ]
    for reviewer in payload.get("reviewers", []):
        review_data = reviewer.get("review", {}) if isinstance(reviewer, Mapping) else {}
        lines.extend([f"## {reviewer.get('model', 'unknown model')}", "", f"Decision: `{review_data.get('decision', 'needs-human')}`", ""])
        findings = review_data.get("blocking_findings", [])
        if findings:
            lines.append("Blocking findings:")
            lines.extend(f"- {str(item)}" for item in findings)
            lines.append("")
        edits = review_data.get("suggested_edits", [])
        if edits:
            lines.append("Suggested edits:")
            lines.extend(f"- {str(item)}" for item in edits)
            lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {value}" for value in payload.get("limitations", []))
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


__all__ = [
    "VolcConfig",
    "VolcReviewError",
    "call_reviewer",
    "review_task",
    "write_review",
]
