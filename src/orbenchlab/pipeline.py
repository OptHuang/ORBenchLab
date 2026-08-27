"""Deterministic, non-interactive OR task pipeline summaries.

The execution layers already produce source-intake bundles, screening records,
and Harbor/Volc result reports.  This module is the small final-mile
orchestrator: it discovers those artifacts, joins them by task/family, and
writes one human-facing task card plus a machine-readable manifest per task.

It intentionally does not invent task content, read raw trajectories, or call
a model.  A candidate with missing provenance, failed controls, or only
single-run evidence is still emitted, but its status is conservative and its
limitations are visible.  That makes ``orbench pipeline run`` safe for a daily
unattended job whose only human interaction is reading the final cards.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .core import schema as schema_mod
from .core.errors import ORBenchError


TASK_CARD_SCHEMA = "task_card.schema.json"
PIPELINE_SCHEMA_VERSION = "orbenchlab.pipeline.v1"
TASK_CARD_SCHEMA_VERSION = "orbenchlab.task-card.v1"
_SUPPORTED_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
_EVIDENCE_LEVELS = frozenset({"E0", "E1", "E2", "E3", "E4", "E5"})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PipelineError(ORBenchError):
    """Invalid pipeline input or output."""

    exit_code = 8


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _value_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _validate_harbor_receipt(document: Mapping[str, Any], source: Path) -> None:
    """Fail closed before a Harbor receipt can unlock task promotion.

    The raw Harbor job is validated when the receipt is authored.  At intake we
    revalidate the complete checksummed summary instead of trusting a marker field or
    two user-controlled ``gate=pass`` strings.
    """

    if (
        document.get("schema_version") != "orbenchlab.screening-report.v1"
        or document.get("harbor_receipt_schema_version") != "orbenchlab.harbor-controls.v1"
    ):
        raise PipelineError(f"unsupported Harbor receipt schema in {source}")
    supplied_digest = document.get("report_digest")
    unsigned = {key: value for key, value in document.items() if key != "report_digest"}
    if not isinstance(supplied_digest, str) or supplied_digest != _value_digest(unsigned):
        raise PipelineError(f"Harbor receipt digest mismatch in {source}")
    digests = [
        document.get("task_tree_digest"),
        document.get("authoring_task_tree_digest"),
        document.get("executed_task_tree_digest"),
    ]
    if any(not isinstance(value, str) or not _DIGEST_RE.fullmatch(value) for value in digests):
        raise PipelineError(f"Harbor receipt task-tree digest is missing or malformed in {source}")
    if len(set(digests)) != 1:
        raise PipelineError(f"Harbor receipt authoring/executed task digests differ in {source}")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise PipelineError(f"Harbor receipt must contain exactly one task in {source}")
    item = tasks[0]
    task = item.get("task")
    controls = item.get("control_gates")
    if not isinstance(task, str) or not task or not isinstance(controls, Mapping):
        raise PipelineError(f"Harbor receipt task/control identity is malformed in {source}")
    if item.get("evidence_level") != "E3" or item.get("arms") not in ({}, None):
        raise PipelineError(f"Harbor control receipt must be E3 and contain no model arms in {source}")
    digest_fields = (
        "job_result_digest",
        "trial_result_digest",
        "ctrf_digest",
        "reward_digest",
        "artifact_manifest_digest",
    )
    count_keys = ("tests", "passed", "failed", "skipped", "pending", "other")
    for name, expected_reward in (("oracle", 1.0), ("nop", 0.0)):
        control = controls.get(name)
        if not isinstance(control, Mapping) or control.get("gate") != "pass":
            raise PipelineError(f"Harbor {name} gate did not pass in {source}")
        if control.get("control") != name or control.get("reward") != expected_reward:
            raise PipelineError(f"Harbor {name} reward/control semantics are invalid in {source}")
        if any(
            not isinstance(control.get(field), str)
            or not _DIGEST_RE.fullmatch(str(control[field]))
            for field in digest_fields
        ):
            raise PipelineError(f"Harbor {name} evidence digests are missing in {source}")
        counts = control.get("ctrf_summary")
        if not isinstance(counts, Mapping) or any(
            not isinstance(counts.get(key), int) or int(counts[key]) < 0 for key in count_keys
        ):
            raise PipelineError(f"Harbor {name} CTRF counts are malformed in {source}")
        if counts["tests"] <= 0 or sum(counts[key] for key in count_keys[1:]) != counts["tests"]:
            raise PipelineError(f"Harbor {name} CTRF counts are inconsistent in {source}")
        if name == "oracle" and counts["passed"] != counts["tests"]:
            raise PipelineError(f"Harbor oracle did not pass every test in {source}")
        if name == "nop" and counts["failed"] <= 0:
            raise PipelineError(f"Harbor NOP did not fail the verifier in {source}")
        observed_task = str(control.get("task_name", "")).rsplit("/", 1)[-1].replace("-", "_")
        if observed_task != task.replace("-", "_"):
            raise PipelineError(f"Harbor {name} task identity mismatch in {source}")


def _load(path: Path) -> Any:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"could not read {path}: {type(exc).__name__}") from None


def _files(inputs: Sequence[str | Path] | None) -> list[Path]:
    """Expand explicit files/directories without following symlinks."""

    found: list[Path] = []
    for raw in inputs or ():
        path = Path(raw)
        if path.is_symlink():
            raise PipelineError(f"pipeline input may not be a symlink: {path}")
        if path.is_file():
            if path.suffix.lower() in _SUPPORTED_SUFFIXES:
                found.append(path)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    continue
                if child.is_file() and child.suffix.lower() in _SUPPORTED_SUFFIXES:
                    found.append(child)
            continue
        raise PipelineError(f"pipeline input not found: {path}")
    return sorted(set(found), key=lambda p: p.as_posix())


def _source_ref(genome: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = genome.get("source")
    if not isinstance(raw, Mapping):
        return None
    # Keep only provenance fields.  In particular, never copy a source summary
    # or arbitrary extracted text into a final card.
    keys = (
        "title",
        "url",
        "related_code_url",
        "intake_snapshot_id",
        "intake_snapshot_digest",
        "intake_item_uid",
        "source_content_digest",
        "intake_status",
    )
    return {key: raw[key] for key in keys if key in raw}


def _difficulty_axes(genome: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = genome.get("difficulty_axes")
    if isinstance(raw, Mapping):
        rows = []
        for name in sorted(raw):
            value = raw[name]
            if isinstance(value, Mapping):
                rows.append(
                    {
                        "name": str(name),
                        "levels": list(value.get("levels") or []),
                        "meaning": value.get("meaning"),
                        "expected_direction": value.get("expected_direction"),
                    }
                )
            else:
                rows.append({"name": str(name), "levels": list(value) if isinstance(value, list) else []})
        return rows
    raw = genome.get("difficulty")
    if isinstance(raw, Mapping) and isinstance(raw.get("knobs"), Mapping):
        return [
            {"name": str(name), "levels": [bounds.get("min"), bounds.get("max")]}
            for name, bounds in sorted(raw["knobs"].items())
            if isinstance(bounds, Mapping)
        ]
    return []


def _purpose(genome: Mapping[str, Any], task: str) -> str:
    for key in ("design_goal", "purpose", "description", "title"):
        value = genome.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return task


def _normalise_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    def number(key: str) -> float | None:
        value = arm.get(key)
        if not isinstance(value, (int, float)):
            return None
        result = float(value)
        if not math.isfinite(result) or result < 0 or result > 1:
            raise PipelineError(f"invalid {key}: expected a finite rate in [0, 1]")
        return result

    def integer(key: str) -> int:
        value = arm.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    n = integer("n")
    complete = integer("complete")
    metric_n = integer("metric_n")
    if min(n, complete, metric_n) < 0 or complete > n or metric_n > complete:
        raise PipelineError("invalid screening counts: require 0 <= metric_n <= complete <= n")
    solve_rate = number("solve_rate")
    quality_rate = number("quality_pass_rate")
    feasibility = number("mean_feasibility")
    solve_n = integer("solve_n") if "solve_n" in arm else (metric_n if solve_rate is not None else 0)
    quality_n = integer("quality_n") if "quality_n" in arm else (metric_n if quality_rate is not None else 0)
    feasibility_n = integer("feasibility_n") if "feasibility_n" in arm else (metric_n if feasibility is not None else 0)
    if any(value < 0 or value > complete for value in (solve_n, quality_n, feasibility_n)):
        raise PipelineError("invalid per-metric denominator: require 0 <= metric_n <= complete")
    infra = list(arm.get("infra_exceptions") or []) if isinstance(arm.get("infra_exceptions"), list) else []
    failures = list(arm.get("failure_modes") or []) if isinstance(arm.get("failure_modes"), list) else []
    return {
        "n": n,
        "complete": complete,
        "metric_n": metric_n,
        "solve_n": solve_n,
        "quality_n": quality_n,
        "feasibility_n": feasibility_n,
        "solve_rate": solve_rate,
        "quality_pass_rate": quality_rate,
        "mean_feasibility": feasibility,
        "infra_exceptions": sorted(str(x) for x in infra),
        "failure_modes": sorted(str(x) for x in failures),
    }


def _report_rows(document: Any, source: Path) -> list[dict[str, Any]]:
    """Convert both ORBenchLab and ORWorkbench screening shapes to task rows."""

    if not isinstance(document, Mapping):
        return []
    # ORBenchLab screening report: one row per task, with model arms.
    tasks = document.get("tasks")
    if isinstance(tasks, list):
        document_kind = (
            "harbor-controls"
            if document.get("harbor_receipt_schema_version")
            else "model-screening"
        )
        if document_kind == "harbor-controls":
            _validate_harbor_receipt(document, source)
        rows: list[dict[str, Any]] = []
        for item in tasks:
            if not isinstance(item, Mapping) or not item.get("task"):
                continue
            arms = item.get("arms") if isinstance(item.get("arms"), Mapping) else {}
            evidence_level = item.get("evidence_level")
            if evidence_level is not None and str(evidence_level) not in _EVIDENCE_LEVELS:
                raise PipelineError(f"unsupported evidence level in {source}: {evidence_level}")
            observed_gap = item.get("discrimination_index_observed_gap")
            if observed_gap is not None and (
                not isinstance(observed_gap, (int, float))
                or not math.isfinite(float(observed_gap))
                or float(observed_gap) < 0
                or float(observed_gap) > 1
            ):
                raise PipelineError(f"invalid observed model gap in {source}")
            rows.append(
                {
                    "task": str(item["task"]),
                    "family": str(item.get("family") or item["task"]),
                    "arms": {str(name): _normalise_arm(value) for name, value in arms.items() if isinstance(value, Mapping)},
                    "controls": dict(item.get("control_gates")) if isinstance(item.get("control_gates"), Mapping) else None,
                    "decision": item.get("decision"),
                    "evidence_level": evidence_level,
                    "task_tree_digest": item.get("task_tree_digest") or document.get("task_tree_digest"),
                    "observed_gap": observed_gap,
                    "limitations": [str(x) for x in item.get("limitations", []) if isinstance(x, str)],
                    "source_report": str(source),
                    "report_digest": _sha256(source),
                    "kind": document_kind,
                }
            )
        return rows
    # ORWorkbench screening_record: use the family as task identity and keep
    # control cells as evidence. It has no model arms, by design.
    family = document.get("family")
    if family:
        levels = document.get("levels") if isinstance(document.get("levels"), list) else []
        grid = document.get("grid") if isinstance(document.get("grid"), Mapping) else {}
        limitations = [str(x) for x in document.get("limitations", []) if isinstance(x, str)]
        return [
            {
                "task": str(family),
                "family": str(family),
                "arms": {},
                "controls": {
                    "decision": document.get("decision"),
                    "overall": document.get("overall") if isinstance(document.get("overall"), Mapping) else {},
                    "levels": levels,
                    "grid": dict(grid),
                    "gates": document.get("gates") if isinstance(document.get("gates"), Mapping) else {},
                },
                "decision": document.get("decision"),
                "evidence_level": "E3" if document.get("evidence_source_class") else None,
                "observed_gap": (document.get("overall") or {}).get("discrimination_index") if isinstance(document.get("overall"), Mapping) else None,
                "limitations": limitations,
                "source_report": str(source),
                "report_digest": _sha256(source),
                "kind": "control-screening",
            }
        ]
    return []


def _merge_performance(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        for model, raw_arm in sorted((row.get("arms") or {}).items()):
            arm = _normalise_arm(raw_arm)
            entry = grouped.setdefault(
                model,
                {
                    "model": model,
                    "n_observed": 0,
                    "n_complete": 0,
                    "metric_n": 0,
                    "solve_n": 0,
                    "quality_n": 0,
                    "feasibility_n": 0,
                    "solve_successes": 0.0,
                    "quality_successes": 0.0,
                    "feasibility_weight": 0.0,
                    "infra_exceptions": set(),
                    "failure_modes": set(),
                    "evidence_levels": set(),
                    "source_reports": set(),
                },
            )
            entry["n_observed"] += arm["n"]
            entry["n_complete"] += arm["complete"]
            entry["metric_n"] += arm["metric_n"]
            entry["solve_n"] += arm["solve_n"]
            entry["quality_n"] += arm["quality_n"]
            entry["feasibility_n"] += arm["feasibility_n"]
            if arm["solve_rate"] is not None:
                entry["solve_successes"] += arm["solve_rate"] * arm["solve_n"]
            if arm["quality_pass_rate"] is not None:
                entry["quality_successes"] += arm["quality_pass_rate"] * arm["quality_n"]
            if arm["mean_feasibility"] is not None:
                entry["feasibility_weight"] += arm["mean_feasibility"] * arm["feasibility_n"]
            entry["infra_exceptions"].update(arm["infra_exceptions"])
            entry["failure_modes"].update(arm["failure_modes"])
            if row.get("evidence_level"):
                entry["evidence_levels"].add(str(row["evidence_level"]))
            entry["source_reports"].add(str(row["source_report"]))

    result = []
    for model, entry in sorted(grouped.items()):
        solve_n = entry["solve_n"]
        quality_n = entry["quality_n"]
        feasibility_n = entry["feasibility_n"]
        result.append(
            {
                "model": model,
                "n_observed": entry["n_observed"],
                "n_complete": entry["n_complete"],
                "metric_n": entry["metric_n"],
                "solve_n": solve_n,
                "quality_n": quality_n,
                "feasibility_n": feasibility_n,
                "solve_rate": round(entry["solve_successes"] / solve_n, 6) if solve_n else None,
                "quality_pass_rate": round(entry["quality_successes"] / quality_n, 6) if quality_n else None,
                "mean_feasibility": round(entry["feasibility_weight"] / feasibility_n, 6) if feasibility_n else None,
                "infra_exceptions": sorted(entry["infra_exceptions"]),
                "failure_modes": sorted(entry["failure_modes"]),
                "evidence_levels": sorted(entry["evidence_levels"]),
                "source_reports": sorted(entry["source_reports"]),
            }
        )
    return result


def _decision(rows: list[dict[str, Any]], performance: list[dict[str, Any]]) -> str:
    decision_rows = [row for row in rows if row.get("kind") != "harbor-controls"]
    decisions = {str(row["decision"]) for row in decision_rows if row.get("decision")}
    if not decisions:
        return "quarantine"
    if "collect-more-evidence" in decisions:
        return "collect-more-evidence"
    infra = any(model["infra_exceptions"] for model in performance)
    if infra:
        return "collect-more-evidence"
    gaps = [
        row.get("observed_gap")
        for row in decision_rows
        if isinstance(row.get("observed_gap"), (int, float))
    ]
    base_performance = [
        row
        for row in performance
        if "@hint-" not in str(row["model"]) or str(row["model"]).endswith("@hint-0")
    ]
    repeated = len(base_performance) >= 2 and min(
        (m["solve_n"] for m in base_performance), default=0
    ) >= 5
    base_models = {str(row["model"]).split("@hint-", 1)[0] for row in base_performance}
    harbor_rows = [row for row in rows if row.get("kind") == "harbor-controls"]
    harbor_ready = len(harbor_rows) == 1
    harbor_digest = harbor_rows[0].get("task_tree_digest") if harbor_ready else None
    model_rows = [row for row in rows if row.get("kind") == "model-screening" and row.get("arms")]
    digest_ready = bool(model_rows) and isinstance(harbor_digest, str) and all(
        row.get("task_tree_digest") == harbor_digest for row in model_rows
    )
    # A later/contradictory revise-or-drop signal wins unless the aggregate
    # evidence independently meets the repeated positive-gap condition below.
    if "revise-or-drop" in decisions and not (
        repeated and gaps and all(float(gap) > 0 for gap in gaps)
    ):
        return "revise-or-drop"
    if (
        "review-promising" in decisions
        and any(float(gap) > 0 for gap in gaps)
        and repeated
        and len(base_models) >= 2
        and harbor_ready
        and digest_ready
    ):
        return "review-promising"
    if "keep" in decisions and not performance and harbor_ready:
        return "keep"
    return "collect-more-evidence"


def _dedupe_report_paths(paths: Sequence[Path]) -> list[Path]:
    """Avoid counting a strict copy and its canonical alias twice."""

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        digest = _sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
    return unique


def _summary_markdown(card: Mapping[str, Any]) -> str:
    lines = [f"# {card['title']}", "", f"**Status:** `{card['decision']}`", ""]
    lines += ["## 任务是做什么的", "", str(card["purpose"]), ""]
    lines += ["## 难度如何调控", ""]
    axes = card["difficulty"]["axes"]
    if axes:
        lines += ["| 轴 | levels | 说明 |", "| --- | --- | --- |"]
        for axis in axes:
            levels = ", ".join(str(x) for x in axis.get("levels", [])) or "未声明"
            lines.append(f"| `{axis['name']}` | {levels} | {axis.get('meaning') or axis.get('expected_direction') or ''} |")
    else:
        lines += ["未提供可执行的难度轴；任务自动进入 quarantine。"]
    lines.append("")
    interventions = card["difficulty"].get("interventions")
    if interventions:
        lines += ["## 可控干预", "", f"`{json.dumps(interventions, ensure_ascii=False, sort_keys=True)}`", ""]
    lines += ["## 模型表现", ""]
    models = card["performance"]["models"]
    if models:
        lines += ["| 模型/route | 尝试 | 完成 | solve (n) | quality (n) | feasibility (n) | 失败模式 | 基础设施异常 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
        for model in models:
            fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
            lines.append(
                f"| `{model['model']}` | {model['n_observed']} | {model['n_complete']} | {fmt(model['solve_rate'])} ({model['solve_n']}) | {fmt(model['quality_pass_rate'])} ({model['quality_n']}) | {fmt(model['mean_feasibility'])} ({model['feasibility_n']}) | {', '.join(model['failure_modes']) or '—'} | {', '.join(model['infra_exceptions']) or '—'} |"
            )
    else:
        lines += ["当前只有 scripted/control screening，没有模型能力指标。"]
    lines.append("")
    harbor_controls = [
        controls
        for controls in card["performance"]["control_screenings"]
        if isinstance(controls, Mapping)
        and all(
            isinstance(controls.get(name), Mapping)
            and isinstance(controls[name].get("reward"), (int, float))
            for name in ("oracle", "nop")
        )
    ]
    if len(harbor_controls) == 1:
        controls = harbor_controls[0]
        oracle = controls["oracle"]
        nop = controls["nop"]
        lines += [
            "## Harbor 打包控制",
            "",
            f"- Oracle：reward `{oracle['reward']}`，{oracle['ctrf_summary']['passed']}/{oracle['ctrf_summary']['tests']} tests passed。",
            f"- NOP：reward `{nop['reward']}`，{nop['ctrf_summary']['failed']}/{nop['ctrf_summary']['tests']} tests failed（按预期拒绝）。",
            "",
        ]
    lines += ["## 证据边界", "", f"evidence level: `{card['evidence']['level']}`", ""]
    for limitation in card["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_cards(
    *,
    task_inputs: Sequence[str | Path] | None = None,
    screening_inputs: Sequence[str | Path] | None = None,
    intake_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Build cards from discovered genomes and screening reports."""

    genome_docs: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _files(task_inputs):
        document = _load(path)
        if not isinstance(document, Mapping) or not document.get("family"):
            continue
        family = str(document["family"])
        genome_docs[family] = (dict(document), path)

    report_rows: list[dict[str, Any]] = []
    report_paths = _dedupe_report_paths(_files(screening_inputs))
    for path in report_paths:
        document = _load(path)
        report_rows.extend(_report_rows(document, path))

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows:
        by_task[str(row["task"])].append(row)
    for family in genome_docs:
        by_task.setdefault(family, [])

    intake = _load(Path(intake_path)) if intake_path else None
    cards: list[dict[str, Any]] = []
    for task in sorted(by_task):
        rows = by_task[task]
        family = str(rows[0].get("family") or task) if rows else task
        genome, genome_path = genome_docs.get(family, ({}, None))
        performance = _merge_performance(rows)
        control_rows = [row["controls"] for row in rows if row.get("controls")]
        evidence_levels = sorted({str(row["evidence_level"]) for row in rows if row.get("evidence_level")})
        task_tree_digests = {
            str(row["task_tree_digest"])
            for row in rows
            if isinstance(row.get("task_tree_digest"), str) and row["task_tree_digest"]
        }
        limitations = sorted({lim for row in rows for lim in row.get("limitations", [])})
        digest_conflict = len(task_tree_digests) > 1
        harbor_rows = [row for row in rows if row.get("kind") == "harbor-controls"]
        model_rows = [
            row for row in rows if row.get("kind") == "model-screening" and row.get("arms")
        ]
        digest_incomplete = bool(harbor_rows and model_rows) and (
            len(harbor_rows) != 1
            or any(not isinstance(row.get("task_tree_digest"), str) for row in model_rows)
        )
        if digest_conflict:
            limitations.append("screening/Harbor reports bind to different task-tree digests.")
        if digest_incomplete:
            limitations.append("model/Harbor evidence is missing a unique shared task-tree digest.")
        if not genome:
            limitations.append("没有匹配到 task genome；任务内容和难度旋钮未经过 task-authoring gate。")
        if not rows:
            limitations.append("没有 screening record；尚无模型或 verifier 结果。")
        if intake is not None:
            if not isinstance(intake, Mapping):
                raise PipelineError("intake document must be an object")
            limitations.append("source intake 仅提供 metadata；论文正文、许可证和任务语义仍需自动 gate/人工复核。")
        if harbor_rows:
            limitations = [
                value
                for value in limitations
                if value
                not in {
                    "Volc model screening only; no Harbor acceptance.",
                    "Oracle/NOP controls are local verifier controls, not Harbor packaging acceptance.",
                }
            ]
            limitations.append("Harbor receipt is a checksummed local artifact, not a cryptographic signature; pipeline inputs must come from a trusted artifact store.")
        decision = _decision(rows, performance)
        if digest_conflict or digest_incomplete:
            decision = "quarantine"
        source = _source_ref(genome)
        if source is None:
            source = {"status": "unbound"}
        if not genome:
            # A raw benchmark result is useful evidence, but without a bound
            # task description and declared difficulty it is not eligible for
            # automatic promotion.  Keep its performance numbers visible and
            # quarantine the task for the next authoring pass.
            decision = "quarantine"
        card: dict[str, Any] = {
            "task_card_schema_version": TASK_CARD_SCHEMA_VERSION,
            "task_id": task,
            "family": family,
            "title": str(genome.get("title") or task),
            "purpose": _purpose(genome, task),
            "source": source,
            "difficulty": {
                "axes": _difficulty_axes(genome),
                "interventions": genome.get("interventions", {}),
                "declared": bool(_difficulty_axes(genome)),
            },
            "performance": {
                "models": performance,
                "control_screenings": control_rows,
                "observed_gaps": [gap for gap in (row.get("observed_gap") for row in rows) if isinstance(gap, (int, float))],
            },
            "decision": decision,
            "evidence": {
                "level": max(evidence_levels) if evidence_levels else "E0",
                "task_genome_path": str(genome_path) if genome_path else None,
                "task_genome_digest": _sha256(genome_path) if genome_path else None,
                "source_reports": sorted({str(row["source_report"]) for row in rows}),
                "report_digests": sorted({str(row["report_digest"]) for row in rows}),
            },
            "limitations": limitations,
            "intake": {
                "path": str(intake_path) if intake_path else None,
                "present": intake is not None,
            },
        }
        card["summary_markdown"] = _summary_markdown(card)
        schema = schema_mod.load_schema(schema_mod.schemas_dir() / TASK_CARD_SCHEMA)
        schema_mod.validate(card, schema, name=f"task card {task}")
        cards.append(card)
    return cards, report_paths


def run(
    *,
    out: str | Path,
    task_inputs: Sequence[str | Path] | None = None,
    screening_inputs: Sequence[str | Path] | None = None,
    intake_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write cards, a readable index, and a content manifest atomically enough for cron."""

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    cards, report_paths = build_cards(
        task_inputs=task_inputs, screening_inputs=screening_inputs, intake_path=intake_path
    )
    cards_payload = {"pipeline_schema_version": PIPELINE_SCHEMA_VERSION, "cards": cards}
    summary = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "task_count": len(cards),
        "decision_counts": {
            decision: sum(card["decision"] == decision for card in cards)
            for decision in sorted({card["decision"] for card in cards})
        },
        "model_task_count": sum(bool(card["performance"]["models"]) for card in cards),
        "quarantined_count": sum(card["decision"] == "quarantine" for card in cards),
        "input_reports": [str(path) for path in report_paths],
        "input_report_digests": {str(path): _sha256(path) for path in report_paths},
        # Filled with the digest of the exact bytes written below.  Hashing the
        # canonical in-memory object here used to make the summary/manifest
        # disagree with the pretty-printed JSON artifact on disk.
        "task_cards_digest": None,
        "limitations": [
            "This command summarizes existing evidence; it does not call a model or author a task.",
            "E4 live intervention is not inferred from E2/E3 reports.",
        ],
    }
    markdown = ["# ORBenchLab 自动任务总览", "", f"任务数：{len(cards)}", ""]
    for card in cards:
        markdown += [f"## {card['task_id']} — `{card['decision']}`", "", card["summary_markdown"], ""]
    cards_path = output / "task-cards.json"
    summary_path = output / "pipeline-summary.json"
    markdown_path = output / "task-cards.md"
    cards_path.write_text(
        json.dumps(cards_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary["task_cards_digest"] = _sha256(cards_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    # The manifest is a byte-level ledger of the artifacts actually handed to
    # the reviewer, including indentation and the final newline.
    manifest = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "files": {
            path.name: _sha256(path)
            for path in (cards_path, summary_path, markdown_path)
        },
    }
    (output / "pipeline-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"out": str(output), "task_count": len(cards), "decision_counts": summary["decision_counts"], "written": sorted(str(path) for path in output.iterdir())}
