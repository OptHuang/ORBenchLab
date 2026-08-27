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


class PipelineError(ORBenchError):
    """Invalid pipeline input or output."""

    exit_code = 8


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
        return float(value) if isinstance(value, (int, float)) else None

    def integer(key: str) -> int:
        value = arm.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    n = integer("n")
    complete = integer("complete")
    metric_n = integer("metric_n")
    infra = list(arm.get("infra_exceptions") or []) if isinstance(arm.get("infra_exceptions"), list) else []
    return {
        "n": n,
        "complete": complete,
        "metric_n": metric_n,
        "solve_rate": number("solve_rate"),
        "quality_pass_rate": number("quality_pass_rate"),
        "mean_feasibility": number("mean_feasibility"),
        "infra_exceptions": sorted(str(x) for x in infra),
    }


def _report_rows(document: Any, source: Path) -> list[dict[str, Any]]:
    """Convert both ORBenchLab and ORWorkbench screening shapes to task rows."""

    if not isinstance(document, Mapping):
        return []
    # ORBenchLab screening report: one row per task, with model arms.
    tasks = document.get("tasks")
    if isinstance(tasks, list):
        rows: list[dict[str, Any]] = []
        for item in tasks:
            if not isinstance(item, Mapping) or not item.get("task"):
                continue
            arms = item.get("arms") if isinstance(item.get("arms"), Mapping) else {}
            rows.append(
                {
                    "task": str(item["task"]),
                    "family": str(item.get("family") or item["task"]),
                    "arms": {str(name): _normalise_arm(value) for name, value in arms.items() if isinstance(value, Mapping)},
                    "decision": item.get("decision"),
                    "evidence_level": item.get("evidence_level"),
                    "observed_gap": item.get("discrimination_index_observed_gap"),
                    "limitations": [str(x) for x in item.get("limitations", []) if isinstance(x, str)],
                    "source_report": str(source),
                    "report_digest": _sha256(source),
                    "kind": "model-screening",
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
                    "solve_successes": 0.0,
                    "quality_successes": 0.0,
                    "feasibility_weight": 0.0,
                    "infra_exceptions": set(),
                    "evidence_levels": set(),
                    "source_reports": set(),
                },
            )
            n = arm["metric_n"] or arm["complete"] or arm["n"]
            entry["n_observed"] += arm["n"]
            entry["n_complete"] += arm["complete"]
            entry["metric_n"] += n
            if arm["solve_rate"] is not None:
                entry["solve_successes"] += arm["solve_rate"] * n
            if arm["quality_pass_rate"] is not None:
                entry["quality_successes"] += arm["quality_pass_rate"] * n
            if arm["mean_feasibility"] is not None:
                entry["feasibility_weight"] += arm["mean_feasibility"] * n
            entry["infra_exceptions"].update(arm["infra_exceptions"])
            if row.get("evidence_level"):
                entry["evidence_levels"].add(str(row["evidence_level"]))
            entry["source_reports"].add(str(row["source_report"]))

    result = []
    for model, entry in sorted(grouped.items()):
        n = entry["metric_n"]
        result.append(
            {
                "model": model,
                "n_observed": entry["n_observed"],
                "n_complete": entry["n_complete"],
                "metric_n": n,
                "solve_rate": round(entry["solve_successes"] / n, 6) if n else None,
                "quality_pass_rate": round(entry["quality_successes"] / n, 6) if n else None,
                "mean_feasibility": round(entry["feasibility_weight"] / n, 6) if n else None,
                "infra_exceptions": sorted(entry["infra_exceptions"]),
                "evidence_levels": sorted(entry["evidence_levels"]),
                "source_reports": sorted(entry["source_reports"]),
            }
        )
    return result


def _decision(rows: list[dict[str, Any]], performance: list[dict[str, Any]]) -> str:
    decisions = {str(row["decision"]) for row in rows if row.get("decision")}
    if not decisions:
        return "quarantine"
    if "collect-more-evidence" in decisions:
        return "collect-more-evidence"
    infra = any(model["infra_exceptions"] for model in performance)
    if infra:
        return "collect-more-evidence"
    gaps = [row.get("observed_gap") for row in rows if isinstance(row.get("observed_gap"), (int, float))]
    repeated = bool(performance) and min((m["metric_n"] for m in performance), default=0) >= 3
    # A later/contradictory revise-or-drop signal wins unless the aggregate
    # evidence independently meets the repeated positive-gap condition below.
    if "revise-or-drop" in decisions and not (
        repeated and gaps and all(float(gap) > 0 for gap in gaps)
    ):
        return "revise-or-drop"
    if "review-promising" in decisions and any(float(gap) > 0 for gap in gaps) and repeated:
        return "review-promising"
    if "keep" in decisions and not performance:
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
    lines += ["## 模型表现", ""]
    models = card["performance"]["models"]
    if models:
        lines += ["| 模型/route | n | 完成 | solve rate | quality pass | 平均 feasibility | 异常 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for model in models:
            fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
            lines.append(
                f"| `{model['model']}` | {model['metric_n']} | {model['n_complete']} | {fmt(model['solve_rate'])} | {fmt(model['quality_pass_rate'])} | {fmt(model['mean_feasibility'])} | {', '.join(model['infra_exceptions']) or '—'} |"
            )
    else:
        lines += ["当前只有 scripted/control screening，没有模型能力指标。"]
    lines.append("")
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
        genome = genome_docs.get(family, ({}, None))[0]
        performance = _merge_performance(rows)
        control_rows = [row["controls"] for row in rows if row.get("controls")]
        evidence_levels = sorted({str(row["evidence_level"]) for row in rows if row.get("evidence_level")})
        limitations = sorted({lim for row in rows for lim in row.get("limitations", [])})
        if not genome:
            limitations.append("没有匹配到 task genome；任务内容和难度旋钮未经过 task-authoring gate。")
        if not rows:
            limitations.append("没有 screening record；尚无模型或 verifier 结果。")
        if intake is not None:
            if not isinstance(intake, Mapping):
                raise PipelineError("intake document must be an object")
            limitations.append("source intake 仅提供 metadata；论文正文、许可证和任务语义仍需自动 gate/人工复核。")
        decision = _decision(rows, performance)
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
        "task_cards_digest": "sha256:" + hashlib.sha256(_canonical(cards_payload)).hexdigest(),
        "limitations": [
            "This command summarizes existing evidence; it does not call a model or author a task.",
            "E4 live intervention is not inferred from E2/E3 reports.",
        ],
    }
    markdown = ["# ORBenchLab 自动任务总览", "", f"任务数：{len(cards)}", ""]
    for card in cards:
        markdown += [f"## {card['task_id']} — `{card['decision']}`", "", card["summary_markdown"], ""]
    manifest = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "files": {
            "task-cards.json": "sha256:" + hashlib.sha256(_canonical(cards_payload)).hexdigest(),
            "pipeline-summary.json": "sha256:" + hashlib.sha256(_canonical(summary)).hexdigest(),
            "task-cards.md": "sha256:" + hashlib.sha256(("\n".join(markdown) + "\n").encode()).hexdigest(),
        },
    }
    (output / "task-cards.json").write_text(json.dumps(cards_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "pipeline-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "task-cards.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output / "pipeline-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"out": str(output), "task_count": len(cards), "decision_counts": summary["decision_counts"], "written": sorted(str(path) for path in output.iterdir())}
