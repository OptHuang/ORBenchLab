#!/usr/bin/env python3
"""Derive a conservative, prompt-free screening report from Harbor results.

This is deliberately a first-pass report: it computes verifier-grounded
feasibility/quality rates and a model-arm gap, but does not call the gap a
capability ranking.  A task is promoted only after repeated runs and
trajectory review.  Raw prompts, tool output, and hidden verifier data are
never copied to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.name).encode())
        try:
            h.update(path.read_bytes())
        except OSError:
            continue
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _task_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.split("/", 1)[-1]


def _timestamp_seconds(value: Any) -> float | None:
    """Parse an ISO timestamp without retaining any trace content."""
    if not isinstance(value, str):
        return None
    try:
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _trajectory_summary(path: Path) -> dict[str, Any]:
    """Return a prompt-free structural summary of one ATIF trajectory.

    This intentionally records counts and lengths only.  It is useful for
    replay triage (E2), but it is not a causal diagnosis and never copies
    messages, tool arguments, observations, or verifier output.
    """
    empty: dict[str, Any] = {
        "present": False,
        "schema_version": None,
        "step_count": 0,
        "tool_call_count": 0,
        "tool_name_counts": {},
        "reasoning_step_count": 0,
        "reasoning_chars": 0,
        "observation_chars": 0,
        "elapsed_seconds": None,
        "step_prompt_tokens": 0,
        "step_completion_tokens": 0,
        "evidence_level": "E0",
    }
    value = _read(path)
    if value is None:
        return empty
    steps = value.get("steps")
    if not isinstance(steps, list):
        return empty
    tool_counts: dict[str, int] = defaultdict(int)
    timestamps: list[float] = []
    reasoning_steps = 0
    reasoning_chars = 0
    observation_chars = 0
    tool_call_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        stamp = _timestamp_seconds(step.get("timestamp"))
        if stamp is not None:
            timestamps.append(stamp)
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            reasoning_steps += 1
            reasoning_chars += len(reasoning)
        observation = step.get("observation")
        if isinstance(observation, str):
            observation_chars += len(observation)
        calls = step.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                tool_call_count += 1
                name = call.get("function_name") or call.get("name") or "unknown"
                tool_counts[str(name)] += 1
        metrics = step.get("metrics")
        if isinstance(metrics, dict):
            p = metrics.get("prompt_tokens")
            c = metrics.get("completion_tokens")
            if isinstance(p, (int, float)) and math.isfinite(p):
                prompt_tokens += int(p)
            if isinstance(c, (int, float)) and math.isfinite(c):
                completion_tokens += int(c)
    elapsed = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    return {
        "present": True,
        "schema_version": value.get("schema_version"),
        "step_count": len(steps),
        "tool_call_count": tool_call_count,
        "tool_name_counts": dict(sorted(tool_counts.items())),
        "reasoning_step_count": reasoning_steps,
        "reasoning_chars": reasoning_chars,
        "observation_chars": observation_chars,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "step_prompt_tokens": prompt_tokens,
        "step_completion_tokens": completion_tokens,
        "evidence_level": "E2",
    }


def _trial_row(path: Path) -> dict[str, Any] | None:
    value = _read(path)
    if value is None:
        return None
    config = value.get("config") or {}
    agent = config.get("agent") or {}
    agent_info = value.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    model = agent.get("model_name") or model_info.get("name") or "unknown"
    rewards = (value.get("verifier_result") or {}).get("rewards") or {}
    feas = rewards.get("feasibility")
    quality = rewards.get("quality")
    verifier_observed = isinstance(feas, (int, float))
    exception = value.get("exception_info")
    if isinstance(exception, dict):
        exception = (
            exception.get("exception_type")
            or exception.get("type")
            or exception.get("name")
        )
    else:
        exception = None
    # A timed-out trial may still leave a verifier result.  Keep that
    # verifier observation for analysis, but do not call the trial complete;
    # the infrastructure exception must remain visible to screening.
    complete = bool(value.get("finished_at")) and verifier_observed and not exception
    start = value.get("started_at")
    finish = value.get("finished_at")
    return {
        "task": _task_name(value.get("task_name")) or path.parent.name,
        "trial_name": value.get("trial_name") or path.parent.name,
        "model": str(model),
        "feasibility": float(feas) if isinstance(feas, (int, float)) else None,
        "quality": float(quality) if isinstance(quality, (int, float)) else None,
        "complete": complete,
        "verifier_observed": verifier_observed,
        "exception": str(exception) if exception else None,
        "input_tokens": (value.get("agent_result") or {}).get("n_input_tokens"),
        "output_tokens": (value.get("agent_result") or {}).get("n_output_tokens"),
        "cost_usd": (value.get("agent_result") or {}).get("cost_usd"),
        "started_at": start,
        "finished_at": finish,
        # Keep reports portable and avoid publishing host-specific paths.
        "result_ref": path.parent.name,
        "trajectory": _trajectory_summary(path.parent / "agent" / "trajectory.json"),
    }


def collect(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(job_dir.glob("*/result.json")):
        row = _trial_row(path)
        if row is not None:
            rows.append(row)
    return rows


def _rate(values: list[float | None], predicate) -> float | None:
    usable = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not usable:
        return None
    return sum(predicate(v) for v in usable) / len(usable)


def _finite(values: list[float | None]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _completed_values(rows: list[dict[str, Any]], field: str) -> list[float | None]:
    """Return metric values from completed, non-exception trials only.

    Harbor can write a verifier result while a trial is subsequently marked
    timed out.  That result is useful as an observed fact, but treating it as
    a successful/failed completed attempt would mix infrastructure failure
    with model performance and bias the screening denominator.
    """
    return [
        row.get(field)
        for row in rows
        if bool(row.get("complete")) and not row.get("exception")
    ]


def build_report(job_dir: Path) -> dict[str, Any]:
    aggregate = _read(job_dir / "result.json") or {}
    rows = collect(job_dir)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["task"]][row["model"]].append(row)

    tasks: list[dict[str, Any]] = []
    for task in sorted(groups):
        arms: dict[str, Any] = {}
        for model, arm_rows in sorted(groups[task].items()):
            feas = _completed_values(arm_rows, "feasibility")
            quality = _completed_values(arm_rows, "quality")
            finite_feas = _finite(feas)
            complete = sum(bool(r["complete"]) for r in arm_rows)
            infra_errors = sorted({r["exception"] for r in arm_rows if r["exception"]})
            arms[model] = {
                "n": len(arm_rows),
                "complete": complete,
                "metric_n": len(finite_feas),
                "solve_rate": _rate(feas, lambda v: v >= 1.0 - 1e-12),
                "quality_pass_rate": _rate(quality, lambda v: v >= 2.0 - 1e-9),
                "mean_feasibility": (
                    sum(finite_feas) / len(finite_feas)
                    if finite_feas
                    else None
                ),
                "infra_exceptions": infra_errors,
            }
        rates = [v["solve_rate"] for v in arms.values() if v["solve_rate"] is not None]
        gap = max(rates) - min(rates) if len(rates) >= 2 else None
        has_infra = any(v["infra_exceptions"] for v in arms.values())
        if len(rates) < 2 or any(v["n"] < 1 for v in arms.values()):
            decision = "collect-more-evidence"
        elif has_infra:
            decision = "collect-more-evidence"
        elif gap is not None and 0.0 < gap <= 1.0:
            decision = "review-promising"
        else:
            decision = "revise-or-drop"
        tasks.append(
            {
                "task": task,
                "arms": arms,
                "discrimination_index_observed_gap": gap,
                "decision": decision,
                "evidence_level": "E3"
                if all(
                    all(bool(r.get("verifier_observed")) for r in groups[task][model])
                    for model in arms
                )
                else "E1",
                "limitations": [
                    "single screening pass is not a model ranking",
                    "no causal intervention evidence",
                    "trajectory fields are prompt-free structural E2 summaries",
                ],
            }
        )

    result_paths = [p for p in job_dir.glob("*/result.json")]
    return {
        "schema_version": "orbenchlab.screening-report.v1",
        "job_id": aggregate.get("id"),
        "job_name": job_dir.name,
        "job_finished_at": aggregate.get("finished_at"),
        "aggregate_stats": aggregate.get("stats") or {},
        "result_set_sha256": _digest(result_paths),
        "n_trials_observed": len(rows),
        "models_observed": sorted({r["model"] for r in rows}),
        "tasks": tasks,
        "raw_rows": rows,
        "claim_boundary": "Observed verifier outcomes; no causal or model-wide claim.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harbor screening report",
        "",
        f"- Job: `{report.get('job_name')}`",
        f"- Observed trials: **{report.get('n_trials_observed', 0)}**",
        f"- Evidence boundary: {report.get('claim_boundary')}",
        "",
        "| Task | Model arms (n / solve rate / quality-pass rate) | Observed gap | Decision | Evidence |",
        "|---|---|---:|---|---|",
    ]
    for task in report.get("tasks", []):
        arms = "; ".join(
            f"{m} ({v['n']} / {v['solve_rate'] if v['solve_rate'] is not None else 'NA'} / {v['quality_pass_rate'] if v['quality_pass_rate'] is not None else 'NA'})"
            for m, v in task["arms"].items()
        )
        gap = task.get("discrimination_index_observed_gap")
        lines.append(
            f"| `{task['task']}` | {arms} | {gap if gap is not None else 'NA'} | {task['decision']} | {task['evidence_level']} |"
        )
    lines += [
        "",
        "## Prompt-free trajectory structure",
        "",
        "Counts below support E2 behavioral triage only; they do not identify a causal bottleneck.",
        "",
        "| Trial | Model | Steps | Tool calls | Reasoning steps | Elapsed (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report.get("raw_rows", []):
        trajectory = row.get("trajectory") or {}
        lines.append(
            f"| `{row['result_ref']}` | {row['model']} | "
            f"{trajectory.get('step_count', 0)} | {trajectory.get('tool_call_count', 0)} | "
            f"{trajectory.get('reasoning_step_count', 0)} | "
            f"{trajectory.get('elapsed_seconds') if trajectory.get('elapsed_seconds') is not None else 'NA'} |"
        )
    lines += ["", "This report intentionally omits prompts, tool output, and hidden verifier data.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    job_dir = args.job_dir.expanduser().resolve()
    if not job_dir.is_dir():
        parser.error(f"job directory does not exist: {job_dir}")
    report = build_report(job_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "trials": report["n_trials_observed"], "tasks": len(report["tasks"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
