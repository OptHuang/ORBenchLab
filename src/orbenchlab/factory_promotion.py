"""Deterministic promotion phase for a semantic-complete agent factory.

After the autopilot's semantic DAG and trusted Harbor barriers finish, this
module carries the run to its formal end without human re-invocation: it
re-runs the static TB-Science gate over the selected task, locates the
digest-matched Harbor control and calibration evidence the barriers already
produced (never re-spending on evidence that exists), obtains the independent
Volc semantic review, builds the deterministic pipeline task card, executes
the fail-closed finalizer, and writes one operator-facing final report that
binds every receipt.  Nothing here trusts agent prose; a missing or failing
gate yields ``promoted: false`` with machine-readable reasons.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from . import (
    agentic_factory,
    factory_finalize,
    factory_review,
    factory_supervisor,
    harbor_model_matrix,
    pipeline,
    task_authoring,
    volc_rollout,
)
from .core.errors import ORBenchError


class FactoryPromotionError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.factory-promotion.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _value_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FactoryPromotionError(f"promotion evidence is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise FactoryPromotionError(f"promotion evidence is malformed: {path}") from None
    if not isinstance(value, dict):
        raise FactoryPromotionError(f"promotion evidence root must be an object: {path}")
    return value


def _reviewer_models(plan: Mapping[str, Any]) -> list[str]:
    models: list[str] = []
    for stage_id in ("task-review-science", "task-review-verifier"):
        stage = next(
            (row for row in plan["stages"] if row["id"] == stage_id), None
        )
        if stage is not None:
            models.append(str(stage["model"]))
    return models


def _locate_runtime_evidence(
    *,
    evidence_root: Path,
    runtime_digest: str,
    out: Path,
) -> tuple[Path, Path] | None:
    """Find digest-matched Harbor control and calibration receipts.

    Reuses the exact receipts the autopilot barriers already validated; a
    selected variant gets its screening envelope built once from the variant's
    validated matrix and controls.  Returns ``(controls_path, screening_path)``
    or ``None`` when no evidence binds the selected task tree.
    """

    baseline_controls = evidence_root / "baseline" / "controls" / "harbor-control-screening.json"
    baseline_screening = evidence_root / "baseline" / "matrix" / "screening-report.json"
    if baseline_controls.is_file() and baseline_screening.is_file():
        screening = _load_json(baseline_screening)
        if screening.get("task_tree_digest") == runtime_digest:
            return baseline_controls, baseline_screening
    variants_root = evidence_root / "difficulty" / "variants"
    if variants_root.is_dir():
        for variant_dir in sorted(variants_root.iterdir()):
            controls_path = variant_dir / "controls" / "harbor-control-screening.json"
            matrix_path = variant_dir / "matrix" / "harbor-model-matrix.json"
            if not controls_path.is_file() or not matrix_path.is_file():
                continue
            matrix = _load_json(matrix_path)
            if matrix.get("task_tree_digest") != runtime_digest:
                continue
            checked = harbor_model_matrix._validated_receipt(matrix)
            controls = _load_json(controls_path)
            screening_dir = out / "calibration"
            screening_path = screening_dir / "screening-report.json"
            if not screening_path.is_file():
                harbor_model_matrix.build_screening_report(
                    checked, harbor_controls=controls, out=screening_dir
                )
            return controls_path, screening_path
    return None


def _observed_costs(state: Mapping[str, Any]) -> dict[str, Any]:
    barriers = state.get("barriers") if isinstance(state, Mapping) else None
    costs: dict[str, Any] = {}
    if isinstance(barriers, Mapping):
        baseline = barriers.get("baseline")
        if isinstance(baseline, Mapping) and isinstance(
            baseline.get("observed_usage"), Mapping
        ):
            costs["baseline"] = dict(baseline["observed_usage"])
        difficulty = barriers.get("difficulty")
        if isinstance(difficulty, Mapping) and isinstance(
            difficulty.get("observed_usage"), list
        ):
            costs["difficulty_variants"] = [
                dict(row) for row in difficulty["observed_usage"] if isinstance(row, Mapping)
            ]
    return costs


def _agent_artifact(workdir: Path, relative: str) -> dict[str, Any] | None:
    path = workdir / relative
    if not path.is_file() or path.is_symlink():
        return None
    return {
        "path": relative,
        "content_digest": _file_digest(path),
        "evidence_level": "E1-agent-authored",
    }


def run_promotion(
    *,
    plan: Mapping[str, Any],
    workdir: str | Path,
    factory_out: str | Path,
    evidence_root: str | Path,
    out: str | Path,
    provider_env: Mapping[str, str],
    state: Mapping[str, Any],
    review_executable: str | Path | None = None,
    review_timeout_sec: float = 600.0,
    review_max_budget_usd: float = 1.0,
    max_review_tokens: int = 2400,
) -> dict[str, Any]:
    """Run or resume the deterministic post-semantic promotion chain."""

    checked = agentic_factory.validate_plan(plan)
    workspace = Path(workdir).resolve()
    factory_root = Path(factory_out).resolve()
    evidence = Path(evidence_root).resolve()
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = factory_root / "factory-plan.json"
    run_path = factory_root / "factory-run.json"
    summary_relative = "factory/final/task-review-summary.json"
    genome_relative = "factory/final/task-genome.json"
    genome_path = workspace / genome_relative
    summary_doc = _load_json(workspace / summary_relative)
    selected_relative = str(summary_doc.get("selected_task") or "")
    if not selected_relative:
        raise FactoryPromotionError("final synthesis did not select a task")
    task = agentic_factory._artifact_path(workspace, selected_relative)
    # Reuse the supervisor's hardened binding: the selected task must be a
    # completed factory artifact, and genome/summary must bind it exactly.
    factory_supervisor._bind_factory_outputs(
        plan_path=plan_path,
        factory_run_path=run_path,
        workspace=workspace,
        task=task,
        genome_path=genome_path,
    )
    paper_provenance = workspace / "factory-input" / "paper-provenance.json"
    genome = _load_json(genome_path)
    factory_supervisor._validate_selected_genome(
        genome,
        summary=summary_doc,
        workspace=workspace,
        task=task,
        paper_provenance=paper_provenance,
    )
    runtime_digest = volc_rollout._task_tree_digest(task)
    authoring_digest = task_authoring._task_tree_digest(task)

    gates: dict[str, Any] = {}

    static_json = root / "static" / "authoring-receipt.json"
    if not static_json.is_file():
        receipt = task_authoring.validate_task(task, paper_provenance=paper_provenance)
        task_authoring.write_receipt(receipt, root / "static")
    static_receipt = _load_json(static_json)
    gates["static"] = {
        "status": "pass" if static_receipt.get("decision") != "blocked" else "fail",
        "decision": static_receipt.get("decision"),
        "receipt": str(static_json),
        "receipt_digest": static_receipt.get("receipt_digest"),
    }

    located = _locate_runtime_evidence(
        evidence_root=evidence, runtime_digest=runtime_digest, out=root
    )
    if located is None:
        gates["runtime_evidence"] = {
            "status": "fail",
            "reason": "promotion_evidence_missing",
            "detail": (
                "no validated Harbor control/calibration receipt binds the selected "
                f"task tree {runtime_digest}; promotion never silently relaunches paid jobs"
            ),
        }
        harbor_json: Path | None = None
        calibration_json: Path | None = None
    else:
        harbor_json, calibration_json = located
        gates["runtime_evidence"] = {
            "status": "pass",
            "harbor_receipt": str(harbor_json),
            "harbor_receipt_digest": _file_digest(harbor_json),
            "calibration_receipt": str(calibration_json),
            "calibration_receipt_digest": _file_digest(calibration_json),
        }

    semantic_json = root / "semantic" / "volc-authoring-review.json"
    reviewers = list(dict.fromkeys(_reviewer_models(checked)))
    semantic_validator = factory_finalize._semantic_validator(
        authoring_digest,
        str(static_receipt.get("receipt_digest") or ""),
        str(
            task_authoring._load_document(paper_provenance).get("source_content_digest")
            or ""
        ),
        semantic_root=semantic_json.parent,
    )

    def revalidate_semantic() -> None:
        # Never treat the mere presence of the JSON as a pass: promotion runs
        # the exact finalize validator (digest + binding + full passing rubric)
        # before showing a pass, on both fresh and resume paths.
        try:
            factory_finalize._load(semantic_json)
            semantic_validator(_load_json(semantic_json))
            gates["semantic_review"] = {"status": "pass", "receipt": str(semantic_json)}
        except (factory_finalize.FactoryFinalizeError, FactoryPromotionError, ValueError, KeyError, TypeError) as exc:
            gates["semantic_review"] = {
                "status": "fail",
                "reason": f"semantic_review_revalidation_failed:{type(exc).__name__}",
                "receipt": str(semantic_json),
            }

    if semantic_json.is_file():
        revalidate_semantic()
    elif gates["static"]["status"] != "pass":
        gates["semantic_review"] = {
            "status": "blocked",
            "reason": "upstream_static_blocked",
        }
    elif len(set(reviewers)) < 2:
        gates["semantic_review"] = {
            "status": "blocked",
            "reason": "reviewer_models_unavailable",
        }
    elif not str(provider_env.get("ANTHROPIC_BASE_URL", "")).strip() or not str(
        provider_env.get("ANTHROPIC_AUTH_TOKEN", "")
    ).strip():
        gates["semantic_review"] = {
            "status": "blocked",
            "reason": "provider_env_missing",
        }
    elif not review_executable:
        gates["semantic_review"] = {
            "status": "blocked",
            "reason": "review_executable_unavailable",
        }
    else:
        try:
            review = factory_review.review_task_via_sessions(
                task,
                paper_provenance_path=paper_provenance,
                static_receipt_path=static_json,
                models=reviewers,
                provider_env=provider_env,
                out=root / "semantic",
                max_budget_usd=review_max_budget_usd,
                timeout_sec=review_timeout_sec,
                executable=review_executable,
            )
            factory_review.write_review(review, root / "semantic")
            revalidate_semantic()
        except (factory_review.FactoryReviewError, agentic_factory.AgenticFactoryError) as exc:
            gates["semantic_review"] = {
                "status": "blocked",
                "reason": f"review_session_failed:{type(exc).__name__}",
            }
    if semantic_json.is_file():
        review_doc = _load_json(semantic_json)
        gates["semantic_review"]["receipt_digest"] = _file_digest(semantic_json)
        gates["semantic_review"]["aggregate_decision"] = review_doc.get(
            "aggregate_decision"
        )
        gates["semantic_review"]["review_mechanism"] = review_doc.get("review_mechanism")
        gates["semantic_review"]["session_receipt_digests"] = [
            row.get("session_receipt_digest")
            for row in review_doc.get("session_bindings", [])
            if isinstance(row, Mapping)
        ]

    cards_json = root / "cards" / "task-cards.json"
    screening_inputs = [
        path for path in (harbor_json, calibration_json) if path is not None
    ]
    if not cards_json.is_file():
        pipeline.run(
            out=root / "cards",
            task_inputs=[genome_path],
            screening_inputs=screening_inputs,
        )
    gates["cards"] = {
        "status": "pass" if cards_json.is_file() else "fail",
        "receipt": str(cards_json),
        "receipt_digest": _file_digest(cards_json) if cards_json.is_file() else None,
    }

    final_dir = root / "final"
    prerequisites = all(
        path is not None and Path(path).is_file()
        for path in (static_json, semantic_json, harbor_json, calibration_json, cards_json)
    )
    if prerequisites:
        final = factory_finalize.build_receipt(
            plan_path=plan_path,
            factory_run_path=run_path,
            workdir=workspace,
            task_dir=task,
            static_receipt_path=static_json,
            semantic_review_path=semantic_json,
            harbor_receipt_path=harbor_json,
            calibration_receipt_path=calibration_json,
            final_summary_path=cards_json,
        )
        factory_finalize.write_receipt(final, final_dir)
    else:
        final = {
            "schema_version": factory_finalize.SCHEMA_VERSION,
            "decision": "not-promoted",
            "promoted": False,
            "evidence_level": "E1",
            "failure_class": "prerequisite_blocked",
            "gates": [],
        }
        _atomic_json(final_dir / "factory-finalization.json", final)
    gates["finalize"] = {
        "status": "pass" if final.get("promoted") else "fail",
        "decision": final.get("decision"),
        "receipt": str(final_dir / "factory-finalization.json"),
        "receipt_digest": _file_digest(final_dir / "factory-finalization.json"),
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "selected_task": selected_relative,
        "task_id": volc_rollout._task_id(task),
        "runtime_task_tree_digest": runtime_digest,
        "authoring_task_tree_digest": authoring_digest,
        "reviewer_models": reviewers,
        "semantic_review_mechanism": "cli-agent-session",
        "semantic_review_liability_usd": round(
            len(reviewers) * float(review_max_budget_usd), 6
        ),
        "gates": gates,
        "promoted": bool(final.get("promoted")),
        "decision": final.get("decision"),
        "evidence_level": final.get("evidence_level"),
    }
    report_paths = write_final_report(
        promotion=summary,
        workdir=workspace,
        factory_out=factory_root,
        evidence_root=evidence,
        out=root,
        state=state,
        genome=genome,
        summary_doc=summary_doc,
        static_receipt=static_receipt,
        calibration_path=calibration_json,
        final_receipt=final,
    )
    summary["final_report"] = {
        key: str(path) for key, path in report_paths.items()
    }
    summary["final_report_digest"] = _file_digest(report_paths["markdown"])
    summary["promotion_digest"] = _value_digest(
        {key: value for key, value in summary.items() if key != "promotion_digest"}
    )
    _atomic_json(root / "promotion-summary.json", summary)
    return summary


def _format_rate(cell: Mapping[str, Any]) -> str:
    rate = cell.get("solve_rate")
    if rate is None:
        return "n/a"
    wilson = cell.get("wilson_95") or []
    interval = (
        f" (95% CI {wilson[0]:.2f}–{wilson[1]:.2f})" if len(wilson) == 2 else ""
    )
    return f"{float(rate):.2f} ({cell.get('passed', '?')}/{cell.get('n', '?')}){interval}"


def write_final_report(
    *,
    promotion: Mapping[str, Any],
    workdir: Path,
    factory_out: Path,
    evidence_root: Path,
    out: Path,
    state: Mapping[str, Any],
    genome: Mapping[str, Any],
    summary_doc: Mapping[str, Any],
    static_receipt: Mapping[str, Any],
    calibration_path: Path | None,
    final_receipt: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the deterministic operator-facing final report."""

    calibration = _load_json(calibration_path) if calibration_path else None
    difficulty_path = evidence_root / "difficulty" / "difficulty-matrix.json"
    difficulty = _load_json(difficulty_path) if difficulty_path.is_file() else None
    capability_path = (
        evidence_root / "baseline" / "trusted-source" / "runtime-capability.json"
    )
    capability = _load_json(capability_path) if capability_path.is_file() else None
    intervention_capability_path = (
        evidence_root / "intervention" / "trusted-source" / "runtime-capability.json"
    )
    intervention_capability = (
        _load_json(intervention_capability_path)
        if intervention_capability_path.is_file()
        else None
    )
    intervention_study_path = (
        evidence_root / "intervention" / "trusted-source" / "intervention-study.json"
    )
    intervention_study = (
        _load_json(intervention_study_path)
        if intervention_study_path.is_file()
        else None
    )
    agent_artifacts = {
        name: _agent_artifact(workdir, relative)
        for name, relative in (
            ("trajectory_diagnosis", "factory/analysis/trajectory-diagnosis.md"),
            ("intervention_study", "factory/analysis/intervention-study.md"),
            ("calibration_index", "factory/calibration/calibration-index.md"),
            ("difficulty_lattice", "factory/difficulty/difficulty-lattice.md"),
        )
    }
    costs = _observed_costs(state)

    lines: list[str] = [
        f"# Factory final report — {promotion['task_id']}",
        "",
        f"**Promotion decision:** `{promotion['decision']}` "
        f"(promoted: `{promotion['promoted']}`, evidence level: `{promotion.get('evidence_level')}`)",
        "",
        "## 任务是什么",
        "",
        str(genome.get("design_goal") or genome.get("title") or promotion["task_id"]),
        "",
        f"- 任务目录：`{promotion['selected_task']}`",
        f"- 任务树 digest（runtime）：`{promotion['runtime_task_tree_digest']}`",
        f"- Task summary（agent 撰写，E1）：{str(summary_doc.get('task_summary', ''))[:800]}",
        "",
        "## 来源 provenance",
        "",
    ]
    source = genome.get("source") if isinstance(genome.get("source"), Mapping) else {}
    for key in ("title", "url", "paper_provenance_digest", "source_content_digest"):
        if source.get(key):
            lines.append(f"- {key}: `{source[key]}`")
    lines += [
        "",
        "## 验证等级与 gate 结果",
        "",
        "| Gate | 状态 | 细节 |",
        "| --- | --- | --- |",
    ]
    for name, gate in promotion["gates"].items():
        detail = gate.get("decision") or gate.get("reason") or gate.get("aggregate_decision") or ""
        lines.append(f"| `{name}` | `{gate.get('status')}` | {detail} |")
    static_counts = static_receipt.get("counts") or {}
    lines += [
        "",
        f"静态 gate（39 项 TB-Science implementation rubric）：`{static_counts}`；"
        f"decision `{static_receipt.get('decision')}`。`review` 项为语义准则，仍需人工/评审确认。",
        "",
        "## Frontier vs weak 表现（真实 Harbor 重复 rollouts，E3）",
        "",
    ]
    if calibration is not None:
        rows = calibration.get("tasks") or [{}]
        arms = rows[0].get("arms") if isinstance(rows[0], Mapping) else {}
        discrimination = rows[0].get("discrimination") if isinstance(rows[0], Mapping) else {}
        lines += ["| Arm | n | 完成 | solve rate | 失败模式 | infra 异常 |", "| --- | ---: | ---: | --- | --- | --- |"]
        for arm_name, arm in sorted((arms or {}).items()):
            rate = arm.get("solve_rate")
            lines.append(
                f"| `{arm_name}` | {arm.get('n')} | {arm.get('complete')} | "
                f"{'n/a' if rate is None else f'{rate:.2f}'} | "
                f"{', '.join(arm.get('failure_modes') or []) or '—'} | "
                f"{', '.join(arm.get('infra_exceptions') or []) or '—'} |"
            )
        lines += [
            "",
            f"- 区分度（observed gap）：`{rows[0].get('discrimination_index_observed_gap')}`",
            f"- 保守 95% Wilson 下界 gap：`{(discrimination or {}).get('gap_95_lower_bound')}`",
            f"- promising：`{(discrimination or {}).get('promising')}`",
        ]
    else:
        lines.append("没有绑定到所选任务树的校准 receipt；见 gate `runtime_evidence`。")
    lines += ["", "## 难度维度与变体校准", ""]
    if difficulty is not None:
        genome_axes = difficulty.get("difficulty_genome") or {}
        primary = difficulty.get("primary_axis") or {}
        lines.append(
            f"- 主轴：`{primary.get('name')}`（预期方向：{primary.get('expected_direction')}）"
        )
        for axis in difficulty.get("secondary_axes") or []:
            lines.append(
                f"- 次轴：`{axis.get('name')}` — {axis.get('meaning') or axis.get('expected_direction') or ''}"
            )
        lines += [
            "",
            "| Variant | level | axis levels | frontier | weak | gap 下界 |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
        for row in difficulty.get("variants") or []:
            cells = row.get("cells") or {}
            frontier_cell = cells.get(difficulty.get("frontier_model"), {})
            weak_cell = cells.get(difficulty.get("weak_model"), {})
            lines.append(
                f"| `{row.get('variant_id')}` | {json.dumps(row.get('level'), ensure_ascii=False)} | "
                f"{json.dumps(row.get('axis_levels'), ensure_ascii=False)} | "
                f"{_format_rate(frontier_cell)} | {_format_rate(weak_cell)} | "
                f"{(row.get('model_gap') or {}).get('wilson_lower_bound')} |"
            )
        monotonic = difficulty.get("monotonicity") or {}
        lines += [
            "",
            f"- 单调性（预期 solve rate 随难度不增）：`{monotonic.get('passed')}`；"
            f"weak 模型存在严格下降：`{monotonic.get('weak_model_has_strict_drop')}`",
            f"- 难度矩阵 decision：`{difficulty.get('decision')}`（{difficulty.get('evaluation_mode')}）",
            f"- difficulty genome（machine-readable）：`{difficulty_path}`"
            f"，variants={len((genome_axes.get('variants') or []))}",
        ]
    else:
        lines.append("难度矩阵 receipt 缺失（difficulty barrier 未完成或所选任务无变体证据）。")
    lines += ["", "## 轨迹与卡点分析（分级）", ""]
    for name, artifact in agent_artifacts.items():
        if artifact is None:
            lines.append(f"- `{name}`：缺失。")
        else:
            lines.append(
                f"- `{name}`（{artifact['evidence_level']}，描述性）：`{artifact['path']}` "
                f"digest `{artifact['content_digest'][:23]}…`"
            )
    if capability is not None:
        lines += [
            "",
            "## 干预证据等级",
            "",
            f"- Harbor 轨迹：checkpoint capability `{capability.get('checkpoint_capability')}`；"
            f"same-checkpoint 注入 `{capability.get('same_checkpoint_hint_injection')}`（独立全新 restart，E3 描述性）。",
        ]
        if intervention_capability is not None:
            lines.append(
                f"- Same-session 通道（非 Harbor-native）：支持 "
                f"`{intervention_capability.get('same_session_hint_injection')}`；"
                f"study 状态 `{intervention_capability.get('study_status')}`"
                f"（原因 `{intervention_capability.get('study_reason')}`）；"
                f"证据等级 `{intervention_capability.get('study_evidence_level')}`；"
                f"可主张 E4 因果 `{intervention_capability.get('causal_intervention_claim_available')}`。"
            )
        if intervention_study is not None:
            arms = intervention_study.get("arms") or {}
            control = (arms.get("control") or {}).get("pass_rate")
            treatment = (arms.get("treatment") or {}).get("pass_rate")
            lines.append(
                f"- 受控 study：control pass rate `{control}`，treatment pass rate "
                f"`{treatment}`，所有注入被确认 "
                f"`{intervention_study.get('all_treatment_injections_confirmed')}`，"
                f"receipts 全部复验 `{intervention_study.get('all_treatment_receipts_verified')}`。"
            )
        else:
            lines.append(
                "- 未运行受控 same-session study（见能力 receipt 的 machine 状态）；"
                "restart-with-hint 不是 E4。"
            )
    lines += ["", "## 成本与用量（provider 上报口径）", ""]
    if costs:
        lines.append(f"`{json.dumps(costs, ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("autopilot 状态中没有记录到 usage 汇总。")
    lines += [
        "",
        "## 未决事项 / 失败",
        "",
    ]
    open_items = [
        f"gate `{name}`：{gate.get('reason') or gate.get('decision')}"
        for name, gate in promotion["gates"].items()
        if gate.get("status") not in {"pass", "reused"}
    ]
    review_count = int((static_receipt.get("counts") or {}).get("review", 0))
    if review_count:
        open_items.append(
            f"{review_count} 项静态 rubric 语义准则仍标记为 review（人工/TB-Science 评审所有）"
        )
    for row in final_receipt.get("limitations") or []:
        open_items.append(str(row))
    lines.extend(f"- {item}" for item in open_items or ["无。"])
    lines += [
        "",
        "## 可复现命令",
        "",
        "```bash",
        "# 1. 重新运行确定性静态 gate",
        f"orbench task-author validate --task-dir {workdir / promotion['selected_task']} \\",
        f"  --paper-provenance {workdir / 'factory-input' / 'paper-provenance.json'} \\",
        "  --out /tmp/recheck-static",
        "# 2. 独立重放 fail-closed finalizer（不重启任何付费 job）",
        "orbench agent-factory finalize \\",
        f"  --plan {factory_out / 'factory-plan.json'} \\",
        f"  --factory-run {factory_out / 'factory-run.json'} \\",
        f"  --workdir {workdir} \\",
        f"  --task-dir {workdir / promotion['selected_task']} \\",
        f"  --static-receipt {out / 'static' / 'authoring-receipt.json'} \\",
        f"  --semantic-review {out / 'semantic' / 'volc-authoring-review.json'} \\",
        f"  --harbor-receipt {promotion['gates'].get('runtime_evidence', {}).get('harbor_receipt')} \\",
        f"  --calibration-receipt {promotion['gates'].get('runtime_evidence', {}).get('calibration_receipt')} \\",
        f"  --final-summary {out / 'cards' / 'task-cards.json'} \\",
        "  --out /tmp/recheck-final",
        "```",
        "",
        "本报告由 deterministic harness 从签名 receipts 生成；agent 撰写的内容一律标注 E1。",
    ]
    report_json = {
        "schema_version": "orbenchlab.factory-final-report.v1",
        "promotion": dict(promotion),
        "source": dict(source),
        "static_counts": dict(static_counts),
        "calibration_receipt_digest": (
            _file_digest(calibration_path) if calibration_path else None
        ),
        "difficulty_receipt_digest": (
            _file_digest(difficulty_path) if difficulty_path.is_file() else None
        ),
        "runtime_capability": capability,
        "agent_artifacts": agent_artifacts,
        "observed_costs": costs,
        "open_items": open_items,
    }
    markdown_path = out / "final-report.md"
    json_path = out / "final-report.json"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(json_path, report_json)
    return {"markdown": markdown_path, "json": json_path}


__all__ = [
    "FactoryPromotionError",
    "SCHEMA_VERSION",
    "run_promotion",
    "write_final_report",
]
