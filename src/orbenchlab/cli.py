"""The ``orbench`` command line.

Every subcommand is a thin wrapper over the library, so anything the CLI can do
is testable without spawning a process. Machine-readable output goes to stdout
as JSON; human progress notes go to stderr; failures exit non-zero with a
specific code from :mod:`orbenchlab.core.errors`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .campaign import compile as compile_mod
from .campaign import spec as spec_mod
from .core import schema as schema_mod
from .core.errors import ORBenchError
from .core.urls import validate_https_base_url
from .integrations import registry
from .report import render as render_mod
from .report.model import NormalizedRollout
from . import workflow as workflow_mod
from . import execution as execution_mod
from . import export as export_mod
from . import source_intake as intake_mod
from . import pipeline as pipeline_mod
from . import task_authoring as authoring_mod
from . import volc_review as volc_review_mod
from . import volc_rollout as volc_rollout_mod
from . import harbor_controls as harbor_controls_mod
from . import authoring_loop as authoring_loop_mod

PROG = "orbench"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except ORBenchError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except BrokenPipeError:  # pragma: no cover - shell pipelines
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Control plane for operations-research agent benchmarks. Registers upstream "
            "integrations, compiles campaigns into plans with stable run ids, delegates "
            "execution to pinned upstream runners, and renders evidence-labelled reports."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command")

    _add_integrations(sub)
    _add_integration(sub)
    _add_campaign(sub)
    _add_report(sub)
    _add_schema(sub)
    _add_doctor(sub)
    _add_run(sub)
    _add_export(sub)
    _add_intake(sub)
    _add_pipeline(sub)
    _add_task_author(sub)
    _add_task_screen(sub)
    _add_harbor_receipt(sub)
    return parser


# --------------------------------------------------------------------------- #
# run -- the practical one-command path
# --------------------------------------------------------------------------- #


def _add_doctor(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "doctor",
        help="check the real runner, benchmark checkout and credential transport",
    )
    parser.add_argument("integration", choices=["oragentbench"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--model", default="")
    parser.add_argument("--scaffold-version", default="")
    parser.add_argument(
        "--auth-mode",
        choices=["api-key", "codex-auth-json", "codex-login"],
        default="api-key",
    )
    parser.add_argument("--model-base-url", default="")
    parser.set_defaults(handler=_cmd_doctor)


def _resolved_model_base_url(args: argparse.Namespace) -> str:
    if args.agent in execution_mod.CONTROL_SCAFFOLDS:
        return ""
    candidate = str(args.model_base_url).strip() if args.model_base_url else ""
    if not candidate and args.auth_mode == "codex-auth-json":
        candidate = execution_mod.discover_codex_base_url()
    if (
        not candidate
        and args.auth_mode != "codex-login"
        and args.agent in execution_mod.ROUTE_PINNED_SCAFFOLDS
    ):
        # MODEL_BASE_URL is configuration, not a secret. Resolve it without
        # placing it on argv or stdout; validation errors never echo the value.
        candidate = os.environ.get("MODEL_BASE_URL", "")
    return validate_https_base_url(candidate) if candidate else ""


def _cmd_doctor(args: argparse.Namespace) -> int:
    base_url = _resolved_model_base_url(args)
    if args.agent not in execution_mod.CONTROL_SCAFFOLDS:
        execution_mod.validate_pinned_scaffold_version(args.scaffold_version)
    report = execution_mod.oragentbench_preconditions(
        source=Path(args.source),
        task_name=args.task,
        scaffold=args.agent,
        model=args.model,
        auth_mode=args.auth_mode,
        model_base_url=base_url,
        require_docker=True,
        require_harbor=True,
        require_secrets=args.agent not in execution_mod.CONTROL_SCAFFOLDS,
    )
    _print_json(
        {
            "ok": report.ok,
            "integration": "oragentbench",
            "agent": args.agent,
            "auth_mode": args.auth_mode,
            "model_base_url_configured": bool(base_url),
            "checks": report.to_dict(),
        }
    )
    return 0 if report.ok else 5


def _add_run(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "run",
        help="prepare or execute one benchmark evaluation in an auditable workspace",
        description=(
            "One-command lifecycle for a supported benchmark. Safe by default: without "
            "--execute it only inspects, plans and writes a preflight bundle."
        ),
    )
    parser.add_argument("integration", choices=["oragentbench"])
    parser.add_argument("--source", required=True, help="validated upstream checkout")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--agent",
        default="oracle",
        help="oracle/nop control, or a recorded scaffold such as codex or claude-code",
    )
    parser.add_argument("--model", default="", help="pinned model id for a model agent")
    parser.add_argument(
        "--scaffold-version",
        default="",
        help="exact released CLI version baked into a paid scaffold image",
    )
    parser.add_argument(
        "--auth-mode",
        choices=["api-key", "codex-auth-json", "codex-login"],
        default="api-key",
        help=(
            "credential transport; api-key is executable. codex-login and "
            "codex-auth-json are prepare/doctor only until a broker exists"
        ),
    )
    parser.add_argument(
        "--model-base-url",
        default="",
        help=(
            "non-secret pinned HTTPS provider URL; only its normalized digest is "
            "recorded in campaign artifacts"
        ),
    )
    parser.add_argument("--date", required=True, help="explicit YYYY-MM-DD identity input")
    parser.add_argument("--workspace", default="orbench-runs")
    parser.add_argument("--wall-clock-sec", type=int, default=2700)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=25.0,
        help=(
            "declared audit envelope only; enforce the real spend ceiling at the "
            "model provider (wall-clock-sec is enforced locally)"
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--prepare-only",
        action="store_true",
        help="inspect, plan and preflight only (the default)",
    )
    action.add_argument("--execute", action="store_true", help="start the upstream runner")
    parser.add_argument(
        "--acknowledge-cost",
        default="",
        help="model runs require the literal i-accept-model-costs",
    )
    parser.set_defaults(handler=_cmd_run)


def _cmd_run(args: argparse.Namespace) -> int:
    base_url = _resolved_model_base_url(args)
    prepared = workflow_mod.prepare_oragentbench_run(
        source=args.source,
        task=args.task,
        agent=args.agent,
        model=args.model,
        scaffold_version=args.scaffold_version,
        date=args.date,
        workspace=args.workspace,
        wall_clock_sec=args.wall_clock_sec,
        max_cost_usd=args.max_cost_usd,
        auth_mode=args.auth_mode,
        model_base_url=base_url,
    )
    payload: dict[str, Any] = {
        "campaign_id": prepared.campaign_id,
        "integration": "oragentbench",
        "run_root": str(prepared.run_root),
        "resumed": prepared.resumed,
        "state": "prepared",
        "preconditions": prepared.preconditions.to_dict(),
    }
    if args.execute:
        ingested = workflow_mod.execute_prepared_run(
            prepared, acknowledge_cost=args.acknowledge_cost
        )
        payload.update(
            {
                "state": "completed",
                "trials": ingested.trials,
                "orphans": ingested.orphans,
                "normalized": str(ingested.normalized_path),
                "report": str(ingested.report_dir / "summary.md"),
            }
        )
    _print_json(payload)
    return 0


def _add_export(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "export",
        help="export a completed run as a sanitized, integrity-checked share bundle",
        description=(
            "Verify a completed ORAgentBench workspace and atomically export only its "
            "normalized rollout, rendered report and newly derived path-free metadata. "
            "Raw jobs, logs, credentials and the host-local manifest are never copied."
        ),
    )
    parser.add_argument("--run-root", required=True, help="completed local run workspace")
    parser.add_argument("--destination", required=True, help="new share-bundle directory")
    parser.set_defaults(handler=_cmd_export)


def _cmd_export(args: argparse.Namespace) -> int:
    result = export_mod.export_shareable_run(args.run_root, args.destination)
    _print_json(result.to_dict())
    return 0


# --------------------------------------------------------------------------- #
# source intake -- public metadata to a human review queue
# --------------------------------------------------------------------------- #


def _add_intake(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "intake",
        help="collect public OR source metadata into an auditable review queue",
        description=(
            "Fetches configured RSS/arXiv/GitHub metadata, de-duplicates it and writes "
            "a human-review queue. No model calls, task authoring, publication, or raw "
            "source writes are performed."
        ),
    )
    inner = parser.add_subparsers(dest="subcommand")

    validate_parser = inner.add_parser(
        "validate", help="validate a source-feed configuration without fetching it"
    )
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(handler=_cmd_intake_validate)

    collect_parser = inner.add_parser(
        "collect", help="fetch feeds and write intake.json plus review_queue.jsonl"
    )
    collect_parser.add_argument("--config", required=True)
    collect_parser.add_argument("--out", required=True, help="new or idempotent bundle directory")
    collect_parser.add_argument(
        "--previous",
        default="",
        help="prior intake.json or bundle directory used for cross-day de-duplication",
    )
    collect_parser.add_argument("--timeout-sec", type=int, default=20)
    collect_parser.add_argument("--max-bytes", type=int, default=2_000_000)
    collect_parser.add_argument(
        "--created-at",
        default="",
        help="optional ISO-8601 timestamp (otherwise current UTC time)",
    )
    collect_parser.set_defaults(handler=_cmd_intake_collect)

    bind_parser = inner.add_parser(
        "bind-paper",
        help="bind one checksummed intake item to exact local paper bytes",
    )
    bind_parser.add_argument("--intake", required=True, help="intake bundle directory or intake.json")
    bind_parser.add_argument("--item-uid", required=True)
    bind_parser.add_argument("--source-file", required=True, help="local PDF/Markdown paper; read only")
    bind_parser.add_argument(
        "--license-status",
        required=True,
        choices=("pending-human", "registry-resolved"),
    )
    bind_parser.add_argument("--out", required=True, help="paper-provenance.json or output directory")
    bind_parser.set_defaults(handler=_cmd_intake_bind_paper)


def _cmd_intake_validate(args: argparse.Namespace) -> int:
    feeds = intake_mod.load_feed_config(args.config)
    _print_json(
        {
            "valid": True,
            "config": str(Path(args.config)),
            "feeds": [feed.to_dict() for feed in feeds],
            "network_requests": 0,
            "model_calls": 0,
        }
    )
    return 0


def _cmd_intake_collect(args: argparse.Namespace) -> int:
    feeds = intake_mod.load_feed_config(args.config)
    result = intake_mod.collect(
        feeds,
        previous=args.previous or None,
        created_at=args.created_at or None,
        timeout_sec=args.timeout_sec,
        max_bytes=args.max_bytes,
    )
    paths = intake_mod.write_bundle(result, args.out)
    _print_json(
        {
            "intake_id": result.intake_id,
            "feeds": len(result.feeds),
            "feed_errors": result.feed_errors,
            "items": len(result.items),
            "new_or_updated": len(result.review_queue),
            "model_calls": 0,
            "task_authoring": "disabled",
            "written": {key: str(path) for key, path in paths.items()},
        }
    )
    # A partial snapshot is useful and is left on disk, but CI/automation must
    # notice that at least one configured source failed.
    return 8 if result.has_errors else 0


def _cmd_intake_bind_paper(args: argparse.Namespace) -> int:
    binding = intake_mod.bind_paper(
        args.intake,
        item_uid=args.item_uid,
        source_file=args.source_file,
        license_status=args.license_status,
    )
    path = intake_mod.write_paper_binding(binding, args.out)
    _print_json(
        {
            "item_uid": binding["intake_item_uid"],
            "source_content_digest": binding["source_content_digest"],
            "license_status": binding["license_status"],
            "binding_digest": binding["binding_digest"],
            "model_calls": 0,
            "raw_sources_modified": False,
            "written": str(path),
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# unattended final task cards
# --------------------------------------------------------------------------- #


def _add_pipeline(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "pipeline",
        help="build unattended per-task cards from existing evidence",
        description=(
            "Join task genomes, screening reports and optional intake metadata into "
            "deterministic task cards. This is a post-run summarizer: it never calls "
            "a model, reads raw trajectories, or invents a task."
        ),
    )
    inner = parser.add_subparsers(dest="subcommand")
    run_parser = inner.add_parser(
        "run", help="write task-cards.json, task-cards.md and pipeline manifest"
    )
    run_parser.add_argument("--tasks", action="append", default=[], help="task genome file or directory")
    run_parser.add_argument("--screenings", action="append", default=[], help="screening/report file or directory")
    run_parser.add_argument(
        "--evidence-root", default="artifacts",
        help="fallback root searched for *screening*.json when --screenings is omitted",
    )
    run_parser.add_argument("--intake", default="", help="optional intake.json provenance bundle")
    run_parser.add_argument(
        "--intake-config", default="",
        help="optional feed config; collect metadata automatically before writing cards",
    )
    run_parser.add_argument("--previous", default="", help="prior intake bundle for cross-day deduplication")
    run_parser.add_argument("--intake-out", default="", help="intake bundle directory (default: OUT/intake)")
    run_parser.add_argument("--created-at", default="", help="optional deterministic intake timestamp")
    run_parser.add_argument("--out", required=True, help="output directory")
    run_parser.set_defaults(handler=_cmd_pipeline_run)


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    task_inputs = list(args.tasks)
    if not task_inputs and Path("docs/task-genomes").is_dir():
        task_inputs = ["docs/task-genomes"]
    screening_inputs = list(args.screenings)
    if not screening_inputs and Path(args.evidence_root).is_dir():
        screening_inputs = [
            str(path)
            for path in sorted(Path(args.evidence_root).rglob("*screening*.json"))
            if path.is_file() and not path.is_symlink()
        ]
    intake_path = args.intake or None
    intake_error = False
    intake_result: dict[str, Any] | None = None
    if args.intake_config:
        feeds = intake_mod.load_feed_config(args.intake_config)
        collected = intake_mod.collect(
            feeds,
            previous=args.previous or None,
            created_at=args.created_at or None,
        )
        intake_paths = intake_mod.write_bundle(collected, args.intake_out or (out / "intake"))
        intake_path = str(intake_paths["intake"])
        intake_error = collected.has_errors
        intake_result = {
            "intake_id": collected.intake_id,
            "feeds": len(collected.feeds),
            "items": len(collected.items),
            "feed_errors": collected.feed_errors,
            "written": {key: str(path) for key, path in intake_paths.items()},
        }
    result = pipeline_mod.run(
        out=out,
        task_inputs=task_inputs,
        screening_inputs=screening_inputs,
        intake_path=intake_path,
    )
    if intake_result is not None:
        result["intake"] = intake_result
    _print_json(result)
    return 8 if intake_error else 0


# --------------------------------------------------------------------------- #
# paper -> Terminal-Bench Science task authoring gate
# --------------------------------------------------------------------------- #


def _add_task_author(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "task-author",
        help="validate a paper-backed task against the TB-Science authoring gate",
        description=(
            "Run deterministic local checks over a candidate Terminal-Bench Science task. "
            "Semantic rubric criteria remain review items; this command never calls a model."
        ),
    )
    inner = parser.add_subparsers(dest="subcommand")
    validate_parser = inner.add_parser("validate", help="write an authoring round receipt")
    validate_parser.add_argument("--task-dir", required=True, help="candidate task directory")
    validate_parser.add_argument(
        "--paper-provenance", default="", help="JSON/YAML with title, URL, digest and license status"
    )
    validate_parser.add_argument("--round", type=int, default=1, help="authoring iteration number")
    validate_parser.add_argument("--previous", default="", help="previous authoring receipt for round linkage")
    validate_parser.add_argument("--out", required=True, help="receipt output directory")
    validate_parser.set_defaults(handler=_cmd_task_author_validate)

    review_parser = inner.add_parser(
        "review",
        help="ask independent Volcengine agents to review a static authoring receipt",
        description=(
            "Run structured, evidence-bounded authoring reviews through the Volcengine "
            "Anthropic-compatible endpoint. Raw prompts, responses and credentials are not stored."
        ),
    )
    review_parser.add_argument("--task-dir", required=True, help="candidate task directory")
    review_parser.add_argument("--paper-provenance", required=True, help="JSON/YAML paper provenance")
    review_parser.add_argument("--receipt", required=True, help="static authoring receipt JSON")
    review_parser.add_argument("--round", type=int, default=1, help="authoring iteration number")
    review_parser.add_argument(
        "--models",
        default="",
        help="at least two distinct comma-separated Volc reviewer model ids",
    )
    review_parser.add_argument("--timeout-sec", type=int, default=120, help="per-model HTTP timeout")
    review_parser.add_argument("--max-tokens", type=int, default=2400, help="per-model output token cap")
    review_parser.add_argument("--out", required=True, help="review output directory")
    review_parser.set_defaults(handler=_cmd_task_author_review)

    iterate_parser = inner.add_parser(
        "iterate",
        help="run bounded Volc author/reviewer rounds over a copied strict skeleton",
    )
    iterate_parser.add_argument("--seed-task", required=True)
    iterate_parser.add_argument("--paper-provenance", required=True)
    iterate_parser.add_argument(
        "--paper-derivation",
        required=True,
        help="bounded audited UTF-8 paper-to-task derivation evidence",
    )
    iterate_parser.add_argument("--author-model", default="ark-code-latest")
    iterate_parser.add_argument("--review-model", action="append", required=True)
    iterate_parser.add_argument("--max-rounds", type=int, default=3)
    iterate_parser.add_argument("--max-author-tokens", type=int, default=2400)
    iterate_parser.add_argument("--max-review-tokens", type=int, default=2400)
    iterate_parser.add_argument("--timeout-sec", type=int, default=120)
    iterate_parser.add_argument("--out", required=True)
    iterate_parser.set_defaults(handler=_cmd_task_author_iterate)


def _cmd_task_author_validate(args: argparse.Namespace) -> int:
    receipt = authoring_mod.validate_task(
        args.task_dir,
        paper_provenance=args.paper_provenance or None,
        round_number=args.round,
        previous_receipt=args.previous or None,
    )
    paths = authoring_mod.write_receipt(receipt, args.out)
    _print_json(
        {
            "decision": receipt["decision"],
            "round": receipt["round"],
            "counts": receipt["counts"],
            "receipt_digest": receipt["receipt_digest"],
            "written": {key: str(path) for key, path in paths.items()},
        }
    )
    return 8 if receipt["decision"] == "blocked" else 0


def _cmd_task_author_review(args: argparse.Namespace) -> int:
    paper = authoring_mod._load_document(Path(args.paper_provenance))
    receipt_path = Path(args.receipt)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise volc_review_mod.VolcReviewError("authoring receipt is not valid UTF-8 JSON") from None
    if not isinstance(receipt, dict):
        raise volc_review_mod.VolcReviewError("authoring receipt must be a JSON object")
    config = volc_review_mod.VolcConfig.from_env(timeout_sec=args.timeout_sec)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    review = volc_review_mod.review_task(
        args.task_dir,
        paper_provenance=paper,
        receipt=receipt,
        config=config,
        models=models,
        round_number=args.round,
        max_tokens=args.max_tokens,
    )
    paths = volc_review_mod.write_review(review, args.out)
    written = json.loads(paths["json"].read_text(encoding="utf-8"))
    _print_json(
        {
            "aggregate_decision": written["aggregate_decision"],
            "round": written["round"],
            "models": written["models"],
            "review_count": written["review_count"],
            "evidence_level": written["evidence_level"],
            "review_digest": written["review_digest"],
            "written": {key: str(path) for key, path in paths.items()},
        }
    )
    return 8 if written["aggregate_decision"] == "blocked-static-gate" else 0


def _cmd_task_author_iterate(args: argparse.Namespace) -> int:
    config = volc_review_mod.VolcConfig.from_env(timeout_sec=args.timeout_sec)
    run = authoring_loop_mod.iterate(
        args.seed_task,
        paper_provenance=args.paper_provenance,
        paper_derivation=args.paper_derivation,
        config=config,
        author_model=args.author_model,
        review_models=args.review_model,
        max_rounds=args.max_rounds,
        max_author_tokens=args.max_author_tokens,
        max_review_tokens=args.max_review_tokens,
        out=args.out,
    )
    _print_json(
        {
            "status": run["status"],
            "stop_reason": run["stop_reason"],
            "rounds": len(run["rounds"]),
            "seed_unchanged": run["seed_unchanged"],
            "final_task": run["final_task"],
            "run_digest": run["run_digest"],
            "written": {
                "run": str(Path(args.out) / "run.json"),
                "manifest": str(Path(args.out) / "run-manifest.json"),
            },
        }
    )
    return 0 if run["status"] == "promising-needs-harbor" else 8


# --------------------------------------------------------------------------- #
# Volcengine task-local model screening
# --------------------------------------------------------------------------- #


def _add_task_screen(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "task-screen",
        help="run a strict task candidate with Volcengine models in a no-network container",
        description=(
            "Ask Volcengine models for a bounded solver.py, execute it in the task's "
            "pinned test image, and write an outcome-grounded screening report. "
            "This is not Harbor acceptance."
        ),
    )
    parser.add_argument("--task-dir", action="append", required=True, help="strict task directory; repeat for a suite")
    parser.add_argument("--test-image", action="append", required=True, help="matching Docker verifier image; repeat in task order")
    parser.add_argument("--out", required=True, help="screening output directory")
    parser.add_argument("--models", default="", help="comma-separated Volc model ids")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--hint-level", type=int, default=0, help="contract reminder level (0=none, 1=exact output reminder)")
    parser.add_argument("--hint-levels", default="", help="comma-separated matrix; overrides --hint-level")
    parser.add_argument("--controls", default="oracle,nop", help="comma-separated zero-model controls")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.set_defaults(handler=_cmd_task_screen)


def _cmd_task_screen(args: argparse.Namespace) -> int:
    config = volc_review_mod.VolcConfig.from_env(timeout_sec=args.timeout_sec)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    controls = [value.strip() for value in args.controls.split(",") if value.strip()]
    if len(args.task_dir) != len(args.test_image):
        raise volc_rollout_mod.VolcRolloutError(
            "--task-dir and --test-image must be repeated the same number of times"
        )
    try:
        hint_levels = (
            [int(value.strip()) for value in args.hint_levels.split(",") if value.strip()]
            if args.hint_levels
            else [args.hint_level]
        )
    except ValueError:
        raise volc_rollout_mod.VolcRolloutError("--hint-levels must contain integers") from None
    tasks = list(zip(args.task_dir, args.test_image, strict=True))
    if len(tasks) == 1:
        report = volc_rollout_mod.run_rollout(
            tasks[0][0],
            config=config,
            models=models,
            test_image=tasks[0][1],
            out=args.out,
            repetitions=args.repetitions,
            hint_levels=hint_levels,
            controls=controls,
            timeout_sec=args.timeout_sec,
            max_tokens=args.max_tokens,
        )
    else:
        report = volc_rollout_mod.run_suite(
            tasks,
            config=config,
            models=models,
            out=args.out,
            repetitions=args.repetitions,
            hint_levels=hint_levels,
            controls=controls,
            timeout_sec=args.timeout_sec,
            max_tokens=args.max_tokens,
        )
    _print_json(
        {
            "tasks": [row["task"] for row in report["tasks"]],
            "decisions": {row["task"]: row["decision"] for row in report["tasks"]},
            "evidence_levels": {row["task"]: row["evidence_level"] for row in report["tasks"]},
            "arms": {row["task"]: row["arms"] for row in report["tasks"]},
            "report_digest": report["report_digest"],
            "written": {"json": str(Path(args.out) / "screening-report.json"), "markdown": str(Path(args.out) / "screening-report.md")},
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# real Harbor Oracle/NOP evidence
# --------------------------------------------------------------------------- #


def _add_harbor_receipt(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "harbor-receipt",
        help="validate completed Harbor Oracle/NOP jobs into a pipeline receipt",
        description=(
            "Fail closed unless both jobs contain one clean completed trial, a consistent "
            "reward, a valid CTRF report, and an artifact manifest."
        ),
    )
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--executed-task-dir", required=True)
    parser.add_argument("--oracle-job", required=True)
    parser.add_argument("--nop-job", required=True)
    parser.add_argument("--out", required=True)
    parser.set_defaults(handler=_cmd_harbor_receipt)


def _cmd_harbor_receipt(args: argparse.Namespace) -> int:
    receipt = harbor_controls_mod.build_receipt(
        args.task_dir,
        executed_task_dir=args.executed_task_dir,
        oracle_job=args.oracle_job,
        nop_job=args.nop_job,
    )
    paths = harbor_controls_mod.write_receipt(receipt, args.out)
    row = receipt["tasks"][0]
    _print_json(
        {
            "task": row["task"],
            "evidence_level": row["evidence_level"],
            "control_gates": row["control_gates"],
            "report_digest": receipt["report_digest"],
            "written": {key: str(path) for key, path in paths.items()},
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# integrations
# --------------------------------------------------------------------------- #


def _add_integrations(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("integrations", help="list registered integrations")
    inner = parser.add_subparsers(dest="subcommand")
    list_parser = inner.add_parser("list", help="list registered integrations")
    list_parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    list_parser.set_defaults(handler=_cmd_integrations_list)


def _cmd_integrations_list(args: argparse.Namespace) -> int:
    rows = registry.summary_rows()
    if args.json:
        _print_json({"integrations": rows})
        return 0
    widths = {
        "name": max(4, *(len(row["name"]) for row in rows)),
        "kind": max(4, *(len(row["kind"]) for row in rows)),
    }
    header = f"{'NAME'.ljust(widths['name'])}  {'KIND'.ljust(widths['kind'])}  REQUIRES"
    print(header)
    print("-" * len(header))
    for row in rows:
        requires = []
        if row["requires_model_api_key"]:
            requires.append("model-key")
        if row["requires_solver_license"]:
            requires.append("solver-license")
        if row["requires_self_hosted_runner"]:
            requires.append("self-hosted-runner")
        if row["performance_scored"]:
            requires.append("perf-isolated-site")
        print(
            f"{row['name'].ljust(widths['name'])}  {row['kind'].ljust(widths['kind'])}  "
            f"{', '.join(requires) or 'none'}"
        )
    print()
    for row in rows:
        print(f"{row['name']}: {row['upstream_repo']} @ {row['pinned_commit']}")
    return 0


def _add_integration(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("integration", help="inspect or describe one integration")
    inner = parser.add_subparsers(dest="subcommand")

    inspect_parser = inner.add_parser(
        "inspect",
        help="statically inspect an upstream checkout and emit a machine-readable report",
        description=(
            "Reads an upstream checkout and reports what it found. Makes no model calls, "
            "executes no benchmark and reads no credentials."
        ),
    )
    inspect_parser.add_argument("name", help=f"integration name ({', '.join(registry.names())})")
    inspect_parser.add_argument(
        "--source", required=True, help="path to an upstream checkout to inspect"
    )
    inspect_parser.add_argument("--json", dest="json_out", help="also write the report to this path")
    inspect_parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="exit non-zero when the report is degraded, not only when it failed",
    )
    inspect_parser.set_defaults(handler=_cmd_integration_inspect)

    describe_parser = inner.add_parser("describe", help="print an integration's declaration")
    describe_parser.add_argument("name")
    describe_parser.set_defaults(handler=_cmd_integration_describe)


def _cmd_integration_inspect(args: argparse.Namespace) -> int:
    report = registry.inspect(args.name, Path(args.source))
    payload = report.to_dict()
    _print_json(payload)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}", file=sys.stderr)
    if report.status == "failed":
        failures = ", ".join(check.id for check in report.failures())
        print(f"{PROG}: inspection failed: {failures}", file=sys.stderr)
        return 3
    if args.fail_on_warn and report.status == "degraded":
        print(f"{PROG}: inspection degraded and --fail-on-warn was set", file=sys.stderr)
        return 3
    return 0


def _cmd_integration_describe(args: argparse.Namespace) -> int:
    _print_json(registry.describe(args.name))
    return 0


# --------------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------------- #


def _add_campaign(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("campaign", help="validate and plan campaigns")
    inner = parser.add_subparsers(dest="subcommand")

    validate_parser = inner.add_parser("validate", help="validate a campaign spec")
    validate_parser.add_argument("spec", help="path to a campaign spec YAML file")
    validate_parser.add_argument("--sites-dir", default="sites")
    validate_parser.set_defaults(handler=_cmd_campaign_validate)

    plan_parser = inner.add_parser(
        "plan",
        help="compile a spec into a plan, a plan ledger and Harbor job configs",
        description=(
            "Deterministic: the same spec always produces byte-identical output. "
            "Planning executes nothing."
        ),
    )
    plan_parser.add_argument("spec")
    plan_parser.add_argument("--out", required=True, help="output directory")
    plan_parser.add_argument("--sites-dir", default="sites")
    plan_parser.add_argument("--json", action="store_true", help="print the plan to stdout")
    plan_parser.set_defaults(handler=_cmd_campaign_plan)


def _load_validated_spec(args: argparse.Namespace) -> spec_mod.CampaignSpec:
    raw = spec_mod.load_spec(args.spec)
    return spec_mod.validate(raw, sites_dir=args.sites_dir, source_path=args.spec)


def _cmd_campaign_validate(args: argparse.Namespace) -> int:
    spec = _load_validated_spec(args)
    integration = registry.describe(spec.integration)
    _print_json(
        {
            "valid": True,
            "spec": args.spec,
            "slug": spec.slug,
            "integration": spec.integration,
            "integration_kind": integration["kind"],
            "site": spec.site,
            "evidence_intent": spec.evidence_intent.value,
            "planned_runs": spec.n_planned_runs,
            "makes_model_calls": spec.makes_model_calls,
            "zero_cost": not spec.makes_model_calls,
        }
    )
    return 0


def _cmd_campaign_plan(args: argparse.Namespace) -> int:
    spec = _load_validated_spec(args)
    compiled = compile_mod.compile_campaign(spec)
    written = compile_mod.write_plan(compiled, args.out)
    if args.json:
        _print_json(compiled.plan_dict())
    else:
        _print_json(
            {
                "campaign_id": compiled.campaign_id,
                "runs": len(compiled.runs),
                "jobs": len(compiled.jobs),
                "shards": spec.shards,
                "written": sorted(str(path) for path in written.values()),
            }
        )
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _add_report(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("report", help="build reports from normalized data")
    inner = parser.add_subparsers(dest="subcommand")

    build_parser = inner.add_parser(
        "build",
        help="render a report from a normalized rollout slice",
        description=(
            "Renders summary.md, summary.json and evidence_index.json. The evidence label "
            "is downgraded automatically when its preconditions are unmet, and comparative "
            "wording is rejected in single-rollout output."
        ),
    )
    build_parser.add_argument("--input", required=True, help="normalized rollout JSON")
    build_parser.add_argument("--out", required=True, help="output directory")
    build_parser.add_argument(
        "--require-label",
        choices=["exploratory", "partial", "validated"],
        help="exit non-zero if the effective label is weaker than this",
    )
    build_parser.set_defaults(handler=_cmd_report_build)


def _cmd_report_build(args: argparse.Namespace) -> int:
    rollout = NormalizedRollout.load(args.input)
    report = render_mod.build_report(rollout)
    paths = render_mod.write_report(report, args.out)
    _print_json(
        {
            "campaign_id": report.campaign_id,
            "intended_label": report.intended_label.value,
            "effective_label": report.effective_label.value,
            "downgrade_reasons": list(report.downgrade_reasons),
            "comparative_claims_allowed": report.comparisons_allowed,
            "claims": len(report.claims),
            "written": {key: str(path) for key, path in paths.items()},
        }
    )
    if args.require_label:
        from .core.evidence import EvidenceLabel

        required = EvidenceLabel(args.require_label)
        if report.effective_label.rank < required.rank:
            print(
                f"{PROG}: effective evidence label {report.effective_label.value!r} is weaker "
                f"than the required {required.value!r}",
                file=sys.stderr,
            )
            return 4
    return 0


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def _add_schema(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("schema", help="list and apply the shipped JSON schemas")
    inner = parser.add_subparsers(dest="subcommand")

    list_parser = inner.add_parser("list", help="list shipped schemas")
    list_parser.set_defaults(handler=_cmd_schema_list)

    validate_parser = inner.add_parser("validate", help="validate a document against a schema")
    validate_parser.add_argument("document")
    validate_parser.add_argument(
        "--schema", required=True, help="schema file name, e.g. normalized_rollout.schema.json"
    )
    validate_parser.set_defaults(handler=_cmd_schema_validate)


def _cmd_schema_list(args: argparse.Namespace) -> int:
    schemas = []
    for path in schema_mod.iter_schema_paths():
        schema = schema_mod.load_schema(path)
        schemas.append(
            {
                "file": path.name,
                "title": schema.get("title"),
                "description": schema.get("description"),
            }
        )
    _print_json({"schemas": schemas})
    return 0


def _cmd_schema_validate(args: argparse.Namespace) -> int:
    schema_path = schema_mod.schemas_dir() / args.schema
    if not schema_path.is_file():
        raise schema_mod.SchemaError(
            f"unknown schema {args.schema!r}; available: "
            f"{[p.name for p in schema_mod.iter_schema_paths()]}"
        )
    schema = schema_mod.load_schema(schema_path)
    document = json.loads(Path(args.document).read_text(encoding="utf-8"))
    schema_mod.validate(document, schema, name=args.document)
    _print_json({"valid": True, "document": args.document, "schema": args.schema})
    return 0


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
