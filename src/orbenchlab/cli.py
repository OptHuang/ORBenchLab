"""The ``orbench`` command line.

Every subcommand is a thin wrapper over the library, so anything the CLI can do
is testable without spawning a process. Machine-readable output goes to stdout
as JSON; human progress notes go to stderr; failures exit non-zero with a
specific code from :mod:`orbenchlab.core.errors`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .campaign import compile as compile_mod
from .campaign import spec as spec_mod
from .core import schema as schema_mod
from .core.errors import ORBenchError
from .integrations import registry
from .report import render as render_mod
from .report.model import NormalizedRollout

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
            "integrations, compiles campaigns into plans with stable run ids, and renders "
            "evidence-labelled reports. It does not execute benchmarks."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command")

    _add_integrations(sub)
    _add_integration(sub)
    _add_campaign(sub)
    _add_report(sub)
    _add_schema(sub)
    return parser


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
