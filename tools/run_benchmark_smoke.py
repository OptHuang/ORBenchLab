#!/usr/bin/env python3
"""Run a benchmark smoke through its own upstream execution path.

This is the one place in the repository that starts an upstream process. It
does not implement execution: it validates inputs against the checkout, builds
the command upstream documents for itself, checks the preconditions, hands the
argv over, and writes a small sanitized receipt.

Everything except the handover is a pure function in `orbenchlab.execution`,
which is why the exact commands can be unit-tested without a key, a licence,
Docker or a paid call.

Two safety properties hold by construction:

* **`--dry-run` is the default.** Nothing starts unless `--execute` is passed
  *and*, for a paid path, `--acknowledge-cost i-accept-model-costs` is too.
* **Preconditions fail closed.** A missing checkout, task, secret, licence,
  image or isolation guarantee stops the run. Nothing degrades into a cheaper
  approximation that still writes a receipt.

Examples
--------
Zero cost — print the exact command without running anything::

    python tools/run_benchmark_smoke.py oragentbench \\
        --source upstream/ORAgentBench \\
        --task additive_microfactory_order_planning \\
        --scaffold claude-code --model deepseek-v4-pro \\
        --campaign-slug oab-smoke --date 2026-08-22 \\
        --dataset-digest sha256:<64 hex>

Zero cost — let upstream's wrapper prove the config is consumable::

    ... --upstream-dry-run --execute

Real run, on a configured self-hosted runner::

    ... --execute --acknowledge-cost i-accept-model-costs
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orbenchlab import execution  # noqa: E402
from orbenchlab.campaign import compile as compile_mod  # noqa: E402
from orbenchlab.campaign import spec as spec_mod  # noqa: E402
from orbenchlab.core.errors import ORBenchError, PreconditionError  # noqa: E402
from orbenchlab.integrations import registry  # noqa: E402

COST_ACKNOWLEDGEMENT = "i-accept-model-costs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark_smoke.py",
        description=(
            "Validate inputs, build the upstream command, and optionally run it. "
            "Dry run by default; paid paths additionally require an explicit "
            "cost acknowledgement."
        ),
    )
    parser.add_argument(
        "integration", choices=["oragentbench", "frontieror"], help="which benchmark"
    )
    parser.add_argument("--source", required=True, help="path to the upstream checkout")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually start the upstream process (default: print the command only)",
    )
    parser.add_argument(
        "--acknowledge-cost",
        default="",
        help=f"must be {COST_ACKNOWLEDGEMENT!r} to execute a path that makes model calls",
    )
    parser.add_argument("--receipt", help="write a sanitized receipt to this path")
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter used for upstream commands"
    )
    parser.add_argument(
        "--require-preconditions",
        action="store_true",
        help=(
            "fail a dry run whose preconditions are unmet. Execution always enforces "
            "them; this makes a check-only run enforce them too."
        ),
    )
    parser.add_argument(
        "--allow-missing-tooling",
        action="store_true",
        help=(
            "skip the docker/harbor presence checks. For building a command on a "
            "machine that will never run it; refused together with --execute."
        ),
    )

    oab = parser.add_argument_group("oragentbench")
    oab.add_argument("--task", help="task directory name under harbor_tasks/")
    oab.add_argument("--scaffold", default="claude-code", help="agent scaffold or a control")
    oab.add_argument("--model", default="", help="pinned model id")
    oab.add_argument("--campaign-slug", default="oragentbench-agent-smoke")
    oab.add_argument("--date", default="", help="campaign date, YYYY-MM-DD")
    oab.add_argument(
        "--dataset-digest",
        default="",
        help="dataset digest from 'orbench integration inspect'; computed from the "
        "checkout when omitted",
    )
    oab.add_argument("--seed", type=int, default=1)
    oab.add_argument("--wall-clock-sec", type=int, default=2700)
    oab.add_argument("--max-cost-usd", type=float, default=25.0)
    oab.add_argument("--site", default="local-docker")
    oab.add_argument(
        "--output-root",
        default="out/benchmark-smoke",
        help="deterministic root for the compiled plan and the Harbor jobs directory",
    )
    oab.add_argument(
        "--upstream-dry-run",
        action="store_true",
        help=(
            "pass --dry-run to upstream's wrapper: it prints the transformed config and "
            "the harbor command, then exits without running Harbor. Zero cost."
        ),
    )

    fo = parser.add_argument_group("frontieror")
    fo.add_argument("--paper-id", help="paper id from paper_meta_info.json")
    fo.add_argument("--primary-model", default="", help="full provider/model route")
    fo.add_argument("--stage1-instances", nargs="+", default=["tiny"])
    fo.add_argument("--dev-set", nargs="+", default=["large_1"])
    fo.add_argument("--test-set", nargs="+", default=["large_2"])
    fo.add_argument("--run-id", default="orbench-agent-smoke")
    fo.add_argument("--cpus", type=int, default=1)
    fo.add_argument("--memory", default="128G")
    fo.add_argument("--coral-agent-count", type=int, default=1)
    fo.add_argument("--coral-attempts", type=int, default=10)
    fo.add_argument("--coral-max-steps", type=int, default=10)
    fo.add_argument("--coral-max-seconds", default="auto")
    fo.add_argument(
        "--extra",
        default="",
        help=(
            "additional upstream agent flags as one shell-quoted string, e.g. "
            "--extra '--wls-egress off'. A single string keeps argparse from "
            "swallowing the flags before they can be validated. Anything outside "
            "upstream's documented interface, or any flag from its fixed trusted "
            "profile, is refused."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except ORBenchError as exc:
        print(f"run_benchmark_smoke: error: {exc}", file=sys.stderr)
        return exc.exit_code


def _dispatch(args: argparse.Namespace) -> int:
    if args.allow_missing_tooling and args.execute:
        print(
            "run_benchmark_smoke: error: --allow-missing-tooling cannot be combined with "
            "--execute; a run needs the tooling it claims to skip checking for",
            file=sys.stderr,
        )
        return 5
    if args.integration == "oragentbench":
        return _oragentbench(args)
    return _frontieror(args)


# --------------------------------------------------------------------------- #
# ORAgentBench
# --------------------------------------------------------------------------- #


def _oragentbench(args: argparse.Namespace) -> int:
    source = execution.validate_oragentbench_source(Path(args.source))
    if not args.task:
        raise ORBenchError("--task is required for oragentbench")
    execution.validate_oragentbench_task(source, args.task)

    digest = args.dataset_digest
    if not digest:
        # Inspecting the checkout is free and gives the content-addressed digest
        # that pins the run ids to this exact dataset state.
        digest = registry.inspect("oragentbench", source).facts["dataset_digest"]

    date = args.date or _require_date()
    raw_spec = execution.oragentbench_agent_campaign_spec(
        slug=args.campaign_slug,
        date=date,
        dataset_digest=digest,
        task_name=args.task,
        scaffold=args.scaffold,
        model=args.model,
        seeds=(args.seed,),
        wall_clock_sec=args.wall_clock_sec,
        max_cost_usd=args.max_cost_usd,
        site=args.site,
    )

    # Deterministic output root. The campaign id is a pure function of the spec,
    # and `jobs_dir` is excluded from that function precisely so the root can be
    # named after the id without the id moving when the root is written back in.
    campaign_spec = spec_mod.validate(raw_spec, sites_dir=REPO_ROOT / "sites")
    campaign_id = compile_mod.compile_campaign(campaign_spec).campaign_id

    campaign_root = (Path(args.output_root) / campaign_id).resolve()
    raw_spec["harbor"]["jobs_dir"] = str(campaign_root / "jobs")
    campaign_spec = spec_mod.validate(raw_spec, sites_dir=REPO_ROOT / "sites")
    compiled = compile_mod.compile_campaign(campaign_spec)
    if compiled.campaign_id != campaign_id:
        raise ORBenchError(
            "campaign id changed when the output root was applied "
            f"({campaign_id} -> {compiled.campaign_id}); the output location must not "
            "be part of campaign identity"
        )
    written = compile_mod.write_plan(compiled, campaign_root / "plan")

    job_configs = sorted((campaign_root / "plan" / "jobs").glob("*.yaml"))
    if len(job_configs) != 1:
        raise ORBenchError(
            f"expected exactly one job config for a single-task smoke, got {len(job_configs)}"
        )

    command = execution.oragentbench_agent_command(
        source=source,
        job_config=job_configs[0],
        python=args.python,
        skip_build=True,
        dry_run=args.upstream_dry_run,
        # The credentials reach Harbor through the job config, so the command
        # carries their names rather than their values.
        required_env=campaign_spec.agents[0].secret_names,
    )
    preconditions = execution.oragentbench_preconditions(
        source=source,
        task_name=args.task,
        scaffold=args.scaffold,
        model=args.model,
        require_docker=not args.allow_missing_tooling and not args.upstream_dry_run,
        # Upstream's wrapper needs neither harbor nor a credential for its own
        # dry run: it stops after printing the transformed config and command.
        require_harbor=not args.allow_missing_tooling and not args.upstream_dry_run,
        require_secrets=not args.upstream_dry_run,
    )

    return _finish(
        args=args,
        integration="oragentbench",
        mode="agent" if args.scaffold not in execution.CONTROL_SCAFFOLDS else "controls",
        command=command,
        preconditions=preconditions,
        campaign_id=compiled.campaign_id,
        output_root=str(campaign_root),
        notes=[
            f"plan written to {written['plan']}",
            f"ledger written to {written['plan_ledger']}",
            f"harbor jobs_dir: {raw_spec['harbor']['jobs_dir']}",
            "raw Harbor job bundles stay on this host; only this receipt leaves it",
        ],
    )


def _require_date() -> str:
    raise ORBenchError(
        "--date is required (YYYY-MM-DD). Campaign ids are derived from it, and an id "
        "that changes with the wall clock is not an id."
    )


# --------------------------------------------------------------------------- #
# FrontierOR
# --------------------------------------------------------------------------- #


def _frontieror(args: argparse.Namespace) -> int:
    source = execution.validate_frontieror_source(Path(args.source))
    if not args.paper_id:
        raise ORBenchError("--paper-id is required for frontieror")

    command = execution.frontieror_agent_command(
        source=source,
        paper_id=args.paper_id,
        primary_model=args.primary_model,
        stage1_instances=args.stage1_instances,
        dev_instances=args.dev_set,
        test_instances=args.test_set,
        run_id=args.run_id,
        cpus=args.cpus,
        memory=args.memory,
        coral_agent_count=args.coral_agent_count,
        coral_attempts=args.coral_attempts,
        coral_max_steps=args.coral_max_steps,
        coral_max_seconds=args.coral_max_seconds,
        extra=shlex.split(args.extra),
        python=args.python,
    )
    preconditions = execution.frontieror_preconditions(
        source=source,
        require_docker=not args.allow_missing_tooling,
        require_perf_isolation=not args.allow_missing_tooling,
        require_images=not args.allow_missing_tooling,
        available_images=execution.docker_images_present() if not args.allow_missing_tooling else (),
    )

    return _finish(
        args=args,
        integration="frontieror",
        mode="agent",
        command=command,
        preconditions=preconditions,
        campaign_id=None,
        output_root=None,
        notes=[
            "upstream appends its fixed trusted profile (coral, docker isolation, "
            "staged_qte, proxied model access, anti-hack); this command supplies none "
            "of those flags and refuses them as inputs",
            "official scoring requires a performance-isolated host; a run anywhere "
            "else is exploratory and must not be published as an official score",
        ],
    )


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #


def _finish(
    *,
    args: argparse.Namespace,
    integration: str,
    mode: str,
    command: execution.UpstreamCommand,
    preconditions: execution.PreconditionReport,
    campaign_id: str | None,
    output_root: str | None,
    notes: list[str],
) -> int:
    print(f"# upstream command ({command.provenance})", file=sys.stderr)
    print(f"# cwd: {command.cwd}", file=sys.stderr)
    print(f"+ {command.shell()}", file=sys.stderr)

    for item in preconditions.satisfied:
        print(f"  [ok]      {item}", file=sys.stderr)
    for item in preconditions.missing:
        print(f"  [MISSING] {item}", file=sys.stderr)

    exit_code: int | None = None
    if args.execute:
        preconditions.raise_if_unmet(f"{integration} {mode}")
        if command.makes_model_calls and args.acknowledge_cost != COST_ACKNOWLEDGEMENT:
            # Same class as every other fail-closed condition, so a caller can
            # branch on one exit code for "a prerequisite was absent".
            raise PreconditionError(
                f"this path makes model calls; re-run with "
                f"--acknowledge-cost {COST_ACKNOWLEDGEMENT}"
            )
        print("# executing", file=sys.stderr)
        exit_code = subprocess.run(list(command.argv), cwd=command.cwd).returncode
        print(f"# upstream exited {exit_code}", file=sys.stderr)
    else:
        notes = [*notes, "dry run: the command above was built and validated, not executed"]

    receipt = execution.build_receipt(
        integration=integration,
        mode=mode,
        command=command,
        campaign_id=campaign_id,
        preconditions=preconditions,
        exit_code=exit_code,
        # A single rollout supports case diagnosis and nothing more, whatever
        # the machine it ran on.
        evidence_label="exploratory",
        output_root=output_root,
        notes=notes,
    )
    execution.assert_receipt_is_shareable(receipt)
    blob = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    print(blob, end="")
    if args.receipt:
        path = Path(args.receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(blob, encoding="utf-8")
        print(f"# receipt written to {path}", file=sys.stderr)

    if exit_code is not None:
        return exit_code
    if args.require_preconditions and not preconditions.ok:
        print(
            f"run_benchmark_smoke: {len(preconditions.missing)} precondition(s) unmet "
            "and --require-preconditions was set",
            file=sys.stderr,
        )
        return 5
    # A dry run's job is to build and validate the command. Unmet preconditions
    # are reported in the receipt; they are not a failure of the dry run itself.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
