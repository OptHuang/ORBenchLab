"""ORAgentBench integration — Harbor-native, no adapter.

ORAgentBench already ships Harbor task packages (``harbor_tasks/<task>/task.toml``
plus ``instruction.md``, ``environment/``, ``solution/``, ``tests/``), an
official aggregation metric (``metrics/per_dimension_reward.py``), Harbor job
configs under ``experiments/config/``, expert skills and a difficulty table.
Harbor consumes all of that directly.

So this integration writes **no adapter and materialises no second copy of the
tasks**. It inspects the upstream checkout, records the facts a campaign needs
(dataset digest, reward keys, multi-step tasks, control configs) and reports
where the upstream contract diverges from what hidden-verifier grading assumes.
Divergences are reported as warnings against the upstream repository, not
silently patched here — patching them here would fork the benchmark.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from ..core.errors import IntegrationError
from .base import (
    IntegrationKind,
    InspectionReport,
    find_repo_root,
    find_vendored,
    read_source_revision,
)

NAME = "oragentbench"
KIND = IntegrationKind.HARBOR_NATIVE

UPSTREAM_REPO = "https://github.com/ORAgentBench/ORAgentBench"
#: Commit inspected while building this integration. CI clones exactly this.
PINNED_COMMIT = "c9eb952435a4352f33daa2a35efe0f8c76d31b28"

TASKS_DIR = "harbor_tasks"
METRIC_SCRIPT = "metrics/per_dimension_reward.py"
ORACLE_JOB_CONFIG = "experiments/config/run_oracle_all.yaml"
DIFFICULTY_FILE = "difficulty.json"

#: Reward keys ORAgentBench's own ``tests/test.sh`` writes to
#: ``/logs/verifier/reward.json``. Discovered by inspection, not assumed.
EXPECTED_REWARD_KEYS = ("feasibility", "quality")


def describe() -> dict[str, Any]:
    return {
        "name": NAME,
        "kind": KIND.value,
        "upstream_repo": UPSTREAM_REPO,
        "pinned_commit": PINNED_COMMIT,
        "summary": (
            "ORAgentBench is already a Harbor-native benchmark. ORBenchLab consumes its "
            "official harbor_tasks, metrics, skills and job configs unchanged."
        ),
        "we_do_not_own": [
            "task packages (harbor_tasks/**)",
            "verifier logic (tests/test.sh, tests/evaluate_solution.py)",
            "reference metrics (tests/reference_metrics.json)",
            "reward aggregation (metrics/per_dimension_reward.py)",
            "expert skills (skills/**)",
        ],
        "we_own": [
            "campaign spec -> Harbor JobConfig compilation",
            "stable external run ids and the plan ledger",
            "evidence labelling and report rendering",
        ],
        "controls": {
            "oracle": "harbor built-in agent 'oracle' (zero model cost)",
            "nop": "harbor built-in agent 'nop' (zero model cost)",
            "upstream_oracle_job_config": ORACLE_JOB_CONFIG,
        },
        "requires": {
            "secrets": [],
            "model_api_key": False,
            "solver_license": False,
            "self_hosted_runner": True,
            "self_hosted_reason": "docker build + container execution for rollouts",
            "runner_labels": ["self-hosted", "orbench-exec"],
        },
        "performance_scored": False,
    }


def inspect(source: Path) -> InspectionReport:
    """Statically inspect an ORAgentBench checkout."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise IntegrationError(f"source is not a directory: {source}")

    report = InspectionReport(
        integration=NAME,
        integration_kind=KIND,
        source=str(source),
        source_revision=read_source_revision(source),
    )

    tasks_root = source / TASKS_DIR
    if not tasks_root.is_dir():
        report.add(
            "harbor_native_task_packages",
            "fail",
            f"{TASKS_DIR}/ not found; this checkout is not a Harbor-native ORAgentBench tree",
            expected_path=TASKS_DIR,
        )
        report.facts["task_count"] = 0
        _record_decisions(report)
        return report

    task_dirs = sorted(p for p in tasks_root.iterdir() if (p / "task.toml").is_file())
    report.add(
        "harbor_native_task_packages",
        "pass" if task_dirs else "fail",
        f"found {len(task_dirs)} Harbor task packages under {TASKS_DIR}/",
        task_count=len(task_dirs),
        tasks_dir=TASKS_DIR,
    )
    if not task_dirs:
        _record_decisions(report)
        return report

    parsed = _parse_tasks(task_dirs)
    _check_task_names(report, parsed)
    _check_dataset_digest(report, parsed)
    _check_reward_channel(report, parsed)
    _check_verifier_isolation(report, parsed)
    _check_multi_step(report, parsed)
    _check_solver_license(report, parsed)
    _check_official_assets(report, source)
    _check_no_vendored_copy(report)

    report.facts.update(
        {
            "task_count": len(parsed),
            "task_names": [entry["task_name"] for entry in parsed],
            "base_images": sorted({entry["base_image"] for entry in parsed if entry["base_image"]}),
        }
    )
    _record_decisions(report)
    return report


def _parse_tasks(task_dirs: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        toml_path = task_dir / "task.toml"
        raw = toml_path.read_bytes()
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            entries.append(
                {
                    "dir": task_dir.name,
                    "task_name": None,
                    "parse_error": str(exc),
                    "toml_digest": hashlib.sha256(raw).hexdigest(),
                    "steps": [],
                    "base_image": None,
                    "requires_gurobi": None,
                    "allow_internet": None,
                    "verifier_environment_mode": None,
                    "reward_keys": [],
                    "writes_reward_json": False,
                }
            )
            continue
        steps = data.get("steps") or []
        entries.append(
            {
                "dir": task_dir.name,
                "path": task_dir,
                "task_name": (data.get("task") or {}).get("name"),
                "toml_digest": hashlib.sha256(raw).hexdigest(),
                "steps": [step.get("name") for step in steps],
                "base_image": (data.get("metadata") or {}).get("base_image"),
                "requires_gurobi": (data.get("metadata") or {}).get("requires_gurobi"),
                "allow_internet": (data.get("environment") or {}).get("allow_internet"),
                "verifier_environment_mode": (data.get("verifier") or {}).get("environment_mode"),
                **_scan_verifier_scripts(task_dir, steps),
            }
        )
    return entries


def _scan_verifier_scripts(task_dir: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the reward keys the upstream verifier actually emits.

    Single-step tasks carry ``tests/test.sh``; multi-step tasks carry one per
    step under ``steps/<name>/tests/test.sh``.
    """
    scripts = []
    root_script = task_dir / "tests" / "test.sh"
    if root_script.is_file():
        scripts.append(root_script)
    for step in steps:
        step_name = step.get("name")
        if not step_name:
            continue
        step_script = task_dir / "steps" / step_name / "tests" / "test.sh"
        if step_script.is_file():
            scripts.append(step_script)

    reward_keys: set[str] = set()
    writes_reward_json = False
    for script in scripts:
        text = script.read_text(encoding="utf-8", errors="replace")
        if "/logs/verifier/reward.json" in text:
            writes_reward_json = True
        for key in EXPECTED_REWARD_KEYS:
            if f'"{key}"' in text:
                reward_keys.add(key)
    return {
        "verifier_scripts": [str(p.relative_to(task_dir)) for p in scripts],
        "reward_keys": sorted(reward_keys),
        "writes_reward_json": writes_reward_json,
    }


def _check_task_names(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    unnamed = [entry["dir"] for entry in parsed if not entry["task_name"]]
    namespaced = [
        entry["task_name"]
        for entry in parsed
        if entry["task_name"] and "/" in str(entry["task_name"])
    ]
    if unnamed:
        report.add(
            "task_name_declared",
            "fail",
            f"{len(unnamed)} task(s) have no [task].name; Harbor cannot register them in a dataset",
            unnamed_tasks=unnamed[:10],
        )
    else:
        report.add(
            "task_name_declared",
            "pass",
            f"all {len(parsed)} tasks declare [task].name, "
            f"{len(namespaced)} of them namespaced as <org>/<task>",
            namespace_prefixes=sorted({str(n).split("/")[0] for n in namespaced}),
        )


def _check_dataset_digest(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    """Content-address the task set so run ids are pinned to a dataset state."""
    digest = hashlib.sha256()
    for entry in sorted(parsed, key=lambda e: e["dir"]):
        digest.update(entry["dir"].encode("utf-8"))
        digest.update(entry["toml_digest"].encode("utf-8"))
    dataset_digest = f"sha256:{digest.hexdigest()}"
    report.facts["dataset_digest"] = dataset_digest
    report.add(
        "dataset_digest",
        "pass",
        "dataset digest computed over sorted (task dir, task.toml sha256) pairs",
        dataset_digest=dataset_digest,
        method="sha256(concat(sorted(dir_name || sha256(task.toml))))",
    )


def _check_reward_channel(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    with_reward_json = [e for e in parsed if e["writes_reward_json"]]
    all_keys = sorted({key for entry in parsed for key in entry["reward_keys"]})
    without = [e["dir"] for e in parsed if not e["writes_reward_json"]]
    report.facts["reward_keys"] = all_keys
    report.facts["tasks_writing_reward_json"] = len(with_reward_json)
    if not with_reward_json:
        report.add(
            "reward_channel",
            "fail",
            "no task writes /logs/verifier/reward.json; the multi-dimensional reward "
            "channel this integration reads is absent",
        )
        return
    status = "pass" if not without else "warn"
    report.add(
        "reward_channel",
        status,
        f"{len(with_reward_json)}/{len(parsed)} tasks write /logs/verifier/reward.json "
        f"with keys {all_keys}",
        reward_keys=all_keys,
        tasks_without_reward_json=without[:10],
        note=(
            "Harbor accepts any str->number object in reward.json without schema "
            "validation, so a typo becomes a silently wrong metric. Ingest must "
            "warn on unknown keys."
        ),
    )


def _check_verifier_isolation(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    declared = [e["dir"] for e in parsed if e["verifier_environment_mode"]]
    internet = [e["dir"] for e in parsed if e["allow_internet"] is True]

    if declared:
        report.add(
            "verifier_environment_mode",
            "pass",
            f"{len(declared)}/{len(parsed)} tasks declare [verifier].environment_mode",
            declared_count=len(declared),
        )
    else:
        report.add(
            "verifier_environment_mode",
            "warn",
            f"no task declares [verifier].environment_mode; all {len(parsed)} tasks fall back "
            "to the Harbor default",
            impact=(
                "separate-mode verifiers are what make hidden grading and "
                "'harbor trial regrade' possible; this is an upstream property, "
                "not something ORBenchLab may patch locally"
            ),
        )

    if internet:
        report.add(
            "agent_network_isolation",
            "warn",
            f"{len(internet)}/{len(parsed)} tasks set [environment].allow_internet = true",
            impact=(
                "an agent with network access can in principle retrieve solutions; "
                "campaigns that need anti-speculation isolation must say so explicitly "
                "and cannot claim isolation from this upstream state"
            ),
            examples=internet[:5],
        )
    else:
        report.add(
            "agent_network_isolation",
            "pass",
            "no task enables unrestricted internet access for the agent phase",
        )


def _check_multi_step(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    multi = [e for e in parsed if e["steps"]]
    report.facts["multi_step_task_count"] = len(multi)
    report.facts["multi_step_tasks"] = [e["dir"] for e in multi]
    if not multi:
        report.add("multi_step_tasks", "pass", "no multi-step tasks present")
        return
    report.add(
        "multi_step_tasks",
        "warn",
        f"{len(multi)}/{len(parsed)} tasks are multi-step ([[steps]])",
        multi_step_tasks=[e["dir"] for e in multi],
        impact=(
            "Harbor 'regrade' does not support multi-step tasks, so these cannot be "
            "re-scored through the official path; plan for an offline re-score or "
            "exclude them from re-scored campaigns"
        ),
    )


def _check_solver_license(report: InspectionReport, parsed: list[dict[str, Any]]) -> None:
    needs = [e["dir"] for e in parsed if e["requires_gurobi"] is True]
    undeclared = [e["dir"] for e in parsed if e["requires_gurobi"] is None]
    grant = "none" if not needs else "verifier"
    report.facts["solver_license_grant"] = grant
    report.facts["tasks_requiring_gurobi"] = len(needs)
    report.add(
        "solver_license_grant",
        "pass" if not undeclared else "warn",
        f"inferred solver_license_grant='{grant}' "
        f"({len(needs)} task(s) declare requires_gurobi = true)",
        tasks_requiring_gurobi=needs[:10],
        tasks_without_declaration=undeclared[:10],
        note=(
            "grant is per-phase least privilege: the compiler injects a license only "
            "into the phase a task declares, never into both for convenience"
        ),
    )


def _check_official_assets(report: InspectionReport, source: Path) -> None:
    metric = source / METRIC_SCRIPT
    report.add(
        "official_metric_script",
        "pass" if metric.is_file() else "fail",
        f"{METRIC_SCRIPT} {'found' if metric.is_file() else 'missing'}",
        path=METRIC_SCRIPT,
        note="consumed as a Harbor uv-script metric; ORBenchLab does not reimplement it",
    )
    if metric.is_file():
        report.facts["metric_script"] = METRIC_SCRIPT
        report.facts["metric_script_digest"] = (
            f"sha256:{hashlib.sha256(metric.read_bytes()).hexdigest()}"
        )

    oracle_cfg = source / ORACLE_JOB_CONFIG
    if oracle_cfg.is_file():
        text = oracle_cfg.read_text(encoding="utf-8", errors="replace")
        uses_oracle = "name: oracle" in text
        report.add(
            "oracle_control_config",
            "pass" if uses_oracle else "warn",
            f"{ORACLE_JOB_CONFIG} present"
            + ("; declares the built-in oracle agent" if uses_oracle else "; oracle agent not found in it"),
            path=ORACLE_JOB_CONFIG,
            model_cost="zero — 'oracle' is a Harbor built-in deterministic agent",
        )
        report.facts["oracle_job_config"] = ORACLE_JOB_CONFIG
    else:
        report.add(
            "oracle_control_config",
            "warn",
            f"{ORACLE_JOB_CONFIG} not found; the zero-cost oracle control must be compiled from scratch",
            path=ORACLE_JOB_CONFIG,
        )

    # The nop control is a Harbor built-in, so its absence upstream is expected
    # and must not be reported as an upstream defect.
    report.add(
        "nop_control_available",
        "skip",
        "the 'nop' control agent is a Harbor built-in; it is not provided by, and "
        "cannot be verified from, this upstream repository",
        verify_with="harbor's AgentName enum at the pinned Harbor version",
    )

    difficulty = source / DIFFICULTY_FILE
    if difficulty.is_file():
        report.add(
            "difficulty_metadata",
            "pass",
            f"{DIFFICULTY_FILE} present; per-task difficulty available for stratified sampling",
            path=DIFFICULTY_FILE,
        )
        report.facts["difficulty_file"] = DIFFICULTY_FILE
    else:
        report.add(
            "difficulty_metadata",
            "warn",
            f"{DIFFICULTY_FILE} not found; campaigns cannot stratify by difficulty",
        )


def _check_no_vendored_copy(report: InspectionReport) -> None:
    """Assert ORBenchLab itself ships no copy of the upstream task set.

    A second copy of the tasks is the failure mode this integration exists to
    avoid: it would drift from upstream and quietly become a different
    benchmark.
    """
    repo_root = find_repo_root(Path(__file__))
    if repo_root is None:
        report.add(
            "no_vendored_task_copy",
            "skip",
            "ORBenchLab repository root not locatable from the installed package; "
            "run the equivalent check from a source checkout",
        )
        return
    vendored = find_vendored(repo_root, (TASKS_DIR,))
    report.add(
        "no_vendored_task_copy",
        "pass" if not vendored else "fail",
        (
            f"ORBenchLab contains no vendored {TASKS_DIR}/ copy"
            if not vendored
            else f"ORBenchLab vendors upstream tasks at {vendored}"
        ),
        repo_root=str(repo_root),
        vendored_paths=vendored,
    )


def _record_decisions(report: InspectionReport) -> None:
    report.decisions = {
        "adapter_required": False,
        "adapter_rationale": (
            "upstream already ships Harbor task packages; writing an adapter would "
            "create a second, drifting copy of the benchmark"
        ),
        "task_materialization": "none — Harbor consumes upstream harbor_tasks/ in place",
        "verifier_ownership": "upstream (tests/test.sh + tests/evaluate_solution.py)",
        "metric_ownership": "upstream (metrics/per_dimension_reward.py as a Harbor uv-script metric)",
        "controls": ["oracle (harbor built-in)", "nop (harbor built-in)"],
        "control_model_cost_usd": 0.0,
    }
