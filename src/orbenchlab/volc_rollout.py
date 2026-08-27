"""Run a paper-derived task candidate with a Volcengine model.

This is a deliberately small screening runner for the strict task fixtures in
``examples/tasks``.  The model sees only the public task contract and input
files, returns one ``solver.py`` file, and the candidate is evaluated in a
no-network Docker container.  We persist hashes and verifier aggregates, not
the prompt or the model's raw response.

The runner is not a replacement for Harbor.  Its result is outcome-grounded
screening evidence (E3 for a completed verifier run); Harbor remains the
acceptance path and repeated/interventional trajectory work is a later stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.errors import ORBenchError
from .volc_review import VolcConfig, VolcReviewError, call_reviewer


class VolcRolloutError(ORBenchError):
    """A model screening run could not be completed safely."""

    exit_code = 8


MAX_SOLVER_BYTES = 120_000
MIN_DISCRIMINATION_REPETITIONS = 5
_VISIBLE_SUFFIXES = {".json", ".jsonl", ".md", ".toml", ".yaml", ".yml"}


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task_tree_digest(task_dir: Path) -> str:
    """Bind a receipt to the complete regular-file task tree."""

    rows = []
    for path in sorted(task_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows.append(
                {
                    "path": path.relative_to(task_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "digest": _file_digest(path),
                }
            )
    return _digest(rows)


def _task_id(task_dir: Path) -> str:
    task = task_dir / "task.toml"
    if not task.is_file():
        raise VolcRolloutError("task directory must contain task.toml")
    try:
        document = tomllib.loads(task.read_text(encoding="utf-8"))
        name = document.get("task", {}).get("name", "")
        if isinstance(name, str) and name.strip():
            return name.rsplit("/", 1)[-1].replace("-", "_")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        pass
    return task_dir.name.replace("-", "_")


def _visible_context(task_dir: Path) -> dict[str, Any]:
    """Return the public context an agent would receive, excluding tests/oracle."""

    paths: list[Path] = []
    for name in ("task.toml", "instruction.md", "README.md"):
        path = task_dir / name
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    data = task_dir / "data"
    if data.is_dir() and not data.is_symlink():
        paths.extend(
            path
            for path in sorted(data.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and not any(part.startswith(".") for part in path.relative_to(data).parts)
            and path.suffix.lower() in _VISIBLE_SUFFIXES
        )
    files: list[dict[str, Any]] = []
    for path in paths:
        rel = path.relative_to(task_dir).as_posix()
        raw = path.read_bytes()
        if len(raw) > 64_000:
            raise VolcRolloutError(f"visible task file is too large: {rel}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise VolcRolloutError(f"visible task file is not UTF-8: {rel}") from None
        files.append({"path": rel, "bytes": len(raw), "digest": _file_digest(path), "content": text})
    return {"task": _task_id(task_dir), "files": files}


def _extract_solver(parsed: Mapping[str, Any]) -> str:
    value = parsed.get("solver_py")
    if value is None and isinstance(parsed.get("files"), Mapping):
        files = parsed["files"]
        value = files.get("solver.py") or files.get("submission/solver.py")
    if value is None:
        value = parsed.get("solver") or parsed.get("code")
    if not isinstance(value, str) or not value.strip():
        raise VolcRolloutError("Volc model response did not contain solver_py")
    if len(value.encode("utf-8")) > MAX_SOLVER_BYTES:
        raise VolcRolloutError("generated solver.py exceeds the size limit")
    return value


def _hint(task_dir: Path, level: int) -> str:
    if level <= 0:
        return ""
    task = task_dir.name
    if task == "alphaevolve-scheduling":
        base = (
            "\nHint level 1 (contract reminder): the verifier expects a top-level "
            "schema_version of alphaevolve-scheduling.solution.v1 and an "
            "object with schedules[level_id][str(seed)], and each schedule has "
            "an operations list containing operation_id, job_id, operation_index, "
            "machine, start, end plus an integer makespan. Use the exact seed "
            "duration rule in the instruction/data; do not invent a list schema."
        )
        if level >= 2:
            base += " Use a stable algorithm (for example exhaustive job-order permutations on these small fixtures) and verify every bound before writing the file; do not use Python's randomized hash()."
        return base
    if task == "vrp-recovery":
        return (
            "\nHint level 1 (contract reminder): solver.py must write all three files "
            "under /root/submission (initial_routes.json, replans.jsonl, "
            "event_audit.json); preserve frozen prefixes, change only affected "
            "vehicles, avoid closed edges, and recompute each integer objective."
        )
    return "\nHint level 1 (contract reminder): follow every output filename and field in instruction.md exactly."


def _prompt(task_dir: Path, *, trial: int, hint_level: int = 0) -> str:
    context = _visible_context(task_dir)
    return (
        "Act as a benchmark coding agent. Solve the supplied strict task contract. "
        "Return JSON only: {\"solver_py\": \"<complete Python source>\"}. "
        "The source must be self-contained, read only files under /root, write the "
        "required submission artifacts, accept the exact CLI arguments required by "
        "the instruction when applicable, and use no network or package downloads. "
        "Do not include Markdown fences or extra keys. This is screening trial "
        f"{trial}; do not assume hidden tests or reveal test files.\n\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
        + _hint(task_dir, hint_level)
    )


def _copy_visible_inputs(task_dir: Path, stage: Path) -> None:
    for name in ("instance.json", "events.jsonl"):
        source = task_dir / "data" / name
        if source.is_file():
            shutil.copy2(source, stage / name)
    source_data = task_dir / "data"
    if source_data.is_dir():
        shutil.copytree(source_data, stage / "data", dirs_exist_ok=True)
    (stage / "submission").mkdir(parents=True, exist_ok=True)
    tests = task_dir / "tests"
    if not tests.is_dir():
        raise VolcRolloutError("task directory is missing tests/")


def _run_container(
    task_dir: Path,
    stage: Path,
    *,
    test_image: str,
    timeout_sec: int,
    expected_tests: int | None = None,
) -> dict[str, Any]:
    if not test_image.strip():
        raise VolcRolloutError("--test-image is required for model screening")
    tests = task_dir.resolve() / "tests"
    stage = stage.resolve()
    logs = stage / "logs" / "verifier"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "ORBENCH_TASK_ROOT=/root",
        "-v",
        f"{stage}:/root",
        "-v",
        f"{stage / 'logs'}:/logs",
        "-v",
        f"{tests}:/tests:ro",
        test_image,
        "sh",
        "/tests/test.sh",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        raise VolcRolloutError("docker executable is not available") from None
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": None,
            "timeout_sec": timeout_sec,
            "receipt_valid": False,
            "error_type": "VerifierTimeout",
        }
    ctrf = logs / "ctrf.json"
    reward = logs / "reward.txt"
    summary: dict[str, Any] = {
        "status": "infra_error",
        "returncode": completed.returncode,
        "timeout_sec": timeout_sec,
        "stdout_digest": _digest(completed.stdout[-8000:]),
        "stderr_digest": _digest(completed.stderr[-8000:]),
        "receipt_valid": False,
    }
    if completed.returncode != 0:
        summary["error_type"] = "VerifierEntrypointError"
        return summary
    if not ctrf.is_file() or not reward.is_file():
        summary["error_type"] = "VerifierReceiptMissing"
        return summary
    try:
        document = json.loads(ctrf.read_text(encoding="utf-8"))
        reward_text = reward.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        summary["error_type"] = "VerifierReceiptMalformed"
        return summary
    results = document.get("results") if isinstance(document, Mapping) else None
    ctrf_summary = results.get("summary") if isinstance(results, Mapping) else None
    required = ("tests", "passed", "failed", "skipped", "pending", "other")
    if not isinstance(ctrf_summary, Mapping) or any(
        not isinstance(ctrf_summary.get(key), int) or int(ctrf_summary[key]) < 0
        for key in required
    ):
        summary["error_type"] = "VerifierReceiptMalformed"
        return summary
    counts = {key: int(ctrf_summary[key]) for key in required}
    if counts["tests"] <= 0 or sum(counts[key] for key in required[1:]) != counts["tests"]:
        summary["error_type"] = "VerifierReceiptInconsistent"
        return summary
    if expected_tests is not None and counts["tests"] != expected_tests:
        summary["error_type"] = "VerifierTestCountMismatch"
        return summary
    if reward_text not in {"0", "0.0", "1", "1.0"}:
        summary["error_type"] = "VerifierRewardMalformed"
        return summary
    reward_pass = float(reward_text) == 1.0
    ctrf_pass = counts["passed"] == counts["tests"]
    ctrf_fail = counts["failed"] > 0
    if (reward_pass and not ctrf_pass) or (not reward_pass and not ctrf_fail):
        summary["error_type"] = "VerifierReceiptInconsistent"
        return summary
    summary.update(
        {
            "status": "pass" if reward_pass else "fail",
            "receipt_valid": True,
            "reward": 1.0 if reward_pass else 0.0,
            "ctrf": {"summary": counts, "digest": _file_digest(ctrf)},
            "reward_digest": _file_digest(reward),
        }
    )
    return summary


_OUTCOME_STATUSES = frozenset({"pass", "fail", "invalid_output"})


def _summarize_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate trials without mislabelling verifier failures as infrastructure.

    A provider/Docker setup error or verifier timeout has no metric denominator.
    A malformed model artifact or completed verifier failure is a real task
    outcome and therefore counts as a completed failed attempt.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for row in trials:
        model = str(row["model"])
        hint_level = int(row.get("hint_level", 0))
        arm = f"{model}@hint-{hint_level}"
        entry = grouped.setdefault(
            arm,
            {
                "model_id": model,
                "hint_level": hint_level,
                "n": 0,
                "complete": 0,
                "metric_n": 0,
                "passed": 0,
                "infra_exceptions": set(),
                "failure_modes": set(),
            },
        )
        entry["n"] += 1
        status = str(row.get("status", "infra_error"))
        if status in _OUTCOME_STATUSES:
            entry["complete"] += 1
            entry["metric_n"] += 1
            if status == "pass":
                entry["passed"] += 1
            else:
                entry["failure_modes"].add(str(row.get("failure_mode") or status))
        else:
            entry["infra_exceptions"].add(str(row.get("error_type") or status))
            if status == "timeout":
                entry["failure_modes"].add("verifier_timeout_inconclusive")

    arms: dict[str, Any] = {}
    for arm, entry in sorted(grouped.items()):
        metric_n = entry["metric_n"]
        rate = entry["passed"] / metric_n if metric_n else None
        arms[arm] = {
            "model_id": entry["model_id"],
            "hint_level": entry["hint_level"],
            "n": entry["n"],
            "complete": entry["complete"],
            "metric_n": metric_n,
            "solve_n": metric_n,
            "quality_n": 0,
            "feasibility_n": 0,
            "solve_rate": rate,
            "quality_pass_rate": None,
            "mean_feasibility": None,
            "infra_exceptions": sorted(entry["infra_exceptions"]),
            "failure_modes": sorted(entry["failure_modes"]),
        }
    return arms


def _wilson_interval(successes: int, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or successes < 0 or successes > n:
        raise VolcRolloutError("Wilson interval requires 0 <= successes <= n")
    rate = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (rate + z2 / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _discrimination_summary(
    arms: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    *,
    repetitions: int,
) -> dict[str, Any]:
    """Conservative model separation on the rectangular unassisted cell."""

    cells: list[dict[str, Any]] = []
    for model in models:
        arm = arms.get(f"{model}@hint-0")
        if not isinstance(arm, Mapping):
            continue
        n = int(arm.get("solve_n", 0))
        rate = arm.get("solve_rate")
        if not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
            continue
        successes = int(round(float(rate) * n))
        if n <= 0 or abs(successes / n - float(rate)) > 1e-9:
            continue
        lower, upper = _wilson_interval(successes, n)
        cells.append(
            {
                "model": str(model),
                "n": n,
                "successes": successes,
                "solve_rate": float(rate),
                "wilson_95": [round(lower, 6), round(upper, 6)],
                "infra_exceptions": list(arm.get("infra_exceptions") or []),
            }
        )
    rectangular = len(cells) == len(models) and len(cells) >= 2 and all(
        cell["n"] == repetitions and not cell["infra_exceptions"] for cell in cells
    )
    result: dict[str, Any] = {
        "comparison_schema_version": "orbenchlab.model-discrimination.v1",
        "cell": "hint-0",
        "models": cells,
        "minimum_repetitions": MIN_DISCRIMINATION_REPETITIONS,
        "rectangular": rectangular,
        "observed_gap": None,
        "gap_95_lower_bound": None,
        "promising": False,
        "reason": "at least two complete model arms are required",
    }
    if not rectangular:
        if len(cells) >= 2:
            result["reason"] = "model arms are not a complete equal-budget rectangle"
        return result
    ordered = sorted(cells, key=lambda cell: (cell["solve_rate"], cell["model"]))
    worst, best = ordered[0], ordered[-1]
    observed_gap = best["solve_rate"] - worst["solve_rate"]
    lower_bound = best["wilson_95"][0] - worst["wilson_95"][1]
    result.update(
        {
            "best_model": best["model"],
            "baseline_model": worst["model"],
            "observed_gap": round(observed_gap, 6),
            "gap_95_lower_bound": round(lower_bound, 6),
        }
    )
    if repetitions < MIN_DISCRIMINATION_REPETITIONS:
        result["reason"] = "fewer than five repetitions per unassisted model arm"
    elif lower_bound <= 0:
        result["reason"] = "95% Wilson intervals do not establish positive separation"
    else:
        result["promising"] = True
        result["reason"] = "positive conservative model separation on the unassisted cell"
    return result


def _run_trial(
    root: Path,
    *,
    config: VolcConfig,
    model: str,
    trial: int,
    hint_level: int,
    test_image: str,
    output: Path,
    timeout_sec: int,
    max_tokens: int,
    expected_tests: int,
) -> dict[str, Any]:
    """Run one phase-labelled trial and preserve only sanitized evidence."""

    task = _task_id(root)
    common = {"model": model, "trial": trial, "hint_level": hint_level}
    system = (
        "You are a strict terminal benchmark coding agent. Follow the task contract exactly. "
        "Never claim verifier success; return only the requested JSON object."
    )
    try:
        api = call_reviewer(
            config,
            model=model,
            system=system,
            user=_prompt(root, trial=trial, hint_level=hint_level),
            max_tokens=max_tokens,
        )
    except VolcReviewError as exc:
        return {**common, "status": "infra_error", "phase": "provider", "error_type": type(exc).__name__}

    api_evidence = {
        "response_digest": api.get("response_digest"),
        "request_digest": api.get("request_digest"),
        "elapsed_sec": api.get("elapsed_sec"),
        "usage": api.get("usage", {}),
    }
    try:
        solver = _extract_solver(api["parsed"])
    except VolcRolloutError:
        return {
            **common,
            **api_evidence,
            "status": "invalid_output",
            "phase": "candidate",
            "failure_mode": "missing_or_invalid_solver_py",
        }

    trial_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{task}-{model.replace('/', '_')}-h{hint_level}-{trial}-",
            dir=output,
        )
    )
    solver_path = trial_dir / "submission" / "solver.py"
    try:
        _copy_visible_inputs(root, trial_dir)
        solver_path.write_text(solver, encoding="utf-8")
        solver_digest = _file_digest(solver_path)
        candidate = output / (
            f"{task}--{model.replace('/', '_')}--hint-{hint_level}--trial-{trial}.solver.py"
        )
        candidate.write_text(solver, encoding="utf-8")
        try:
            result = _run_container(
                root,
                trial_dir,
                test_image=test_image,
                timeout_sec=timeout_sec,
                expected_tests=expected_tests,
            )
        except VolcRolloutError as exc:
            return {
                **common,
                **api_evidence,
                "status": "infra_error",
                "phase": "verifier-infrastructure",
                "error_type": type(exc).__name__,
                "solver_digest": solver_digest,
            }
        status = str(result.get("status") or "fail")
        record = {
            **common,
            **api_evidence,
            "status": status,
            "phase": "verifier",
            "solver_digest": solver_digest,
            "verifier": result,
        }
        if status in {"infra_error", "timeout"}:
            record["error_type"] = str(result.get("error_type") or "VerifierEvidenceError")
        if status != "pass" and status != "infra_error":
            record["failure_mode"] = "verifier_timeout" if status == "timeout" else "verifier_failed"
        return record
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def _run_control_trial(
    root: Path,
    *,
    control: str,
    test_image: str,
    output: Path,
    timeout_sec: int,
    expected_tests: int | None = None,
) -> dict[str, Any]:
    """Run deterministic oracle/NOP controls through the same verifier image."""

    if control not in {"oracle", "nop"}:
        raise VolcRolloutError(f"unsupported task-screen control: {control}")
    task = _task_id(root)
    stage = Path(tempfile.mkdtemp(prefix=f"{task}-control-{control}-", dir=output))
    try:
        _copy_visible_inputs(root, stage)
        solver = stage / "submission" / "solver.py"
        if control == "oracle":
            source = root / "solution" / "solver.py"
            if not source.is_file() or source.is_symlink():
                return {
                    "kind": "control",
                    "control": control,
                    "status": "infra_error",
                    "phase": "control-setup",
                    "error_type": "OracleSolverMissing",
                }
            shutil.copy2(source, solver)
        else:
            solver.write_text("raise SystemExit(0)\n", encoding="utf-8")
        try:
            result = _run_container(
                root,
                stage,
                test_image=test_image,
                timeout_sec=timeout_sec,
                expected_tests=expected_tests,
            )
        except VolcRolloutError as exc:
            return {
                "kind": "control",
                "control": control,
                "status": "infra_error",
                "phase": "verifier-infrastructure",
                "error_type": type(exc).__name__,
            }
        record = {
            "kind": "control",
            "control": control,
            "status": str(result.get("status") or "fail"),
            "phase": "verifier",
            "solver_digest": _file_digest(solver),
            "verifier": result,
        }
        if record["status"] in {"infra_error", "timeout"}:
            record["error_type"] = str(result.get("error_type") or "VerifierEvidenceError")
        return record
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _control_gates(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {
        name: {"gate": "missing", "observed_status": None, "verifier": None}
        for name in ("oracle", "nop")
    }
    for row in trials:
        control = str(row.get("control", ""))
        status = str(row.get("status", "infra_error"))
        if control not in gates:
            continue
        verifier = row.get("verifier") if isinstance(row.get("verifier"), Mapping) else {}
        receipt_valid = bool(verifier.get("receipt_valid"))
        if gates[control]["observed_status"] is not None:
            gate = "duplicate"
        elif status in {"infra_error", "timeout"} or not receipt_valid:
            gate = "infrastructure-error"
        elif control == "oracle":
            gate = "pass" if status == "pass" else "fail"
        elif control == "nop":
            gate = "pass" if status == "fail" else "fail"
        else:
            gate = "unsupported"
        gates[control] = {
            "gate": gate,
            "observed_status": status,
            "verifier": row.get("verifier"),
        }
    return gates


def run_rollout(
    task_dir: str | Path,
    *,
    config: VolcConfig,
    models: Sequence[str],
    test_image: str,
    out: str | Path,
    repetitions: int = 1,
    hint_level: int = 0,
    hint_levels: Sequence[int] | None = None,
    controls: Sequence[str] = ("oracle", "nop"),
    timeout_sec: int = 120,
    max_tokens: int = 2400,
) -> dict[str, Any]:
    """Run one or more Volc model trials and emit a screening report."""

    root = Path(task_dir)
    if not root.is_dir() or root.is_symlink():
        raise VolcRolloutError("task directory must be a real directory")
    selected_hints = [int(value) for value in (hint_levels if hint_levels is not None else [hint_level])]
    if repetitions <= 0 or timeout_sec <= 0 or max_tokens <= 0 or not selected_hints or min(selected_hints) < 0:
        raise VolcRolloutError("repetitions, timeout and max_tokens must be positive")
    selected_controls = [str(value).strip() for value in controls]
    if len(selected_controls) != 2 or set(selected_controls) != {"oracle", "nop"}:
        raise VolcRolloutError("task screening requires exactly one oracle and one nop control")
    selected = [str(model).strip() for model in models if str(model).strip()] or [config.default_model]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    task = _task_id(root)
    oracle_trial = _run_control_trial(
        root,
        control="oracle",
        test_image=test_image,
        output=output,
        timeout_sec=timeout_sec,
    )
    oracle_verifier = oracle_trial.get("verifier")
    oracle_ctrf = oracle_verifier.get("ctrf") if isinstance(oracle_verifier, Mapping) else None
    oracle_summary = oracle_ctrf.get("summary") if isinstance(oracle_ctrf, Mapping) else None
    expected_tests = int(oracle_summary.get("tests", 0)) if isinstance(oracle_summary, Mapping) else 0
    nop_trial = _run_control_trial(
        root,
        control="nop",
        test_image=test_image,
        output=output,
        timeout_sec=timeout_sec,
        expected_tests=expected_tests or None,
    )
    control_by_name = {"oracle": oracle_trial, "nop": nop_trial}
    control_trials = [control_by_name[name] for name in selected_controls]
    gates = _control_gates(control_trials)
    control_failure = any(value.get("gate") != "pass" for value in gates.values())
    trials: list[dict[str, Any]] = []
    if not control_failure:
        for model in selected:
            for current_hint in selected_hints:
                for trial in range(1, repetitions + 1):
                    trials.append(
                        _run_trial(
                            root,
                            config=config,
                            model=model,
                            trial=trial,
                            hint_level=current_hint,
                            test_image=test_image,
                            output=output,
                            timeout_sec=timeout_sec,
                            max_tokens=max_tokens,
                            expected_tests=expected_tests,
                        )
                    )
    arms = _summarize_trials(trials)
    discrimination = _discrimination_summary(arms, selected, repetitions=repetitions)
    evidence_level = "E3" if not control_failure else "E1"
    decision = (
        "revise-or-drop"
        if control_failure
        else "review-promising" if discrimination["promising"] else "collect-more-evidence"
    )
    limitations = [
        "Volc model screening only; no Harbor acceptance.",
        "Oracle/NOP controls are local verifier controls, not Harbor packaging acceptance.",
        "One task-local verifier outcome per trial; no causal intervention claim.",
        "Difficulty and hint monotonicity are not inferred from the model comparison.",
    ]
    if not discrimination["promising"]:
        limitations.append(str(discrimination["reason"]))
    report = {
        "schema_version": "orbenchlab.screening-report.v1",
        "task": task,
        "tasks": [
            {
                "task": task,
                "family": task,
                "arms": arms,
                "control_gates": gates,
                "discrimination_index_observed_gap": discrimination["observed_gap"],
                "discrimination": discrimination,
                "decision": decision,
                "evidence_level": evidence_level,
                "limitations": limitations,
            }
        ],
        "trials": [*control_trials, *trials],
        "provider": config.public_dict(),
        "test_image": test_image,
        "hint_levels": selected_hints,
        "run_contract": {
            "models": selected,
            "repetitions": repetitions,
            "hint_levels": selected_hints,
            "max_tokens": max_tokens,
            "timeout_sec": timeout_sec,
            "test_image": test_image,
        },
        "task_tree_digest": _task_tree_digest(root),
    }
    report["report_digest"] = _digest(report)
    path = output / "screening-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "screening-report.md").write_text(
        "# Volcengine task screening\n\n"
        f"- Task: `{task}`\n- Evidence: `{evidence_level}`\n- Decision: `{decision}`\n"
        f"- Models: `{', '.join(selected)}`\n- Hint levels: `{selected_hints}`\n"
        f"- Repetitions per cell: `{repetitions}`\n\n"
        "Raw prompts and model responses are not persisted. See screening-report.json for digests and verifier aggregates.\n",
        encoding="utf-8",
    )
    return report


def run_suite(
    tasks: Sequence[tuple[str | Path, str]],
    *,
    config: VolcConfig,
    models: Sequence[str],
    out: str | Path,
    repetitions: int = 1,
    hint_levels: Sequence[int] = (0,),
    controls: Sequence[str] = ("oracle", "nop"),
    timeout_sec: int = 120,
    max_tokens: int = 2400,
) -> dict[str, Any]:
    """Run the same bounded model matrix over several strict tasks."""

    if not tasks:
        raise VolcRolloutError("task screening suite requires at least one task")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    members: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for task_dir, test_image in tasks:
        root = Path(task_dir)
        task = _task_id(root)
        if task in seen:
            raise VolcRolloutError(f"duplicate task identity in screening suite: {task}")
        seen.add(task)
        member_out = output / task
        report = run_rollout(
            root,
            config=config,
            models=models,
            test_image=test_image,
            out=member_out,
            repetitions=repetitions,
            hint_levels=hint_levels,
            controls=controls,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
        )
        task_rows.extend(
            {**dict(row), "task_tree_digest": report["task_tree_digest"]}
            for row in report["tasks"]
        )
        trials.extend({"task": task, **dict(row)} for row in report["trials"])
        members.append(
            {
                "task": task,
                "test_image": test_image,
                "report": str(member_out / "screening-report.json"),
                "report_digest": report["report_digest"],
            }
        )
    suite = {
        "schema_version": "orbenchlab.screening-report.v1",
        "suite_schema_version": "orbenchlab.volc-screening-suite.v1",
        "tasks": task_rows,
        "trials": trials,
        "members": members,
        "provider": config.public_dict(),
        "models": [str(model) for model in models] or [config.default_model],
        "hint_levels": [int(value) for value in hint_levels],
        "repetitions_per_cell": repetitions,
        "limitations": [
            "Suite cells are independent restart-with-hint screenings, not same-checkpoint E4 interventions.",
            "Task-local no-network verifier outcomes are not Harbor acceptance.",
        ],
    }
    suite["report_digest"] = _digest(suite)
    (output / "screening-report.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Volcengine task screening suite",
        "",
        f"- Tasks: `{', '.join(sorted(seen))}`",
        f"- Models: `{', '.join(suite['models'])}`",
        f"- Hint levels: `{suite['hint_levels']}`",
        f"- Repetitions per cell: `{repetitions}`",
        "- Evidence ceiling: `E3`",
        "",
        "Each member directory contains its candidate solvers and sanitized trial report.",
    ]
    (output / "screening-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return suite


__all__ = [
    "MIN_DISCRIMINATION_REPETITIONS",
    "VolcRolloutError",
    "run_rollout",
    "run_suite",
]
