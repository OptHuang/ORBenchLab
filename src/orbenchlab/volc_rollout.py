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
import os
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
_VISIBLE_SUFFIXES = {".json", ".jsonl", ".md", ".toml", ".yaml", ".yml"}


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in _VISIBLE_SUFFIXES
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


def _run_container(task_dir: Path, stage: Path, *, test_image: str, timeout_sec: int) -> dict[str, Any]:
    if not test_image.strip():
        raise VolcRolloutError("--test-image is required for model screening")
    tests = task_dir / "tests"
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
        return {"status": "timeout", "returncode": None, "timeout_sec": timeout_sec}
    ctrf = logs / "ctrf.json"
    summary: dict[str, Any] = {
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "timeout_sec": timeout_sec,
        "stdout_digest": _digest(completed.stdout[-8000:]),
        "stderr_digest": _digest(completed.stderr[-8000:]),
    }
    if ctrf.is_file():
        try:
            document = json.loads(ctrf.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            document = None
        if isinstance(document, Mapping):
            summary["ctrf"] = {
                key: document.get(key)
                for key in ("summary", "tests", "results")
                if key in document
            }
    return summary


def run_rollout(
    task_dir: str | Path,
    *,
    config: VolcConfig,
    models: Sequence[str],
    test_image: str,
    out: str | Path,
    repetitions: int = 1,
    hint_level: int = 0,
    timeout_sec: int = 120,
    max_tokens: int = 2400,
) -> dict[str, Any]:
    """Run one or more Volc model trials and emit a screening report."""

    root = Path(task_dir)
    if not root.is_dir() or root.is_symlink():
        raise VolcRolloutError("task directory must be a real directory")
    if repetitions <= 0 or timeout_sec <= 0 or max_tokens <= 0 or hint_level < 0:
        raise VolcRolloutError("repetitions, timeout and max_tokens must be positive")
    selected = [str(model).strip() for model in models if str(model).strip()] or [config.default_model]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    task = _task_id(root)
    trials: list[dict[str, Any]] = []
    system = (
        "You are a strict terminal benchmark coding agent. Follow the task contract exactly. "
        "Never claim verifier success; return only the requested JSON object."
    )
    for model in selected:
        for trial in range(1, repetitions + 1):
            trial_dir = Path(tempfile.mkdtemp(prefix=f"{task}-{model.replace('/', '_')}-{trial}-", dir=output))
            try:
                api = call_reviewer(
                    config,
                    model=model,
                    system=system,
                    user=_prompt(root, trial=trial, hint_level=hint_level),
                    max_tokens=max_tokens,
                )
                solver = _extract_solver(api["parsed"])
                _copy_visible_inputs(root, trial_dir)
                (trial_dir / "submission" / "solver.py").write_text(solver, encoding="utf-8")
                result = _run_container(root, trial_dir, test_image=test_image, timeout_sec=timeout_sec)
                solver_path = trial_dir / "submission" / "solver.py"
                trial_record = {
                    "model": model,
                    "trial": trial,
                    "hint_level": hint_level,
                    "status": result.get("status"),
                    "solver_digest": _file_digest(solver_path),
                    "response_digest": api.get("response_digest"),
                    "request_digest": api.get("request_digest"),
                    "elapsed_sec": api.get("elapsed_sec"),
                    "usage": api.get("usage", {}),
                    "verifier": result,
                }
                # Keep a reproducible candidate source, but no prompt/response body.
                candidate = output / f"{task}--{model.replace('/', '_')}--trial-{trial}.solver.py"
                candidate.write_text(solver, encoding="utf-8")
            except (VolcReviewError, VolcRolloutError) as exc:
                trial_record = {
                    "model": model,
                    "trial": trial,
                    "hint_level": hint_level,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            finally:
                shutil.rmtree(trial_dir, ignore_errors=True)
            trials.append(trial_record)
    by_model: dict[str, dict[str, Any]] = {}
    for row in trials:
        entry = by_model.setdefault(model := str(row["model"]), {"n": 0, "complete": 0, "metric_n": 0, "passed": 0, "infra_exceptions": []})
        entry["n"] += 1
        if row.get("status") == "pass":
            entry["complete"] += 1
            entry["passed"] += 1
        else:
            entry["infra_exceptions"].append(str(row.get("error_type") or row.get("status")))
        entry["metric_n"] += 1
    arms: dict[str, Any] = {}
    for model, entry in sorted(by_model.items()):
        n = entry["metric_n"]
        arms[model] = {
            "n": entry["n"],
            "complete": entry["complete"],
            "metric_n": n,
            "solve_rate": entry["passed"] / n if n else None,
            "quality_pass_rate": entry["passed"] / n if n else None,
            "mean_feasibility": entry["passed"] / n if n else None,
            "infra_exceptions": sorted(set(entry["infra_exceptions"])),
        }
    report = {
        "schema_version": "orbenchlab.screening-report.v1",
        "task": task,
        "tasks": [
            {
                "task": task,
                "family": task,
                "arms": arms,
                "discrimination_index_observed_gap": None,
                "decision": "collect-more-evidence",
                "evidence_level": "E3",
                "limitations": [
                    "Volc model screening only; no Harbor acceptance.",
                    "One task-local verifier outcome per trial; no causal intervention claim.",
                    "Observed model gap is not computed without a controlled baseline arm.",
                ],
            }
        ],
        "trials": trials,
        "provider": config.public_dict(),
        "test_image": test_image,
        "hint_level": hint_level,
        "task_tree_digest": _digest(_visible_context(root)),
    }
    report["report_digest"] = _digest(report)
    path = output / "screening-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "screening-report.md").write_text(
        "# Volcengine task screening\n\n"
        f"- Task: `{task}`\n- Evidence: `E3`\n- Decision: `collect-more-evidence`\n"
        f"- Models: `{', '.join(selected)}`\n- Repetitions: `{repetitions}`\n\n"
        "Raw prompts and model responses are not persisted. See screening-report.json for digests and verifier aggregates.\n",
        encoding="utf-8",
    )
    return report


__all__ = ["VolcRolloutError", "run_rollout"]
