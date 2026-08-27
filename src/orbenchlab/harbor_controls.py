"""Validate and summarize real Harbor Oracle/NOP control jobs.

The local Volc screening controls exercise a verifier image, but they do not
prove that Harbor can discover the oracle, build both environments, transfer
artifacts, and collect a reward.  This module turns completed Harbor job trees
into a small fail-closed receipt that the final task-card pipeline can ingest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .core.errors import ORBenchError
from .volc_rollout import _digest, _task_id, _task_tree_digest


class HarborControlError(ORBenchError):
    """A Harbor control job is incomplete or internally inconsistent."""

    exit_code = 8


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise HarborControlError(f"invalid Harbor JSON receipt: {path}") from None
    if not isinstance(value, Mapping):
        raise HarborControlError(f"Harbor JSON receipt must be an object: {path}")
    return value


def _one_trial(job_dir: Path) -> Path:
    candidates = sorted(
        path.parent.parent
        for path in job_dir.glob("*/verifier/ctrf.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise HarborControlError(
            f"Harbor control job must contain exactly one completed CTRF trial: {job_dir}"
        )
    return candidates[0]


def _validate_job(job_dir: str | Path, *, control: str) -> dict[str, Any]:
    root = Path(job_dir)
    if control not in {"oracle", "nop"}:
        raise HarborControlError(f"unsupported Harbor control: {control}")
    job_result_path = root / "result.json"
    job_result = _load_json(job_result_path)
    stats = job_result.get("stats")
    if (
        job_result.get("n_total_trials") != 1
        or not isinstance(stats, Mapping)
        or stats.get("n_completed_trials") != 1
        or stats.get("n_errored_trials") != 0
    ):
        raise HarborControlError(f"Harbor {control} job is not one clean completed trial")

    trial = _one_trial(root)
    trial_result_path = trial / "result.json"
    trial_result = _load_json(trial_result_path)
    reward_path = trial / "verifier/reward.txt"
    ctrf_path = trial / "verifier/ctrf.json"
    manifest_path = trial / "artifacts/manifest.json"
    if not reward_path.is_file() or not ctrf_path.is_file() or not manifest_path.is_file():
        raise HarborControlError(f"Harbor {control} trial is missing reward, CTRF, or manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise HarborControlError(f"Harbor {control} artifact manifest is malformed") from None
    submission_entries = [
        entry
        for entry in manifest
        if isinstance(entry, Mapping)
        and str(entry.get("source", "")).startswith("/root/submission/")
    ] if isinstance(manifest, list) else []
    expected_artifact_status = "ok" if control == "oracle" else "failed"
    if not submission_entries or any(
        entry.get("status") != expected_artifact_status for entry in submission_entries
    ):
        raise HarborControlError(
            f"Harbor {control} artifact manifest does not match the expected control behavior"
        )
    try:
        reward = float(reward_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError, ValueError):
        raise HarborControlError(f"Harbor {control} reward is malformed") from None
    ctrf = _load_json(ctrf_path)
    results = ctrf.get("results")
    summary = results.get("summary") if isinstance(results, Mapping) else None
    keys = ("tests", "passed", "failed", "skipped", "pending", "other")
    if not isinstance(summary, Mapping) or any(
        not isinstance(summary.get(key), int) or int(summary[key]) < 0 for key in keys
    ):
        raise HarborControlError(f"Harbor {control} CTRF summary is malformed")
    counts = {key: int(summary[key]) for key in keys}
    if counts["tests"] <= 0 or sum(counts[key] for key in keys[1:]) != counts["tests"]:
        raise HarborControlError(f"Harbor {control} CTRF counts are inconsistent")
    expected = 1.0 if control == "oracle" else 0.0
    valid_semantics = (
        reward == expected
        and (
            (control == "oracle" and counts["passed"] == counts["tests"])
            or (control == "nop" and counts["failed"] > 0)
        )
    )
    rewards = trial_result.get("verifier_result")
    trial_rewards = rewards.get("rewards") if isinstance(rewards, Mapping) else None
    if (
        not valid_semantics
        or not isinstance(trial_rewards, Mapping)
        or trial_rewards.get("reward") != expected
        or trial_result.get("exception_info") is not None
    ):
        raise HarborControlError(f"Harbor {control} reward and CTRF do not prove the expected gate")
    return {
        "gate": "pass",
        "control": control,
        "reward": reward,
        "ctrf_summary": counts,
        "job_id": job_result.get("id"),
        "trial_name": trial.name,
        "task_name": trial_result.get("task_name"),
        "job_result_digest": _file_digest(job_result_path),
        "trial_result_digest": _file_digest(trial_result_path),
        "ctrf_digest": _file_digest(ctrf_path),
        "reward_digest": _file_digest(reward_path),
        "artifact_manifest_digest": _file_digest(manifest_path),
    }


def build_receipt(
    task_dir: str | Path,
    *,
    executed_task_dir: str | Path,
    oracle_job: str | Path,
    nop_job: str | Path,
) -> dict[str, Any]:
    root = Path(task_dir)
    if not root.is_dir() or root.is_symlink():
        raise HarborControlError("task directory must be a real directory")
    executed_root = Path(executed_task_dir)
    if not executed_root.is_dir() or executed_root.is_symlink():
        raise HarborControlError("executed task snapshot must be a real directory")
    task = _task_id(root)
    if _task_id(executed_root) != task:
        raise HarborControlError("executed task snapshot identity does not match task-dir")
    authoring_digest = _task_tree_digest(root)
    executed_digest = _task_tree_digest(executed_root)
    if executed_digest != authoring_digest:
        raise HarborControlError("executed Harbor task snapshot differs from task-dir")
    controls = {
        "oracle": _validate_job(oracle_job, control="oracle"),
        "nop": _validate_job(nop_job, control="nop"),
    }
    if any(
        str(value.get("task_name", "")).rsplit("/", 1)[-1].replace("-", "_") != task
        for value in controls.values()
    ):
        raise HarborControlError("Harbor control job task identity does not match task-dir")
    receipt: dict[str, Any] = {
        "schema_version": "orbenchlab.screening-report.v1",
        "harbor_receipt_schema_version": "orbenchlab.harbor-controls.v1",
        "task_tree_digest": executed_digest,
        "authoring_task_tree_digest": authoring_digest,
        "executed_task_tree_digest": executed_digest,
        "tasks": [
            {
                "task": task,
                "family": task,
                "arms": {},
                "control_gates": controls,
                "decision": "collect-more-evidence",
                "evidence_level": "E3",
                "discrimination_index_observed_gap": None,
                "limitations": [
                    "Harbor Oracle/NOP packaging controls only; no model discrimination claim.",
                    "One control trial per arm; repeatability is not established.",
                ],
            }
        ],
    }
    receipt["report_digest"] = _digest(receipt)
    return receipt


def write_receipt(receipt: Mapping[str, Any], out: str | Path) -> dict[str, Path]:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "harbor-control-screening.json"
    markdown_path = output / "harbor-control-screening.md"
    json_path.write_text(
        json.dumps(dict(receipt), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    row = receipt["tasks"][0]
    controls = row["control_gates"]
    markdown_path.write_text(
        "# Harbor control receipt\n\n"
        f"- Task: `{row['task']}`\n"
        f"- Oracle: reward `{controls['oracle']['reward']}`, "
        f"{controls['oracle']['ctrf_summary']['passed']}/{controls['oracle']['ctrf_summary']['tests']} passed\n"
        f"- NOP: reward `{controls['nop']['reward']}`, "
        f"{controls['nop']['ctrf_summary']['failed']}/{controls['nop']['ctrf_summary']['tests']} failed\n"
        "- Evidence: `E3` Harbor verifier outcome; no model comparison claim.\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


__all__ = ["HarborControlError", "build_receipt", "write_receipt"]
