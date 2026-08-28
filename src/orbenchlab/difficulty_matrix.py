"""Trusted difficulty-lattice evidence from repeated Harbor model matrices."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import harbor_model_matrix, pipeline, volc_rollout
from .core.errors import ORBenchError


class DifficultyMatrixError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.difficulty-matrix.v1"
VARIANT_SCHEMA_VERSION = "orbenchlab.variant-manifest.v1"
PREREGISTRATION_SCHEMA_VERSION = "orbenchlab.difficulty-preregistration.v1"


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DifficultyMatrixError(f"JSON evidence must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise DifficultyMatrixError(f"invalid JSON evidence: {path}") from None
    if not isinstance(value, Mapping):
        raise DifficultyMatrixError(f"JSON evidence root must be an object: {path}")
    return value


def _safe_relative(value: Any) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in raw
    ):
        raise DifficultyMatrixError(f"unsafe variant path: {raw!r}")
    return raw


def _normalise_secondary_axes(
    document: Mapping[str, Any], *, primary_name: str
) -> list[dict[str, Any]]:
    raw = document.get("secondary_axes", [])
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or len(raw) > 8:
        raise DifficultyMatrixError("secondary_axes must be a bounded list of axis objects")
    axes: list[dict[str, Any]] = []
    names: set[str] = {primary_name}
    for row in raw:
        if not isinstance(row, Mapping):
            raise DifficultyMatrixError("secondary axis must be an object")
        name = str(row.get("name") or "").strip()
        if not name or name in names:
            raise DifficultyMatrixError("secondary axis names must be unique and non-primary")
        names.add(name)
        levels = row.get("levels", [])
        if not isinstance(levels, list) or len(levels) > 16:
            raise DifficultyMatrixError("secondary axis levels must be a bounded list")
        axes.append(
            {
                "name": name,
                "meaning": str(row.get("meaning") or "").strip() or None,
                "expected_direction": str(row.get("expected_direction") or "").strip()
                or None,
                "levels": list(levels),
            }
        )
    return axes


def load_variant_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    document = _load(manifest_path)
    axis = document.get("primary_axis")
    variants = document.get("variants")
    evaluation_mode = document.get("evaluation_mode", "exploratory")
    if (
        document.get("schema_version") != VARIANT_SCHEMA_VERSION
        or not isinstance(axis, Mapping)
        or not isinstance(axis.get("name"), str)
        or not axis["name"].strip()
        or not isinstance(axis.get("expected_direction"), str)
        or not axis["expected_direction"].strip()
        or not isinstance(axis.get("ordered_levels"), list)
        or len(axis["ordered_levels"]) < 3
        or len({_digest(level) for level in axis["ordered_levels"]}) != len(axis["ordered_levels"])
        or not isinstance(variants, list)
        or len(variants) < 3
        or evaluation_mode not in {"exploratory", "held-out-confirmation"}
    ):
        raise DifficultyMatrixError("variant manifest lacks one ordered primary axis")
    primary_name = axis["name"].strip()
    secondary_axes = _normalise_secondary_axes(document, primary_name=primary_name)
    declared_axes = {primary_name, *(row["name"] for row in secondary_axes)}
    by_level: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    paths: set[str] = set()
    for row in variants:
        if not isinstance(row, Mapping):
            raise DifficultyMatrixError("variant manifest row must be an object")
        variant_id = str(row.get("variant_id") or "").strip()
        relative = _safe_relative(row.get("relative_path"))
        level_key = _digest(row.get("level"))
        axis_levels = row.get("axis_levels")
        if (
            not variant_id
            or variant_id in ids
            or relative in paths
            or level_key in by_level
            or not isinstance(axis_levels, Mapping)
            or axis_levels.get(axis["name"]) != row.get("level")
        ):
            raise DifficultyMatrixError("variant ids, paths and primary levels must be unique and bound")
        if secondary_axes and any(
            str(name) not in declared_axes for name in axis_levels
        ):
            raise DifficultyMatrixError(
                f"variant {variant_id} sets an undeclared difficulty axis"
            )
        ids.add(variant_id)
        paths.add(relative)
        by_level[level_key] = {
            "variant_id": variant_id,
            "relative_path": relative,
            "level": row.get("level"),
            "axis_levels": dict(axis_levels),
        }
    ordered = []
    for level in axis["ordered_levels"]:
        row = by_level.get(_digest(level))
        if row is None:
            raise DifficultyMatrixError("ordered primary level has no variant")
        ordered.append(row)
    if len(ordered) != len(variants):
        raise DifficultyMatrixError("variant manifest must contain exactly one row per ordered level")
    return {
        "schema_version": VARIANT_SCHEMA_VERSION,
        "primary_axis": {
            "name": primary_name,
            "expected_direction": axis["expected_direction"].strip(),
            "ordered_levels": list(axis["ordered_levels"]),
        },
        "secondary_axes": secondary_axes,
        "variants": ordered,
        "evaluation_mode": evaluation_mode,
        "manifest_digest": _file_digest(manifest_path),
    }


def build_preregistration(
    *,
    manifest_path: str | Path,
    variants_root: str | Path,
    frontier_model: str,
    weak_model: str,
    repetitions: int,
    max_budget_usd: float,
    max_turns: int,
    max_job_attempts: int,
    provider_route_digest: str,
    claude_executable_digest: str,
) -> dict[str, Any]:
    """Freeze a held-out lattice and its evaluation contract before jobs launch."""

    manifest = load_variant_manifest(manifest_path)
    if manifest["evaluation_mode"] != "held-out-confirmation":
        raise DifficultyMatrixError("held-out preregistration requires a held-out manifest")
    if (
        not frontier_model
        or not weak_model
        or frontier_model == weak_model
        or repetitions < 5
        or not 0 < max_budget_usd <= 100
        or max_turns < 1
        or not 1 <= max_job_attempts <= 5
        or not isinstance(provider_route_digest, str)
        or not isinstance(claude_executable_digest, str)
    ):
        raise DifficultyMatrixError("held-out preregistration contract is invalid")
    root = Path(variants_root).resolve()
    variants = []
    for order, row in enumerate(manifest["variants"]):
        task = (root / row["relative_path"]).resolve()
        if not task.is_relative_to(root) or task.is_symlink() or not (task / "task.toml").is_file():
            raise DifficultyMatrixError("held-out preregistration variant is unsafe")
        variants.append(
            {
                "variant_id": row["variant_id"],
                "relative_path": row["relative_path"],
                "order": order,
                "level": row["level"],
                "task_tree_digest": volc_rollout._task_tree_digest(task),
            }
        )
    digests = [row["task_tree_digest"] for row in variants]
    if len(set(digests)) != len(digests):
        raise DifficultyMatrixError(
            "held-out preregistration variants must have distinct task trees"
        )
    receipt: dict[str, Any] = {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "phase": "before-held-out-harbor-jobs",
        "variant_manifest_digest": manifest["manifest_digest"],
        "primary_axis": manifest["primary_axis"],
        "variants": variants,
        "frontier_model": frontier_model,
        "weak_model": weak_model,
        "repetitions_per_cell": repetitions,
        "agent_contract": {
            "max_budget_usd_per_trial": max_budget_usd,
            "max_turns_per_trial": max_turns,
            "max_job_attempts_per_model": max_job_attempts,
            "budget_enforcement": "claude-cli-max-budget-usd",
            "claude_executable_digest": claude_executable_digest,
        },
        "provider_route_digest": provider_route_digest,
        "fixed_confirmation_criteria": {
            "monotonic_solve_rates": "both models nonincreasing",
            "weak_span": "at least one strict weak-model drop",
            "nondegenerate_cell": True,
            "separation": "frontier Wilson lower bound exceeds weak Wilson upper bound",
        },
    }
    receipt["preregistration_digest"] = _digest(receipt)
    return receipt


def write_preregistration(receipt: Mapping[str, Any], out: str | Path) -> Path:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "difficulty-preregistration.json"
    if path.exists():
        existing = _load(path)
        if existing != dict(receipt):
            raise DifficultyMatrixError("refusing to replace another held-out preregistration")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(receipt), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _control_binding(control: Mapping[str, Any], *, task_digest: str) -> dict[str, Any]:
    try:
        pipeline._validate_harbor_receipt(control, Path("harbor-control-screening.json"))
    except pipeline.PipelineError as exc:
        raise DifficultyMatrixError(f"variant Harbor controls failed strict validation: {exc}") from None
    if control.get("task_tree_digest") != task_digest:
        raise DifficultyMatrixError("variant Harbor controls bind another task tree")
    tasks = control["tasks"]
    gates = tasks[0]["control_gates"]
    return {
        "report_digest": control["report_digest"],
        "task_id": tasks[0]["task"],
        "oracle": gates["oracle"],
        "nop": gates["nop"],
    }


def build_receipt(
    *,
    manifest_path: str | Path,
    variants_root: str | Path,
    evidence: Mapping[str, Mapping[str, str | Path]],
    frontier_model: str,
    weak_model: str,
    held_out: bool = False,
    preregistration_path: str | Path | None = None,
    base_task_tree_digest: str | None = None,
) -> dict[str, Any]:
    """Recompute an ordered difficulty matrix from raw per-variant evidence."""

    manifest = load_variant_manifest(manifest_path)
    root = Path(variants_root).resolve()
    if frontier_model == weak_model or not frontier_model or not weak_model:
        raise DifficultyMatrixError("difficulty matrix requires distinct frontier and weak models")
    expected_held_out = manifest["evaluation_mode"] == "held-out-confirmation"
    if bool(held_out) != expected_held_out:
        raise DifficultyMatrixError("held-out decision must be predeclared by the variant manifest")
    preregistration = None
    if held_out:
        if preregistration_path is None:
            raise DifficultyMatrixError("held-out confirmation requires a trusted preregistration")
        preregistration = _load(Path(preregistration_path))
        unsigned_preregistration = {
            key: value
            for key, value in preregistration.items()
            if key != "preregistration_digest"
        }
        if (
            preregistration.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION
            or preregistration.get("preregistration_digest")
            != _digest(unsigned_preregistration)
            or preregistration.get("phase") != "before-held-out-harbor-jobs"
            or preregistration.get("variant_manifest_digest") != manifest["manifest_digest"]
            or preregistration.get("frontier_model") != frontier_model
            or preregistration.get("weak_model") != weak_model
        ):
            raise DifficultyMatrixError("held-out preregistration binding is invalid")
    if set(evidence) != {row["variant_id"] for row in manifest["variants"]}:
        raise DifficultyMatrixError("evidence map must contain exactly the manifest variants")
    variant_rows = []
    raw_trials = []
    common_models: list[str] | None = None
    common_repetitions: int | None = None
    common_agent_contract: Mapping[str, Any] | None = None
    common_provider_route: str | None = None
    for order, variant in enumerate(manifest["variants"]):
        variant_id = variant["variant_id"]
        paths = evidence.get(variant_id)
        if not isinstance(paths, Mapping):
            raise DifficultyMatrixError(f"missing evidence for variant {variant_id}")
        if set(paths) != {"model_matrix", "controls"}:
            raise DifficultyMatrixError("variant evidence requires exactly model_matrix and controls")
        if any(not isinstance(paths[key], (str, Path)) for key in ("model_matrix", "controls")):
            raise DifficultyMatrixError("variant evidence paths must be filesystem paths")
        candidate = root / variant["relative_path"]
        current = root
        unsafe_link = False
        for part in PurePosixPath(variant["relative_path"]).parts:
            current = current / part
            unsafe_link = unsafe_link or current.is_symlink()
        task = candidate.resolve()
        if unsafe_link or not task.is_relative_to(root) or not (task / "task.toml").is_file():
            raise DifficultyMatrixError("variant task path is missing or escaped its root")
        matrix = harbor_model_matrix._validated_receipt(
            _load(Path(paths["model_matrix"]))
        )
        task_digest = volc_rollout._task_tree_digest(task)
        task_id = volc_rollout._task_id(task)
        if matrix.get("task_tree_digest") != task_digest or matrix.get("task") != task_id:
            raise DifficultyMatrixError("variant model matrix binds another task tree")
        if held_out and matrix.get("preregistration_digest") != preregistration.get(
            "preregistration_digest"
        ):
            raise DifficultyMatrixError("held-out matrix did not bind the preregistration")
        controls = _control_binding(_load(Path(paths["controls"])), task_digest=task_digest)
        if controls.get("task_id") != task_id:
            raise DifficultyMatrixError("variant controls bind another task identity")
        raw_models = matrix.get("models")
        if not isinstance(raw_models, list):
            raise DifficultyMatrixError("variant matrix models are malformed")
        models = list(raw_models)
        raw_repetitions = matrix.get("repetitions")
        if not isinstance(raw_repetitions, int) or isinstance(raw_repetitions, bool):
            raise DifficultyMatrixError("variant matrix repetitions are malformed")
        repetitions = raw_repetitions
        agent_contract = matrix.get("agent")
        provider_route = matrix.get("provider_route_digest")
        if (
            len(models) != 2
            or len(set(models)) != 2
            or set(models) != {frontier_model, weak_model}
            or repetitions < 5
            or matrix.get("evidence_level") != "E3"
            or matrix.get("checkpoint_capability") is not False
            or not isinstance(agent_contract, Mapping)
            or agent_contract.get("budget_enforcement") != "claude-cli-max-budget-usd"
            or not isinstance(provider_route, str)
        ):
            raise DifficultyMatrixError("variant matrix lacks two requested models with >=5 repetitions")
        if common_models is None:
            common_models, common_repetitions = models, repetitions
            common_agent_contract = dict(agent_contract)
            common_provider_route = provider_route
        elif (
            set(common_models) != set(models)
            or common_repetitions != repetitions
            or dict(agent_contract) != dict(common_agent_contract or {})
            or provider_route != common_provider_route
        ):
            raise DifficultyMatrixError("variant matrices do not share one equal budget rectangle")
        trials = [
            {
                **dict(trial),
                "variant_id": variant_id,
                "level": variant["level"],
                "order": order,
            }
            for trial in matrix["trials"]
        ]
        if len(trials) != len(models) * repetitions:
            raise DifficultyMatrixError("variant raw trial rectangle is incomplete")
        trial_keys = {(row.get("model_id"), row.get("attempt")) for row in trials}
        expected_keys = {(model, attempt) for model in models for attempt in range(1, repetitions + 1)}
        if trial_keys != expected_keys or any(
            row.get("status") not in {"pass", "fail"}
            or not isinstance(row.get("trial_id"), str)
            or not isinstance(row.get("trial_result_digest"), str)
            or not isinstance(row.get("ctrf_digest"), str)
            or not isinstance(row.get("reward_digest"), str)
            for row in trials
        ):
            raise DifficultyMatrixError("variant raw trials are duplicate or inconclusive")
        cells = {}
        for model in models:
            model_rows = [row for row in trials if row["model_id"] == model]
            successes = sum(row["status"] == "pass" for row in model_rows)
            lower, upper = volc_rollout._wilson_interval(successes, repetitions)
            cells[model] = {
                "n": repetitions,
                "passed": successes,
                "solve_rate": successes / repetitions,
                "wilson_95": [round(lower, 6), round(upper, 6)],
            }
        variant_rows.append(
            {
                **variant,
                "order": order,
                "task_tree_digest": task_digest,
                "matrix_receipt_digest": matrix["receipt_digest"],
                "control_report_digest": controls["report_digest"],
                "cells": cells,
            }
        )
        raw_trials.extend(trials)
    assert common_repetitions is not None
    variant_digests = [row["task_tree_digest"] for row in variant_rows]
    if len(set(variant_digests)) != len(variant_digests):
        raise DifficultyMatrixError(
            "difficulty variants must be pairwise distinct task trees; identical "
            "variants cannot carry a difficulty claim"
        )
    if base_task_tree_digest is not None and base_task_tree_digest in variant_digests:
        raise DifficultyMatrixError(
            "a difficulty variant is byte-identical to the base task; renaming "
            "is not a difficulty change"
        )
    if held_out:
        assert preregistration is not None
        prereg_variants = preregistration.get("variants")
        prereg_agent = preregistration.get("agent_contract")
        expected_variants = [
            {
                "variant_id": row["variant_id"],
                "relative_path": row["relative_path"],
                "order": row["order"],
                "level": row["level"],
                "task_tree_digest": row["task_tree_digest"],
            }
            for row in variant_rows
        ]
        if (
            prereg_variants != expected_variants
            or preregistration.get("repetitions_per_cell") != common_repetitions
            or preregistration.get("provider_route_digest") != common_provider_route
            or not isinstance(prereg_agent, Mapping)
            or prereg_agent.get("max_budget_usd_per_trial")
            != common_agent_contract.get("max_budget_usd_per_trial")
            or prereg_agent.get("max_turns_per_trial")
            != common_agent_contract.get("max_turns_per_trial")
            or prereg_agent.get("max_job_attempts_per_model")
            != common_agent_contract.get("max_job_attempts_per_model")
            or prereg_agent.get("claude_executable_digest")
            != common_agent_contract.get("executable_digest")
        ):
            raise DifficultyMatrixError("held-out preregistration differs from executed matrices")
    monotonic_models = {}
    for model in (frontier_model, weak_model):
        rates = [row["cells"][model]["solve_rate"] for row in variant_rows]
        monotonic_models[model] = {
            "rates": rates,
            "nonincreasing": all(left >= right for left, right in zip(rates, rates[1:])),
            "strict_drop": any(left > right for left, right in zip(rates, rates[1:])),
        }
    separation_levels = []
    for row in variant_rows:
        frontier = row["cells"][frontier_model]
        weak = row["cells"][weak_model]
        gap = frontier["solve_rate"] - weak["solve_rate"]
        lower = frontier["wilson_95"][0] - weak["wilson_95"][1]
        if lower > 0:
            separation_levels.append(row["variant_id"])
        row["model_gap"] = {
            "observed": round(gap, 6),
            "wilson_lower_bound": round(lower, 6),
        }
    monotonic = all(row["nonincreasing"] for row in monotonic_models.values())
    has_span = monotonic_models[weak_model]["strict_drop"]
    nondegenerate = any(
        0 < cell["solve_rate"] < 1
        for row in variant_rows
        for cell in row["cells"].values()
    )
    promising = monotonic and has_span and bool(separation_levels) and nondegenerate
    decision = (
        "confirmed-promising"
        if promising and held_out
        else "exploratory-promising"
        if promising
        else "quarantine"
    )
    difficulty_genome = {
        "primary_axis": manifest["primary_axis"],
        "secondary_axes": manifest.get("secondary_axes", []),
        "variants": [
            {
                "variant_id": row["variant_id"],
                "level": row["level"],
                "axis_levels": dict(row.get("axis_levels", {})),
                "task_tree_digest": row["task_tree_digest"],
            }
            for row in variant_rows
        ],
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "variant_manifest_digest": manifest["manifest_digest"],
        "primary_axis": manifest["primary_axis"],
        "secondary_axes": manifest.get("secondary_axes", []),
        "difficulty_genome": difficulty_genome,
        "base_task_tree_digest": base_task_tree_digest,
        "frontier_model": frontier_model,
        "weak_model": weak_model,
        "repetitions_per_cell": common_repetitions,
        "held_out_confirmation": bool(held_out),
        "evaluation_mode": manifest["evaluation_mode"],
        "preregistration_digest": (
            preregistration.get("preregistration_digest") if preregistration else None
        ),
        "rectangular": True,
        "variants": variant_rows,
        "raw_trials": raw_trials,
        "monotonicity": {
            "expected": "nonincreasing solve rate as ordered difficulty rises",
            "models": monotonic_models,
            "passed": monotonic,
            "weak_model_has_strict_drop": has_span,
        },
        "discrimination": {
            "separated_levels": separation_levels,
            "passed": bool(separation_levels),
            "criterion": "frontier Wilson lower bound exceeds weak Wilson upper bound",
        },
        "nondegenerate_cell_present": nondegenerate,
        "decision": decision,
        "evidence_level": "E3",
        "checkpoint_capability": False,
        "limitations": [
            "Fresh Harbor restarts only; no same-checkpoint E4 intervention evidence.",
            "Exploratory variant selection is not confirmatory unless held_out_confirmation is true.",
        ],
    }
    receipt["receipt_digest"] = _digest(receipt)
    return receipt


def write_receipt(receipt: Mapping[str, Any], out: str | Path) -> Path:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "difficulty-matrix.json"
    lock_path = output / ".difficulty-matrix.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            existing = _load(path)
            if existing.get("receipt_digest") != receipt.get("receipt_digest"):
                raise DifficultyMatrixError("refusing to overwrite another difficulty receipt")
            return path
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(dict(receipt), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


__all__ = [
    "DifficultyMatrixError",
    "SCHEMA_VERSION",
    "PREREGISTRATION_SCHEMA_VERSION",
    "VARIANT_SCHEMA_VERSION",
    "build_preregistration",
    "build_receipt",
    "load_variant_manifest",
    "write_receipt",
    "write_preregistration",
]
