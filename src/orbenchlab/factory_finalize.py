"""Independent, fail-closed finalization for semantic factory runs.

Agent-authored files never promote themselves.  This module joins a validated
E1 factory chain to deterministic static, Harbor, and repeated-model receipts;
the strongest possible result is an E3 release candidate for human review.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from . import agentic_factory, pipeline, task_authoring, volc_rollout
from .core import schema as schema_mod
from .core.errors import ORBenchError
from .volc_review import REQUIRED_REVIEW_CRITERIA


class FactoryFinalizeError(ORBenchError):
    exit_code = 8


SCHEMA_VERSION = "orbenchlab.factory-finalization.v1"
_DIGEST_PREFIX = "sha256:"


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


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_DIGEST_PREFIX)
        and len(value) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in value[len(_DIGEST_PREFIX) :])
    )


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FactoryFinalizeError("evidence file is missing or is a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise FactoryFinalizeError("evidence file is not valid UTF-8 JSON") from None
    if not isinstance(value, Mapping):
        raise FactoryFinalizeError("evidence root must be an object")
    return value


def _gate(name: str, path: Path, validator: Callable[[Mapping[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    try:
        document = _load(path)
        details = validator(document)
        return {
            "name": name,
            "status": "pass",
            "evidence_path": str(path),
            "evidence_digest": _file_digest(path),
            **details,
        }
    except (FactoryFinalizeError, agentic_factory.AgenticFactoryError, pipeline.PipelineError, OSError, KeyError, TypeError, ValueError) as exc:
        return {
            "name": name,
            "status": "fail",
            "evidence_path": str(path),
            "evidence_digest": _file_digest(path) if path.is_file() and not path.is_symlink() else None,
            "reason": type(exc).__name__,
        }


def _static_validator(task_digest: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(document: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = {key: value for key, value in document.items() if key != "receipt_digest"}
        criteria = document.get("implementation_criteria")
        provenance = document.get("provenance_checks")
        if (
            document.get("authoring_schema_version") != task_authoring.AUTHORING_SCHEMA_VERSION
            or document.get("receipt_digest") != _value_digest(unsigned)
            or document.get("task_tree_digest") != task_digest
            or document.get("decision") not in {"ready-for-human-review", "ready-for-harbor-validation"}
            or not isinstance(criteria, list)
            or len(criteria) != len(task_authoring.IMPLEMENTATION_CRITERIA)
            or {row.get("name") for row in criteria if isinstance(row, Mapping)}
            != set(task_authoring.IMPLEMENTATION_CRITERIA)
            or any(not isinstance(row, Mapping) or row.get("status") == "fail" for row in criteria)
            or not isinstance(provenance, list)
            or not provenance
            or any(
                not isinstance(row, Mapping) or row.get("status") == "fail"
                for row in provenance
            )
        ):
            raise FactoryFinalizeError("static authoring receipt did not pass completely")
        return {"task_tree_digest": task_digest, "receipt_digest": document["receipt_digest"]}

    return validate


def _semantic_validator(task_digest: str, static_digest: str, paper_digest: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(document: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = {key: value for key, value in document.items() if key != "review_digest"}
        models, reviewers = document.get("models"), document.get("reviewers")
        if (document.get("schema_version") != "orbenchlab.volc-authoring-review.v1"
            or document.get("review_digest") != _value_digest(unsigned)
            or document.get("task_tree_digest") != task_digest
            or document.get("static_receipt_digest") != static_digest
            or not paper_digest or document.get("paper_digest") != paper_digest
            or document.get("aggregate_decision") != "promising-needs-harbor"
            or not isinstance(models, list) or len(models) < 2 or len(set(models)) != len(models)
            or document.get("review_count") != len(models)
            or not isinstance(reviewers, list) or len(reviewers) != len(models)):
            raise FactoryFinalizeError("semantic review receipt binding failed")
        for model, reviewer in zip(models, reviewers, strict=True):
            review = reviewer.get("review") if isinstance(reviewer, Mapping) else None
            criteria = review.get("criteria") if isinstance(review, Mapping) else None
            if (not isinstance(reviewer, Mapping) or reviewer.get("model") != model
                or not isinstance(review, Mapping) or review.get("decision") != "promising"
                or review.get("shape_complete") is not True or review.get("rubric_complete") is not True
                or not isinstance(criteria, list) or len(criteria) != len(REQUIRED_REVIEW_CRITERIA)
                or {row.get("name") for row in criteria if isinstance(row, Mapping)} != REQUIRED_REVIEW_CRITERIA
                or any(not isinstance(row, Mapping) or row.get("status") != "pass" or not str(row.get("evidence", "")).strip() for row in criteria)):
                raise FactoryFinalizeError("semantic reviewer lacks a complete passing rubric")
        return {"task_tree_digest": task_digest, "review_digest": document["review_digest"], "models": models}
    return validate


def _harbor_validator(task_digest: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(document: Mapping[str, Any]) -> dict[str, Any]:
        pipeline._validate_harbor_receipt(document, Path("harbor-receipt.json"))
        if document.get("task_tree_digest") != task_digest:
            raise FactoryFinalizeError("Harbor receipt binds another task tree")
        row = document["tasks"][0]
        return {
            "task_tree_digest": task_digest,
            "report_digest": document["report_digest"],
            "task_id": row["task"],
            "evidence_level": "E3",
        }

    return validate


def _calibration_validator(task_digest: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(document: Mapping[str, Any]) -> dict[str, Any]:
        supplied = document.get("report_digest")
        unsigned = {key: value for key, value in document.items() if key != "report_digest"}
        rows = document.get("tasks")
        if (
            document.get("schema_version") != "orbenchlab.screening-report.v1"
            or supplied != _value_digest(unsigned)
            or document.get("task_tree_digest") != task_digest
            or not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], Mapping)
        ):
            raise FactoryFinalizeError("calibration receipt schema/digest/task binding failed")
        row = rows[0]
        arms = row.get("arms")
        discrimination = row.get("discrimination")
        trials = document.get("trials")
        contract = document.get("run_contract")
        if (
            document.get("task") != row.get("task")
            or row.get("evidence_level") != "E3"
            or row.get("decision") != "review-promising"
            or not isinstance(arms, Mapping)
            or not isinstance(discrimination, Mapping)
            or not isinstance(trials, list)
            or not isinstance(contract, Mapping)
        ):
            raise FactoryFinalizeError("calibration is not a complete E3 rollout receipt")
        baseline = [
            arm
            for arm in arms.values()
            if isinstance(arm, Mapping) and arm.get("hint_level") == 0
        ]
        models = {str(arm.get("model_id")) for arm in baseline if arm.get("model_id")}
        baseline_trials = [
            trial
            for trial in trials
            if isinstance(trial, Mapping)
            and trial.get("model") in models
            and trial.get("hint_level") == 0
        ]
        trial_ids = {
            (trial.get("model"), trial.get("hint_level"), trial.get("trial"))
            for trial in baseline_trials
        }
        for trial in baseline_trials:
            status = trial.get("status")
            if (
                status not in volc_rollout._OUTCOME_STATUSES
                or not _is_digest(trial.get("request_digest"))
                or not _is_digest(trial.get("response_digest"))
            ):
                raise FactoryFinalizeError("baseline trial lacks provider outcome evidence")
            if status in {"pass", "fail"}:
                verifier = trial.get("verifier")
                if (
                    trial.get("phase") != "verifier"
                    or not _is_digest(trial.get("solver_digest"))
                    or not isinstance(verifier, Mapping)
                    or verifier.get("receipt_valid") is not True
                    or verifier.get("status") != status
                ):
                    raise FactoryFinalizeError("baseline verifier outcome evidence is incomplete")
        if (
            len(models) < 2
            or len(baseline) < 2
            or len(trial_ids) != len(baseline_trials)
            or any(
                not isinstance(arm.get("solve_n"), int)
                or arm["solve_n"] < 5
                or arm.get("infra_exceptions") not in ([], ())
                for arm in baseline
            )
        ):
            raise FactoryFinalizeError("calibration lacks two clean repeated baseline model arms")
        repetitions = {int(arm["solve_n"]) for arm in baseline}
        if (
            len(repetitions) != 1
            or set(contract.get("models") or []) != models
            or int(contract.get("repetitions", 0)) != next(iter(repetitions))
            or 0 not in (contract.get("hint_levels") or [])
        ):
            raise FactoryFinalizeError("calibration is not an equal-budget baseline rectangle")
        recomputed_arms = volc_rollout._summarize_trials(
            [trial for trial in trials if isinstance(trial, Mapping) and trial.get("model")]
        )
        if dict(arms) != recomputed_arms:
            raise FactoryFinalizeError("calibration arms do not match raw trial outcomes")
        recomputed = volc_rollout._discrimination_summary(
            recomputed_arms,
            sorted(models),
            repetitions=next(iter(repetitions)),
        )
        if (
            dict(discrimination) != recomputed
            or recomputed.get("rectangular") is not True
            or recomputed.get("promising") is not True
            or row.get("discrimination_index_observed_gap") != recomputed.get("observed_gap")
        ):
            raise FactoryFinalizeError("calibration discrimination does not recompute as promising")
        return {
            "task_tree_digest": task_digest,
            "report_digest": supplied,
            "task_id": row.get("task"),
            "models": sorted(models),
            "minimum_repetitions": next(iter(repetitions)),
            "observed_gap": recomputed["observed_gap"],
            "gap_95_lower_bound": recomputed["gap_95_lower_bound"],
            "evidence_level": "E3",
        }

    return validate


def _summary_validator(
    *, task_id: str | None, evidence_digests: set[str]
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def validate(document: Mapping[str, Any]) -> dict[str, Any]:
        cards = document.get("cards")
        if (
            document.get("pipeline_schema_version") != pipeline.PIPELINE_SCHEMA_VERSION
            or not isinstance(cards, list)
            or len(cards) != 1
            or not isinstance(cards[0], Mapping)
        ):
            raise FactoryFinalizeError("final summary must be one deterministic pipeline task card")
        card = cards[0]
        schema = schema_mod.load_schema(schema_mod.schemas_dir() / pipeline.TASK_CARD_SCHEMA)
        schema_mod.validate(card, schema, name="factory final task card")
        difficulty = card.get("difficulty")
        performance = card.get("performance")
        evidence = card.get("evidence")
        if (
            card.get("task_id") != task_id
            or card.get("decision") != "review-promising"
            or not isinstance(difficulty, Mapping)
            or difficulty.get("declared") is not True
            or not difficulty.get("axes")
            or not isinstance(performance, Mapping)
            or len(performance.get("models") or []) < 2
            or not isinstance(evidence, Mapping)
            or evidence.get("level") != "E3"
            or not evidence_digests.issubset(set(evidence.get("report_digests") or []))
        ):
            raise FactoryFinalizeError("final task card does not bind promising E3 task evidence")
        return {
            "summary_content_digest": _value_digest(document),
            "task_id": task_id,
            "decision": card["decision"],
        }

    return validate


def build_receipt(
    *,
    plan_path: str | Path,
    factory_run_path: str | Path,
    workdir: str | Path,
    task_dir: str | Path,
    static_receipt_path: str | Path,
    semantic_review_path: str | Path,
    harbor_receipt_path: str | Path,
    calibration_receipt_path: str | Path,
    final_summary_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic finalization receipt; missing gates never promote."""

    plan_file = Path(plan_path)
    run_file = Path(factory_run_path)
    workspace = Path(workdir)
    task = Path(task_dir)
    plan: Mapping[str, Any] | None = None
    run: Mapping[str, Any] | None = None
    factory_error: str | None = None
    try:
        plan = agentic_factory.load_plan(plan_file)
        agentic_factory._require_hard_budget_profiles(plan)
        run = agentic_factory._load_run(run_file, plan)
        agentic_factory._validate_workspace_binding(workspace.resolve(), plan)
        agentic_factory._validate_run_chain(run_file.parent, plan, run, workspace=workspace.resolve())
        if run.get("status") != "semantic-complete-e1":
            raise FactoryFinalizeError("factory has not reached semantic-complete-e1")
    except (agentic_factory.AgenticFactoryError, FactoryFinalizeError, OSError) as exc:
        factory_error = type(exc).__name__
    factory_gate = {
        "name": "semantic_factory",
        "status": "pass" if factory_error is None else "fail",
        "evidence_path": str(run_file),
        "evidence_digest": _file_digest(run_file) if run_file.is_file() else None,
        "reason": factory_error,
        "evidence_level": "E1",
        "factory_id": plan.get("factory_id") if plan else None,
        "completion_digest": run.get("completion_digest") if run else None,
    }
    try:
        if not task.is_dir() or task.is_symlink():
            raise FactoryFinalizeError("task directory is missing")
        authoring_task_digest = task_authoring._task_tree_digest(task)
        runtime_task_digest = volc_rollout._task_tree_digest(task)
        task_id = volc_rollout._task_id(task)
    except (OSError, FactoryFinalizeError) as exc:
        authoring_task_digest = ""
        runtime_task_digest = ""
        task_id = None
        factory_gate["status"] = "fail"
        factory_gate["reason"] = type(exc).__name__

    try:
        static_digest = str(_load(Path(static_receipt_path)).get("receipt_digest") or "")
    except FactoryFinalizeError:
        static_digest = ""
    try:
        paper_digest = str(task_authoring._load_document(task / "paper-provenance.json").get("source_content_digest") or "")
    except (OSError, task_authoring.TaskAuthoringError):
        paper_digest = ""
    evidence_gates = [
        factory_gate,
        _gate("static_authoring", Path(static_receipt_path), _static_validator(authoring_task_digest)),
        _gate("semantic_review", Path(semantic_review_path), _semantic_validator(authoring_task_digest, static_digest, paper_digest)),
        _gate("harbor_oracle_nop", Path(harbor_receipt_path), _harbor_validator(runtime_task_digest)),
        _gate("model_calibration", Path(calibration_receipt_path), _calibration_validator(runtime_task_digest)),
    ]
    report_digests = {
        str(gate["evidence_digest"])
        for gate in evidence_gates
        if gate["name"] in {"harbor_oracle_nop", "model_calibration"}
        and _is_digest(gate.get("evidence_digest"))
    }
    gates = [
        *evidence_gates,
        _gate(
            "final_summary",
            Path(final_summary_path),
            _summary_validator(task_id=task_id, evidence_digests=report_digests),
        ),
    ]
    passed = all(gate["status"] == "pass" for gate in gates)
    bound_ids = {
        gate.get("task_id") for gate in gates if gate.get("task_id") is not None
    }
    identity_ok = bool(task_id) and bound_ids == {task_id}
    identity_gate = {
        "name": "cross_evidence_identity",
        "status": "pass" if identity_ok else "fail",
        "task_id": task_id,
        "authoring_task_tree_digest": authoring_task_digest or None,
        "runtime_task_tree_digest": runtime_task_digest or None,
        "reason": None if identity_ok else "task identity or digest mismatch",
    }
    gates.append(identity_gate)
    passed = passed and identity_ok
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "eligible-for-human-release-review" if passed else "not-promoted",
        "promoted": passed,
        "evidence_level": "E3" if passed else "E1",
        "task_id": task_id,
        "authoring_task_tree_digest": authoring_task_digest or None,
        "runtime_task_tree_digest": runtime_task_digest or None,
        "factory_id": plan.get("factory_id") if plan else None,
        "plan_digest": plan.get("plan_digest") if plan else None,
        "gates": gates,
        "limitations": [
            "This receipt is not TB-Science acceptance, publication, or E4 causal evidence.",
            "Digests are integrity bindings, not cryptographic signatures; inputs require a trusted artifact store.",
        ],
    }
    receipt["receipt_digest"] = _value_digest(receipt)
    return receipt


def write_receipt(receipt: Mapping[str, Any], out: str | Path) -> Path:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "factory-finalization.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(json.dumps(dict(receipt), indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    os.replace(temporary, destination)
    return destination


__all__ = ["FactoryFinalizeError", "SCHEMA_VERSION", "build_receipt", "write_receipt"]
