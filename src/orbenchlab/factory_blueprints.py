"""Opinionated autonomous-agent DAGs for the OR task factory.

The prompts deliberately assign semantic judgment to coding-agent sessions.
The resulting plan remains an E1 authoring process until existing static,
verifier, Harbor and repeated-model gates validate its artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agentic_factory import AgenticFactoryError, compile_plan


WORKSPACE_SCHEMA_VERSION = "orbenchlab.paper-factory-workspace.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_PAPER_TEXT_BYTES = 64 * 1024 * 1024


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise AgenticFactoryError(f"factory seed contains a symlink: {relative}")
        if path.is_file():
            rows.append({"path": relative, "content_digest": _digest_bytes(path.read_bytes())})
    if not rows:
        raise AgenticFactoryError("factory seed task is empty")
    return _digest_bytes(_canonical(rows))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical(dict(payload)) + b"\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_provenance(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AgenticFactoryError("paper provenance is not valid UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise AgenticFactoryError("paper provenance root must be an object")
    unsigned = {key: item for key, item in value.items() if key != "binding_digest"}
    if (
        value.get("paper_provenance_schema_version") != "orbenchlab.paper-provenance.v1"
        or not _DIGEST.fullmatch(str(value.get("source_content_digest", "")))
        or value.get("binding_digest") != _digest_bytes(_canonical(unsigned))
    ):
        raise AgenticFactoryError("paper provenance binding is stale or malformed")
    return value


def _make_inputs_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise AgenticFactoryError("factory input snapshot contains a symlink")
        if path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o555 if executable else 0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _extract_paper_text(
    paper: Path,
    *,
    executable: str | Path = "pdftotext",
    timeout_sec: float = 120.0,
    max_output_bytes: int = _MAX_PAPER_TEXT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    """Extract a bounded, page-addressable snapshot before any model session.

    Agent sessions should reason over this one immutable extraction instead of
    repeatedly spending model budget invoking PDF tools. The original PDF is
    retained so an auditor can still check difficult anchors.
    """

    if timeout_sec <= 0 or not 1 <= max_output_bytes <= _MAX_PAPER_TEXT_BYTES:
        raise AgenticFactoryError("paper text extraction bounds are invalid")
    requested = str(executable)
    resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
    if not resolved or not Path(resolved).is_file():
        raise AgenticFactoryError("pdftotext executable is unavailable")
    resolved_path = Path(resolved).resolve()
    safe_env = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL")
        if name in os.environ
    }
    try:
        version_run = subprocess.run(
            [str(resolved_path), "-v"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=safe_env,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise AgenticFactoryError("pdftotext version probe failed") from None
    version_lines = version_run.stdout.decode("utf-8", errors="replace").splitlines()
    version = version_lines[0][:256] if version_lines else "unknown"
    command = [str(resolved_path), "-layout", "-enc", "UTF-8", str(paper.resolve()), "-"]
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                env=safe_env,
                start_new_session=True,
            )
        except OSError:
            raise AgenticFactoryError("pdftotext launch failed") from None
        deadline = time.monotonic() + timeout_sec
        failure: str | None = None
        while process.poll() is None:
            if time.monotonic() >= deadline:
                failure = "wall_clock_timeout"
                break
            if os.fstat(stdout_file.fileno()).st_size > max_output_bytes:
                failure = "output_limit_exceeded"
                break
            time.sleep(0.05)
        if failure is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise AgenticFactoryError(f"pdftotext failed closed: {failure}")
        if process.returncode != 0:
            raise AgenticFactoryError("pdftotext rejected the bound paper")
        output_size = os.fstat(stdout_file.fileno()).st_size
        if output_size > max_output_bytes:
            raise AgenticFactoryError("pdftotext failed closed: output_limit_exceeded")
        stdout_file.seek(0)
        raw = stdout_file.read(max_output_bytes + 1)
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise AgenticFactoryError("pdftotext output is not valid UTF-8") from None
    pages = decoded.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages or not any(page.strip() for page in pages):
        raise AgenticFactoryError("pdftotext produced no paper text")
    rendered = "".join(
        f"=== PDF PAGE {index} ===\n{page.strip()}\n\n"
        for index, page in enumerate(pages, start=1)
    ).encode("utf-8")
    receipt = {
        "schema_version": "orbenchlab.paper-text-extraction.v1",
        "source_content_digest": _digest_bytes(paper.read_bytes()),
        "extractor": "pdftotext",
        "extractor_version": version,
        "executable_digest": _digest_bytes(resolved_path.read_bytes()),
        "argv_template": ["-layout", "-enc", "UTF-8", "<BOUND_PAPER>", "-"],
        "timeout_sec": timeout_sec,
        "max_output_bytes": max_output_bytes,
        "page_count": len(pages),
        "text_content_digest": _digest_bytes(rendered),
    }
    receipt["receipt_digest"] = _digest_bytes(_canonical(receipt))
    return rendered, receipt


def prepare_workspace(
    *,
    paper_file: str | Path,
    paper_provenance: str | Path,
    seed_task: str | Path,
    workdir: str | Path,
    pdftotext_executable: str | Path = "pdftotext",
    paper_text_timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Create an idempotent, checksummed input workspace for autonomous agents."""

    paper = Path(paper_file)
    provenance_path = Path(paper_provenance)
    seed = Path(seed_task)
    root = Path(workdir)
    if paper.is_symlink() or not paper.is_file():
        raise AgenticFactoryError("paper_file must be a regular non-symlink file")
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise AgenticFactoryError("paper_provenance must be a regular non-symlink file")
    if seed.is_symlink() or not seed.is_dir():
        raise AgenticFactoryError("seed_task must be a real directory")
    provenance = _load_provenance(provenance_path)
    paper_digest = _digest_bytes(paper.read_bytes())
    provenance_digest = _digest_bytes(provenance_path.read_bytes())
    if paper_digest != provenance["source_content_digest"]:
        raise AgenticFactoryError("paper bytes do not match bound provenance")
    seed_digest = _tree_digest(seed)
    paper_text, extraction_receipt = _extract_paper_text(
        paper,
        executable=pdftotext_executable,
        timeout_sec=paper_text_timeout_sec,
    )
    paper_text_digest = _digest_bytes(paper_text)
    unsigned_manifest = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "source_binding_digest": provenance["binding_digest"],
        "paper_content_digest": paper_digest,
        "paper_provenance_digest": provenance_digest,
        "paper_text_digest": paper_text_digest,
        "paper_text_extraction": extraction_receipt,
        "seed_task_tree_digest": seed_digest,
        "inputs": {
            "paper": "factory-input/paper.pdf",
            "paper_text": "factory-input/paper.txt",
            "paper_provenance": "factory-input/paper-provenance.json",
            "seed_task": "factory-input/seed-task",
        },
    }
    manifest = {
        **unsigned_manifest,
        "workspace_binding_digest": _digest_bytes(_canonical(unsigned_manifest)),
    }
    input_root = root / "factory-input"
    manifest_path = input_root / "workspace-manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise AgenticFactoryError("existing factory workspace manifest is malformed") from None
        if existing != manifest:
            raise AgenticFactoryError("refusing to replace a factory workspace with different inputs")
        if (
            _digest_bytes((input_root / "paper.pdf").read_bytes()) != paper_digest
            or _digest_bytes((input_root / "paper.txt").read_bytes()) != paper_text_digest
            or _digest_bytes((input_root / "paper-provenance.json").read_bytes())
            != provenance_digest
            or _tree_digest(input_root / "seed-task") != seed_digest
        ):
            raise AgenticFactoryError("existing factory workspace inputs failed digest validation")
        _make_inputs_read_only(input_root)
        return manifest
    if root.exists() and any(root.iterdir()):
        raise AgenticFactoryError("new factory workdir must be empty")
    root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir()
    shutil.copy2(paper, input_root / "paper.pdf")
    (input_root / "paper.txt").write_bytes(paper_text)
    shutil.copy2(provenance_path, input_root / "paper-provenance.json")
    shutil.copytree(seed, input_root / "seed-task", symlinks=False)
    _atomic_json(manifest_path, manifest)
    _make_inputs_read_only(input_root)
    return manifest


def _stage(
    stage_id: str,
    role: str,
    prompt: str,
    output: str,
    *,
    model: str,
    profile: str,
    depends_on: Sequence[str] = (),
    kind: str = "json",
    timeout_sec: int = 1800,
    max_attempts: int = 2,
    max_budget_usd: float = 1.0,
    max_output_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "role": role,
        "profile": profile,
        "model": model,
        "prompt": prompt,
        "depends_on": list(depends_on),
        "timeout_sec": timeout_sec,
        "max_attempts": max_attempts,
        "max_budget_usd": max_budget_usd,
        "max_output_bytes": max_output_bytes,
        "required_outputs": [{"path": output, "kind": kind}],
    }


def paper_to_benchmark_plan(
    *,
    source_binding_digest: str,
    author_model: str,
    reviewer_models: Sequence[str],
    frontier_model: str,
    weak_model: str,
    profile: str = "claude-code",
) -> dict[str, Any]:
    """Compile the default autonomous paper-to-calibrated-task semantic DAG.

    Runtime stages are told to use the repository's existing ORBench/Harbor
    commands.  Their JSON is still untrusted until the deterministic pipeline
    consumes the raw jobs and independently writes/validates its receipts.
    """

    reviewers = [str(value).strip() for value in reviewer_models if str(value).strip()]
    if len(reviewers) < 2 or len(set(reviewers)) < 2:
        raise AgenticFactoryError("paper factory requires at least two distinct reviewer models")
    common = (
        "Use only files inside this workspace. The bound inputs are listed in "
        "factory-input/workspace-manifest.json. paper.txt is the deterministic, page-marked text "
        "snapshot; use it before opening paper.pdf, which is retained only for difficult anchor checks. "
        "Work autonomously; do not ask a human to choose "
        "a task, rubric, hint or model. Cite source pages/sections for paper-derived claims. "
        "Unknown evidence must remain explicitly unknown."
    )
    stages = [
        _stage(
            "paper-derive-primary",
            "paper scientist",
            common
            + " Read the complete paper. Extract its executable scientific core, assumptions, "
            "available code/data, candidate terminal interactions and non-derivable claims. Write "
            "a structured evidence map with page/section anchors and no task design yet. For this "
            "stage, inspect only paper.txt, paper.pdf and paper-provenance.json: do not inspect seed-task, "
            "run solvers/tests, or evaluate an existing task. Do not re-extract the complete PDF: use "
            "the page markers in paper.txt, write the required JSON promptly, validate it locally, and stop.",
            "factory/evidence/paper-derivation-primary.json",
            model=author_model,
            profile=profile,
            max_budget_usd=2.0,
        ),
        _stage(
            "paper-derive-critic",
            "independent paper evidence auditor",
            common
            + " Audit paper-derivation-primary.json against the complete paper. Correct unsupported "
            "claims, missing assumptions, licenses, reproducibility blockers and source anchors.",
            "factory/evidence/paper-derivation-critic.json",
            model=reviewers[0],
            profile=profile,
            depends_on=("paper-derive-primary",),
            max_budget_usd=2.0,
        ),
        _stage(
            "task-design-a",
            "OR benchmark task designer A",
            common
            + " Propose several strict Terminal-Bench Science tasks from the audited evidence. For "
            "each give terminal contract, scientific value, verifier oracle, anti-shortcut design, "
            "estimated model bottlenecks and orthogonal difficulty controls.",
            "factory/design/task-design-a.json",
            model=author_model,
            profile=profile,
            depends_on=("paper-derive-critic",),
        ),
        _stage(
            "task-design-b",
            "adversarial OR benchmark task designer B",
            common
            + " Independently propose paper-faithful tasks and attack likely triviality, leakage, "
            "brittleness, solver dependence and verifier gaming. Prefer tasks with multiple "
            "controllable difficulty axes.",
            "factory/design/task-design-b.json",
            model=reviewers[1],
            profile=profile,
            depends_on=("paper-derive-critic",),
        ),
        _stage(
            "task-design-synthesis",
            "senior task selection agent",
            common
            + " Compare both designs and autonomously select one task. Record rejected alternatives, "
            "a full rubric/test plan, Oracle and NOP behavior, expected bottlenecks, and a preliminary "
            "difficulty lattice. Selection is provisional until runtime calibration.",
            "factory/design/task-design-selected.json",
            model=author_model,
            profile=profile,
            depends_on=("task-design-a", "task-design-b"),
        ),
        _stage(
            "task-author-v1",
            "Terminal-Bench Science task implementer",
            common
            + " Copy factory-input/seed-task to factory/tasks/task-v1 and replace it with the selected "
            "paper-backed task. Implement task.toml, environment, instruction, solution, data and strict "
            "rubric tests. Run all feasible local static/tests; leave no placeholder or human TODO.",
            "factory/tasks/task-v1",
            model=author_model,
            profile=profile,
            depends_on=("task-design-synthesis",),
            kind="directory",
            timeout_sec=3600,
        ),
        _stage(
            "task-review-science",
            "independent scientific-faithfulness reviewer",
            common
            + " Review task-v1 against the paper evidence and TB-Science expectations. Inspect every "
            "file, run useful checks, and write prioritized, source-grounded defects. Do not edit task-v1.",
            "factory/reviews/task-review-science.json",
            model=reviewers[0],
            profile=profile,
            depends_on=("task-author-v1",),
        ),
        _stage(
            "task-review-verifier",
            "independent verifier and anti-cheating reviewer",
            common
            + " Attack task-v1's verifier, rubric, hidden/public separation, determinism, resource limits, "
            "Oracle/NOP semantics and shortcut resistance. Run tests where possible. Do not edit task-v1.",
            "factory/reviews/task-review-verifier.json",
            model=reviewers[1],
            profile=profile,
            depends_on=("task-author-v1",),
        ),
        _stage(
            "task-repair-v2",
            "senior task repair agent",
            common
            + " Copy task-v1 to factory/tasks/task-v2, resolve every supported finding from both reviews, "
            "and run the repository's deterministic task-authoring gate plus task tests. Preserve a "
            "machine-readable repair ledger inside task-v2/data.",
            "factory/tasks/task-v2",
            model=author_model,
            profile=profile,
            depends_on=("task-review-science", "task-review-verifier"),
            kind="directory",
            timeout_sec=3600,
            max_attempts=3,
        ),
        _stage(
            "runtime-controls",
            "Harbor runtime evidence engineer",
            common
            + " Use the existing ORBenchLab and Harbor commands to run real Oracle and NOP controls on "
            "task-v2. Preserve raw job directories and write a runtime index with exact commands, task "
            "digest, job paths and the generated harbor-control receipt. Never fabricate a passing gate.",
            "factory/runtime/control-index.json",
            model=author_model,
            profile=profile,
            depends_on=("task-repair-v2",),
            timeout_sec=7200,
            max_attempts=2,
        ),
        _stage(
            "pilot-frontier",
            "frontier-model rollout operator",
            common
            + f" Run repeated Harbor rollouts of model {frontier_model!r} at the declared no-hint baseline. "
            "Preserve raw trajectories, verifier outcomes, budgets and run identities in a pilot index.",
            "factory/runtime/pilot-frontier.json",
            model=frontier_model,
            profile=profile,
            depends_on=("runtime-controls",),
            timeout_sec=10_800,
        ),
        _stage(
            "pilot-weak",
            "weak-model rollout operator",
            common
            + f" Run repeated Harbor rollouts of model {weak_model!r} under the exact same baseline budget. "
            "Preserve raw trajectories, verifier outcomes, budgets and run identities in a pilot index.",
            "factory/runtime/pilot-weak.json",
            model=weak_model,
            profile=profile,
            depends_on=("runtime-controls",),
            timeout_sec=10_800,
        ),
        _stage(
            "trajectory-diagnosis",
            "agent trajectory analyst",
            common
            + " Analyse the frontier and weak pilot traces with verifier outcomes. Separate observations "
            "from hypotheses; identify repeated bottlenecks and candidate intervention anchors. Evidence "
            "is E3 unless a real same-checkpoint continuation experiment is available.",
            "factory/analysis/trajectory-diagnosis.json",
            model=reviewers[0],
            profile=profile,
            depends_on=("pilot-frontier", "pilot-weak"),
            timeout_sec=3600,
        ),
        _stage(
            "intervention-study",
            "controlled intervention experimenter",
            common
            + " For each proposed bottleneck, inspect whether the runtime exposes a real resumable "
            "checkpoint. If yes, run repeated same-checkpoint continuations with a fixed hint ladder and "
            "budgets. If not, record E4 as unavailable and run no restart-with-hint masquerading as causal.",
            "factory/analysis/intervention-study.json",
            model=author_model,
            profile=profile,
            depends_on=("trajectory-diagnosis",),
            timeout_sec=10_800,
        ),
        _stage(
            "difficulty-design",
            "benchmark difficulty architect",
            common
            + " Convert validated bottlenecks into orthogonal difficulty axes: instance scale/structure, "
            "information in instructions, tool/budget limits, verifier tolerance/coverage and hint levels. "
            "Specify monotonicity expectations and anti-confounding checks for a variant lattice.",
            "factory/difficulty/difficulty-lattice.json",
            model=reviewers[1],
            profile=profile,
            depends_on=("intervention-study",),
        ),
        _stage(
            "variant-author",
            "difficulty variant implementer",
            common
            + " Copy task-v2 into factory/tasks/variants and implement the smallest useful lattice of "
            "versioned variants. Each variant must state exactly one or a controlled combination of changed "
            "axes, preserve paper fidelity, pass static checks and keep distinct task identities.",
            "factory/tasks/variants",
            model=author_model,
            profile=profile,
            depends_on=("difficulty-design",),
            kind="directory",
            timeout_sec=7200,
        ),
        _stage(
            "calibration",
            "multi-model calibration operator",
            common
            + " Run repeated Harbor calibration over all variants with frontier and weak models under equal "
            "budgets. Produce pass counts, uncertainty, infra exclusions, monotonicity checks and model-gap "
            "metrics. Quarantine variants lacking repeatability or clean Oracle/NOP controls.",
            "factory/calibration/calibration-index.json",
            model=author_model,
            profile=profile,
            depends_on=("variant-author",),
            timeout_sec=21_600,
        ),
        _stage(
            "final-synthesis",
            "benchmark release summarizer",
            common
            + " Produce the human review packet: what the task is, paper provenance, verifier contract, "
            "difficulty axes/variants, frontier-vs-weak repeated outcomes, trajectory bottlenecks, intervention "
            "evidence level, costs, limitations and exact accepted/quarantined artifacts. Never promote from "
            "agent opinion alone.",
            "factory/final/task-review-summary.json",
            model=reviewers[0],
            profile=profile,
            depends_on=("calibration",),
            timeout_sec=3600,
        ),
    ]
    return compile_plan(
        name="autonomous paper to calibrated OR benchmark",
        source_binding_digest=source_binding_digest,
        stages=stages,
        workspace_manifest="factory-input/workspace-manifest.json",
    )


__all__ = ["WORKSPACE_SCHEMA_VERSION", "paper_to_benchmark_plan", "prepare_workspace"]
