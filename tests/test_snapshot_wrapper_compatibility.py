"""Compatibility contract between source snapshots and the pinned paid wrapper."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

from orbenchlab import workflow


def test_snapshot_copy_can_receive_injected_skills_without_mutating_snapshot(
    upstream_fixtures: Path, tmp_path: Path
) -> None:
    """The upstream wrapper can edit its copied task while the source stays bound."""
    source = tmp_path / "ORAgentBench"
    shutil.copytree(upstream_fixtures / "oragentbench_min", source)
    source_skills = source / "harbor_tasks" / "single_task" / "environment" / "skills"
    source_skills.mkdir(parents=True)
    (source_skills / "stale.txt").write_text("stale\n", encoding="utf-8")

    prepared = workflow.prepare_oragentbench_run(
        source=source,
        task="single_task",
        agent="oracle",
        model="",
        date="2026-08-24",
        workspace=tmp_path / "runs",
        wall_clock_sec=20,
    )
    snapshot_task = prepared.source / "harbor_tasks" / "single_task"
    before = prepared.source_snapshot_digest

    # shutil.copytree preserves source modes.  The pinned wrapper then replaces
    # environment/skills and writes skills_dir into task.toml in this copy.
    copied_task = tmp_path / "wrapper-work" / "single_task"
    shutil.copytree(snapshot_task, copied_task)
    copied_skills = copied_task / "environment" / "skills"
    shutil.rmtree(copied_skills)
    copied_skills.mkdir(parents=True)
    task_toml = copied_task / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            "[environment]\n", '[environment]\nskills_dir = "/skills"\n'
        ),
        encoding="utf-8",
    )

    snapshot_task_mode = snapshot_task.stat().st_mode
    snapshot_toml_mode = (snapshot_task / "task.toml").stat().st_mode
    unrelated_mode = (prepared.source / "skills" / "README.md").stat().st_mode
    assert snapshot_task_mode & stat.S_IWUSR
    assert snapshot_toml_mode & stat.S_IWUSR
    assert not (unrelated_mode & stat.S_IWUSR)
    assert workflow._source_snapshot_digest(prepared.source) == before
