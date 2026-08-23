"""Behavior tests for the persistent self-hosted workspace boundary."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def test_workspace_lane_is_private_even_with_collaborative_umask(
    repo_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "persistent-runs"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    script = repo_root / "scripts" / "validate-run-workspace.sh"
    env = {
        **os.environ,
        "GITHUB_WORKSPACE": "",
        "ORBENCH_MIN_FREE_KB": "1",
        "RUNNER_TEMP": "",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            'umask 0002; exec "$@"',
            "orbench-workspace-test",
            str(script),
            str(root),
            "acceptance",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    workspace = Path(result.stdout.strip())
    assert workspace == (root / "acceptance").resolve()
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
