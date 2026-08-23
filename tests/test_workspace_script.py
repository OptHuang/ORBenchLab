"""Behavior tests for the persistent self-hosted workspace boundary."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid
from pathlib import Path


def test_workspace_lane_is_private_even_with_collaborative_umask(
    repo_root: Path,
) -> None:
    # pytest's tmp_path is rooted below /tmp on GitHub-hosted Linux runners.
    # That location is intentionally rejected by the production validator, so
    # this positive permission test must use a distinct, non-temporary root.
    root = repo_root.parent / f".orbenchlab-workspace-test-{uuid.uuid4().hex}"
    try:
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
    finally:
        shutil.rmtree(root, ignore_errors=True)
