from __future__ import annotations

import json
from pathlib import Path

from orbenchlab import harbor_launcher


def test_launches_and_resumes_exact_oracle_nop_controls(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        '[task]\nname = "terminal-bench-science/demo-task"\n',
        encoding="utf-8",
    )
    harbor = tmp_path / "harbor"
    harbor.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
agent = value('--agent')
job = pathlib.Path(value('--jobs-dir')) / value('--job-name')
trial = job / ('task__' + agent)
trial.joinpath('verifier').mkdir(parents=True)
trial.joinpath('artifacts').mkdir(parents=True)
reward = 1.0 if agent == 'oracle' else 0.0
passed = 3 if agent == 'oracle' else 0
failed = 0 if agent == 'oracle' else 3
job.joinpath('result.json').write_text(json.dumps({'id': agent + '-id', 'n_total_trials': 1, 'stats': {'n_completed_trials': 1, 'n_errored_trials': 0}}))
trial.joinpath('result.json').write_text(json.dumps({'task_name': 'terminal-bench-science/demo-task', 'verifier_result': {'rewards': {'reward': reward}}, 'exception_info': None}))
trial.joinpath('verifier/ctrf.json').write_text(json.dumps({'results': {'summary': {'tests': 3, 'passed': passed, 'failed': failed, 'skipped': 0, 'pending': 0, 'other': 0}}}))
trial.joinpath('verifier/reward.txt').write_text(str(reward) + '\\n')
trial.joinpath('artifacts/manifest.json').write_text(json.dumps([{'source': '/root/submission/solver.py', 'status': 'ok' if agent == 'oracle' else 'failed'}]))
""",
        encoding="utf-8",
    )
    harbor.chmod(0o755)
    output = tmp_path / "out"
    first = harbor_launcher.launch_controls(task, harbor_executable=harbor, out=output)
    second = harbor_launcher.launch_controls(task, harbor_executable=harbor, out=output)
    assert first["report_digest"] == second["report_digest"]
    assert first["tasks"][0]["control_gates"]["oracle"]["gate"] == "pass"
    assert first["tasks"][0]["control_gates"]["nop"]["gate"] == "pass"
    assert (output / "harbor-control-screening.json").is_file()
    assert len(list((output / "jobs").glob("*/result.json"))) == 2
