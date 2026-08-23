#!/usr/bin/env bash
set -euo pipefail

# Install ORBenchLab into a repository-local virtual environment.  The caller
# may pin another interpreter/venv without editing this script:
#   ORBENCH_PYTHON=$HOME/.local/bin/python3.12 ORBENCH_VENV=.venv ./scripts/bootstrap-runner.sh
python_bin="${ORBENCH_PYTHON:-python3}"
venv_dir="${ORBENCH_VENV:-.venv}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: Python interpreter not found: $python_bin" >&2
  exit 2
fi

if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "error: ORBenchLab requires Python >= 3.11 ($python_bin is too old)" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$python_bin" -m venv "$repo_root/$venv_dir"
"$repo_root/$venv_dir/bin/python" -m pip install -e "$repo_root"
"$repo_root/$venv_dir/bin/orbench" --version

echo "runner ready: $repo_root/$venv_dir/bin/orbench"
