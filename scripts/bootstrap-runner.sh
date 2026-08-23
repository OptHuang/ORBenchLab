#!/usr/bin/env bash
set -euo pipefail

# Install ORBenchLab into a repository-local virtual environment.  The caller
# may pin another interpreter/venv without editing this script:
#   ORBENCH_PYTHON=$HOME/.local/bin/python3.12 ORBENCH_VENV=.venv ./scripts/bootstrap-runner.sh
python_bin="${ORBENCH_PYTHON:-}"
venv_dir="${ORBENCH_VENV:-.venv}"

if [[ -n "$python_bin" ]]; then
  python_candidates=("$python_bin")
else
  # Non-interactive runners often omit user-local bins from PATH. Prefer an
  # explicit modern interpreter, then inspect the conventional user-local
  # location, and only then try the unversioned command.
  python_candidates=(
    python3.13 python3.12 python3.11
    "${HOME:?HOME must be set}/.local/bin/python3.13"
    "$HOME/.local/bin/python3.12"
    "$HOME/.local/bin/python3.11"
    python3
  )
fi

python_bin=""
for candidate in "${python_candidates[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done

if [[ -z "$python_bin" ]]; then
  if [[ -n "${ORBENCH_PYTHON:-}" ]]; then
    echo "error: ORBENCH_PYTHON is missing or older than Python 3.11" >&2
  else
    echo "error: ORBenchLab requires Python >= 3.11; no compatible interpreter was found" >&2
  fi
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$python_bin" -m venv "$repo_root/$venv_dir"
"$repo_root/$venv_dir/bin/python" -m pip install -e "$repo_root"

# `uv tool install harbor` keeps the executable outside the ordinary PATH on
# some non-interactive runners (notably under
# $HOME/.local/share/uv/tools/harbor/bin). Resolve it without assuming a user
# name and expose it through the project venv, whose bin directory is already
# stable after activation. Never overwrite an existing venv command.
harbor_bin=""
if command -v harbor >/dev/null 2>&1; then
  harbor_bin="$(command -v harbor)"
else
  uv_harbor_dir="${ORBENCH_HARBOR_BIN_DIR:-${HOME:?HOME must be set}/.local/share/uv/tools/harbor/bin}"
  if [[ -x "$uv_harbor_dir/harbor" ]]; then
    harbor_bin="$uv_harbor_dir/harbor"
  fi
fi

venv_harbor="$repo_root/$venv_dir/bin/harbor"
if [[ -n "$harbor_bin" && ! -e "$venv_harbor" && ! -L "$venv_harbor" ]]; then
  if [[ "$harbor_bin" != /* ]]; then
    harbor_bin="$(cd "$(dirname "$harbor_bin")" && pwd -P)/$(basename "$harbor_bin")"
  fi
  ln -s "$harbor_bin" "$venv_harbor"
fi

"$repo_root/$venv_dir/bin/orbench" --version

echo "runner ready: $repo_root/$venv_dir/bin/orbench"
if [[ -x "$venv_harbor" ]]; then
  "$venv_harbor" --version
  echo "Harbor ready via project venv: $venv_harbor"
else
  echo "warning: Harbor was not found; prepare/report commands work, but ORAgentBench execution will fail doctor" >&2
  echo 'hint: install Harbor or set ORBENCH_HARBOR_BIN_DIR (for uv, usually $HOME/.local/share/uv/tools/harbor/bin), then rerun bootstrap' >&2
fi
