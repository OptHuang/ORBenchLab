#!/usr/bin/env bash
# Validate and return a durable, runner-owned ORBenchLab workspace lane.
set -euo pipefail

root="${1:-}"
lane="${2:-}"
minimum_free_kb="${ORBENCH_MIN_FREE_KB:-20971520}"

if [[ -z "$root" || "$root" != /* ]]; then
  echo "error: ORBENCH_RUNS_ROOT must be a non-empty absolute path" >&2
  exit 2
fi
if [[ ! "$lane" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]]; then
  echo "error: workspace lane must be a short lowercase identifier" >&2
  exit 2
fi
is_at_or_below() {
  local candidate="${1%/}"
  local boundary="${2%/}"
  [[ -n "$boundary" ]] && [[ "$candidate" == "$boundary" || "$candidate" == "$boundary/"* ]]
}
if [[ "$root" == "/" ]] \
  || is_at_or_below "$root" "/tmp" \
  || is_at_or_below "$root" "/var/tmp" \
  || is_at_or_below "$root" "${RUNNER_TEMP:-}" \
  || is_at_or_below "$root" "${GITHUB_WORKSPACE:-}"; then
  echo "error: ORBENCH_RUNS_ROOT must be durable and dedicated, not a temporary or checkout root" >&2
  exit 2
fi
stat_uid() {
  stat -c '%u' "$1" 2>/dev/null || stat -f '%u' "$1"
}
stat_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}
if [[ -L "$root" ]]; then
  echo "error: ORBENCH_RUNS_ROOT may not be a symlink" >&2
  exit 2
fi
mkdir -p "$root"
root="$(realpath "$root")"
if [[ "$(stat_uid "$root")" != "$(id -u)" ]]; then
  echo "error: ORBENCH_RUNS_ROOT is not owned by the runner account" >&2
  exit 2
fi
mode="$(stat_mode "$root")"
if (( (8#$mode & 0022) != 0 )); then
  echo "error: ORBENCH_RUNS_ROOT must not be group/world writable" >&2
  exit 2
fi
available_kb="$(df -Pk "$root" | awk 'NR==2 {print $4}')"
if [[ ! "$available_kb" =~ ^[0-9]+$ ]] || (( available_kb < minimum_free_kb )); then
  echo "error: ORBENCH_RUNS_ROOT has less than the required free space" >&2
  exit 2
fi

workspace="$root/$lane"
if [[ ! -e "$workspace" && ! -L "$workspace" ]]; then
  # Shared research hosts commonly use umask 0002.  The lane is a private
  # evidence workspace, so create the final directory with an explicit mode
  # instead of creating 0775 and then rejecting our own output.  If another
  # process wins this creation race, mkdir fails and the caller retries.
  mkdir -m 700 "$workspace"
fi
if [[ -L "$workspace" || "$(stat_uid "$workspace")" != "$(id -u)" ]]; then
  echo "error: workspace lane is unsafe or not runner-owned" >&2
  exit 2
fi
lane_mode="$(stat_mode "$workspace")"
if (( (8#$lane_mode & 0022) != 0 )); then
  echo "error: workspace lane must not be group/world writable" >&2
  exit 2
fi
printf '%s\n' "$(realpath "$workspace")"
