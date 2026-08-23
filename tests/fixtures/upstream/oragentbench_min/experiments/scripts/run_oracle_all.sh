#!/usr/bin/env bash
set -euo pipefail
# Minimal stand-in for the pinned upstream contract used by command tests.
config=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) config="$2"; shift 2 ;;
    *) shift ;;
  esac
done
test -n "$config"
bash "$(dirname "$0")/../../scripts/build_base_image.sh"
exec harbor run -c "$config"
