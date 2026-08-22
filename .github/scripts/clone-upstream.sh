#!/usr/bin/env bash
# Clone an upstream benchmark at the commit its integration module pins.
#
#   clone-upstream.sh <integration> <parent-dir> [ref-override]
#
# Two properties matter and are easy to get wrong, which is why this is one
# script rather than copy-pasted steps:
#
#   1. It fetches the *exact pinned commit*, not a branch. No tracking, nothing
#      newer, no surprise when upstream moves.
#   2. It uses the directory name each project requires. ORAgentBench resolves
#      the relative dataset path `ORAgentBench/harbor_tasks` against its
#      checkout's parent, so a clone named anything else produces a job config
#      whose dataset path does not exist.
#
# The pinned commit comes from the integration registry, so the workflow and
# the library cannot disagree about which commit is current.
set -euo pipefail

INTEGRATION="${1:?usage: clone-upstream.sh <integration> <parent-dir> [ref]}"
PARENT="${2:?usage: clone-upstream.sh <integration> <parent-dir> [ref]}"
REF_OVERRIDE="${3:-}"

case "$INTEGRATION" in
  oragentbench)
    REPO="https://github.com/ORAgentBench/ORAgentBench"
    # Load-bearing: upstream's path resolver expects this exact name.
    DIRNAME="ORAgentBench"
    ;;
  frontieror)
    REPO="https://github.com/Minw913/FrontierOR"
    DIRNAME="FrontierOR"
    ;;
  *)
    echo "error: unknown integration '$INTEGRATION'" >&2
    exit 2
    ;;
esac

PINNED="$(INTEGRATION="$INTEGRATION" python3 -c '
import os
from orbenchlab.integrations import registry

print(registry.describe(os.environ["INTEGRATION"])["pinned_commit"])
')"

REF="$PINNED"
if [ -n "$REF_OVERRIDE" ]; then
  REF="$REF_OVERRIDE"
  echo "::notice::drift check — cloning $INTEGRATION at $REF instead of the pin $PINNED"
fi

DEST="$PARENT/$DIRNAME"
mkdir -p "$DEST"
# Explicit paths rather than `cd`: each workflow step gets a fresh shell, so a
# directory change here would not carry to the next one anyway.
git -C "$DEST" init -q
git -C "$DEST" remote add origin "$REPO" 2>/dev/null || git -C "$DEST" remote set-url origin "$REPO"
git -C "$DEST" fetch -q --depth 1 origin "$REF"
git -C "$DEST" checkout -q FETCH_HEAD

ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
echo "cloned $INTEGRATION -> $DEST at $ACTUAL"
if [ -z "$REF_OVERRIDE" ] && [ "$ACTUAL" != "$PINNED" ]; then
  echo "::error::checked out $ACTUAL but the integration pins $PINNED"
  exit 1
fi
