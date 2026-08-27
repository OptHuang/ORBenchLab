#!/bin/sh
set -eu
export ORBENCH_TASK_ROOT=/root
mkdir -p /root/submission
cp /solution/solver.py /root/submission/solver.py
exec python /solution/solver.py
