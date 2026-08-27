#!/bin/sh
set -eu
mkdir -p /root/submission
cp /solution/solver.py /root/submission/solver.py
exec python /solution/solver.py \
  --instance /root/instance.json \
  --output /root/submission/solution.json
