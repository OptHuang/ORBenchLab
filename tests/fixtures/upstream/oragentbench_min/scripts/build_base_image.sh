#!/usr/bin/env bash
set -euo pipefail
docker build -t "${1:-oragentbench-base:py311-scip}" .
