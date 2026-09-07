#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/bayes_frontier
pids=()
for limit in 0.60 0.70; do
  python3 continuous_mpc_gate.py \
    --arms bayes \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 10 \
    --case-offset 40 \
    --population 512 \
    --iterations 4 \
    --chance-limit "$limit" \
    --probability-weight 0.0 \
    --output "results/bayes_frontier/bayes_${limit}.json" \
    > "results/bayes_frontier/bayes_${limit}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
