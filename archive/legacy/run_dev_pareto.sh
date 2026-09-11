#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/dev_pareto

arms=(sensor deterministic fixed bayes worst gt)
pids=()
for arm in "${arms[@]}"; do
  python3 continuous_mpc_gate.py \
    --arms "$arm" \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 10 \
    --case-offset 40 \
    --population 512 \
    --iterations 4 \
    --chance-limit 0.50 \
    --probability-weight 0.0 \
    --output "results/dev_pareto/${arm}.json" \
    > "results/dev_pareto/${arm}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
