#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/fixed_frontier
pids=()
for radius in 0.35 0.50 0.65; do
  python3 continuous_mpc_gate.py \
    --arms fixed \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 10 \
    --case-offset 40 \
    --population 512 \
    --iterations 4 \
    --chance-limit 0.50 \
    --probability-weight 0.0 \
    --fixed-uncertainty-radius "$radius" \
    --output "results/fixed_frontier/fixed_${radius}.json" \
    > "results/fixed_frontier/fixed_${radius}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
