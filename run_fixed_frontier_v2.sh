#!/usr/bin/env bash
set -euo pipefail

out=results/fixed_frontier_v2_dev_500_599
mkdir -p "$out"
radii=(0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60)

run_radius() {
  local radius="$1"
  local label="${radius/./}"
  python3 continuous_mpc_gate.py \
    --arms fixed \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 100 \
    --case-offset 500 \
    --population 512 \
    --iterations 4 \
    --v-max 1.20 \
    --chance-limit 0.50 \
    --near-chance-limit 0.15 \
    --probability-weight 1.0 \
    --human-margin 0.10 \
    --workers 2 \
    --fixed-uncertainty-radius "$radius" \
    --output "$out/fixed_${label}.json" \
    > "$out/fixed_${label}.log" 2>&1
}

# Five concurrent radii x two workers keeps the 16-core host responsive.
for start in 0 5; do
  pids=()
  for ((i=start; i<start+5; i++)); do
    run_radius "${radii[$i]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if ((status != 0)); then
    exit "$status"
  fi
done

python3 analyze_fixed_frontier.py "$out" \
  --case-offset 500 \
  --expected-episodes 100 \
  --output "$out/frontier.json" \
  > "$out/frontier.log"
