#!/usr/bin/env bash
set -euo pipefail

out=results/v4_final_900_999
mkdir -p "$out"
pids=()

run_arm() {
  local label="$1"
  local arm="$2"
  shift 2
  python3 continuous_mpc_gate.py \
    --arms "$arm" \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 100 \
    --case-offset 900 \
    --population 512 \
    --iterations 4 \
    --v-max 1.20 \
    --chance-limit 0.50 \
    --near-chance-limit 0.15 \
    --probability-weight 1.0 \
    --human-margin 0.10 \
    --workers 2 \
    "$@" \
    --output "$out/${label}.json" \
    > "$out/${label}.log" 2>&1 &
  pids+=("$!")
}

run_arm sensor sensor
run_arm deterministic deterministic
run_arm fixed_035 fixed --fixed-uncertainty-radius 0.35
run_arm fixed_065 fixed --fixed-uncertainty-radius 0.65
run_arm bayes bayes
run_arm worst worst --unknown-margin 0.46
run_arm gt gt

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status == 0 )); then
  python3 analyze_formal.py "$out" \
    --case-offset 900 \
    --episodes 100 \
    --output "$out/analysis.json" \
    > "$out/analysis.log"
fi
exit "$status"
