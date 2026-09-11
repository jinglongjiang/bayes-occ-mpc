#!/usr/bin/env bash
set -euo pipefail

out=results/v6_fresh_1500_1899
mkdir -p "$out"

run_arm() {
  local label="$1"
  local arm="$2"
  shift 2
  python3 continuous_mpc_gate.py \
    --arms "$arm" \
    --human-counts 20 \
    --scenarios circle_crossing \
    --episodes 400 \
    --case-offset 1500 \
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

run_batch() {
  pids=()
  "$@"
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

primary_batch() {
  run_arm sensor sensor
  run_arm posterior_mean deterministic
  run_arm bayes bayes
  run_arm worst_040 worst --unknown-margin 0.40
  run_arm gt gt
}

# These five radii are the complete nondominated set selected only on
# development cases 500--599. Test outcomes may not remove any candidate.
fixed_batch() {
  run_arm fixed_020 fixed --fixed-uncertainty-radius 0.20
  run_arm fixed_035 fixed --fixed-uncertainty-radius 0.35
  run_arm fixed_040 fixed --fixed-uncertainty-radius 0.40
  run_arm fixed_045 fixed --fixed-uncertainty-radius 0.45
  run_arm fixed_060 fixed --fixed-uncertainty-radius 0.60
}

run_batch primary_batch
run_batch fixed_batch

python3 analyze_paired.py \
  --input bayes="$out/bayes.json" \
  --input sensor="$out/sensor.json" \
  --input posterior_mean="$out/posterior_mean.json" \
  --input worst_040="$out/worst_040.json" \
  --input gt="$out/gt.json" \
  --input fixed_020="$out/fixed_020.json" \
  --input fixed_035="$out/fixed_035.json" \
  --input fixed_040="$out/fixed_040.json" \
  --input fixed_045="$out/fixed_045.json" \
  --input fixed_060="$out/fixed_060.json" \
  --case-offset 1500 \
  --episodes 400 \
  --output "$out/analysis.json" \
  > "$out/analysis.log"
