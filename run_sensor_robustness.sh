#!/usr/bin/env bash
set -euo pipefail

calibration=results/belief_calibration_dev_500_549.json
out=results/sensor_robustness_2500_2599
mkdir -p "$out"

mapfile -t radii < <(python3 - "$calibration" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))["split_conformal"]
for key in ("visible_radial_buffer_by_horizon", "hidden_radial_buffer_by_horizon"):
    print(",".join(f"{value:.17g}" for value in p[key]))
PY
)

run_condition() {
  local label="$1"
  local pos="$2"
  local vel="$3"
  local detect="$4"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 continuous_mpc_gate.py \
      --arms deterministic,fixed,conformal,bayes \
      --human-counts 20 \
      --scenarios circle_crossing \
      --episodes 100 \
      --case-offset 2500 \
      --population 512 \
      --iterations 4 \
      --v-max 1.20 \
      --chance-limit 0.50 \
      --near-chance-limit 0.15 \
      --probability-weight 1.0 \
      --human-margin 0.10 \
      --fixed-uncertainty-radius 0.20 \
      --conformal-visible-radii "${radii[0]}" \
      --conformal-hidden-radii "${radii[1]}" \
      --position-noise-std "$pos" \
      --velocity-noise-std "$vel" \
      --detection-probability "$detect" \
      --workers 6 \
      --output "$out/$label.json" > "$out/$label.log" 2>&1
}

run_condition moderate 0.05 0.10 0.90 &
p1=$!
run_condition severe 0.10 0.20 0.80 &
p2=$!
wait "$p1"
wait "$p2"
