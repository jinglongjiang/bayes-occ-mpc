#!/usr/bin/env bash
set -euo pipefail

calibration=results/belief_calibration_dev_500_549.json
out=results/latency_single_process_2800_2819
mkdir -p "$out"

mapfile -t radii < <(python3 - "$calibration" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))["split_conformal"]
for key in ("visible_radial_buffer_by_horizon", "hidden_radial_buffer_by_horizon"):
    print(",".join(f"{value:.17g}" for value in p[key]))
PY
)

run_arm() {
  local label="$1"
  local arm="$2"
  shift 2
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 continuous_mpc_gate.py \
      --arms "$arm" \
      --human-counts 20 \
      --scenarios circle_crossing \
      --episodes 20 \
      --case-offset 2800 \
      --population 512 \
      --iterations 4 \
      --v-max 1.20 \
      --chance-limit 0.50 \
      --near-chance-limit 0.15 \
      --probability-weight 1.0 \
      --human-margin 0.10 \
      --conformal-visible-radii "${radii[0]}" \
      --conformal-hidden-radii "${radii[1]}" \
      --workers 1 \
      "$@" \
      --output "$out/$label.json" > "$out/$label.log" 2>&1
}

run_arm sensor sensor
run_arm posterior_mean deterministic
run_arm fixed_020 fixed --fixed-uncertainty-radius 0.20
run_arm conformal conformal
run_arm bayes bayes
run_arm gt gt
