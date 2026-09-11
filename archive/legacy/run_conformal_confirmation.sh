#!/usr/bin/env bash
set -euo pipefail

calibration=results/belief_calibration_dev_500_549.json
out=results/conformal_confirmation_2000_2399
mkdir -p "$out"

mapfile -t radii < <(python3 - "$calibration" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))["split_conformal"]
for key in ("visible_radial_buffer_by_horizon", "hidden_radial_buffer_by_horizon"):
    values = p[key]
    if len(values) != 16 or any(value is None for value in values):
        raise SystemExit(f"invalid conformal radii: {key}")
    print(",".join(f"{value:.17g}" for value in values))
PY
)

common=(
  --human-counts 20
  --scenarios circle_crossing
  --episodes 400
  --case-offset 2000
  --population 512
  --iterations 4
  --v-max 1.20
  --chance-limit 0.50
  --near-chance-limit 0.15
  --probability-weight 1.0
  --human-margin 0.10
  --conformal-visible-radii "${radii[0]}"
  --conformal-hidden-radii "${radii[1]}"
  --workers 2
)

run_arm() {
  local label="$1"
  local arm="$2"
  shift 2
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 continuous_mpc_gate.py \
      --arms "$arm" "${common[@]}" "$@" \
      --output "$out/$label.json" > "$out/$label.log" 2>&1 &
  pids+=("$!")
}

pids=()
run_arm bayes bayes
run_arm conformal conformal
run_arm posterior_mean deterministic
run_arm fixed_020 fixed --fixed-uncertainty-radius 0.20
run_arm sensor sensor
run_arm gt gt
for pid in "${pids[@]}"; do
  wait "$pid"
done

python3 analyze_paired.py \
  --input bayes="$out/bayes.json" \
  --input conformal="$out/conformal.json" \
  --input posterior_mean="$out/posterior_mean.json" \
  --input fixed_020="$out/fixed_020.json" \
  --input sensor="$out/sensor.json" \
  --input gt="$out/gt.json" \
  --case-offset 2000 \
  --episodes 400 \
  --output "$out/analysis.json" > "$out/analysis.log"
