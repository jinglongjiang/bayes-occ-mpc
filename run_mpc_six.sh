#!/usr/bin/env bash
set -euo pipefail

calibration=results/belief_calibration_dev_500_549.json
out=results/mpc_six_2400_2499
mkdir -p "$out"

mapfile -t radii < <(python3 - "$calibration" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))["split_conformal"]
for key in ("visible_radial_buffer_by_horizon", "hidden_radial_buffer_by_horizon"):
    print(",".join(f"{value:.17g}" for value in p[key]))
PY
)

labels=(baseline_circle baseline_square dense_circle dense_square large_circle large_square)
sims=(circle_crossing square_crossing circle_crossing square_crossing circle_crossing square_crossing)
counts=(5 10 10 20 12 20)
sizes=(4.0 10.0 4.0 10.0 6.0 14.0)
pids=()

for i in "${!labels[@]}"; do
  geometry=()
  if [[ "${sims[$i]}" == circle_crossing ]]; then
    geometry=(--circle-radius "${sizes[$i]}")
  else
    geometry=(--square-width "${sizes[$i]}")
  fi
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 continuous_mpc_gate.py \
      --arms sensor,deterministic,fixed,conformal,bayes,gt \
      --human-counts "${counts[$i]}" \
      --scenarios "${sims[$i]}" \
      "${geometry[@]}" \
      --time-limit 25 \
      --episodes 100 \
      --case-offset 2400 \
      --population 512 \
      --iterations 4 \
      --v-max 1.0 \
      --chance-limit 0.50 \
      --near-chance-limit 0.15 \
      --probability-weight 1.0 \
      --human-margin 0.10 \
      --fixed-uncertainty-radius 0.20 \
      --conformal-visible-radii "${radii[0]}" \
      --conformal-hidden-radii "${radii[1]}" \
      --workers 2 \
      --output "$out/${labels[$i]}.json" \
      > "$out/${labels[$i]}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
