#!/usr/bin/env bash
set -euo pipefail

out=results/external_six_2400_2499
mkdir -p "$out"

labels=(baseline_circle baseline_square dense_circle dense_square large_circle large_square)
sims=(circle_crossing square_crossing circle_crossing square_crossing circle_crossing square_crossing)
counts=(5 10 10 20 12 20)
sizes=(4.0 10.0 4.0 10.0 6.0 14.0)
orca_pids=()

run_orca() {
  local i="$1"
  local geometry=()
  if [[ "${sims[$i]}" == circle_crossing ]]; then
    geometry=(--circle-radius "${sizes[$i]}")
  else
    geometry=(--square-width "${sizes[$i]}")
  fi
  python3 external_baseline_eval.py \
    --controller orca \
    --human-count "${counts[$i]}" \
    --scenario "${sims[$i]}" \
    "${geometry[@]}" \
    --time-limit 25 \
    --episodes 100 \
    --case-offset 2400 \
    --output "$out/orca_${labels[$i]}.json" \
    > "$out/orca_${labels[$i]}.log" 2>&1
}

for i in "${!labels[@]}"; do
  run_orca "$i" &
  orca_pids+=("$!")
done
for pid in "${orca_pids[@]}"; do
  wait "$pid"
done

# Two GPU processes at a time avoid both VRAM pressure and unfair latency
# inflation while still keeping the six-scenario run short.
for start in 0 2 4; do
  mamba_pids=()
  for ((i=start; i<start+2; i++)); do
    if [[ "${sims[$i]}" == circle_crossing ]]; then
      geometry=(--circle-radius "${sizes[$i]}")
    else
      geometry=(--square-width "${sizes[$i]}")
    fi
    docker exec -w /workspace/bayes_occ_mpc mamba_env \
      /root/miniconda3/envs/mamba/bin/python external_baseline_eval.py \
      --controller mamba \
      --crowdnav-root /workspace/nav_data/mamba/camrl/CrowdNav \
      --checkpoint /workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav/runs/mamba_vl/rl_model_ep9000_t24.pth \
      --human-count "${counts[$i]}" \
      --scenario "${sims[$i]}" \
      "${geometry[@]}" \
      --time-limit 25 \
      --episodes 100 \
      --case-offset 2400 \
      --output "/workspace/bayes_occ_mpc/$out/mamba_${labels[$i]}.json" \
      > "$out/mamba_${labels[$i]}.log" 2>&1 &
    mamba_pids+=("$!")
  done
  for pid in "${mamba_pids[@]}"; do
    wait "$pid"
  done
done
