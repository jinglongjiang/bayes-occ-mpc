#!/usr/bin/env bash
set -euo pipefail

out=results/cem_stability_2600_2699
mkdir -p "$out"
pids=()

for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 continuous_mpc_gate.py \
      --arms bayes \
      --human-counts 20 \
      --scenarios circle_crossing \
      --episodes 100 \
      --case-offset 2600 \
      --population 512 \
      --iterations 4 \
      --v-max 1.20 \
      --chance-limit 0.50 \
      --near-chance-limit 0.15 \
      --probability-weight 1.0 \
      --human-margin 0.10 \
      --planner-seed-offset "$seed" \
      --workers 4 \
      --output "$out/seed_$seed.json" > "$out/seed_$seed.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payloads = [json.loads((root / f"seed_{seed}.json").read_text()) for seed in range(3)]
rows = [{int(row["case_id"]): row for row in payload["episodes"]} for payload in payloads]
cases = sorted(rows[0])
if any(sorted(item) != cases for item in rows):
    raise SystemExit("seed runs are not paired")
output = {
    "summary": {f"seed_{i}": p["summary"][0] for i, p in enumerate(payloads)},
    "event_agreement_all_three": sum(
        len({item[case]["event"] for item in rows}) == 1 for case in cases
    ) / len(cases),
    "success_rate_range": [
        min(p["summary"][0]["sr"] for p in payloads),
        max(p["summary"][0]["sr"] for p in payloads),
    ],
    "collision_rate_range": [
        min(p["summary"][0]["cr"] for p in payloads),
        max(p["summary"][0]["cr"] for p in payloads),
    ],
}
(root / "analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True))
print(json.dumps(output, indent=2, sort_keys=True))
PY
