# Corrected continuous-executor attribution

600 development episodes; fixed Gaussian belief and point 4. Timing is measured under two-worker concurrency.

| Scene | Arm | SR | audited CR | TR | penalized time s | plan p95 ms | fallback episodes | fallback steps / total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| circle5 | current | 98.0% | 2.0% | 0.0% | 10.485 | 60.09 | 7 | 18/4034 |
| circle5 | single_mode | 99.0% | 1.0% | 0.0% | 10.255 | 59.67 | 3 | 9/4022 |
| circle5 | brake | 94.0% | 6.0% | 0.0% | 11.072 | 60.28 | 7 | 13/3937 |
| square20 | current | 72.0% | 27.0% | 1.0% | 16.837 | 166.71 | 46 | 137/4511 |
| square20 | single_mode | 65.0% | 33.0% | 2.0% | 17.402 | 166.38 | 51 | 167/4255 |
| square20 | brake | 58.0% | 41.0% | 1.0% | 18.492 | 164.42 | 46 | 99/3980 |

## circle5__current_vs_single_mode

Rows=current, columns=variant.

| Event | reach_goal | collision | timeout |
|---|---:|---:|---:|
| reach_goal | 98 | 0 | 0 |
| collision | 1 | 1 | 0 |
| timeout | 0 | 0 | 0 |

```json
{
  "reach_goal": {
    "current_only": 0,
    "variant_only": 1,
    "variant_minus_current": 0.01,
    "bootstrap_ci95": [
      0.0,
      0.03
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "collision": {
    "current_only": 1,
    "variant_only": 0,
    "variant_minus_current": -0.01,
    "bootstrap_ci95": [
      -0.03,
      0.0
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "timeout": {
    "current_only": 0,
    "variant_only": 0,
    "variant_minus_current": 0.0,
    "bootstrap_ci95": [
      0.0,
      0.0
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "time": {
    "variant_minus_current": -0.23,
    "bootstrap_ci95": [
      -0.6475,
      0.115
    ]
  }
}
```

All changed cases: [{"case": 20179, "current": "collision", "variant": "reach_goal", "current_time": 25.0, "variant_time": 10.25}]

## circle5__current_vs_brake

Rows=current, columns=variant.

| Event | reach_goal | collision | timeout |
|---|---:|---:|---:|
| reach_goal | 94 | 4 | 0 |
| collision | 0 | 2 | 0 |
| timeout | 0 | 0 | 0 |

```json
{
  "reach_goal": {
    "current_only": 4,
    "variant_only": 0,
    "variant_minus_current": -0.04,
    "bootstrap_ci95": [
      -0.08,
      -0.01
    ],
    "mcnemar_p": 0.125,
    "holm_p_12_binary_comparisons": 1.0
  },
  "collision": {
    "current_only": 0,
    "variant_only": 4,
    "variant_minus_current": 0.04,
    "bootstrap_ci95": [
      0.01,
      0.08
    ],
    "mcnemar_p": 0.125,
    "holm_p_12_binary_comparisons": 1.0
  },
  "timeout": {
    "current_only": 0,
    "variant_only": 0,
    "variant_minus_current": 0.0,
    "bootstrap_ci95": [
      0.0,
      0.0
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "time": {
    "variant_minus_current": 0.5875,
    "bootstrap_ci95": [
      0.1425,
      1.1825
    ]
  }
}
```

All changed cases: [{"case": 20100, "current": "reach_goal", "variant": "collision", "current_time": 10.5, "variant_time": 25.0}, {"case": 20115, "current": "reach_goal", "variant": "collision", "current_time": 10.0, "variant_time": 25.0}, {"case": 20144, "current": "reach_goal", "variant": "collision", "current_time": 10.75, "variant_time": 25.0}, {"case": 20176, "current": "reach_goal", "variant": "collision", "current_time": 10.0, "variant_time": 25.0}]

## square20__current_vs_single_mode

Rows=current, columns=variant.

| Event | reach_goal | collision | timeout |
|---|---:|---:|---:|
| reach_goal | 61 | 9 | 2 |
| collision | 3 | 24 | 0 |
| timeout | 1 | 0 | 0 |

```json
{
  "reach_goal": {
    "current_only": 11,
    "variant_only": 4,
    "variant_minus_current": -0.07,
    "bootstrap_ci95": [
      -0.15,
      0.0
    ],
    "mcnemar_p": 0.11846923828124999,
    "holm_p_12_binary_comparisons": 1.0
  },
  "collision": {
    "current_only": 3,
    "variant_only": 9,
    "variant_minus_current": 0.06,
    "bootstrap_ci95": [
      0.0,
      0.13
    ],
    "mcnemar_p": 0.14599609375,
    "holm_p_12_binary_comparisons": 1.0
  },
  "timeout": {
    "current_only": 1,
    "variant_only": 2,
    "variant_minus_current": 0.01,
    "bootstrap_ci95": [
      -0.02,
      0.04
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "time": {
    "variant_minus_current": 0.565,
    "bootstrap_ci95": [
      -0.345,
      1.5
    ]
  }
}
```

All changed cases: [{"case": 21003, "current": "timeout", "variant": "reach_goal", "current_time": 25.0, "variant_time": 11.5}, {"case": 21005, "current": "reach_goal", "variant": "collision", "current_time": 10.5, "variant_time": 25.0}, {"case": 21027, "current": "reach_goal", "variant": "collision", "current_time": 13.25, "variant_time": 25.0}, {"case": 21032, "current": "reach_goal", "variant": "timeout", "current_time": 22.75, "variant_time": 25.0}, {"case": 21040, "current": "reach_goal", "variant": "collision", "current_time": 14.75, "variant_time": 25.0}, {"case": 21041, "current": "reach_goal", "variant": "collision", "current_time": 14.75, "variant_time": 25.0}, {"case": 21047, "current": "collision", "variant": "reach_goal", "current_time": 25.0, "variant_time": 10.25}, {"case": 21052, "current": "reach_goal", "variant": "collision", "current_time": 13.25, "variant_time": 25.0}, {"case": 21053, "current": "reach_goal", "variant": "collision", "current_time": 10.25, "variant_time": 25.0}, {"case": 21064, "current": "reach_goal", "variant": "collision", "current_time": 10.75, "variant_time": 25.0}, {"case": 21067, "current": "reach_goal", "variant": "collision", "current_time": 18.5, "variant_time": 25.0}, {"case": 21069, "current": "collision", "variant": "reach_goal", "current_time": 25.0, "variant_time": 14.25}, {"case": 21073, "current": "reach_goal", "variant": "collision", "current_time": 15.25, "variant_time": 25.0}, {"case": 21075, "current": "reach_goal", "variant": "timeout", "current_time": 16.5, "variant_time": 25.0}, {"case": 21076, "current": "collision", "variant": "reach_goal", "current_time": 25.0, "variant_time": 20.0}]

## square20__current_vs_brake

Rows=current, columns=variant.

| Event | reach_goal | collision | timeout |
|---|---:|---:|---:|
| reach_goal | 58 | 14 | 0 |
| collision | 0 | 27 | 0 |
| timeout | 0 | 0 | 1 |

```json
{
  "reach_goal": {
    "current_only": 14,
    "variant_only": 0,
    "variant_minus_current": -0.14,
    "bootstrap_ci95": [
      -0.21,
      -0.08
    ],
    "mcnemar_p": 0.0001220703125,
    "holm_p_12_binary_comparisons": 0.00146484375
  },
  "collision": {
    "current_only": 0,
    "variant_only": 14,
    "variant_minus_current": 0.14,
    "bootstrap_ci95": [
      0.08,
      0.21
    ],
    "mcnemar_p": 0.0001220703125,
    "holm_p_12_binary_comparisons": 0.00146484375
  },
  "timeout": {
    "current_only": 0,
    "variant_only": 0,
    "variant_minus_current": 0.0,
    "bootstrap_ci95": [
      0.0,
      0.0
    ],
    "mcnemar_p": 1.0,
    "holm_p_12_binary_comparisons": 1.0
  },
  "time": {
    "variant_minus_current": 1.655,
    "bootstrap_ci95": [
      0.88,
      2.5225624999999945
    ]
  }
}
```

All changed cases: [{"case": 21000, "current": "reach_goal", "variant": "collision", "current_time": 12.25, "variant_time": 25.0}, {"case": 21005, "current": "reach_goal", "variant": "collision", "current_time": 10.5, "variant_time": 25.0}, {"case": 21006, "current": "reach_goal", "variant": "collision", "current_time": 16.0, "variant_time": 25.0}, {"case": 21009, "current": "reach_goal", "variant": "collision", "current_time": 14.25, "variant_time": 25.0}, {"case": 21031, "current": "reach_goal", "variant": "collision", "current_time": 15.25, "variant_time": 25.0}, {"case": 21041, "current": "reach_goal", "variant": "collision", "current_time": 14.75, "variant_time": 25.0}, {"case": 21049, "current": "reach_goal", "variant": "collision", "current_time": 10.5, "variant_time": 25.0}, {"case": 21052, "current": "reach_goal", "variant": "collision", "current_time": 13.25, "variant_time": 25.0}, {"case": 21053, "current": "reach_goal", "variant": "collision", "current_time": 10.25, "variant_time": 25.0}, {"case": 21064, "current": "reach_goal", "variant": "collision", "current_time": 10.75, "variant_time": 25.0}, {"case": 21067, "current": "reach_goal", "variant": "collision", "current_time": 18.5, "variant_time": 25.0}, {"case": 21068, "current": "reach_goal", "variant": "collision", "current_time": 13.5, "variant_time": 25.0}, {"case": 21086, "current": "reach_goal", "variant": "collision", "current_time": 20.5, "variant_time": 25.0}, {"case": 21087, "current": "reach_goal", "variant": "collision", "current_time": 8.5, "variant_time": 25.0}]

Prefix agreement passed for all 200 current/brake pairs.
No equivalence or noninferiority claim is inferred from nonsignificance.
